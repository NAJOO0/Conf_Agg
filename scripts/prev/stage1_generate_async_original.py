"""
Stage 1: 원시 데이터 생성 스크립트 (Async 버전 - 개선판)
- vLLM API 폴백 지원 (generate/get_next_response)
- 백프레셔 제어 (Semaphore)
- I/O 최적화 (파일 핸들 재사용, 배치 flush)
- Parquet 안정성 향상 (JSON 직렬화)
- Graceful shutdown 지원
- 메모리 모니터링
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import json
import asyncio
import signal

# PyArrow 임포트
import pyarrow as pa
import pyarrow.parquet as pq

# 메모리 모니터링
import psutil

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
from transformers import AutoTokenizer

# vLLM Async 엔진
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


logger = logging.getLogger(__name__)

# Graceful shutdown을 위한 전역 이벤트
shutdown_event = asyncio.Event()


def setup_environment_defaults(cfg: DictConfig):
    """
    환경변수 기본값 설정 (config 기반)
    사용자가 환경변수로 오버라이드하지 않은 경우에만 적용
    """
    # HuggingFace 캐시 설정
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = cfg.paths.huggingface_cache
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = cfg.paths.huggingface_cache
    
    # vLLM 로깅 레벨
    if "VLLM_LOGGING_LEVEL" not in os.environ:
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    
    # 기본 SNAPSHOT_EVERY (config에서 가져올 수 있다면)
    if "SNAPSHOT_EVERY" not in os.environ:
        os.environ.setdefault("SNAPSHOT_EVERY", "50")
    
    # 기본 FLUSH_EVERY
    if "FLUSH_EVERY" not in os.environ:
        os.environ.setdefault("FLUSH_EVERY", "100")
    
    # 재시작 기능 기본 활성화
    if "RESUME" not in os.environ:
        os.environ.setdefault("RESUME", "true")


def signal_handler(sig, frame):
    """SIGINT/SIGTERM 시그널 핸들러"""
    logger.warning(f"⚠️  Shutdown 신호 수신 (signal={sig})")
    shutdown_event.set()


def simple_extract_topk(gen_logprobs: List[Dict[int, Any]], k: int) -> List[List[float]]:
    """최적화된 logprob 추출 함수 (float32로 메모리 절약)"""
    if not gen_logprobs:
        return []
    
    results = []
    for token_step_dict in gen_logprobs:
        if not token_step_dict:
            results.append([])
            continue
        
        lps = []
        for i, entry in enumerate(token_step_dict.values()):
            if i >= k:
                break
            if hasattr(entry, "logprob"):
                lps.append(float(entry.logprob))
            elif isinstance(entry, dict) and "logprob" in entry:
                lps.append(float(entry["logprob"]))
        
        if lps:
            # float32 변환 (float16보다 안정적이고 Parquet 호환)
            results.append(np.array(lps, dtype=np.float32).tolist())
        else:
            results.append([])
    
    return results


def compute_prompt_token_counts(
    tokenizer: AutoTokenizer, 
    texts: List[str], 
    batch_size: int = 100
) -> List[int]:
    """
    프롬프트 토큰 카운트 계산 (배치 처리, 메모리 효율적)
    """
    prompt_token_counts = []
    
    # return_length 지원 여부 확인
    supports_return_length = False
    try:
        test_result = tokenizer(
            ["test"], 
            add_special_tokens=False, 
            return_length=True
        )
        if "length" in test_result:
            supports_return_length = True
    except (TypeError, KeyError):
        pass
    
    # 배치 단위로 처리
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            if supports_return_length:
                enc = tokenizer(
                    batch, 
                    add_special_tokens=False, 
                    return_length=True
                )
                prompt_token_counts.extend([int(x) for x in enc["length"]])
            else:
                enc = tokenizer(batch, add_special_tokens=False)
                prompt_token_counts.extend([len(ids) for ids in enc.input_ids])
        except Exception as e:
            logger.warning(f"토큰 카운트 계산 실패 (배치 {i//batch_size}): {e}")
            # 폴백: 개별 처리
            for text in batch:
                try:
                    enc = tokenizer([text], add_special_tokens=False)
                    prompt_token_counts.append(len(enc.input_ids[0]))
                except Exception:
                    prompt_token_counts.append(0)
    
    return prompt_token_counts


def apply_chat_template_safe(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
) -> str:
    """
    안전한 chat template 적용 (enable_thinking 호환성 처리)
    """
    try:
        # Qwen2.5 등에서 enable_thinking 지원
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        # enable_thinking 미지원 토크나이저
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    return text


def create_engine_args(cfg: DictConfig) -> AsyncEngineArgs:
    """
    AsyncEngineArgs 생성 (버전 호환성 고려)
    Config 파일의 모든 설정을 반영
    """
    vllm_config = cfg.data.raw_dataset.vllm
    
    # 필수 필드
    engine_kwargs = {
        "model": cfg.model.base_model,
        "tensor_parallel_size": vllm_config.tensor_parallel_size,
        "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
        "max_model_len": vllm_config.max_model_len,
        "dtype": vllm_config.dtype,
        "trust_remote_code": vllm_config.trust_remote_code,
        "max_num_batched_tokens": vllm_config.max_num_batched_tokens,
        "max_num_seqs": vllm_config.max_num_seqs,
        "enforce_eager": vllm_config.enforce_eager,
    }
    
    # 선택적 필드 (버전별로 있을 수도 없을 수도 있음)
    optional_fields = [
        "disable_custom_all_reduce",
        "disable_log_stats",
        "kv_cache_dtype",
        "enable_prefix_caching",
    ]
    
    for field in optional_fields:
        if field in vllm_config and vllm_config[field] is not None:
            engine_kwargs[field] = vllm_config[field]
    
    return AsyncEngineArgs(**engine_kwargs)


def create_parquet_schema() -> pa.Schema:
    """
    명시적 Parquet 스키마 정의 (타입 불일치 방지)
    
    Note: Confidence scores는 ConfidenceCalculator가 동적으로 생성하므로
    여기서는 기본 필드만 정의하고, 실제 스키마는 첫 배치에서 추론
    """
    return pa.schema([
        ("problem_id", pa.string()),
        ("problem_text", pa.string()),
        ("ground_truth", pa.string()),
        ("response_id", pa.string()),
        ("generated_text", pa.string()),
        ("output_token_count", pa.int32()),
        ("prompt_token_count", pa.int32()),
        ("total_token_count", pa.int32()),
        ("logprobs", pa.string()),  # JSON 문자열로 저장
        ("worker_gpu", pa.string()),
        ("worker_replica", pa.string()),
        # Confidence scores는 동적으로 추가됨
        # ConfidenceCalculator.calculate_all_confidence_scores()의 반환값 참조
    ])


class FileHandlers:
    """
    파일 핸들러 관리 클래스 (안전한 리소스 관리)
    """
    def __init__(self, jsonl_path: str, parquet_path: str, flush_every: int = 100):
        self.jsonl_path = jsonl_path
        self.parquet_path = parquet_path
        self.flush_every = flush_every
        
        self.jsonl_f: Optional[Any] = None
        self.jsonl_fd: Optional[int] = None
        self.parquet_writer: Optional[pq.ParquetWriter] = None
        self.parquet_schema: Optional[pa.Schema] = None
        
        self.since_flush = 0
        self.is_closed = False
    
    def open_jsonl(self):
        """JSONL 파일 열기"""
        if self.jsonl_f is None:
            self.jsonl_f = open(
                self.jsonl_path, 
                "a", 
                encoding="utf-8",
                buffering=8192  # 8KB 버퍼
            )
            self.jsonl_fd = self.jsonl_f.fileno()
    
    def write_jsonl(self, row_str: str):
        """JSONL에 한 줄 쓰기 (버퍼링)"""
        self.open_jsonl()
        self.jsonl_f.write(row_str + "\n")
        self.since_flush += 1
        
        # 주기적 flush/fsync
        if self.since_flush >= self.flush_every:
            self.flush_jsonl()
    
    def flush_jsonl(self):
        """JSONL 버퍼 강제 플러시"""
        if self.jsonl_f is not None:
            self.jsonl_f.flush()
            os.fsync(self.jsonl_fd)
            self.since_flush = 0
    
    def open_parquet(self):
        """Parquet writer 초기화 (스키마는 첫 배치에서 추론)"""
        # 스키마가 아직 없으면 나중에 첫 배치에서 추론
        pass
    
    def write_parquet_batch(self, rows: List[Dict[str, Any]]):
        """Parquet에 배치 쓰기 (첫 배치에서 스키마 자동 추론)"""
        if not rows:
            return
        
        try:
            # DataFrame 생성
            df = pd.DataFrame(rows)
            
            # 첫 배치에서 스키마 추론 및 writer 생성
            if self.parquet_writer is None:
                self.parquet_schema = pa.Table.from_pandas(df).schema
                self.parquet_writer = pq.ParquetWriter(
                    self.parquet_path, 
                    self.parquet_schema, 
                    compression="zstd"
                )
                logger.info(f"Parquet 스키마 추론 완료: {len(self.parquet_schema)} 컬럼")
            
            # Arrow Table 생성 및 쓰기
            table = pa.Table.from_pandas(df, schema=self.parquet_schema)
            self.parquet_writer.write_table(table)
        
        except Exception as e:
            logger.error(f"Parquet 쓰기 실패: {e}")
            if rows:
                logger.error(f"실패 데이터 샘플: {json.dumps(rows[0], indent=2, ensure_ascii=False)[:300]}")
    
    def close(self):
        """모든 파일 핸들러 안전하게 닫기"""
        if self.is_closed:
            return
        
        # JSONL 닫기
        if self.jsonl_f is not None:
            self.flush_jsonl()
            self.jsonl_f.close()
            self.jsonl_f = None
        
        # Parquet 닫기
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
        
        self.is_closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


async def feed_requests(
    engine: AsyncLLMEngine,
    texts: List[str],
    sampling_params: SamplingParams,
    shard_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    비동기로 모든 요청을 엔진에 등록 (백프레셔 제어)
    """
    logger.info(f"[Shard {shard_id}] 비동기 요청 등록 시작...")
    
    for i, prompt in enumerate(texts):
        # Shutdown 체크
        if shutdown_event.is_set():
            logger.warning(f"[Shard {shard_id}] Shutdown 신호 감지, 요청 등록 중단")
            break
        
        request_id = f"req_{shard_id}_{i}"
        
        # Semaphore 획득 (in-flight 요청 수 제한)
        await semaphore.acquire()
        
        try:
            await engine.add_request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
        except Exception as e:
            logger.error(f"[Shard {shard_id}] 요청 추가 실패 (Req ID: {request_id}): {e}")
            semaphore.release()  # 실패 시 슬롯 반환
    
    logger.info(f"[Shard {shard_id}] 요청 등록 완료: {len(texts)} 건")


async def collect_results(
    engine: AsyncLLMEngine,
    request_id_to_index: Dict[str, int],
    problems: List[Dict],
    prompt_token_counts: List[int],
    confidence_calculator: ConfidenceCalculator,
    file_handlers: FileHandlers,
    existing_response_ids: set,
    gen_cfg_logprobs: int,
    shard_id: int,
    gpu_id: str,
    semaphore: asyncio.Semaphore,
    snapshot_every: int,
) -> Dict[str, int]:
    """
    비동기로 결과 수집 및 실시간 저장
    """
    finished_requests = 0
    total_appended = 0
    total_skipped = 0
    
    # vLLM API 버전 호환성 체크
    use_generate = hasattr(engine, "generate") and callable(getattr(engine, "generate"))
    
    logger.info(f"[Shard {shard_id}] 결과 수집 시작 (API: {'generate' if use_generate else 'get_next_response'})")
    
    try:
        if use_generate:
            # vLLM 0.6.x 스타일: async for loop
            async for request_output in engine.generate():
                if shutdown_event.is_set():
                    logger.warning(f"[Shard {shard_id}] Shutdown 신호 감지, 결과 수집 중단")
                    break
                
                # 요청 처리
                stats = await process_request_output(
                    request_output=request_output,
                    request_id_to_index=request_id_to_index,
                    problems=problems,
                    prompt_token_counts=prompt_token_counts,
                    confidence_calculator=confidence_calculator,
                    file_handlers=file_handlers,
                    existing_response_ids=existing_response_ids,
                    gen_cfg_logprobs=gen_cfg_logprobs,
                    shard_id=shard_id,
                    gpu_id=gpu_id,
                )
                
                finished_requests += 1
                total_appended += stats["appended"]
                total_skipped += stats["skipped"]
                
                # Semaphore 반환 (in-flight 슬롯 회수)
                semaphore.release()
                
                # 주기적 진행 상황 로깅
                if snapshot_every > 0 and finished_requests % snapshot_every == 0:
                    await log_progress(
                        shard_id=shard_id,
                        finished_requests=finished_requests,
                        total_requests=len(request_id_to_index),
                        total_appended=total_appended,
                        total_skipped=total_skipped,
                    )
        
        else:
            # vLLM 0.10.x 스타일: get_next_response()
            while True:
                if shutdown_event.is_set():
                    logger.warning(f"[Shard {shard_id}] Shutdown 신호 감지, 결과 수집 중단")
                    break
                
                request_output = await engine.get_next_response()
                
                if request_output is None:
                    break
                
                # 요청 처리
                stats = await process_request_output(
                    request_output=request_output,
                    request_id_to_index=request_id_to_index,
                    problems=problems,
                    prompt_token_counts=prompt_token_counts,
                    confidence_calculator=confidence_calculator,
                    file_handlers=file_handlers,
                    existing_response_ids=existing_response_ids,
                    gen_cfg_logprobs=gen_cfg_logprobs,
                    shard_id=shard_id,
                    gpu_id=gpu_id,
                )
                
                finished_requests += 1
                total_appended += stats["appended"]
                total_skipped += stats["skipped"]
                
                # Semaphore 반환
                semaphore.release()
                
                # 주기적 진행 상황 로깅
                if snapshot_every > 0 and finished_requests % snapshot_every == 0:
                    await log_progress(
                        shard_id=shard_id,
                        finished_requests=finished_requests,
                        total_requests=len(request_id_to_index),
                        total_appended=total_appended,
                        total_skipped=total_skipped,
                    )
    
    except Exception as e:
        logger.error(f"[Shard {shard_id}] 결과 수집 중 오류: {e}", exc_info=True)
    
    finally:
        # 마지막 flush
        file_handlers.flush_jsonl()
    
    return {
        "finished_requests": finished_requests,
        "total_appended": total_appended,
        "total_skipped": total_skipped,
    }


async def process_request_output(
    request_output,
    request_id_to_index: Dict[str, int],
    problems: List[Dict],
    prompt_token_counts: List[int],
    confidence_calculator: ConfidenceCalculator,
    file_handlers: FileHandlers,
    existing_response_ids: set,
    gen_cfg_logprobs: int,
    shard_id: int,
    gpu_id: str,
) -> Dict[str, int]:
    """
    단일 요청 출력 처리
    """
    req_id = request_output.request_id
    
    if req_id not in request_id_to_index:
        logger.warning(f"알 수 없는 request_id: {req_id}")
        return {"appended": 0, "skipped": 0}
    
    idx = request_id_to_index[req_id]
    base_meta = problems[idx]
    prompt_tokens = prompt_token_counts[idx] if idx < len(prompt_token_counts) else 0
    
    # 배치 결과 준비
    batch_results_json = []
    batch_results_arrow = []
    
    appended = 0
    skipped = 0
    
    # 각 출력 처리
    for oi, gen in enumerate(request_output.outputs):
        # logprob 추출
        topk = simple_extract_topk(gen.logprobs, gen_cfg_logprobs)
        
        # 신뢰도 계산
        confidence_scores = confidence_calculator.calculate_all_confidence_scores(topk)
        
        response_id = f"{base_meta['problem_id']}_resp_{oi}"
        
        # 재시작 시 기존 결과 스킵
        if response_id in existing_response_ids:
            skipped += 1
            continue
        
        # 토큰 카운트
        output_token_count = len(gen.token_ids) if hasattr(gen, "token_ids") else 0
        total_token_count = prompt_tokens + output_token_count
        
        # 결과 행 생성
        row = {
            "problem_id": base_meta["problem_id"],
            "problem_text": base_meta["problem_text"],
            "ground_truth": base_meta["ground_truth"],
            "response_id": response_id,
            "generated_text": gen.text,
            "output_token_count": output_token_count,
            "prompt_token_count": prompt_tokens,
            "total_token_count": total_token_count,
            "logprobs": json.dumps(topk, ensure_ascii=False),  # JSON 문자열로 저장
            "worker_gpu": gpu_id,
            "worker_replica": f"shard_{shard_id}",
            **confidence_scores,
        }
        
        batch_results_json.append(json.dumps(row, ensure_ascii=False))
        batch_results_arrow.append(row)
        appended += 1
        
        # 재시작 세트에 즉시 추가
        existing_response_ids.add(response_id)
    
    # JSONL 실시간 저장
    if batch_results_json:
        for row_str in batch_results_json:
            file_handlers.write_jsonl(row_str)
    
    # Parquet 실시간 저장
    if batch_results_arrow:
        file_handlers.write_parquet_batch(batch_results_arrow)
    
    return {"appended": appended, "skipped": skipped}


async def log_progress(
    shard_id: int,
    finished_requests: int,
    total_requests: int,
    total_appended: int,
    total_skipped: int,
):
    """
    진행 상황 로깅 (메모리 모니터링 포함)
    """
    # 메모리 사용량
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    
    logger.info(
        f"[Shard {shard_id}] 진행: {finished_requests}/{total_requests} 요청 처리 완료 "
        f"(저장: {total_appended}건, 스킵: {total_skipped}건) | "
        f"메모리: {mem_gb:.2f} GB"
    )


def load_existing_response_ids(
    parquet_path: str,
    jsonl_path: str,
    shard_id: int,
    resume_enabled: bool,
) -> set:
    """
    재시작을 위한 기존 결과 로드
    """
    existing_response_ids = set()
    
    if not resume_enabled:
        return existing_response_ids
    
    # 1. Parquet 우선 시도 (더 빠름)
    if os.path.exists(parquet_path):
        try:
            logger.info(f"[Shard {shard_id}] 기존 Parquet 발견, 재시작 활성화: {parquet_path}")
            df_existing = pd.read_parquet(parquet_path, columns=["response_id"])
            existing_response_ids = set(df_existing["response_id"])
            logger.info(f"Parquet에서 기존 결과 로드: {len(existing_response_ids)}개")
            return existing_response_ids
        except Exception as e:
            logger.warning(f"기존 Parquet 로드 실패, JSONL로 대체: {e}")
            existing_response_ids.clear()
    
    # 2. JSONL 폴백
    if os.path.exists(jsonl_path):
        logger.info(f"[Shard {shard_id}] 기존 JSONL 로드: {jsonl_path}")
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "response_id" in obj:
                            existing_response_ids.add(obj["response_id"])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"JSONL에서 기존 결과 로드: {len(existing_response_ids)}개")
        except Exception as e:
            logger.warning(f"기존 JSONL 로드 실패: {e}")
    
    return existing_response_ids


async def main_worker_async(cfg: DictConfig, args: argparse.Namespace) -> None:
    """
    Async 버전 메인 워커 함수 (개선판)
    """
    # 환경변수 기본값 설정 (config 기반)
    setup_environment_defaults(cfg)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 로깅 설정 (config 기반)
    log_file = os.path.join(
        cfg.paths.log_dir, 
        f"stage1_generate_async_shard_{args.shard_id}.log"
    )
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags + [f"shard_{args.shard_id}"]  # 샤드 태그 추가
    )
    
    app_logger = logging.getLogger("conf_agg_llm")
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.INFO)
    for h in app_logger.handlers:
        vllm_logger.addHandler(h)
    vllm_logger.propagate = False
    vllm_logger.info("[DIAG] vLLM async logger 연결 완료.")
    
    app_logger.info(f"🚀 [Shard {args.shard_id} | GPU {args.gpu_id}] Stage 1 (Async 개선판) 시작")
    app_logger.info(f"=" * 80)
    app_logger.info(f"프로젝트: {cfg.project.name} v{cfg.project.version}")
    app_logger.info(f"모델: {cfg.model.base_model}")
    app_logger.info(f"데이터 경로: {cfg.paths.data_dir}")
    app_logger.info(f"출력 경로: {cfg.paths.output_dir}")
    app_logger.info(f"=" * 80)
    app_logger.info(f"vLLM 설정:")
    app_logger.info(f"  - max_model_len: {cfg.data.raw_dataset.vllm.max_model_len}")
    app_logger.info(f"  - max_num_seqs: {cfg.data.raw_dataset.vllm.max_num_seqs}")
    app_logger.info(f"  - gpu_memory_utilization: {cfg.data.raw_dataset.vllm.gpu_memory_utilization}")
    app_logger.info(f"  - dtype: {cfg.data.raw_dataset.vllm.dtype}")
    app_logger.info(f"  - kv_cache_dtype: {cfg.data.raw_dataset.vllm.get('kv_cache_dtype', 'N/A')}")
    app_logger.info(f"생성 설정:")
    app_logger.info(f"  - num_responses_per_problem: {cfg.data.raw_dataset.generation.num_responses_per_problem}")
    app_logger.info(f"  - temperature: {cfg.data.raw_dataset.generation.temperature}")
    app_logger.info(f"  - max_tokens: {cfg.data.raw_dataset.generation.max_tokens}")
    app_logger.info(f"=" * 80)
    if cfg.experiment.get("log_level") != "INFO":
        app_logger.debug(f"전체 설정:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # 1. AsyncLLMEngine 초기화
        app_logger.info(f"[Shard {args.shard_id}] AsyncLLMEngine 로드 중: {cfg.model.base_model}")
        
        engine_args = create_engine_args(cfg)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=cfg.model.trust_remote_code
        )
        app_logger.info(f"[Shard {args.shard_id}] AsyncLLMEngine 로드 완료.")
        
        # 2. 출력 디렉토리 생성
        output_dir = os.path.join(cfg.paths.data_dir, "generated")
        sample_limit_env = os.environ.get("SAMPLE_LIMIT")
        if sample_limit_env:
            output_dir = os.path.join(output_dir, f"sample_{sample_limit_env}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 원본 데이터셋 로드
        raw_data_path = os.path.join(cfg.paths.data_dir, "raw", "deepscaler.jsonl")
        if not os.path.exists(raw_data_path):
            app_logger.error(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
            return
        
        raw_dataset = RawDataset(raw_data_path)
        app_logger.info(f"전체 원본 데이터셋 로드 완료: {len(raw_dataset)}개 문제")
        
        # 4. 샘플링 설정
        sample_limit = int(sample_limit_env) if sample_limit_env and sample_limit_env.isdigit() else 0
        
        if sample_limit > 0 and sample_limit < len(raw_dataset):
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(raw_dataset), 
                size=sample_limit, 
                replace=False
            )
            selected_indices = sorted(selected_indices)
            app_logger.info(f"SAMPLE_LIMIT 적용: 전체 {len(raw_dataset)}개 중 {sample_limit}개 선택")
            app_logger.info(f"선택된 인덱스 범위: {min(selected_indices)} ~ {max(selected_indices)}")
        else:
            selected_indices = list(range(len(raw_dataset)))
            app_logger.info(f"전체 데이터셋 사용: {len(raw_dataset)}개 문제")
        
        # 5. 문제/프롬프트 구성
        problems: List[Dict] = []
        texts: List[str] = []
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        
        app_logger.info("전체 입력 텍스트 생성 중...")
        for idx in selected_indices:
            problem_data = raw_dataset[idx]
            problem_id = problem_data.get("id", f"problem_{idx}")
            problem_text = problem_data.get("problem", "")
            ground_truth = problem_data.get("answer", "")
            
            messages = [{"role": "user", "content": f"{problem_text}\n\n{instruction}"}]
            text = apply_chat_template_safe(tokenizer, messages)
            
            problems.append({
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
            })
            texts.append(text)
        
        app_logger.info(f"총 {len(texts)}개 프롬프트 준비 완료.")
        
        # 6. 작업 분할 (Sharding)
        my_problems = problems[args.shard_id::args.total_shards]
        my_texts = texts[args.shard_id::args.total_shards]
        app_logger.info(
            f"[Shard {args.shard_id}] 작업 분할 완료: "
            f"{len(my_texts)}개 (1/{args.total_shards})"
        )
        
        # 7. 프롬프트 토큰 카운트 계산 (배치 처리)
        app_logger.info(f"[Shard {args.shard_id}] 프롬프트 토큰 카운트 계산 중...")
        prompt_token_counts = compute_prompt_token_counts(tokenizer, my_texts)
        app_logger.info(f"토큰 카운트 계산 완료.")
        
        # 8. 샘플링 파라미터 구성 (config 기반)
        gen_cfg = cfg.data.raw_dataset.generation
        sampling_params = SamplingParams(
            n=gen_cfg.num_responses_per_problem,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            min_p=gen_cfg.min_p,
            max_tokens=gen_cfg.max_tokens,
            logprobs=gen_cfg.logprobs if gen_cfg.logprobs > 0 else None,  # 0이면 None
            presence_penalty=gen_cfg.presence_penalty,
        )
        gen_cfg_logprobs = gen_cfg.logprobs if gen_cfg.logprobs > 0 else 0
        
        app_logger.info(f"샘플링 파라미터: n={gen_cfg.num_responses_per_problem}, "
                       f"temp={gen_cfg.temperature}, top_p={gen_cfg.top_p}, "
                       f"max_tokens={gen_cfg.max_tokens}, logprobs={gen_cfg_logprobs}")
        
        # 9. 저장 경로 및 옵션 설정
        jsonl_path = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}.jsonl")
        parquet_path = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}.parquet")
        snapshot_every = int(os.environ.get("SNAPSHOT_EVERY", "50"))
        flush_every = int(os.environ.get("FLUSH_EVERY", "100"))
        resume_enabled = os.environ.get("RESUME", "true").lower() in ("1", "true", "yes")
        
        # 10. 재시작 로직: 기존 결과 로드
        existing_response_ids = load_existing_response_ids(
            parquet_path=parquet_path,
            jsonl_path=jsonl_path,
            shard_id=args.shard_id,
            resume_enabled=resume_enabled,
        )
        
        # 11. 요청 식별자 맵핑 생성
        request_id_to_index: Dict[str, int] = {}
        for i in range(len(my_texts)):
            request_id = f"req_{args.shard_id}_{i}"
            request_id_to_index[request_id] = i
        
        # 12. 백프레셔 제어 (Semaphore)
        # config의 max_num_seqs를 기준으로 MAX_INFLIGHT 계산
        max_num_seqs = cfg.data.raw_dataset.vllm.max_num_seqs
        # 환경변수로 오버라이드 가능, 기본값은 max_num_seqs의 2배
        max_inflight = int(os.environ.get("MAX_INFLIGHT", str(max_num_seqs * 2)))
        semaphore = asyncio.Semaphore(max_inflight)
        
        app_logger.info(f"백프레셔 설정: MAX_INFLIGHT={max_inflight}, max_num_seqs={max_num_seqs}")
        
        # 13. 신뢰도 계산기 초기화 (config 기반)
        confidence_calculator = ConfidenceCalculator(
            group_size=cfg.data.raw_dataset.confidence.group_size
        )
        app_logger.info(f"신뢰도 계산기 초기화: group_size={cfg.data.raw_dataset.confidence.group_size}")
        
        # 14. 파일 핸들러 초기화
        with FileHandlers(jsonl_path, parquet_path, flush_every) as file_handlers:
            # 15. Feeder와 Collector 동시 실행
            feeder_task = asyncio.create_task(
                feed_requests(
                    engine=engine,
                    texts=my_texts,
                    sampling_params=sampling_params,
                    shard_id=args.shard_id,
                    semaphore=semaphore,
                )
            )
            
            collector_task = asyncio.create_task(
                collect_results(
                    engine=engine,
                    request_id_to_index=request_id_to_index,
                    problems=my_problems,
                    prompt_token_counts=prompt_token_counts,
                    confidence_calculator=confidence_calculator,
                    file_handlers=file_handlers,
                    existing_response_ids=existing_response_ids,
                    gen_cfg_logprobs=gen_cfg_logprobs,
                    shard_id=args.shard_id,
                    gpu_id=args.gpu_id,
                    semaphore=semaphore,
                    snapshot_every=snapshot_every,
                )
            )
            
            # 두 태스크 완료 대기
            app_logger.info(f"[Shard {args.shard_id}] Feeder와 Collector 시작...")
            feeder_result, collector_result = await asyncio.gather(
                feeder_task, 
                collector_task,
                return_exceptions=True
            )
            
            # 예외 처리
            if isinstance(feeder_result, Exception):
                app_logger.error(f"Feeder 실패: {feeder_result}", exc_info=feeder_result)
            if isinstance(collector_result, Exception):
                app_logger.error(f"Collector 실패: {collector_result}", exc_info=collector_result)
            
            app_logger.info(f"[Shard {args.shard_id}] Feeder와 Collector 완료.")
        
        # 16. 최종 통계 출력
        if isinstance(collector_result, dict):
            finished_requests = collector_result["finished_requests"]
            total_appended = collector_result["total_appended"]
            total_skipped = collector_result["total_skipped"]
            
            app_logger.info(f"✅ [Shard {args.shard_id}] Async Stage 1 완료")
            app_logger.info(f"저장 위치: {parquet_path}")
            app_logger.info(
                f"처리 완료: {finished_requests}개 요청, "
                f"{total_appended}행 저장, {total_skipped}행 스킵"
            )
        
        # 17. 최종 데이터프레임 통계
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            app_logger.info(f"최종 데이터프레임 크기: {len(df)}행")
            app_logger.info(f"문제 수: {df['problem_id'].nunique()}")
            
            if len(df) > 0:
                app_logger.info(
                    f"문제당 평균 응답 수: "
                    f"{len(df) / df['problem_id'].nunique():.1f}"
                )
                
                # 토큰 통계
                if 'total_token_count' in df.columns:
                    try:
                        df_tokens = df['total_token_count'].fillna(0)
                        total_tokens = int(df_tokens.sum())
                        mean_tokens = float(df_tokens.mean())
                        min_tokens = int(df_tokens.min())
                        max_tokens = int(df_tokens.max())
                        
                        app_logger.info(f"전체 토큰 수: {total_tokens:,}")
                        app_logger.info(f"평균 토큰 수: {mean_tokens:.1f}")
                        app_logger.info(f"토큰 수 범위: {min_tokens:,} ~ {max_tokens:,}")
                        
                        # max_tokens 도달 여부 (config 기반)
                        # prompt + generated tokens가 max_model_len에 근접했는지 체크
                        max_model_len = cfg.data.raw_dataset.vllm.max_model_len
                        near_limit_threshold = max_model_len * 0.95  # 95% 도달
                        near_limit_count = int((df_tokens >= near_limit_threshold).sum())
                        app_logger.info(
                            f"토큰 한계 근접 인스턴스 (>={near_limit_threshold:.0f}): {near_limit_count}건"
                        )
                    except Exception as e:
                        app_logger.warning(f"토큰 통계 계산 실패: {e}")
                
                # 샘플 출력
                app_logger.info("=" * 80)
                app_logger.info("샘플 결과 출력 (첫 번째 인스턴스):")
                app_logger.info("=" * 80)
                sample = df.iloc[0]
                app_logger.info(f"Problem ID: {sample['problem_id']}")
                app_logger.info(f"Generated Text: {sample['generated_text'][:500]}...")
                app_logger.info(f"Total Token Count: {sample.get('total_token_count', 'N/A')}")
                
                # Confidence scores (config에 정의된 메서드 기준)
                conf_cols = [col for col in df.columns if col.startswith("confidence_")]
                if conf_cols:
                    app_logger.info("Confidence Scores:")
                    for col in conf_cols:
                        val = sample.get(col, 'N/A')
                        if isinstance(val, (int, float)):
                            app_logger.info(f"  {col}: {val:.4f}")
                        else:
                            app_logger.info(f"  {col}: {val}")
                    
                    # 설정된 메서드 확인
                    expected_methods = cfg.data.raw_dataset.confidence.methods
                    app_logger.info(f"Config에 정의된 신뢰도 메서드: {expected_methods}")
                
                app_logger.info("=" * 80)
    
    except Exception as e:
        app_logger.error(
            f"[Shard {args.shard_id}] 실행 중 오류 발생: {e}", 
            exc_info=True
        )
        raise


if __name__ == "__main__":
    # argparse로 런처의 인수를 받음
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", 
        type=str, 
        required=True, 
        help="Hydra config directory (e.g., ../config)"
    )
    parser.add_argument(
        "--config-name", 
        type=str, 
        required=True, 
        help="Hydra config name (e.g., config)"
    )
    parser.add_argument(
        "--gpu-id", 
        type=str, 
        required=True, 
        help="GPU ID (e.g., '0')"
    )
    parser.add_argument(
        "--shard-id", 
        type=int, 
        required=True, 
        help="Data shard index (0, 1, 2, 3)"
    )
    parser.add_argument(
        "--total-shards", 
        type=int, 
        default=4, 
        help="Total number of shards"
    )
    args = parser.parse_args()
    
    # GPU 격리
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Hydra 초기화
    config_dir = Path(args.config_path).resolve()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    
    cfg = hydra.compose(config_name=args.config_name)
    
    # 비동기 메인 워커 실행
    try:
        asyncio.run(main_worker_async(cfg, args))
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"비동기 메인 워커 실행 실패: {e}", exc_info=True)
        sys.exit(1)