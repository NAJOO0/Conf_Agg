"""
Stage 1: API 기반 원시 데이터 생성 스크립트
vLLM API 서버에 비동기 요청을 보내고 스트리밍으로 저장
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
import asyncio
import aiohttp
import aiofiles
import json
from datetime import datetime
from tqdm.asyncio import tqdm
import time
from dataclasses import dataclass
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RequestData:
    """요청 데이터 클래스"""
    problem_id: str
    problem_text: str
    ground_truth: str
    messages: List[Dict[str, str]]
    server_url: str
    retry_count: int = 0


def simple_extract_topk(gen_logprobs: List[Dict], k: int) -> List[List[float]]:
    """API 응답에서 logprob 추출"""
    if not gen_logprobs:
        return []
    
    results = []
    for token_info in gen_logprobs:
        if not token_info or 'top_logprobs' not in token_info:
            results.append([])
            continue
        
        top_logprobs = token_info['top_logprobs']
        if isinstance(top_logprobs, list):
            # 리스트 형태의 logprobs
            lps = [item.get('logprob', 0.0) for item in top_logprobs[:k]]
        elif isinstance(top_logprobs, dict):
            # 딕셔너리 형태의 logprobs
            lps = list(top_logprobs.values())[:k]
        else:
            lps = []
        
        if lps:
            results.append(np.array(lps, dtype=np.float16).tolist())
        else:
            results.append([])
    
    return results


async def make_api_request(
    session: aiohttp.ClientSession,
    request_data: RequestData,
    sampling_params: Dict,
    timeout: int = 900
) -> Optional[Dict]:
    """vLLM API 서버에 단일 요청"""
    
    payload = {
        "model": sampling_params.get("model", "default"),
        "messages": request_data.messages,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "top_k": sampling_params["top_k"],
        "max_tokens": sampling_params["max_tokens"],
        "n": sampling_params["n"],
        "logprobs": sampling_params["logprobs"],
        "presence_penalty": sampling_params.get("presence_penalty", 0.0),
        "stream": False  # 스트리밍 비활성화 (완료된 응답만 받기)
    }
    
    try:
        async with session.post(
            f"{request_data.server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "request_data": request_data,
                    "response": result
                }
            else:
                logger.warning(f"API 오류 (상태 {response.status}): {request_data.problem_id}")
                return None
                
    except asyncio.TimeoutError:
        logger.warning(f"요청 타임아웃: {request_data.problem_id}")
        return None
    except Exception as e:
        logger.error(f"요청 실패: {request_data.problem_id} - {e}")
        return None


async def request_worker(
    session: aiohttp.ClientSession,
    request_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    sampling_params: Dict,
    max_retries: int = 3
):
    """요청 워커 - 큐에서 요청을 가져와 처리"""
    
    while True:
        try:
            request_data = await request_queue.get()
            
            if request_data is None:  # 종료 신호
                break
            
            # API 요청
            result = await make_api_request(session, request_data, sampling_params)
            
            if result is None and request_data.retry_count < max_retries:
                # 재시도
                request_data.retry_count += 1
                await asyncio.sleep(1)  # 잠시 대기
                await request_queue.put(request_data)
            elif result:
                # 성공
                await result_queue.put(result)
            else:
                # 최종 실패
                logger.error(f"최종 실패: {request_data.problem_id}")
                await result_queue.put({
                    "request_data": request_data,
                    "response": None,
                    "error": True
                })
            
            request_queue.task_done()
            
        except Exception as e:
            logger.error(f"워커 오류: {e}")


async def result_processor(
    result_queue: asyncio.Queue,
    output_file: aiofiles.threadpool.AsyncTextIOWrapper,
    confidence_calculator: ConfidenceCalculator,
    gen_cfg_logprobs: int,
    args: argparse.Namespace,
    total: int
):
    """결과 처리 및 저장 워커"""
    
    processed_count = 0
    buffer = []
    buffer_size = 100
    
    pbar = tqdm(total=total, desc=f"Shard {args.shard_id}")
    
    while True:
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # 버퍼에 데이터가 있으면 저장
            if buffer and processed_count >= total:
                for item in buffer:
                    await output_file.write(json.dumps(item) + '\n')
                await output_file.flush()
                buffer.clear()
                break
            continue
        
        if result is None:  # 종료 신호
            break
        
        request_data = result["request_data"]
        response = result.get("response")
        
        if response and "choices" in response:
            # 각 응답 처리
            for i, choice in enumerate(response["choices"]):
                # logprobs 추출
                logprobs_data = choice.get("logprobs", {})
                if logprobs_data and "content" in logprobs_data:
                    topk = simple_extract_topk(logprobs_data["content"], gen_cfg_logprobs)
                else:
                    topk = []
                
                # 신뢰도 계산
                confidence_scores = confidence_calculator.calculate_all_confidence_scores(topk)
                
                # 결과 생성 (chat/completions 호환)
                output_record = {
                    "problem_id": request_data.problem_id,
                    "problem_text": request_data.problem_text,
                    "ground_truth": request_data.ground_truth,
                    "response_id": f"{request_data.problem_id}_resp_{i}",
                    "generated_text": choice.get("message", {}).get("content", "") or choice.get("text", ""),
                    "output_token_count": choice.get("usage", {}).get("completion_tokens", 0),
                    "logprobs": topk,
                    "worker_gpu": args.gpu_id,
                    "worker_replica": f"shard_{args.shard_id}",
                    **confidence_scores,
                }
                
                buffer.append(output_record)
        
        processed_count += 1
        pbar.update(1)
        
        # 버퍼가 차면 저장
        if len(buffer) >= buffer_size:
            for item in buffer:
                await output_file.write(json.dumps(item) + '\n')
            await output_file.flush()
            logger.info(f"[Shard {args.shard_id}] {processed_count}/{total} 처리 완료")
            buffer.clear()
        
        result_queue.task_done()
    
    # 남은 버퍼 저장
    if buffer:
        for item in buffer:
            await output_file.write(json.dumps(item) + '\n')
        await output_file.flush()
    
    pbar.close()


async def api_generation_worker(cfg: DictConfig, args: argparse.Namespace):
    """API 기반 비동기 생성 워커"""
    
    # 로깅 디렉토리 (ENV 우선, 기본은 config)
    log_dir = os.environ.get("LOG_DIR", getattr(cfg.paths, "log_dir", "/workspace/outputs/logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"stage1_api_shard_{args.shard_id}.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger = logging.getLogger("conf_agg_llm")
    logger.info(f"🚀 [Shard {args.shard_id}] Stage 1: API 기반 데이터 생성 시작")
    
    try:
        # vLLM 서버 정보 로드 (절대 경로 우선)
        servers_json_path = os.environ.get("VLLM_SERVERS_JSON", "/workspace/vllm_servers.json")
        if not os.path.exists(servers_json_path):
            # 호환: 현재 작업 디렉토리도 확인
            if os.path.exists('vllm_servers.json'):
                servers_json_path = 'vllm_servers.json'
            else:
                raise FileNotFoundError(f"vllm_servers.json 파일이 없습니다: {servers_json_path}")
        
        with open(servers_json_path, 'r') as f:
            server_info = json.load(f)
        
        servers = server_info['servers']
        logger.info(f"사용 가능한 서버: {len(servers)}개")
        
        # 데이터 디렉토리 (ENV 우선)
        base_data_dir = os.environ.get("DATA_DIR", getattr(cfg.paths, "data_dir", "/workspace/outputs/data"))
        output_dir = os.path.join(base_data_dir, "generated")
        sample_limit_env = os.environ.get("SAMPLE_LIMIT", "")
        if sample_limit_env:
            output_dir = os.path.join(output_dir, f"sample_{sample_limit_env}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 데이터 경로 (ENV RAW_DATA_PATH로 오버라이드 가능)
        raw_data_path = os.environ.get("RAW_DATA_PATH", os.path.join(base_data_dir, "raw", "deepscaler.jsonl"))
        if not os.path.exists(raw_data_path):
            logger = logging.getLogger("conf_agg_llm")
            logger.error(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
            raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
        raw_dataset = RawDataset(raw_data_path)
        logger.info(f"전체 원본 데이터셋 로드 완료: {len(raw_dataset)}개 문제")
        
        # 샘플링 및 샤딩 (기존과 동일)
        sample_limit = int(sample_limit_env) if sample_limit_env and sample_limit_env.isdigit() else 0
        if sample_limit > 0 and sample_limit < len(raw_dataset):
            np.random.seed(42)
            selected_indices = np.random.choice(len(raw_dataset), size=sample_limit, replace=False)
            selected_indices = sorted(selected_indices)
        else:
            selected_indices = list(range(len(raw_dataset)))
        
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=cfg.data.raw_dataset.vllm.trust_remote_code
        )
        
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # 데이터 준비
        problems = []
        texts = []
        for idx in selected_indices:
            problem_data = raw_dataset[idx]
            problem_id = problem_data.get("id", f"problem_{idx}")
            problem_text = problem_data.get("problem", "")
            ground_truth = problem_data.get("answer", "")
            messages = [{"role": "user", "content": f"{problem_text}\n\n{instruction}"}]
            # 서버에서 채팅 템플릿/토크나이즈를 처리하도록 messages만 전달
            text = None
            problems.append({
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
            })
            texts.append(messages)
        
        # 샤드 분할
        my_problems = problems[args.shard_id::args.total_shards]
        my_texts = texts[args.shard_id::args.total_shards]
        logger.info(f"[Shard {args.shard_id}] {len(my_texts)}개 문제 처리")
        
        # 샘플링 파라미터
        gen_cfg = cfg.data.raw_dataset.generation
        sampling_params = {
            "model": cfg.model.base_model,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "max_tokens": gen_cfg.max_tokens,
            "n": gen_cfg.num_responses_per_problem,
            "logprobs": gen_cfg.logprobs,
            "presence_penalty": gen_cfg.presence_penalty,
        }
        
        # 신뢰도 계산기
        confidence_calculator = ConfidenceCalculator(
            group_size=cfg.data.raw_dataset.confidence.group_size
        )
        
        # 큐 생성
        request_queue = asyncio.Queue(maxsize=1000)
        result_queue = asyncio.Queue(maxsize=1000)
        
        # 요청 데이터 생성 및 큐에 추가
        for i, (problem, messages) in enumerate(zip(my_problems, my_texts)):
            # 라운드 로빈으로 서버 할당
            server = servers[i % len(servers)]
            request_data = RequestData(
                problem_id=problem["problem_id"],
                problem_text=problem["problem_text"],
                ground_truth=problem["ground_truth"],
                messages=messages,
                server_url=server["url"]
            )
            await request_queue.put(request_data)
        
        # 출력 파일
        temp_jsonl = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}_temp.jsonl")
        final_parquet = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}.parquet")
        
        async with aiofiles.open(temp_jsonl, 'a') as output_file:
            # HTTP 세션 생성
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            async with aiohttp.ClientSession(connector=connector) as session:
                
                # 워커 태스크 생성
                num_request_workers = min(50, len(my_texts))  # 동시 요청 워커 수
                
                # 요청 워커들
                request_workers = [
                    asyncio.create_task(
                        request_worker(
                            session, 
                            request_queue, 
                            result_queue, 
                            sampling_params
                        )
                    )
                    for _ in range(num_request_workers)
                ]
                
                # 결과 처리 워커
                processor_task = asyncio.create_task(
                    result_processor(
                        result_queue,
                        output_file,
                        confidence_calculator,
                        gen_cfg.logprobs,
                        args,
                        len(my_texts)
                    )
                )
                
                # 모든 요청 완료 대기
                await request_queue.join()
                
                # 워커 종료
                for _ in request_workers:
                    await request_queue.put(None)
                await asyncio.gather(*request_workers)
                
                # 결과 처리 완료 대기
                await result_queue.join()
                await result_queue.put(None)
                await processor_task
        
        # JSONL → Parquet 변환
        logger.info(f"[Shard {args.shard_id}] Parquet 변환 중...")
        df = pd.read_json(temp_jsonl, lines=True)
        df.to_parquet(final_parquet, index=False, compression="zstd")
        
        # 임시 파일 삭제
        os.remove(temp_jsonl)
        
        logger.info(f"✅ [Shard {args.shard_id}] 완료: {len(df)}개 결과 저장")
        logger.info(f"Parquet 위치: {final_parquet}")
        
        # 통계
        if len(df) > 0:
            logger.info(f"생성된 응답 수: {len(df)}")
            logger.info(f"문제 수: {df['problem_id'].nunique()}")
            logger.info(f"문제당 평균 응답 수: {len(df) / df['problem_id'].nunique():.1f}")
        
    except Exception as e:
        logger.error(f"[Shard {args.shard_id}] 오류: {e}", exc_info=True)
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--gpu-id", type=str, default="0")  # API 방식에서는 사용 안함
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=4)
    args = parser.parse_args()
    
    # Hydra 초기화
    config_dir = Path(args.config_path).resolve()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    cfg = hydra.compose(config_name=args.config_name)
    
    # 비동기 실행
    asyncio.run(api_generation_worker(cfg, args))


if __name__ == "__main__":
    main()