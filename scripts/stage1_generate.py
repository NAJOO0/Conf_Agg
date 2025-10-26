"""
Stage 1: 원시 데이터 생성 스크립트 (단순 데이터 병렬 최적화)
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
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def simple_extract_topk(gen_logprobs: List[Dict[int, Any]], k: int) -> List[List[float]]:
    """최적화된 logprob 추출 함수 (float16으로 메모리 절약)"""
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
            # float16 변환 (메모리 절약)
            results.append(np.array(lps, dtype=np.float16).tolist())
        else:
            results.append([])
    
    return results


def main_worker(cfg: DictConfig, args: argparse.Namespace) -> None:
    """
    오프라인 배치 워커 메인 함수
    - args로부터 gpu_id와 shard_id를 받아 작업을 분할 처리
    """
    
    # 로깅 설정 (샤드별 파일 구분)
    log_file = os.path.join(cfg.paths.log_dir, f"stage1_generate_shard_{args.shard_id}.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    # 올바른 로거 사용
    logger = logging.getLogger("conf_agg_llm")
    
    logger.info(f"🚀 [Shard {args.shard_id} | GPU {args.gpu_id}] Stage 1: 원시 데이터 생성 시작")
    logger.info(f"전체 설정: {OmegaConf.to_yaml(cfg)}")
    
    try:
        # 1. vLLM 모델 로드 (Ray Serve 대신 직접 로드)
        logger.info(f"[Shard {args.shard_id}] vLLM 모델 로드 중: {cfg.model.base_model}")
        vllm_config = cfg.data.raw_dataset.vllm
        llm = LLM(
            model=cfg.model.base_model,
            tensor_parallel_size=1,  # TP=1 (단일 GPU)
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            max_model_len=vllm_config.max_model_len,
            dtype=vllm_config.dtype,
            trust_remote_code=vllm_config.trust_remote_code,
            max_num_batched_tokens=vllm_config.get("max_num_batched_tokens", 16384),
            max_num_seqs=vllm_config.get("max_num_seqs", 256),
            enforce_eager=vllm_config.get("enforce_eager", False),
            disable_custom_all_reduce=vllm_config.get("disable_custom_all_reduce", True),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=vllm_config.trust_remote_code
        )
        logger.info(f"[Shard {args.shard_id}] vLLM 모델 로드 완료.")

        # 디렉토리 생성
        output_dir = os.path.join(cfg.paths.data_dir, f"generated")
        output_dir = os.path.join(output_dir, f"sample_{os.environ.get("SAMPLE_LIMIT")}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 데이터셋 로드 (전체 40K)
        raw_data_path = os.path.join(cfg.paths.data_dir, "raw", "deepscaler.jsonl")
        if not os.path.exists(raw_data_path):
            logger.error(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
            return
        
        raw_dataset = RawDataset(raw_data_path)
        logger.info(f"전체 원본 데이터셋 로드 완료: {len(raw_dataset)}개 문제")
        
        # 샘플 수 제한 (랜덤 샘플링)
        sample_limit_env = os.environ.get("SAMPLE_LIMIT")
        sample_limit = int(sample_limit_env) if sample_limit_env and sample_limit_env.isdigit() else 0
        
        if sample_limit > 0 and sample_limit < len(raw_dataset):
            # 랜덤 샘플링으로 인덱스 선택
            np.random.seed(42)  # 재현 가능한 랜덤 샘플링을 위한 시드 설정
            selected_indices = np.random.choice(len(raw_dataset), size=sample_limit, replace=False)
            selected_indices = sorted(selected_indices)  # 정렬하여 일관성 유지
            logger.info(f"SAMPLE_LIMIT 적용: 전체 {len(raw_dataset)}개 중 랜덤으로 {sample_limit}개 문제 선택")
            logger.info(f"선택된 인덱스 범위: {min(selected_indices)} ~ {max(selected_indices)}")
        else:
            selected_indices = list(range(len(raw_dataset)))
            logger.info(f"전체 데이터셋 사용: {len(raw_dataset)}개 문제")
        
        total_items = len(selected_indices)
        
        # 신뢰도 계산기 초기화
        confidence_calculator = ConfidenceCalculator(
            group_size=cfg.data.raw_dataset.confidence.group_size
        )
        
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        
        problems: List[Dict] = []
        texts: List[str] = []
        
        logger.info("전체 입력 텍스트 생성 중...")
        for idx in selected_indices:
            problem_data = raw_dataset[idx]
            problem_id = problem_data.get("id", f"problem_{idx}")
            problem_text = problem_data.get("problem", "")
            ground_truth = problem_data.get("answer", "")
            messages = [{"role": "user", "content": f"{problem_text}\n\n{instruction}"}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Qwen 계열 모델에 필요할 수 있음
            )
            problems.append({
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
            })
            texts.append(text)
        logger.info(f"총 {len(texts)}개 프롬프트 준비 완료.")

        # 2. 작업 분할 (Sharding)
        # 이 워커(샤드)에 할당된 작업만 필터링
        my_problems = problems[args.shard_id::args.total_shards]
        my_texts = texts[args.shard_id::args.total_shards]
        
        logger.info(f"[Shard {args.shard_id}] 작업 분할 완료. 이 샤드에서 {len(my_texts)}개 문제 처리 (1/{args.total_shards})")

        # 3. 샘플링 파라미터 구성
        gen_cfg = cfg.data.raw_dataset.generation
        sampling_params = SamplingParams(
            n=gen_cfg.num_responses_per_problem,  # 각 문제당 응답 수
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            min_p=gen_cfg.min_p,
            max_tokens=gen_cfg.max_tokens,
            logprobs=gen_cfg.logprobs,  # top-k logprob 저장
        )
        gen_cfg_logprobs = gen_cfg.logprobs

        # 4. vLLM 일괄 추론 실행 (모든 요청을 한 번에 던지고 vLLM이 자동 배치 처리)
        logger.info(f"[Shard {args.shard_id}] vLLM 추론 시작... (입력 {len(my_texts)}개 문제, 각 문제당 {gen_cfg.num_responses_per_problem}개 응답)")
        outputs = llm.generate(my_texts, sampling_params)
        logger.info(f"[Shard {args.shard_id}] vLLM 추론 완료. 총 {len(outputs)}개 프롬프트 결과 수신.")
        
        # 5. 후처리 및 결과 취합 (CPU 작업)
        all_results = []
        logger.info(f"[Shard {args.shard_id}] 결과 후처리 시작...")
        
        pbar = tqdm(total=len(outputs), desc=f"Shard {args.shard_id} Post-processing", ncols=100)
        for pi, req_out in enumerate(outputs):
            base_meta = my_problems[pi]  # 샤드에 할당된 문제 메타데이터
            
            for oi, gen in enumerate(req_out.outputs):
                
                # 최적화된 logprob 추출
                topk = simple_extract_topk(gen.logprobs, gen_cfg_logprobs)
                
                # 신뢰도 계산
                confidence_scores = confidence_calculator.calculate_all_confidence_scores(topk)
                
                all_results.append({
                    "problem_id": base_meta["problem_id"],
                    "problem_text": base_meta["problem_text"],
                    "ground_truth": base_meta["ground_truth"],
                    "response_id": f"{base_meta['problem_id']}_resp_{oi}",
                    "generated_text": gen.text,
                    "output_token_count": len(gen.token_ids) if hasattr(gen, "token_ids") else 0,
                    "logprobs": topk,
                    "worker_gpu": args.gpu_id,  # GPU ID 저장
                    "worker_replica": f"shard_{args.shard_id}",  # Shard ID 저장
                    **confidence_scores,
                })
            pbar.update(1)
        pbar.close()

        # 6. 결과 저장 (샤드별 파일)
        df = pd.DataFrame(all_results)
        parquet_path = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}.parquet")
        df.to_parquet(parquet_path, index=False, compression="zstd")
        
        logger.info(f"✅ [Shard {args.shard_id}] Stage 1 완료: {len(df)}개 결과 저장")
        logger.info(f"Parquet 저장 위치: {parquet_path}")
        
        # 통계 정보 출력
        logger.info(f"생성된 응답 수: {len(df)}")
        logger.info(f"문제 수: {df['problem_id'].nunique()}")
        logger.info(f"문제당 평균 응답 수: {len(df) / df['problem_id'].nunique():.1f}")
        
        if 'output_token_count' in df.columns:
            try:
                total_tokens = int(df['output_token_count'].fillna(0).sum())
                mean_tokens = float(df['output_token_count'].fillna(0).mean())
                min_tokens = int(df['output_token_count'].fillna(0).min()) if len(df) > 0 else 0
                max_tokens = int(df['output_token_count'].fillna(0).max()) if len(df) > 0 else 0
                logger.info(f"응답 토큰 수 합계: {total_tokens}")
                logger.info(f"응답 토큰 수 평균: {mean_tokens:.1f}")
                logger.info(f"응답 토큰 수 최소/최대: {min_tokens}/{max_tokens}")
                
                # max_tokens와 같은 토큰 수를 가진 인스턴스 개수 출력
                max_tokens_limit = gen_cfg.max_tokens
                max_tokens_count = int((df['output_token_count'].fillna(0) == max_tokens_limit).sum())
                logger.info(f"최대 토큰 수({max_tokens_limit})에 도달한 인스턴스 개수: {max_tokens_count}")
            except Exception:
                pass
        
        # 샘플 1개 전체 출력
        if len(df) > 0:
            logger.info("=" * 80)
            logger.info("샘플 결과 출력 (첫 번째 인스턴스):")
            logger.info("=" * 80)
            sample = df.iloc[0]
            logger.info(f"Problem ID: {sample['problem_id']}")
            logger.info(f"Problem Text: {sample['problem_text']}")
            logger.info(f"Ground Truth: {sample['ground_truth']}")
            logger.info(f"Generated Text: {sample['generated_text']}")
            logger.info(f"Output Token Count: {sample['output_token_count']}")
            logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"[Shard {args.shard_id}] 실행 중 오류 발생: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # argparse로 런처의 인수를 받음
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Hydra config directory (e.g., ../config)")
    parser.add_argument("--config-name", type=str, required=True, help="Hydra config name (e.g., config)")
    parser.add_argument("--gpu-id", type=str, required=True, help="GPU ID (e.g., '0')")
    parser.add_argument("--shard-id", type=int, required=True, help="Data shard index (0, 1, 2, 3)")
    parser.add_argument("--total-shards", type=int, default=4, help="Total number of shards")
    args = parser.parse_args()

    # 1. GPU 격리
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 2. Hydra 수동 초기화
    # config_path는 디렉토리이므로 Path 객체로 변환
    config_dir = Path(args.config_path).resolve()
    # hydra.initialize_config_dir은 절대 경로를 사용해야 함
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # Hydra 초기화
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    
    cfg = hydra.compose(config_name=args.config_name)

    # 3. 메인 워커 함수 실행
    main_worker(cfg, args)