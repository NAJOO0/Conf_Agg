"""
Stage 1: 원시 데이터 생성 스크립트 (AsyncLLMEngine으로 continuous batching + 스트리밍 저장)
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm
import asyncio
import aiofiles
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput

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


def process_single_output(
    problem_meta: Dict,
    completion: Any,
    response_idx: int,
    confidence_calculator: ConfidenceCalculator,
    gen_cfg_logprobs: int,
    args: argparse.Namespace
) -> Dict:
    """단일 출력을 처리하여 결과 딕셔너리 반환"""
    
    # 최적화된 logprob 추출
    topk = simple_extract_topk(completion.logprobs, gen_cfg_logprobs)
    
    # 신뢰도 계산
    confidence_scores = confidence_calculator.calculate_all_confidence_scores(topk)
    
    return {
        "problem_id": problem_meta["problem_id"],
        "problem_text": problem_meta["problem_text"],
        "ground_truth": problem_meta["ground_truth"],
        "response_id": f"{problem_meta['problem_id']}_resp_{response_idx}",
        "generated_text": completion.text,
        "output_token_count": len(completion.token_ids) if hasattr(completion, "token_ids") else 0,
        # "logprobs": topk,
        "worker_gpu": args.gpu_id,
        "worker_replica": f"shard_{args.shard_id}",
        **confidence_scores,
    }


async def async_generation_worker(cfg: DictConfig, args: argparse.Namespace):
    """비동기 엔진으로 continuous batching 유지 + 스트리밍 저장"""
    
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
    
    logger.info(f"🚀 [Shard {args.shard_id} | GPU {args.gpu_id}] Stage 1: 원시 데이터 생성 시작 (Async Mode)")
    logger.info(f"전체 설정: {OmegaConf.to_yaml(cfg)}")
    
    try:
        # 1. CUDA 컨텍스트 재확인 (프로세스 간 격리 보장)
        if torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES로 인해 0번이 실제 GPU
            current_device = torch.cuda.current_device()
            logger.info(f"[Shard {args.shard_id}] CUDA 컨텍스트 확인: device={current_device}, GPU={torch.cuda.get_device_name(current_device)}")
        
        # 2. AsyncLLMEngine 설정
        logger.info(f"[Shard {args.shard_id}] AsyncLLMEngine 초기화 중: {cfg.model.base_model}")
        vllm_config = cfg.data.raw_dataset.vllm
        
        engine_args = AsyncEngineArgs(
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
            disable_log_stats=vllm_config.get("disable_log_stats", False),
            enable_prefix_caching=vllm_config.get("enable_prefix_caching", True),  # 프리픽스 캐싱
        )
        
        # kv_cache_dtype이 설정되어 있으면 추가
        if "kv_cache_dtype" in vllm_config:
            engine_args.kv_cache_dtype = vllm_config.kv_cache_dtype
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=vllm_config.trust_remote_code
        )
        logger.info(f"[Shard {args.shard_id}] AsyncLLMEngine 초기화 완료.")
        
        # 디렉토리 생성
        output_dir = os.path.join(cfg.paths.data_dir, "generated")
        sample_limit_env = os.environ.get("SAMPLE_LIMIT", "")
        if sample_limit_env:
            output_dir = os.path.join(output_dir, f"sample_{sample_limit_env}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 데이터셋 로드
        raw_data_path = os.path.join(cfg.paths.data_dir, "raw", "deepscaler.jsonl")
        if not os.path.exists(raw_data_path):
            logger.error(f"원본 데이터 파일을 찾을 수 없습니다: {raw_data_path}")
            return
        
        raw_dataset = RawDataset(raw_data_path)
        logger.info(f"전체 원본 데이터셋 로드 완료: {len(raw_dataset)}개 문제")
        
        # 샘플 수 제한
        sample_limit = int(sample_limit_env) if sample_limit_env and sample_limit_env.isdigit() else 0
        
        if sample_limit > 0 and sample_limit < len(raw_dataset):
            np.random.seed(42)
            selected_indices = np.random.choice(len(raw_dataset), size=sample_limit, replace=False)
            selected_indices = sorted(selected_indices)
            logger.info(f"SAMPLE_LIMIT 적용: 전체 {len(raw_dataset)}개 중 랜덤으로 {sample_limit}개 문제 선택")
        else:
            selected_indices = list(range(len(raw_dataset)))
            logger.info(f"전체 데이터셋 사용: {len(raw_dataset)}개 문제")
        
        # 신뢰도 계산기 초기화
        confidence_calculator = ConfidenceCalculator(
            group_size=cfg.data.raw_dataset.confidence.group_size
        )
        
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # 전체 데이터 준비
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
                enable_thinking=cfg.data.raw_dataset.generation.enable_thinking,
            )
            problems.append({
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
            })
            texts.append(text)
        
        # 작업 분할 (Sharding)
        my_problems = problems[args.shard_id::args.total_shards]
        my_texts = texts[args.shard_id::args.total_shards]
        
        logger.info(f"[Shard {args.shard_id}] 작업 분할 완료. 이 샤드에서 {len(my_texts)}개 문제 처리")
        
        # 샘플링 파라미터 구성
        gen_cfg = cfg.data.raw_dataset.generation
        num_responses_per_problem = gen_cfg.num_responses_per_problem

        # n=1로 고정하여 각 요청이 1개의 응답만 생성하도록 설정
        # 대신 코드에서 문제당 num_responses_per_problem번 호출
        sampling_params = SamplingParams(
            n=1,  # 고정
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            min_p=gen_cfg.min_p,
            max_tokens=gen_cfg.max_tokens,
            logprobs=gen_cfg.logprobs,
            presence_penalty=gen_cfg.presence_penalty,
        )
        gen_cfg_logprobs = gen_cfg.logprobs
        
        # 출력 파일 경로
        temp_jsonl_path = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}_temp.jsonl")
        final_parquet_path = os.path.join(output_dir, f"raw_generated_shard_{args.shard_id}.parquet")
        checkpoint_path = os.path.join(output_dir, f"checkpoint_shard_{args.shard_id}.json")
        
        # 체크포인트 로드
        start_idx = 0
        processed_problems = set()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get('last_index', 0)
                processed_problems = set(checkpoint.get('processed_problems', []))
            logger.info(f"체크포인트에서 재시작: {start_idx}번째 문제부터, 이미 처리된 문제 {len(processed_problems)}개")
        
        # 비동기 파일 핸들러
        async with aiofiles.open(temp_jsonl_path, 'a') as f:
            # 요청 생성기
            async def request_generator() -> AsyncGenerator:
                """모든 요청을 순차적으로 생성 (각 문제마다 num_responses_per_problem번 요청)"""
                for i in range(start_idx, len(my_texts)):
                    problem = my_problems[i]

                    # 이미 처리된 문제는 건너뛰기
                    if problem['problem_id'] in processed_problems:
                        continue

                    text = my_texts[i]

                    # 각 문제마다 num_responses_per_problem번 요청 생성
                    for resp_idx in range(num_responses_per_problem):
                        request_id = f"shard_{args.shard_id}_problem_{i}_resp_{resp_idx}"
                        yield (request_id, text, problem, i, resp_idx)
            
            # 동시 처리 중인 요청 추적
            pending_requests = {}
            results_buffer = []
            buffer_size = cfg.data.raw_dataset.get("buffer_size", 10)  # 버퍼 크기
            save_interval = cfg.data.raw_dataset.get("save_interval", 100)  # N개 배치마다 저장
            max_pending = cfg.data.raw_dataset.get("max_pending_requests", 10)  # 최대 동시 요청 수
            
            processed_count = 0
            total_saved = 0
            last_checkpoint_time = datetime.now()
            
            # 요청 제출 태스크
            async def submit_requests():
                """요청을 엔진에 제출"""
                async for request_id, text, problem, idx, resp_idx in request_generator():
                    # 대기열이 너무 크면 잠시 대기
                    while len(pending_requests) > max_pending:
                        await asyncio.sleep(0.1)

                    # 엔진에 요청 제출
                    results_generator = engine.generate(
                        prompt=text,
                        sampling_params=sampling_params,
                        request_id=request_id
                    )

                    pending_requests[request_id] = {
                        'generator': results_generator,
                        'problem': problem,
                        'index': idx,
                        'response_idx': resp_idx
                    }

                    # 진행 상황 로그
                    if len(pending_requests) % 100 == 0:
                        logger.info(f"[Shard {args.shard_id}] {len(pending_requests)}개 요청 제출됨")

                logger.info(f"[Shard {args.shard_id}] 모든 요청 제출 완료")
            
            # 결과 수집 태스크
            async def collect_results():
                """결과를 수집하고 즉시 저장"""
                nonlocal processed_count, total_saved, last_checkpoint_time

                # 전체 요청 수 = 문제 수 × 응답 수
                total_requests = (len(my_texts) - start_idx) * num_responses_per_problem
                pbar = tqdm(total=total_requests, desc=f"Shard {args.shard_id}", position=args.shard_id)

                while pending_requests or not submit_task.done():
                    if not pending_requests:
                        await asyncio.sleep(0.1)
                        continue

                    # 완료된 요청 처리
                    completed = []

                    for request_id, req_data in list(pending_requests.items()):
                        try:
                            # 결과 가져오기 (논블로킹)
                            output = await anext(req_data['generator'])

                            if output is not None and output.finished:
                                # 응답 처리 (n=1이므로 output.outputs는 1개만 있음)
                                problem = req_data['problem']
                                response_idx = req_data['response_idx']
                                completion = output.outputs[0]

                                result = process_single_output(
                                    problem,
                                    completion,
                                    response_idx,
                                    confidence_calculator,
                                    gen_cfg_logprobs,
                                    args
                                )
                                results_buffer.append(result)

                                completed.append(request_id)
                                processed_count += 1
                                processed_problems.add(problem['problem_id'])
                                pbar.update(1)
                                
                                # 버퍼가 차면 저장
                                if len(results_buffer) >= buffer_size:
                                    for r in results_buffer:
                                        await f.write(json.dumps(r) + '\n')
                                    await f.flush()
                                    total_saved += len(results_buffer)
                                    logger.info(f"[Shard {args.shard_id}] {total_saved}개 결과 저장 완료 ({processed_count}/{total_requests}개 요청 처리, {len(processed_problems)}개 문제 완료)")
                                    results_buffer.clear()
                                
                                # 체크포인트 저장 (1분마다)
                                if (datetime.now() - last_checkpoint_time).seconds > 60:
                                    checkpoint = {
                                        'last_index': req_data['index'],
                                        'processed_problems': list(processed_problems),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    async with aiofiles.open(checkpoint_path, 'w') as ckpt_f:
                                        await ckpt_f.write(json.dumps(checkpoint))
                                    last_checkpoint_time = datetime.now()
                                    logger.info(f"[Shard {args.shard_id}] 체크포인트 저장")
                                
                        except StopAsyncIteration:
                            # 생성 완료
                            completed.append(request_id)
                        except Exception as e:
                            logger.error(f"[Shard {args.shard_id}] 요청 {request_id} 처리 중 오류: {e}")
                            completed.append(request_id)
                    
                    # 완료된 요청 제거
                    for req_id in completed:
                        del pending_requests[req_id]
                    
                    # CPU 양보
                    await asyncio.sleep(0.01)
                
                pbar.close()
                
                # 마지막 버퍼 처리
                if results_buffer:
                    for r in results_buffer:
                        await f.write(json.dumps(r) + '\n')
                    await f.flush()
                    total_saved += len(results_buffer)
                    logger.info(f"[Shard {args.shard_id}] 최종 {len(results_buffer)}개 결과 저장 (총 {total_saved}개 저장 완료)")
                    results_buffer.clear()
            
            # 동시 실행
            submit_task = asyncio.create_task(submit_requests())
            collect_task = asyncio.create_task(collect_results())
            
            await asyncio.gather(submit_task, collect_task)
        
        # JSONL → Parquet 변환
        logger.info(f"[Shard {args.shard_id}] JSONL을 Parquet으로 변환 중...")
        df = pd.read_json(temp_jsonl_path, lines=True)
        df.to_parquet(final_parquet_path, index=False, compression="zstd")
        
        # 임시 파일 삭제
        os.remove(temp_jsonl_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        logger.info(f"✅ [Shard {args.shard_id}] Stage 1 완료: {len(df)}개 결과 저장")
        logger.info(f"Parquet 저장 위치: {final_parquet_path}")
        
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
                
                max_tokens_limit = gen_cfg.max_tokens
                max_tokens_count = int((df['output_token_count'].fillna(0) == max_tokens_limit).sum())
                logger.info(f"최대 토큰 수({max_tokens_limit})에 도달한 인스턴스 개수: {max_tokens_count}")
            except Exception:
                pass
        
        # 샘플 출력
        if len(df) > 0:
            logger.info("=" * 80)
            logger.info("샘플 결과 출력 (첫 번째 인스턴스):")
            logger.info("=" * 80)
            sample = df.iloc[0]
            logger.info(f"Problem ID: {sample['problem_id']}")
            logger.info(f"Problem Text: {sample['problem_text'][:200]}...")
            logger.info(f"Ground Truth: {sample['ground_truth']}")
            logger.info(f"Generated Text: {sample['generated_text'][:500]}...")
            logger.info(f"Output Token Count: {sample.get('output_token_count', 'N/A')}")
            logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"[Shard {args.shard_id}] 실행 중 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 함수 - 동기 래퍼"""
    # argparse로 런처의 인수를 받음
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Hydra config directory")
    parser.add_argument("--config-name", type=str, required=True, help="Hydra config name")
    parser.add_argument("--gpu-id", type=str, required=True, help="GPU ID")
    parser.add_argument("--shard-id", type=int, required=True, help="Data shard index")
    parser.add_argument("--total-shards", type=int, default=4, help="Total number of shards")
    args = parser.parse_args()
    
    # 1. GPU 격리 (가장 먼저 설정)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # 2. CUDA 컨텍스트 명시적 초기화 및 GPU 설정
    # CUDA_VISIBLE_DEVICES 설정 후에는 항상 0번이 실제 GPU가 됨
    if torch.cuda.is_available():
        # 명시적으로 GPU를 설정하여 프로세스 간 격리 보장
        # CUDA_VISIBLE_DEVICES로 인해 0번이 실제 GPU가 됨
        torch.cuda.set_device(0)
        # GPU 메모리 캐시 정리
        torch.cuda.empty_cache()
    
    # 3. Hydra 수동 초기화
    config_dir = Path(args.config_path).resolve()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    
    cfg = hydra.compose(config_name=args.config_name)
    
    # 4. 비동기 워커 실행
    asyncio.run(async_generation_worker(cfg, args))


if __name__ == "__main__":
    main()