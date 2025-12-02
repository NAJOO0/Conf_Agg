#!/usr/bin/env python3
"""
/think 마커가 없는 응답을 삭제하고 부족한 응답을 vLLM으로 재생성하는 스크립트

사용법:
    # 단일 GPU 모드
    python scripts/regenerate_think_responses.py --gpu-id 0

    # 멀티 GPU 모드 (4 GPUs)
    python scripts/regenerate_think_responses.py --gpu-id 0 --shard-id 0 --total-shards 4 &
    python scripts/regenerate_think_responses.py --gpu-id 1 --shard-id 1 --total-shards 4 &
    python scripts/regenerate_think_responses.py --gpu-id 2 --shard-id 2 --total-shards 4 &
    python scripts/regenerate_think_responses.py --gpu-id 3 --shard-id 3 --total-shards 4 &
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import logging
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.utils.logging import setup_logging
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def has_final_answer(text: str) -> bool:
    """생성된 텍스트에 final_answer가 있는지 확인 (\\boxed{} 패턴)"""
    pattern = r'\\boxed\{[^}]+\}'
    return bool(re.search(pattern, text))


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
    max_tokens: int,
    gpu_id: int,
    shard_id: int
) -> tuple[Dict, bool]:
    """
    단일 출력을 처리하여 결과 딕셔너리 반환

    Returns:
        tuple[Dict, bool]: (결과 딕셔너리, 성공 여부)
    """
    # 최적화된 logprob 추출
    topk = simple_extract_topk(completion.logprobs, gen_cfg_logprobs)

    # 신뢰도 계산
    confidence_scores = confidence_calculator.calculate_all_confidence_scores(topk)

    generated_text = completion.text
    output_token_count = len(completion.token_ids) if hasattr(completion, "token_ids") else 0

    # 실패 케이스: max_tokens 도달 + final_answer 없음
    is_success = True
    if output_token_count >= max_tokens and not has_final_answer(generated_text):
        is_success = False

    result = {
        "problem_id": problem_meta["problem_id"],
        "problem_text": problem_meta["problem_text"],
        "ground_truth": problem_meta["ground_truth"],
        "response_id": f"{problem_meta['problem_id']}_resp_regen_{response_idx}",
        "generated_text": generated_text,
        "output_token_count": output_token_count,
        "worker_gpu": gpu_id,
        "worker_replica": f"shard_{shard_id}",
        **confidence_scores,
    }

    return result, is_success


def regenerate_missing_responses(
    cfg,
    args: argparse.Namespace,
    input_path: str,
    output_path: str
):
    """부족한 /think 응답을 재생성"""

    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, f"regenerate_shard_{args.shard_id}.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=False,  # WandB 비활성화
    )

    logger = logging.getLogger("conf_agg_llm")
    logger.info(f"🚀 [Shard {args.shard_id} | GPU {args.gpu_id}] /think 응답 재생성 시작")

    try:
        # 1. 기존 데이터 로드 및 필터링
        logger.info(f"기존 데이터 로드 중: {input_path}")
        df = pd.read_parquet(input_path)
        logger.info(f"원본 데이터: {len(df)}행")

        # /think 마커 확인
        df['has_think'] = df['generated_text'].str.contains('/think', na=False, regex=False)
        has_think_count = df['has_think'].sum()
        no_think_count = (~df['has_think']).sum()
        logger.info(f"/think 있음: {has_think_count}개, 없음: {no_think_count}개")

        # /think 마커가 있는 응답만 유지
        df_filtered = df[df['has_think']].drop(columns=['has_think']).copy()
        logger.info(f"필터링 후 데이터: {len(df_filtered)}행")

        # 2. 문제별 부족한 응답 수 계산
        problem_counts = df_filtered.groupby('problem_id').size().reset_index(name='count')
        problem_info = df_filtered.groupby('problem_id').agg({
            'problem_text': 'first',
            'ground_truth': 'first'
        }).reset_index()
        problem_stats = problem_counts.merge(problem_info, on='problem_id')

        # 32개 미만인 문제들
        missing_problems = problem_stats[problem_stats['count'] < 32].copy()
        missing_problems['need_count'] = 32 - missing_problems['count']

        logger.info(f"32개 미만인 문제 수: {len(missing_problems)}")
        logger.info(f"총 재생성 필요 응답 수: {missing_problems['need_count'].sum()}")

        if len(missing_problems) == 0:
            logger.info("모든 문제가 32개의 /think 응답을 가지고 있습니다. 작업 종료.")
            # 필터링된 데이터 저장
            df_filtered.to_parquet(output_path, index=False)
            logger.info(f"필터링된 데이터 저장 완료: {output_path}")
            return

        # 3. Sharding - 이 워커가 처리할 문제들
        missing_problems_list = missing_problems.to_dict('records')
        my_problems = [p for i, p in enumerate(missing_problems_list) if i % args.total_shards == args.shard_id]

        if len(my_problems) == 0:
            logger.info(f"[Shard {args.shard_id}] 처리할 문제가 없습니다. 작업 종료.")
            return

        total_responses_needed = sum(p['need_count'] for p in my_problems)
        logger.info(f"[Shard {args.shard_id}] 처리할 문제: {len(my_problems)}개, 생성할 응답: {total_responses_needed}개")

        # 4. LLM 엔진 초기화
        logger.info(f"[Shard {args.shard_id}] LLM 엔진 초기화 중: {cfg.model.base_model}")
        vllm_config = cfg.data.raw_dataset.vllm

        llm = LLM(
            model=cfg.model.base_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            max_model_len=vllm_config.max_model_len,
            dtype=vllm_config.dtype,
            trust_remote_code=vllm_config.trust_remote_code,
            max_num_batched_tokens=vllm_config.get("max_num_batched_tokens", 16384),
            max_num_seqs=vllm_config.get("max_num_seqs", 256),
            enforce_eager=vllm_config.get("enforce_eager", False),
            disable_custom_all_reduce=vllm_config.get("disable_custom_all_reduce", True),
            enable_prefix_caching=vllm_config.get("enable_prefix_caching", True),
            kv_cache_dtype=vllm_config.get("kv_cache_dtype", "auto"),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=vllm_config.trust_remote_code
        )
        logger.info(f"[Shard {args.shard_id}] LLM 엔진 초기화 완료")

        # 5. 신뢰도 계산기 초기화
        confidence_calculator = ConfidenceCalculator(
            group_size=cfg.data.raw_dataset.confidence.group_size
        )

        instruction = "Please reason step by step, and put your final answer within \\boxed{}."

        # 6. 샘플링 파라미터 설정
        gen_cfg = cfg.data.raw_dataset.generation
        sampling_params = SamplingParams(
            n=1,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            min_p=gen_cfg.min_p,
            max_tokens=gen_cfg.max_tokens,
            logprobs=gen_cfg.logprobs,
            presence_penalty=gen_cfg.presence_penalty,
        )
        gen_cfg_logprobs = gen_cfg.logprobs
        max_tokens = gen_cfg.max_tokens

        # 7. 프롬프트 생성
        logger.info(f"[Shard {args.shard_id}] 프롬프트 생성 중...")
        prompts = []
        problem_metas = []
        response_indices = []

        for prob_idx, problem in enumerate(my_problems):
            problem_id = problem['problem_id']
            problem_text = problem['problem_text']
            ground_truth = problem['ground_truth']
            need_count = problem['need_count']

            # 프롬프트 생성
            messages = [{"role": "user", "content": f"{problem_text}\n\n{instruction}"}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=cfg.data.raw_dataset.generation.enable_thinking,
            )

            problem_meta = {
                "problem_id": problem_id,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
            }

            # 부족한 개수만큼 반복
            for resp_idx in range(need_count):
                prompts.append(text)
                problem_metas.append(problem_meta)
                response_indices.append(resp_idx)

        logger.info(f"[Shard {args.shard_id}] 총 {len(prompts)}개 프롬프트 생성 완료")

        # 8. 배치 생성
        logger.info(f"[Shard {args.shard_id}] LLM 생성 시작...")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        # 9. 결과 처리
        logger.info(f"[Shard {args.shard_id}] 결과 처리 중...")
        results_buffer = []
        successful_count = 0
        failed_count = 0

        for idx, output in enumerate(tqdm(outputs, desc=f"Processing Shard {args.shard_id}")):
            problem_meta = problem_metas[idx]
            response_idx = response_indices[idx]
            completion = output.outputs[0]

            result, is_success = process_single_output(
                problem_meta,
                completion,
                response_idx,
                confidence_calculator,
                gen_cfg_logprobs,
                max_tokens,
                args.gpu_id,
                args.shard_id
            )

            # /think 마커 확인
            has_think = '/think' in result['generated_text']

            if has_think:
                results_buffer.append(result)
                successful_count += 1
            else:
                # /think 마커가 없으면 실패로 간주
                failed_count += 1
                logger.warning(f"[Shard {args.shard_id}] /think 마커 없음: {problem_meta['problem_id']}_resp_{response_idx}")

        logger.info(f"[Shard {args.shard_id}] 생성 완료: 성공 {successful_count}개, 실패 {failed_count}개")

        # 10. 결과 병합 및 저장
        if results_buffer:
            new_df = pd.DataFrame(results_buffer)
            logger.info(f"[Shard {args.shard_id}] 새로 생성된 응답: {len(new_df)}개")

            # 필터링된 기존 데이터와 병합
            combined_df = pd.concat([df_filtered, new_df], ignore_index=True)

            # 샤드별 임시 파일 저장
            temp_output = output_path.replace('.parquet', f'_shard_{args.shard_id}.parquet')
            combined_df.to_parquet(temp_output, index=False)
            logger.info(f"[Shard {args.shard_id}] 병합 데이터 저장 완료: {temp_output}")
            logger.info(f"[Shard {args.shard_id}] 최종 데이터: {len(combined_df)}행")
        else:
            logger.warning(f"[Shard {args.shard_id}] 생성된 응답이 없습니다.")

    except Exception as e:
        logger.error(f"[Shard {args.shard_id}] 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--shard-id', type=int, default=0, help='Shard ID (0부터 시작)')
    parser.add_argument('--total-shards', type=int, default=1, help='전체 샤드 수')
    parser.add_argument('--input-file', type=str,
                       default='/mnt/data1/projects/Conf_Agg/output_s/generated/sample_800_offset_3200/raw_generated_cleaned_regenerated_shard_0.parquet',
                       help='입력 파일 경로')
    parser.add_argument('--output-file', type=str,
                       default='/mnt/data1/projects/Conf_Agg/output_s/generated/sample_800_offset_3200/raw_generated_cleaned_regenerated_2.parquet',
                       help='출력 파일 경로')
    parser.add_argument('--config-path', type=str,
                       default='config/config.yaml',
                       help='설정 파일 경로')
    args = parser.parse_args()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Hydra config 로드 (defaults 처리 포함)
    config_dir = str((Path(__file__).parent.parent / "config").absolute())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    # 실행
    regenerate_missing_responses(cfg, args, args.input_file, args.output_file)


if __name__ == "__main__":
    main()
