#!/usr/bin/env python3
"""
Train/Validation 데이터의 Prompt Token Count 분포 분석

주요 기능:
1. Train 데이터의 prompt token count 분포 분석
2. Validation 데이터의 prompt token count 분포 분석
3. 전체 데이터의 prompt token count 분포 분석
4. 시각화 및 통계 출력
"""
import os
import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings
from typing import Optional, List, Tuple, Dict, Any
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict
import json

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer
from src.evaluation.math_verifier import MathVerifier

# Global tokenizer for multiprocessing
_global_tokenizer = None

def _init_worker(model_name: str):
    """멀티프로세싱 워커 초기화"""
    global _global_tokenizer
    _global_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

def _tokenize_batch(prompts: List[str]) -> List[int]:
    """배치 토큰화 (멀티프로세싱 워커 함수)"""
    global _global_tokenizer
    try:
        encoded = _global_tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return [len(input_ids) for input_ids in encoded['input_ids']]
    except Exception as e:
        # Fallback: 개별 처리
        counts = []
        for prompt in prompts:
            try:
                tokens = _global_tokenizer.encode(prompt, add_special_tokens=False)
                counts.append(len(tokens))
            except:
                counts.append(len(prompt.split()))
        return counts


# TRL imports (chat template 적용용)
try:
    from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, maybe_apply_chat_template
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logger.warning("TRL을 사용할 수 없습니다. chat template을 적용하지 않습니다.")

# PyArrow 가용성 확인
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


def load_tokenizer(model_name: str):
    """
    Tokenizer 로드 및 chat template 설정
    
    Args:
        model_name: 모델 이름 또는 경로
        
    Returns:
        tokenizer
    """
    logger.info(f"Tokenizer 로드 중: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer 로드 완료")
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        raise
    
    # Chat template 설정 (stage3_train_2.py와 동일하게)
    if tokenizer.chat_template is None:
        if HAS_TRL:
            tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
            logger.info("📝 기본 chat template 적용 (SIMPLE_CHAT_TEMPLATE)")
        else:
            logger.warning("⚠️ TRL이 없어 chat template을 설정하지 않습니다.")
    else:
        logger.info("📝 모델에 이미 chat template이 설정되어 있습니다.")
    
    return tokenizer


def count_tokens_with_tokenizer(tokenizer, text: str, apply_chat_template: bool = False) -> int:
    """
    Tokenizer를 사용하여 정확한 토큰 수 계산
    
    Args:
        tokenizer: tokenizer 인스턴스
        text: 텍스트 문자열 (이미 chat template이 적용된 형태일 수 있음)
        apply_chat_template: chat template 적용 여부 (기본값: False)
                           curation.py에서 이미 적용했으므로 기본값은 False
        
    Returns:
        토큰 수
    """
    if pd.isna(text) or not text:
        return 0
    
    try:
        text_str = str(text)
        
        # chat template 적용 (필요한 경우에만)
        if apply_chat_template:
            text_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": text_str}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        
        # 토큰화
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text_str, add_special_tokens=False)
            return len(tokens)
        elif hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text_str)
            return len(tokens)
        else:
            return len(text_str.split())
    except Exception as e:
        logger.warning(f"토큰 수 계산 실패: {e}")
        return len(str(text).split())


def load_parquet_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parquet 파일을 안정적으로 로드
    
    Args:
        file_path: Parquet 파일 경로
    
    Returns:
        로드된 데이터프레임 또는 None
    """
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    
    if not file_path.endswith('.parquet'):
        logger.error(f"Parquet 파일이 아닙니다: {file_path}")
        return None
    
    logger.info(f"파일 로드 중: {file_path}")
    try:
        # 먼저 기본 pandas read_parquet로 시도 (가장 안정적)
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
        except Exception as e1:
            logger.warning(f"PyArrow engine 실패: {e1}, fastparquet로 재시도...")
            try:
                df = pd.read_parquet(file_path, engine='fastparquet')
            except Exception as e2:
                logger.warning(f"fastparquet 실패: {e2}, 기본 설정으로 재시도...")
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e3:
                    # 마지막 시도: PyArrow로 직접 읽기 (중첩 데이터 제외)
                    if HAS_PYARROW:
                        logger.warning(f"기본 pandas read_parquet 실패: {e3}, PyArrow 직접 읽기 시도...")
                        try:
                            table = pq.read_table(file_path, memory_map=False, columns=['prompt', 'problem_id', 'problem_text', 'ground_truth', 'set_id', 'solutions', 'confidence_scores'])
                            df = table.to_pandas()
                            logger.info("PyArrow로 필수 컬럼만 로드 성공")
                        except Exception as e4:
                            logger.error(f"모든 로드 방법 실패: {e4}")
                            raise e4
                    else:
                        raise e3

        logger.info(f"로드 완료: {len(df)}개 행")
        logger.info(f"컬럼: {df.columns.to_list()}")
        
        # string 타입을 large_string으로 변환 (offset overflow 방지)
        if HAS_PYARROW:
            try:
                logger.info("string 타입을 large_string으로 변환 중...")
                string_cols = df.select_dtypes(include=['string[pyarrow]']).columns
                if not string_cols.empty:
                    for col in string_cols:
                        try:
                            df[col] = df[col].astype('large_string[pyarrow]')
                        except Exception as ce:
                            logger.warning(f"'{col}' 컬럼 large_string 변환 실패: {ce}")
                else:
                    object_cols = df.select_dtypes(include=['object']).columns
                    for col in object_cols:
                        try:
                            if not df[col].empty and isinstance(df[col].dropna().iloc[0], str):
                                df[col] = df[col].astype('large_string[pyarrow]')
                        except Exception as oe:
                            logger.warning(f"'{col}' (object) 컬럼 변환 중 오류 (무시): {oe}")
            except Exception as e:
                logger.warning(f"large_string 변환 단계에서 경고: {e}")

        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
        return None


def calculate_prompt_token_counts(
    df: pd.DataFrame,
    tokenizer,
    apply_chat_template: bool = False,
    batch_size: int = 1000,
    num_workers: Optional[int] = None,
    model_name: str = "Qwen/Qwen3-1.7B"
) -> pd.DataFrame:
    """
    Prompt의 token count 계산 (멀티프로세싱 방식으로 최적화)

    Args:
        df: 데이터프레임
        tokenizer: tokenizer 인스턴스 (메인 프로세스용)
        apply_chat_template: chat template 적용 여부 (기본값: False)
        batch_size: 각 워커가 처리할 배치 크기 (기본값: 1000)
        num_workers: 워커 프로세스 수 (기본값: CPU 코어 수)
        model_name: 모델 이름 (워커 초기화용)

    Returns:
        prompt_token_count 컬럼이 추가된 데이터프레임
    """
    if 'prompt' not in df.columns:
        logger.error(f"❌ 'prompt' 컬럼이 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
        raise ValueError(f"DataFrame에 'prompt' 컬럼이 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")

    # 워커 수 결정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 1개는 메인 프로세스용으로 남김

    # num_workers가 1이면 멀티프로세싱 비활성화
    use_multiprocessing = num_workers > 1

    if use_multiprocessing:
        logger.info(f"Prompt token count 계산 중... (멀티프로세싱, {num_workers} workers)")
    else:
        logger.info(f"Prompt token count 계산 중... (단일 프로세스)")

    start_time = time.time()
    total = len(df)

    # 전체 prompts 가져오기
    all_prompts = df['prompt'].tolist()

    # 전처리: 빈 prompt 처리 및 문자열 변환
    logger.info("전처리 중...")
    valid_prompts = []
    prompt_indices = []

    for idx, prompt in enumerate(all_prompts):
        if pd.isna(prompt) or not prompt:
            continue

        prompt_str = str(prompt)

        # chat template 적용 (필요한 경우)
        if apply_chat_template:
            try:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_str}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception as e:
                logger.warning(f"Chat template 적용 실패 (idx={idx}): {e}")

        valid_prompts.append(prompt_str)
        prompt_indices.append(idx)

    logger.info(f"유효한 prompts: {len(valid_prompts)}/{total}")

    # 결과 배열 초기화
    prompt_token_counts = [0] * total

    if len(valid_prompts) == 0:
        df['prompt_token_count'] = prompt_token_counts
        return df

    # 배치로 나누기
    batches = []
    for i in range(0, len(valid_prompts), batch_size):
        end_idx = min(i + batch_size, len(valid_prompts))
        batches.append(valid_prompts[i:end_idx])

    if use_multiprocessing:
        logger.info(f"멀티프로세싱 토큰화 시작... ({len(batches)} 배치, {num_workers} workers)")
    else:
        logger.info(f"단일 프로세스 토큰화 시작... ({len(batches)} 배치)")

    # 멀티프로세싱 또는 단일 프로세스로 처리
    if use_multiprocessing:
        # 멀티프로세싱 사용
        try:
            logger.info("멀티프로세싱 Pool 생성 중...")
            with Pool(processes=num_workers, initializer=_init_worker, initargs=(model_name,)) as pool:
                results = []
                logger.info("배치 처리 시작...")
                for i, result in enumerate(pool.imap(_tokenize_batch, batches)):
                    results.extend(result)

                    # 진행률 표시
                    if (i + 1) % max(1, len(batches) // 10) == 0 or i == len(batches) - 1:
                        progress = (i + 1) / len(batches) * 100
                        elapsed = time.time() - start_time
                        items_processed = len(results)
                        items_per_sec = items_processed / elapsed if elapsed > 0 else 0
                        eta_seconds = (len(valid_prompts) - items_processed) / items_per_sec if items_per_sec > 0 else 0
                        logger.info(f"진행: {i+1}/{len(batches)} 배치 ({progress:.1f}%) | 속도: {items_per_sec:.0f} items/s | ETA: {eta_seconds:.0f}s")

            logger.info(f"멀티프로세싱 완료, 결과 개수: {len(results)}, 예상: {len(valid_prompts)}")

            # 결과를 원래 위치에 저장
            if len(results) != len(valid_prompts):
                raise ValueError(f"결과 개수 불일치: {len(results)} != {len(valid_prompts)}")

            for orig_idx, count in zip(prompt_indices, results):
                prompt_token_counts[orig_idx] = count

            logger.info("✅ 멀티프로세싱 성공")

        except Exception as e:
            import traceback
            logger.error(f"멀티프로세싱 실패: {e}")
            logger.error(f"상세 에러:\n{traceback.format_exc()}")
            logger.info("Fallback: 단일 프로세스로 처리합니다...")

            # Fallback: 단일 프로세스로 처리
            for i, prompt in enumerate(valid_prompts):
                if i % 1000 == 0:
                    logger.info(f"Fallback 진행: {i}/{len(valid_prompts)}")
                try:
                    tokens = tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_token_counts[prompt_indices[i]] = len(tokens)
                except Exception as e2:
                    logger.warning(f"토큰화 실패 (idx={prompt_indices[i]}): {e2}")
                    prompt_token_counts[prompt_indices[i]] = len(prompt.split())

            logger.info("✅ Fallback 처리 완료")

    else:
        # 단일 프로세스로 대용량 배치 처리
        logger.info("단일 프로세스 배치 토큰화...")
        for i, batch in enumerate(batches):
            # 진행률 표시
            if i % max(1, len(batches) // 10) == 0 or i == len(batches) - 1:
                progress = (i + 1) / len(batches) * 100
                elapsed = time.time() - start_time
                items_processed = (i + 1) * batch_size
                items_per_sec = items_processed / elapsed if elapsed > 0 else 0
                eta_seconds = (len(batches) - i - 1) * batch_size / items_per_sec if items_per_sec > 0 else 0
                logger.info(f"진행: {i+1}/{len(batches)} 배치 ({progress:.1f}%) | 속도: {items_per_sec:.0f} items/s | ETA: {eta_seconds:.0f}s")

            try:
                # 배치 토큰화
                encoded = tokenizer(
                    batch,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )

                # 결과 저장
                start_idx = i * batch_size
                for j, input_ids in enumerate(encoded['input_ids']):
                    orig_idx = prompt_indices[start_idx + j]
                    prompt_token_counts[orig_idx] = len(input_ids)

            except Exception as e:
                logger.warning(f"배치 토큰화 실패 (batch {i}): {e}, 개별 처리...")
                # Fallback: 개별 토큰화
                start_idx = i * batch_size
                for j, prompt in enumerate(batch):
                    try:
                        tokens = tokenizer.encode(prompt, add_special_tokens=False)
                        orig_idx = prompt_indices[start_idx + j]
                        prompt_token_counts[orig_idx] = len(tokens)
                    except:
                        orig_idx = prompt_indices[start_idx + j]
                        prompt_token_counts[orig_idx] = len(prompt.split())

        logger.info("✅ 단일 프로세스 처리 완료")

    elapsed = time.time() - start_time
    logger.info(f"✅ Prompt token count 계산 완료: {total}개 항목, {elapsed:.1f}초 소요 ({total/elapsed:.0f} items/s)")

    # 결과 확인
    non_zero_count = sum(1 for c in prompt_token_counts if c > 0)
    logger.info(f"토큰 수가 0보다 큰 항목: {non_zero_count}/{total}")

    df = df.copy()  # 원본 보호
    df['prompt_token_count'] = prompt_token_counts

    # 추가된 컬럼 확인
    if 'prompt_token_count' not in df.columns:
        logger.error("❌ 'prompt_token_count' 컬럼 추가 실패!")
        raise ValueError("Failed to add 'prompt_token_count' column")

    logger.info(f"✅ 'prompt_token_count' 컬럼 추가 완료 (샘플: {df['prompt_token_count'].head().tolist()})")

    return df


def calculate_correct_solution_distribution(
    df: pd.DataFrame,
    verifier: MathVerifier,
    easy_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    각 instance에 포함된 solution 중 정답인 solution의 개수 분포 계산

    Args:
        df: 데이터프레임 (solutions, ground_truth 컬럼 필요)
        verifier: MathVerifier 인스턴스
        easy_threshold: Easy/Hard 분류 임계값 (정답률 >= threshold이면 Easy)

    Returns:
        정답 개수 분포 정보를 담은 딕셔너리
    """
    logger.info("정답 solution 개수 분포 계산 중...")

    if 'solutions' not in df.columns or 'ground_truth' not in df.columns:
        logger.warning("'solutions' 또는 'ground_truth' 컬럼이 없어 정답 개수 분포를 계산할 수 없습니다.")
        return {
            'distribution': {},
            'total_instances': 0,
            'statistics': {},
            'hard_easy_split': {'hard': 0, 'easy': 0, 'easy_ratio': 0.0, 'easy_percentage': 0.0, 'threshold': easy_threshold}
        }

    correct_counts = []
    total_solution_counts = []
    hard_count = 0
    easy_count = 0

    for idx, row in df.iterrows():
        try:
            # solutions가 JSON 문자열인 경우 파싱
            solutions = row['solutions']
            if isinstance(solutions, str):
                solutions = json.loads(solutions)

            ground_truth = row['ground_truth']

            if not isinstance(solutions, list):
                logger.warning(f"Row {idx}: solutions가 리스트가 아님, 건너뜀")
                continue

            # 정답 개수 카운트
            correct_count = 0
            for solution in solutions:
                if isinstance(solution, dict):
                    final_answer = solution.get('final_answer', '')
                    if verifier.verify_answer(final_answer, ground_truth):
                        correct_count += 1

            correct_counts.append(correct_count)
            total_solutions = len(solutions)
            total_solution_counts.append(total_solutions)

            # Hard/Easy 분류 (정답률 기준)
            accuracy = correct_count / total_solutions if total_solutions > 0 else 0.0
            if accuracy >= easy_threshold:
                easy_count += 1
            else:
                hard_count += 1

        except Exception as e:
            logger.warning(f"Row {idx} 처리 중 오류: {e}")
            continue

    if len(correct_counts) == 0:
        logger.warning("정답 개수를 계산할 수 있는 인스턴스가 없습니다.")
        return {
            'distribution': {},
            'total_instances': 0,
            'statistics': {},
            'hard_easy_split': {'hard': 0, 'easy': 0, 'easy_ratio': 0.0, 'easy_percentage': 0.0, 'threshold': easy_threshold}
        }

    # 정답 개수별 분포 계산
    count_freq = defaultdict(int)
    for count in correct_counts:
        count_freq[count] += 1

    sorted_distribution = dict(sorted(count_freq.items()))

    # 통계 계산
    correct_counts_array = np.array(correct_counts)
    statistics = {
        'mean': float(np.mean(correct_counts_array)),
        'median': float(np.median(correct_counts_array)),
        'std': float(np.std(correct_counts_array)),
        'min': int(np.min(correct_counts_array)),
        'max': int(np.max(correct_counts_array)),
        'q25': float(np.percentile(correct_counts_array, 25)),
        'q75': float(np.percentile(correct_counts_array, 75))
    }

    # 평균 solution 개수
    avg_total_solutions = np.mean(total_solution_counts) if total_solution_counts else 0

    # Hard/Easy 비율 계산
    total_instances = len(correct_counts)
    easy_ratio = easy_count / total_instances if total_instances > 0 else 0.0

    result = {
        'distribution': sorted_distribution,
        'total_instances': total_instances,
        'statistics': statistics,
        'avg_total_solutions': float(avg_total_solutions),
        'hard_easy_split': {
            'hard': hard_count,
            'easy': easy_count,
            'easy_ratio': float(easy_ratio),
            'easy_percentage': float(easy_ratio * 100),
            'threshold': easy_threshold
        }
    }

    logger.info(f"✅ 정답 개수 분포 계산 완료: {len(correct_counts)}개 인스턴스 (Hard: {hard_count}, Easy: {easy_count}, Easy 비율: {easy_ratio*100:.2f}%)")

    return result


def print_correct_solution_distribution(
    distribution_info: Dict[str, Any],
    dataset_name: str,
    prefix: str = ""
):
    """
    정답 solution 개수 분포 출력

    Args:
        distribution_info: calculate_correct_solution_distribution의 결과
        dataset_name: 데이터셋 이름
        prefix: 출력 제목에 추가할 접두사
    """
    if distribution_info['total_instances'] == 0:
        logger.info(f"\n{prefix} {dataset_name}: 정답 개수 분포 데이터 없음")
        return

    title = f"{prefix} {dataset_name}" if prefix else dataset_name
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {title} 정답 Solution 개수 분포 ===")
    logger.info(f"{'='*60}")

    total_instances = distribution_info['total_instances']
    distribution = distribution_info['distribution']
    statistics = distribution_info['statistics']
    avg_total_solutions = distribution_info.get('avg_total_solutions')

    logger.info(f"총 인스턴스 수: {total_instances}")
    if avg_total_solutions is not None:
        logger.info(f"평균 solution 개수: {avg_total_solutions:.2f}")

    # Hard/Easy 분류 정보 출력
    hard_easy_split = distribution_info.get('hard_easy_split')
    if hard_easy_split:
        logger.info(f"\nHard/Easy 분류:")
        logger.info(f"  Hard: {hard_easy_split['hard']}개")
        logger.info(f"  Easy: {hard_easy_split['easy']}개")
        logger.info(f"  Easy 비율: {hard_easy_split['easy_percentage']:.2f}% (threshold: {hard_easy_split['threshold']})")

    logger.info(f"\n정답 개수별 분포:")
    for correct_count, freq in distribution.items():
        percentage = freq / total_instances * 100
        logger.info(f"  정답 {correct_count}개: {freq}개 인스턴스 ({percentage:.2f}%)")

    if statistics:
        logger.info(f"\n통계:")
        logger.info(f"  평균 정답 개수: {statistics['mean']:.2f}")
        logger.info(f"  중앙값: {statistics['median']:.2f}")
        logger.info(f"  표준편차: {statistics['std']:.2f}")
        logger.info(f"  최소값: {statistics['min']}")
        logger.info(f"  최대값: {statistics['max']}")
        logger.info(f"  25% 분위수: {statistics['q25']:.2f}")
        logger.info(f"  75% 분위수: {statistics['q75']:.2f}")


def calculate_filtered_distribution(
    before_dist: Dict[str, Any],
    after_dist: Dict[str, Any]
) -> Dict[str, Any]:
    """
    필터링으로 제거된 인스턴스들의 정답 개수 분포 계산 (before - after)

    Args:
        before_dist: 필터링 전 분포
        after_dist: 필터링 후 분포

    Returns:
        제거된 인스턴스들의 정답 개수 분포
    """
    before_distribution = before_dist.get('distribution', {})
    after_distribution = after_dist.get('distribution', {})

    # before - after 계산
    filtered_distribution = {}
    all_keys = set(before_distribution.keys()) | set(after_distribution.keys())

    for key in all_keys:
        before_count = before_distribution.get(key, 0)
        after_count = after_distribution.get(key, 0)
        filtered_count = before_count - after_count
        if filtered_count > 0:
            filtered_distribution[key] = filtered_count

    # 정렬
    filtered_distribution = dict(sorted(filtered_distribution.items()))

    total_filtered = sum(filtered_distribution.values())

    if total_filtered == 0:
        # threshold 가져오기 (기본값: 0.5)
        threshold = before_dist.get('hard_easy_split', {}).get('threshold', 0.5)
        return {
            'distribution': {},
            'total_instances': 0,
            'statistics': {},
            'hard_easy_split': {'hard': 0, 'easy': 0, 'easy_ratio': 0.0, 'easy_percentage': 0.0, 'threshold': threshold}
        }

    # 통계 계산을 위해 정답 개수 리스트 생성
    correct_counts = []
    for correct_count, freq in filtered_distribution.items():
        correct_counts.extend([int(correct_count)] * freq)

    correct_counts_array = np.array(correct_counts)
    statistics = {
        'mean': float(np.mean(correct_counts_array)),
        'median': float(np.median(correct_counts_array)),
        'std': float(np.std(correct_counts_array)),
        'min': int(np.min(correct_counts_array)),
        'max': int(np.max(correct_counts_array)),
        'q25': float(np.percentile(correct_counts_array, 25)),
        'q75': float(np.percentile(correct_counts_array, 75))
    }

    # Hard/Easy 분류 계산 (before - after)
    before_hard_easy = before_dist.get('hard_easy_split', {'hard': 0, 'easy': 0})
    after_hard_easy = after_dist.get('hard_easy_split', {'hard': 0, 'easy': 0})

    filtered_hard = before_hard_easy['hard'] - after_hard_easy['hard']
    filtered_easy = before_hard_easy['easy'] - after_hard_easy['easy']
    filtered_easy_ratio = filtered_easy / total_filtered if total_filtered > 0 else 0.0

    threshold = before_hard_easy.get('threshold', 0.5)

    return {
        'distribution': filtered_distribution,
        'total_instances': total_filtered,
        'statistics': statistics,
        'hard_easy_split': {
            'hard': filtered_hard,
            'easy': filtered_easy,
            'easy_ratio': float(filtered_easy_ratio),
            'easy_percentage': float(filtered_easy_ratio * 100),
            'threshold': threshold
        }
    }


def calculate_bin_distribution(token_counts: np.ndarray) -> Tuple[List[str], List[int], List[float]]:
    """
    Token count의 bin별 분포 계산
    
    Args:
        token_counts: 토큰 수 배열
        
    Returns:
        (labels, counts, percentages): bin 레이블, 각 bin의 개수, 각 bin의 비율
    """
    non_zero_mask = token_counts > 0
    non_zero_counts = token_counts[non_zero_mask]
    
    if len(non_zero_counts) == 0:
        return [], [], []
    
    max_val = non_zero_counts.max()
    if max_val < 1000:
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, float('inf')]
        labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                 '500-600', '600-700', '700-800', '800-900', '900+']
    elif max_val < 5000:
        bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, float('inf')]
        labels = ['0-500', '500-1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', 
                 '2.5K-3K', '3K-3.5K', '3.5K-4K', '4K+']
    else:
        bins = [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, float('inf')]
        labels = [
            '0-1K', '1K-2K', '2K-4K', '4K-6K', '6K-8K',
            '8K-10K', '10K-12K', '12K-14K', '14K-16K', '16K+'
        ]
    
    counts = []
    percentages = []
    for i in range(len(bins)-1):
        if i < len(bins) - 1:
            count = ((non_zero_counts >= bins[i]) & (non_zero_counts < bins[i+1])).sum()
        else:
            count = (non_zero_counts >= bins[i]).sum()
        percentage = count / len(non_zero_counts) * 100 if len(non_zero_counts) > 0 else 0
        counts.append(count)
        percentages.append(percentage)
    
    return labels, counts, percentages


def print_statistics(token_counts: np.ndarray, dataset_name: str, prefix: str = ""):
    """
    Token count 통계 출력

    Args:
        token_counts: 토큰 수 배열
        dataset_name: 데이터셋 이름
        prefix: 출력 제목에 추가할 접두사
    """
    title = f"{prefix} {dataset_name}" if prefix else dataset_name
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {title} Prompt Token Count 통계 ===")
    logger.info(f"{'='*60}")
    
    non_zero_mask = token_counts > 0
    non_zero_counts = token_counts[non_zero_mask]
    
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"토큰이 있는 데이터 수: {non_zero_mask.sum()} ({non_zero_mask.sum()/len(token_counts)*100:.1f}%)")
    
    if len(non_zero_counts) > 0:
        logger.info(f"\n기본 통계:")
        logger.info(f"  평균 토큰 수 (전체): {token_counts.mean():.2f}")
        logger.info(f"  평균 토큰 수 (토큰 있음): {non_zero_counts.mean():.2f}")
        logger.info(f"  중앙값 (토큰 있음): {np.median(non_zero_counts):.2f}")
        logger.info(f"  표준편차 (토큰 있음): {non_zero_counts.std():.2f}")
        logger.info(f"  최소값 (토큰 있음): {non_zero_counts.min():.2f}")
        logger.info(f"  최대값 (토큰 있음): {non_zero_counts.max():.2f}")
        
        # 분위수
        percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
        logger.info(f"\n분위수 (토큰 있음):")
        for p in percentiles:
            logger.info(f"  {p}%: {np.percentile(non_zero_counts, p):.2f}")
        
        # 구간별 분포
        logger.info(f"\n토큰 수 구간별 분포:")
        labels, counts, percentages = calculate_bin_distribution(token_counts)
        for label, count, percentage in zip(labels, counts, percentages):
            logger.info(f"  {label}: {count}개 ({percentage:.1f}%)")


def visualize_distribution(
    train_counts: np.ndarray,
    valid_counts: Optional[np.ndarray],
    all_counts: np.ndarray,
    output_dir: str,
    prefix: str = ""
):
    """
    Token count 분포 시각화

    Args:
        train_counts: Train 데이터 토큰 수 배열
        valid_counts: Validation 데이터 토큰 수 배열 (None 가능)
        all_counts: 전체 데이터 토큰 수 배열
        output_dir: 출력 디렉토리
        prefix: 파일명에 추가할 접두사
    """
    logger.info(f"시각화 생성 중... {f'({prefix})' if prefix else ''}")
    
    try:
        # 1. 히스토그램 비교 (Train vs Validation)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Train 히스토그램
        ax = axes[0, 0]
        train_non_zero = train_counts[train_counts > 0] if train_counts is not None else np.array([])
        if len(train_non_zero) > 0:
            ax.hist(train_non_zero, bins=50, alpha=0.7, color='#4C78A8', edgecolor='black')
            ax.set_xlabel('Prompt Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Train Prompt Token Count Distribution\n(n={len(train_counts)}, mean={train_non_zero.mean():.1f})')
            ax.grid(True, alpha=0.3)
        
        # Validation 히스토그램
        ax = axes[0, 1]
        if valid_counts is not None:
            valid_non_zero = valid_counts[valid_counts > 0]
            if len(valid_non_zero) > 0:
                ax.hist(valid_non_zero, bins=50, alpha=0.7, color='#E45756', edgecolor='black')
                ax.set_xlabel('Prompt Token Count')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Validation Prompt Token Count Distribution\n(n={len(valid_counts)}, mean={valid_non_zero.mean():.1f})')
                ax.grid(True, alpha=0.3)
        else:
            valid_non_zero = np.array([])
            ax.text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Prompt Token Count Distribution\n(No Data)')
        
        # 전체 히스토그램
        ax = axes[1, 0]
        all_non_zero = all_counts[all_counts > 0] if all_counts is not None else np.array([])
        if len(all_non_zero) > 0:
            ax.hist(all_non_zero, bins=50, alpha=0.7, color='#54A24B', edgecolor='black')
            ax.set_xlabel('Prompt Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'All Prompt Token Count Distribution\n(n={len(all_counts)}, mean={all_non_zero.mean():.1f})')
            ax.grid(True, alpha=0.3)
        
        # 비교 Box Plot
        ax = axes[1, 1]
        data_to_plot = []
        labels = []
        if len(train_non_zero) > 0:
            data_to_plot.append(train_non_zero)
            labels.append('Train')
        if valid_counts is not None and len(valid_non_zero) > 0:
            data_to_plot.append(valid_non_zero)
            labels.append('Validation')
        if len(all_non_zero) > 0:
            data_to_plot.append(all_non_zero)
            labels.append('All')
        
        if data_to_plot:
            ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_ylabel('Prompt Token Count')
            ax.set_title('Prompt Token Count Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{prefix}_prompt_token_distribution.png' if prefix else 'prompt_token_distribution.png'
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {plot_path}")
        
        # 2. CDF (Cumulative Distribution Function) 비교
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(train_non_zero) > 0:
            sorted_train = np.sort(train_non_zero)
            y_train = np.arange(1, len(sorted_train) + 1) / len(sorted_train)
            ax.plot(sorted_train, y_train, label='Train', linewidth=2, color='#4C78A8')
        
        if valid_counts is not None and len(valid_non_zero) > 0:
            sorted_valid = np.sort(valid_non_zero)
            y_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
            ax.plot(sorted_valid, y_valid, label='Validation', linewidth=2, color='#E45756')
        
        if len(all_non_zero) > 0:
            sorted_all = np.sort(all_non_zero)
            y_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
            ax.plot(sorted_all, y_all, label='All', linewidth=2, color='#54A24B')
        
        ax.set_xlabel('Prompt Token Count')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function (CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{prefix}_prompt_token_cdf.png' if prefix else 'prompt_token_cdf.png'
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"CDF 시각화 저장: {plot_path}")
        
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")
        import traceback
        logger.error(f"상세 에러:\n{traceback.format_exc()}")


def apply_balanced_sampling(
    df: pd.DataFrame,
    verifier: MathVerifier,
    set_size: int = 8
) -> pd.DataFrame:
    """
    정답 개수 분포에 따라 데이터 샘플링 (Balanced Sampling)
    - 0개 정답과 set_size개 정답인 그룹을 나머지 그룹(1 ~ set_size-1)의 평균 개수로 다운샘플링
    
    Args:
        df: 데이터프레임
        verifier: 정답 검증기
        set_size: 세트 크기
        
    Returns:
        샘플링된 데이터프레임
    """
    logger.info("\n" + "="*60)
    logger.info("=== Balanced Sampling 적용 ===")
    logger.info("="*60)
    
    # 1. 각 행의 정답 개수 계산
    logger.info("각 행의 정답 개수 계산 중...")
    correct_counts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            solutions = row['solutions']
            if isinstance(solutions, str):
                solutions = json.loads(solutions)
            
            ground_truth = row['ground_truth']
            
            count = 0
            if isinstance(solutions, list):
                for sol in solutions:
                    if isinstance(sol, dict):
                        if verifier.verify_answer(sol.get('final_answer', ''), ground_truth):
                            count += 1
            
            correct_counts.append(count)
            valid_indices.append(idx)
        except Exception as e:
            continue
            
    # 임시 데이터프레임 생성
    temp_df = df.loc[valid_indices].copy()
    temp_df['correct_count'] = correct_counts
    
    # 2. 중간 그룹(1 ~ set_size-1)의 평균 개수 계산
    middle_counts = temp_df[
        (temp_df['correct_count'] > 0) & 
        (temp_df['correct_count'] < set_size)
    ]
    
    if len(middle_counts) == 0:
        logger.warning("⚠️ 중간 분포(1 ~ set_size-1)가 없어 Balanced Sampling을 건너뜁니다.")
        return df
        
    # 각 정답 개수별 빈도 계산
    bucket_counts = middle_counts['correct_count'].value_counts()
    average_count = int(bucket_counts.mean())
    logger.info(f"중간 그룹(1 ~ {set_size-1}개 정답) 통계:")
    logger.info(f"  총 개수: {len(middle_counts)}")
    logger.info(f"  그룹별 평균 개수: {average_count}")
    
    # 3. 샘플링 적용
    final_indices = []
    
    # 0부터 set_size까지 각 그룹별로 처리
    for i in range(set_size + 1):
        bucket_df = temp_df[temp_df['correct_count'] == i]
        count = len(bucket_df)
        
        if count == 0:
            continue
            
        if i == 0 or i == set_size:
            # 양극단 그룹: 평균 개수로 다운샘플링 (초과 시)
            if count > average_count:
                sampled = bucket_df.sample(n=average_count, random_state=42)
                final_indices.extend(sampled.index.tolist())
                logger.info(f"  Bucket {i} (Extreme): {count} -> {average_count} (Downsampled)")
            else:
                final_indices.extend(bucket_df.index.tolist())
                logger.info(f"  Bucket {i} (Extreme): {count} (Kept all)")
        else:
            # 중간 그룹: 모두 유지
            final_indices.extend(bucket_df.index.tolist())
            logger.info(f"  Bucket {i} (Middle):  {count} (Kept all)")
            
    balanced_df = df.loc[final_indices].copy()
    logger.info(f"Balanced Sampling 완료: {len(df)} -> {len(balanced_df)} ({len(balanced_df)/len(df)*100:.1f}%)")
    
    return balanced_df


def main(
    train_path: str,
    validation_path: str,
    model_name: str,
    output_dir: Optional[str] = None,
    max_input_length: Optional[int] = None,
    batch_size: int = 1000,
    num_workers: Optional[int] = None,
    easy_threshold: float = 0.5,
    sample_size: Optional[int] = None,
    set_size: int = 8
):
    """
    메인 함수: Prompt Token Count 분포 분석

    Args:
        train_path: Train 데이터 파일 경로
        validation_path: Validation 데이터 파일 경로
        model_name: 모델 이름 (tokenizer 로드용)
        output_dir: 출력 디렉토리
        max_input_length: 최대 입력 길이 제한 (이 값을 넘는 인스턴스 제거)
        batch_size: 각 워커의 배치 크기 (기본값: 1000)
        num_workers: 멀티프로세싱 워커 수 (기본값: CPU 코어 수 - 1)
        easy_threshold: Easy/Hard 분류 임계값 (기본값: 0.5)
        sample_size: 샘플링할 데이터 개수 (None이면 전체 사용)
        set_size: 세트 크기 (Balanced Sampling용, 기본값: 8)
    """
    logger.info("\n" + "="*60)
    logger.info("=== Prompt Token Count 분포 분석 ===")
    logger.info("="*60)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(train_path))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # 1. Tokenizer 로드
    try:
        tokenizer = load_tokenizer(model_name)
        # curation.py에서 이미 chat template을 적용했으므로 재적용하지 않음
        logger.info("✅ curation.py에서 이미 chat template이 적용된 prompt를 사용합니다.")
        logger.info("   토큰 수 계산 시 chat template을 재적용하지 않습니다.")
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        return
    
    # 2. 데이터 로드
    logger.info("\n데이터 로드 중...")
    train_df = load_parquet_file(train_path)
    if train_df is None:
        logger.error("Train 데이터 로드 실패")
        return
    
    # Balanced Sampling 적용 (Train)
    # sample_size가 지정되어 있으면 Balanced Sampling 대신 단순 샘플링 사용 (또는 둘 다 사용? 요구사항은 Balanced Sampling으로 대체하는 것 같음)
    # 하지만 사용자가 "default 값으로 filtering하는게 아니라... 코드 수정해줘"라고 했으므로
    # sample_size 로직은 유지하되, Balanced Sampling을 우선 적용하는 것이 좋음.
    # 하지만 sample_size는 테스트 용도였으므로, Balanced Sampling을 메인으로 적용.
    
    if sample_size is not None:
        # sample_size가 명시적으로 주어지면 단순 샘플링 (테스트용)
        if len(train_df) > sample_size:
            logger.info(f"Train 데이터 단순 샘플링 (테스트용): {len(train_df)} -> {sample_size}")
            train_df = train_df.sample(n=sample_size, random_state=42)
    else:
        # sample_size가 없으면 Balanced Sampling 적용
        verifier = MathVerifier(timeout=30)
        train_df = apply_balanced_sampling(train_df, verifier, set_size)
    
    # Validation 데이터는 선택적 (없을 수 있음)
    valid_df = None
    if validation_path and validation_path.strip():
        valid_df = load_parquet_file(validation_path)
        if valid_df is None:
            logger.warning("⚠️ Validation 데이터 로드 실패 또는 파일이 없습니다. Validation 데이터 없이 진행합니다.")
            valid_df = None
        else:
            # Validation에도 Balanced Sampling 적용? 보통 Train 분석이 목적이므로 Validation은 그대로 두거나 동일하게 적용.
            # 여기서는 Train 분석이 주 목적이므로 Validation은 단순 로드 또는 sample_size만 적용
            if sample_size is not None and len(valid_df) > sample_size:
                logger.info(f"Validation 데이터 샘플링: {len(valid_df)} -> {sample_size}")
                valid_df = valid_df.sample(n=sample_size, random_state=42)
            logger.info(f"✅ Validation 데이터 로드 완료: {len(valid_df)}개 인스턴스")
    else:
        logger.info("⚠️ Validation 경로가 제공되지 않았습니다. Train 데이터만 분석합니다.")
    
    # 3. Prompt token count 계산
    # curation.py에서 이미 chat template이 적용되었으므로 재적용하지 않음
    logger.info("\n" + "="*60)
    logger.info("Train 데이터 Prompt Token Count 계산 중...")
    train_df = calculate_prompt_token_counts(
        train_df,
        tokenizer,
        apply_chat_template=False,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name
    )

    # Validation 데이터 처리 (있는 경우에만)
    if valid_df is not None:
        logger.info("\n" + "="*60)
        logger.info("Validation 데이터 Prompt Token Count 계산 중...")
        valid_df = calculate_prompt_token_counts(
            valid_df,
            tokenizer,
            apply_chat_template=False,
            batch_size=batch_size,
            num_workers=num_workers,
            model_name=model_name
        )
        
        # Validation 컬럼 존재 확인
        if 'prompt_token_count' not in valid_df.columns:
            logger.error("❌ Validation 데이터에 'prompt_token_count' 컬럼이 없습니다.")
            logger.error(f"Valid DataFrame 컬럼: {valid_df.columns.tolist()}")
            raise ValueError("Validation DataFrame에 'prompt_token_count' 컬럼이 없습니다.")

    # Train 컬럼 존재 확인
    if 'prompt_token_count' not in train_df.columns:
        logger.error("❌ Train 데이터에 'prompt_token_count' 컬럼이 없습니다.")
        logger.error(f"Train DataFrame 컬럼: {train_df.columns.tolist()}")
        raise ValueError("Train DataFrame에 'prompt_token_count' 컬럼이 없습니다.")

    # 4. 필터링 전 통계 출력 및 시각화
    logger.info("\n" + "="*60)
    logger.info("=== 필터링 전 전체 분포 분석 ===")
    logger.info("="*60)

    train_counts_before = train_df['prompt_token_count'].values
    valid_counts_before = valid_df['prompt_token_count'].values if valid_df is not None else None
    all_counts_before = train_counts_before.copy()
    if valid_df is not None:
        all_counts_before = np.concatenate([train_counts_before, valid_counts_before])

    print_statistics(train_counts_before, "Train", prefix="[Before Filtering]")
    if valid_df is not None:
        print_statistics(valid_counts_before, "Validation", prefix="[Before Filtering]")
    print_statistics(all_counts_before, "All (Train + Validation)" if valid_df is not None else "All (Train only)", prefix="[Before Filtering]")

    visualize_distribution(train_counts_before, valid_counts_before, all_counts_before, output_dir, prefix="before_filtering")

    # 4-1. 정답 solution 개수 분포 분석 (필터링 전)
    logger.info("\n" + "="*60)
    logger.info("=== 정답 Solution 개수 분포 분석 (필터링 전) ===")
    logger.info("="*60)

    verifier = MathVerifier(timeout=30)

    train_correct_dist_before = calculate_correct_solution_distribution(train_df, verifier, easy_threshold)
    print_correct_solution_distribution(train_correct_dist_before, "Train", prefix="[Before Filtering]")

    if valid_df is not None:
        valid_correct_dist_before = calculate_correct_solution_distribution(valid_df, verifier, easy_threshold)
        print_correct_solution_distribution(valid_correct_dist_before, "Validation", prefix="[Before Filtering]")

    # 정답 개수 분포를 JSON으로 저장
    correct_dist_before = {
        'train': train_correct_dist_before,
    }
    if valid_df is not None:
        correct_dist_before['validation'] = valid_correct_dist_before

    correct_dist_before_path = os.path.join(output_dir, 'correct_solution_distribution_before_filtering.json')
    with open(correct_dist_before_path, 'w', encoding='utf-8') as f:
        json.dump(correct_dist_before, f, indent=2, ensure_ascii=False)
    logger.info(f"정답 개수 분포 저장: {correct_dist_before_path}")

    # 5. max_input_length 제한 적용 (필터링)
    if max_input_length is not None:
        logger.info("\n" + "="*60)
        logger.info(f"Max Input Length 제한 적용: {max_input_length} 토큰")
        logger.info("="*60)

        train_before = len(train_df)
        train_df = train_df[train_df['prompt_token_count'] <= max_input_length].copy()
        train_removed = train_before - len(train_df)
        
        total_before = train_before
        total_removed = train_removed
        
        logger.info(f"Train: {train_removed}개 제거 ({train_removed/train_before*100:.2f}%)")
        
        if valid_df is not None:
            valid_before = len(valid_df)
            valid_df = valid_df[valid_df['prompt_token_count'] <= max_input_length].copy()
            valid_removed = valid_before - len(valid_df)
            total_before += valid_before
            total_removed += valid_removed
            logger.info(f"Validation: {valid_removed}개 제거 ({valid_removed/valid_before*100:.2f}%)")
        
        logger.info(f"전체: {total_removed}개 제거 ({(total_removed/total_before*100):.2f}%)")
        logger.info(f"남은 인스턴스: Train {len(train_df)}개" + (f", Validation {len(valid_df)}개" if valid_df is not None else ""))

        #filtered train_df and valid_df 저장
        train_df.to_parquet(os.path.join(output_dir, 'train_filtered.parquet'))
        if valid_df is not None:
            valid_df.to_parquet(os.path.join(output_dir, 'valid_filtered.parquet'))

        # 6. 필터링 후 통계 출력 및 시각화
        logger.info("\n" + "="*60)
        logger.info("=== 필터링 후 분포 분석 ===")
        logger.info("="*60)

        train_counts = train_df['prompt_token_count'].values
        valid_counts = valid_df['prompt_token_count'].values if valid_df is not None else None
        all_counts = train_counts.copy()
        if valid_df is not None:
            all_counts = np.concatenate([train_counts, valid_counts])

        print_statistics(train_counts, "Train", prefix="[After Filtering]")
        if valid_df is not None:
            print_statistics(valid_counts, "Validation", prefix="[After Filtering]")
        print_statistics(all_counts, "All (Train + Validation)" if valid_df is not None else "All (Train only)", prefix="[After Filtering]")

        visualize_distribution(train_counts, valid_counts, all_counts, output_dir, prefix="after_filtering")

        # 6-1. 정답 solution 개수 분포 분석 (필터링 후)
        logger.info("\n" + "="*60)
        logger.info("=== 정답 Solution 개수 분포 분석 (필터링 후) ===")
        logger.info("="*60)

        train_correct_dist_after = calculate_correct_solution_distribution(train_df, verifier, easy_threshold)
        print_correct_solution_distribution(train_correct_dist_after, "Train", prefix="[After Filtering]")

        if valid_df is not None:
            valid_correct_dist_after = calculate_correct_solution_distribution(valid_df, verifier, easy_threshold)
            print_correct_solution_distribution(valid_correct_dist_after, "Validation", prefix="[After Filtering]")

        # 정답 개수 분포를 JSON으로 저장
        correct_dist_after = {
            'train': train_correct_dist_after,
        }
        if valid_df is not None:
            correct_dist_after['validation'] = valid_correct_dist_after

        correct_dist_after_path = os.path.join(output_dir, 'correct_solution_distribution_after_filtering.json')
        with open(correct_dist_after_path, 'w', encoding='utf-8') as f:
            json.dump(correct_dist_after, f, indent=2, ensure_ascii=False)
        logger.info(f"정답 개수 분포 저장: {correct_dist_after_path}")

        # 6-2. 필터링으로 제거된 인스턴스들의 정답 개수 분포 분석 (before - after)
        logger.info("\n" + "="*60)
        logger.info("=== 필터링으로 제거된 인스턴스들의 정답 Solution 개수 분포 ===")
        logger.info("="*60)

        train_correct_dist_filtered = calculate_filtered_distribution(train_correct_dist_before, train_correct_dist_after)
        print_correct_solution_distribution(train_correct_dist_filtered, "Train", prefix="[Filtered (Removed)]")

        correct_dist_filtered = {
            'train': train_correct_dist_filtered,
        }

        if valid_df is not None:
            valid_correct_dist_filtered = calculate_filtered_distribution(valid_correct_dist_before, valid_correct_dist_after)
            print_correct_solution_distribution(valid_correct_dist_filtered, "Validation", prefix="[Filtered (Removed)]")
            correct_dist_filtered['validation'] = valid_correct_dist_filtered

        # 필터링된 인스턴스들의 정답 개수 분포를 JSON으로 저장
        correct_dist_filtered_path = os.path.join(output_dir, 'correct_solution_distribution_filtered_removed.json')
        with open(correct_dist_filtered_path, 'w', encoding='utf-8') as f:
            json.dump(correct_dist_filtered, f, indent=2, ensure_ascii=False)
        logger.info(f"필터링된 인스턴스 정답 개수 분포 저장: {correct_dist_filtered_path}")

        # 요약 비교 출력
        logger.info("\n" + "="*60)
        logger.info("=== 필터링 전/후/제거 요약 비교 (Train) ===")
        logger.info("="*60)

        train_before_he = train_correct_dist_before['hard_easy_split']
        train_after_he = train_correct_dist_after['hard_easy_split']
        train_filtered_he = train_correct_dist_filtered['hard_easy_split']

        logger.info(f"Before Filtering: {train_correct_dist_before['total_instances']}개 인스턴스, "
                   f"평균 정답 {train_correct_dist_before['statistics'].get('mean', 0):.2f}개, "
                   f"Hard {train_before_he['hard']}개 / Easy {train_before_he['easy']}개 (Easy {train_before_he['easy_percentage']:.2f}%)")
        logger.info(f"After Filtering:  {train_correct_dist_after['total_instances']}개 인스턴스, "
                   f"평균 정답 {train_correct_dist_after['statistics'].get('mean', 0):.2f}개, "
                   f"Hard {train_after_he['hard']}개 / Easy {train_after_he['easy']}개 (Easy {train_after_he['easy_percentage']:.2f}%)")
        logger.info(f"Filtered (Removed): {train_correct_dist_filtered['total_instances']}개 인스턴스, "
                   f"평균 정답 {train_correct_dist_filtered['statistics'].get('mean', 0):.2f}개, "
                   f"Hard {train_filtered_he['hard']}개 / Easy {train_filtered_he['easy']}개 (Easy {train_filtered_he['easy_percentage']:.2f}%)")

        if valid_df is not None:
            logger.info("\n" + "="*60)
            logger.info("=== 필터링 전/후/제거 요약 비교 (Validation) ===")
            logger.info("="*60)

            valid_before_he = valid_correct_dist_before['hard_easy_split']
            valid_after_he = valid_correct_dist_after['hard_easy_split']
            valid_filtered_he = valid_correct_dist_filtered['hard_easy_split']

            logger.info(f"Before Filtering: {valid_correct_dist_before['total_instances']}개 인스턴스, "
                       f"평균 정답 {valid_correct_dist_before['statistics'].get('mean', 0):.2f}개, "
                       f"Hard {valid_before_he['hard']}개 / Easy {valid_before_he['easy']}개 (Easy {valid_before_he['easy_percentage']:.2f}%)")
            logger.info(f"After Filtering:  {valid_correct_dist_after['total_instances']}개 인스턴스, "
                       f"평균 정답 {valid_correct_dist_after['statistics'].get('mean', 0):.2f}개, "
                       f"Hard {valid_after_he['hard']}개 / Easy {valid_after_he['easy']}개 (Easy {valid_after_he['easy_percentage']:.2f}%)")
            logger.info(f"Filtered (Removed): {valid_correct_dist_filtered['total_instances']}개 인스턴스, "
                       f"평균 정답 {valid_correct_dist_filtered['statistics'].get('mean', 0):.2f}개, "
                       f"Hard {valid_filtered_he['hard']}개 / Easy {valid_filtered_he['easy']}개 (Easy {valid_filtered_he['easy_percentage']:.2f}%)")
    else:
        # 필터링을 하지 않는 경우에는 before 데이터를 그대로 사용
        train_counts = train_counts_before
        valid_counts = valid_counts_before
        all_counts = all_counts_before

    # 7. 결과를 CSV로 저장 (필터링 후 데이터)
    try:
        def safe_stat(arr, func):
            if arr is None:
                return 0
            non_zero = arr[arr > 0] if len(arr) > 0 else np.array([])
            return func(non_zero) if len(non_zero) > 0 else 0
        
        def count_over_max(arr, max_tokens):
            if arr is None or max_tokens is None:
                return 0
            return (arr > max_tokens).sum()
        
        # 데이터셋 리스트 구성
        if valid_df is not None and valid_counts is not None:
            datasets = ['Train', 'Validation', 'All']
            count_arrays = [train_counts, valid_counts, all_counts]
        else:
            datasets = ['Train', 'All']
            count_arrays = [train_counts, all_counts]
        
        # 기본 통계 계산
        total_samples = [len(arr) for arr in count_arrays]
        mean_vals = [safe_stat(arr, lambda x: x.mean()) for arr in count_arrays]
        median_vals = [safe_stat(arr, np.median) for arr in count_arrays]
        std_vals = [safe_stat(arr, lambda x: x.std()) for arr in count_arrays]
        min_vals = [safe_stat(arr, lambda x: x.min()) for arr in count_arrays]
        max_vals = [safe_stat(arr, lambda x: x.max()) for arr in count_arrays]
        p95_vals = [safe_stat(arr, lambda x: np.percentile(x, 95)) for arr in count_arrays]
        p99_vals = [safe_stat(arr, lambda x: np.percentile(x, 99)) for arr in count_arrays]
        
        # max_tokens 초과 개수 계산
        over_max_counts = [count_over_max(arr, max_input_length) for arr in count_arrays]
        over_max_percentages = [
            (over_max_counts[i] / total_samples[i] * 100) if total_samples[i] > 0 else 0
            for i in range(len(datasets))
        ]
        
        # 기본 통계 DataFrame
        summary_data = {
            'dataset': datasets,
            'total_samples': total_samples,
            'mean': mean_vals,
            'median': median_vals,
            'std': std_vals,
            'min': min_vals,
            'max': max_vals,
            'p95': p95_vals,
            'p99': p99_vals,
            f'over_{max_input_length}_count': over_max_counts,
            f'over_{max_input_length}_percentage': [f'{p:.2f}%' for p in over_max_percentages]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # 기본 통계 저장
        summary_path = os.path.join(output_dir, 'prompt_token_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n통계 요약 저장: {summary_path}")
        
        # Bin별 분포 계산 및 저장
        # 모든 데이터셋에서 사용할 bin 결정 (가장 큰 max 값을 기준)
        all_max_vals = [safe_stat(arr, lambda x: x.max()) for arr in count_arrays]
        overall_max = max(all_max_vals) if all_max_vals else 0
        
        # Bin 레이블 결정
        if overall_max < 1000:
            bin_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                         '500-600', '600-700', '700-800', '800-900', '900+']
        elif overall_max < 5000:
            bin_labels = ['0-500', '500-1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', 
                         '2.5K-3K', '3K-3.5K', '3.5K-4K', '4K+']
        else:
            bin_labels = ['0-1K', '1K-2K', '2K-4K', '4K-6K', '6K-8K',
                         '8K-10K', '10K-12K', '12K-14K', '14K-16K', '16K+']
        
        # Bin 분포를 별도 파일로 저장
        bin_dist_data = {'bin_range': bin_labels}
        for dataset_idx, dataset_name in enumerate(datasets):
            labels, counts, percentages = calculate_bin_distribution(count_arrays[dataset_idx])
            # bin_labels와 일치하도록 정렬
            bin_counts = []
            bin_percentages = []
            for bin_label in bin_labels:
                if bin_label in labels:
                    idx = labels.index(bin_label)
                    bin_counts.append(counts[idx])
                    bin_percentages.append(f'{percentages[idx]:.2f}%')
                else:
                    bin_counts.append(0)
                    bin_percentages.append('0.00%')
            bin_dist_data[f'{dataset_name}_count'] = bin_counts
            bin_dist_data[f'{dataset_name}_percentage'] = bin_percentages
        
        bin_dist_df = pd.DataFrame(bin_dist_data)
        bin_dist_path = os.path.join(output_dir, 'prompt_token_bin_distribution.csv')
        bin_dist_df.to_csv(bin_dist_path, index=False)
        logger.info(f"Bin별 분포 저장: {bin_dist_path}")
        
    except Exception as e:
        logger.warning(f"통계 요약 저장 실패: {e}")
        import traceback
        logger.error(f"상세 에러:\n{traceback.format_exc()}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 분석 완료!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Validation 데이터의 Prompt Token Count 분포 분석")
    parser.add_argument("--train-path", type=str, default="/mnt/data1/datasets/nlp/conf_agg/curated_4000_32_naive_diverse_confidence/train_curated.parquet",
                       help="Train 데이터 Parquet 파일 경로")
    parser.add_argument("--validation-path", type=str, default=None,
                       help="Validation 데이터 Parquet 파일 경로")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B",
                       help="모델 이름 또는 경로 (tokenizer 로드용)")
    parser.add_argument("--output-dir", type=str, default="/mnt/data1/datasets/nlp/conf_agg/curated_4000_32_naive_diverse_confidence",
                       help="출력 디렉토리 (기본값: train 파일이 있는 디렉토리)")
    parser.add_argument("--max-input-length", type=int, default=16384,
                       help="최대 입력 길이 제한 (이 값을 넘는 인스턴스 제거, 예: 32768)")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="각 워커의 배치 크기 (기본값: 1000, 메모리 부족 시 줄이기)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="멀티프로세싱 워커 수 (기본값: CPU 코어 수 - 1)")
    parser.add_argument("--easy-threshold", type=float, default=0.5,
                       help="Easy/Hard 분류 임계값 (기본값: 0.5, 정답률 >= threshold이면 Easy)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="샘플링할 데이터 개수 (기본값: None, 설정 시 Balanced Sampling 대신 단순 샘플링 사용)")
    parser.add_argument("--set-size", type=int, default=8,
                       help="세트 크기 (Balanced Sampling용, 기본값: 8)")

    args = parser.parse_args()

    main(
        train_path=args.train_path,
        validation_path=args.validation_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_input_length=args.max_input_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        easy_threshold=args.easy_threshold,
        sample_size=args.sample_size,
        set_size=args.set_size
    )

