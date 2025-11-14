#!/usr/bin/env python3
"""
샤드별로 생성된 Parquet 파일들을 병합하고 토큰 분포를 분석하는 스크립트
1. 샤드 파일들을 하나로 병합
2. output_token_count 분포 분석
3. </think> 이후 생성된 토큰 수 분포 분석
4. generated_text 파싱 및 검증 추가
"""
import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import re
import warnings
import json
import random
import sys
import time

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.math_verifier import MathVerifier

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 빠른 Parquet 로드를 위한 PyArrow 가용성 확인
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


def _fast_read_single_parquet(file_path: str):
    """
    PyArrow로 단일 Parquet를 빠르게 로드하여 pandas DataFrame으로 반환.
    """
    t0 = time.time()
    # ParquetFile.read는 memory_map 인자를 지원하지 않으므로 read_table 사용
    table = pq.read_table(file_path, memory_map=True)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    dt = time.time() - t0
    logger.info(f"PyArrow fast read 완료: {len(df)}행, {dt:.2f}s")
    return df


def _fast_read_dataset_dir(data_dir: str):
    """
    PyArrow Dataset로 디렉토리 내 모든 Parquet를 빠르게 로드하여 pandas DataFrame으로 반환.
    출력 파일이 아직 없다는 전제에서 병합 전에 사용.
    """
    t0 = time.time()
    dataset = ds.dataset(data_dir, format="parquet")
    table = dataset.to_table(use_threads=True)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    dt = time.time() - t0
    logger.info(f"PyArrow dataset fast read 완료: {len(df)}행, {dt:.2f}s")
    return df


def extract_reasoning_content(text: str) -> str:
    """
    <think>부터 </think> 전까지 추출
    
    Args:
        text: generated_text
        
    Returns:
        reasoning 내용 (없으면 빈 문자열)
    """
    if pd.isna(text):
        return ""
    
    text_str = str(text)
    
    start_marker = "<think>"
    end_marker = "</think>"
    
    # 시작 마커 찾기
    start_pos = text_str.find(start_marker)
    if start_pos == -1:
        return ""
    
    # 끝 마커 찾기 (시작 마커 이후부터)
    end_pos = text_str.find(end_marker, start_pos)
    if end_pos == -1:
        return ""
    
    # reasoning 부분 추출 (시작 마커는 포함하지 않음)
    reasoning_start = start_pos + len(start_marker)
    reasoning_content = text_str[reasoning_start:end_pos].strip()
    
    return reasoning_content


def extract_content(text: str) -> str:
    """
    </think> 토큰 이후 값들 추출
    
    Args:
        text: generated_text
        
    Returns:
        </think> 이후 내용
    """
    if pd.isna(text):
        return ""
    
    text_str = str(text)
    marker = "</think>"
    
    marker_pos = text_str.find(marker)
    if marker_pos == -1:
        return ""
    
    # 마커 이후 텍스트 추출
    content = text_str[marker_pos + len(marker):].strip()
    return content


import re
from typing import Optional

def extract_final_answer_from_content(content: str) -> str:
    """
    content에서 \boxed{} 안의 값 추출
    
    Args:
        content: </think> 이후 내용
        
    Returns:
        boxed 안의 값 (없으면 빈 문자열)
    """
    if pd.isna(content) or not content:
        return ""
    
    content_str = str(content)
    
    # \boxed{} 패턴 찾기
    # 여러 패턴 시도: \boxed{}, \boxed{...}, \\boxed{}
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{content}
        r'\\boxed\{((?:[^}]|\\\})*)\}',  # 중첩 중괄호 처리
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content_str)
        if match:
            return match.group(1).strip()
    
    # 패턴을 찾지 못한 경우 content에서 마지막 부분 추출
    if content_str:
        # 간단히 마지막 줄 또는 숫자 추출 시도
        lines = content_str.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 숫자만 추출
            numbers = re.findall(r'-?\d+\.?\d*', last_line)
            if numbers:
                return numbers[-1]
    
    return ""


def parse_and_verify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 generated_text를 파싱하고 검증
    Exact matching vs MathVerifier 비교 추가
    
    Args:
        df: 병합된 데이터프레임
        
    Returns:
        파싱 및 검증된 데이터프레임
    """
    logger.info("\n" + "="*50)
    logger.info("=== generated_text 파싱 및 검증 ===")
    logger.info("="*50)
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return df
    
    if 'ground_truth' not in df.columns:
        logger.error("ground_truth 컬럼이 없습니다.")
        return df
    
    # MathVerifier 초기화
    math_verifier = MathVerifier()
    
    # 새 컬럼 추가
    reasoning_contents = []
    contents = []
    final_answers = []
    is_corrects_exact = []
    is_corrects_math_verifier = []
    
    logger.info(f"총 {len(df)}개 항목 파싱 중...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"진행 중: {idx}/{len(df)}")
        
        generated_text = row['generated_text']
        ground_truth = str(row['ground_truth']).strip()
        
        # 1. reasoning_content 추출
        reasoning_content = extract_reasoning_content(generated_text)
        reasoning_contents.append(reasoning_content)
        
        # 2. content 추출
        if reasoning_content:
            # 2. content 추출
            content = extract_content(generated_text)
            contents.append(content)
            
            # 3. final_answer 추출 (content에서 추출)
            final_answer = extract_final_answer_from_content(content)
            final_answers.append(final_answer)
        else:
            contents.append(generated_text)
            final_answer = extract_final_answer_from_content(generated_text)
            final_answers.append(final_answer)
        
        # 4. Exact matching 검증
        if final_answer:
            is_correct_exact = final_answer.strip().lower() == ground_truth.strip().lower()
        else:
            is_correct_exact = False
        is_corrects_exact.append(is_correct_exact)
        
        # 5. MathVerifier 검증
        if final_answer:
            try:
                is_correct_math = math_verifier.verify_answer(final_answer, ground_truth)
            except Exception as e:
                logger.warning(f"MathVerifier 검증 실패 (idx={idx}): {e}")
                is_correct_math = False
        else:
            is_correct_math = False
        is_corrects_math_verifier.append(is_correct_math)
    
    # 데이터프레임에 새 컬럼 추가
    df['reasoning_content'] = reasoning_contents
    df['content'] = contents
    df['final_answer'] = final_answers
    # ArrowExtensionArray 이슈 방지를 위해 명시적 dtype 설정
    df['is_correct_exact'] = pd.Series(is_corrects_exact, dtype='bool')
    df['is_correct_math_verifier'] = pd.Series(is_corrects_math_verifier, dtype='bool')
    
    # 통계 출력
    logger.info(f"\n파싱 완료!")
    logger.info(f"reasoning_content가 있는 항목: {sum(1 for x in reasoning_contents if x)}개 ({sum(1 for x in reasoning_contents if x)/len(df)*100:.1f}%)")
    logger.info(f"content가 있는 항목: {sum(1 for x in contents if x)}개 ({sum(1 for x in contents if x)/len(df)*100:.1f}%)")
    logger.info(f"final_answer가 있는 항목: {sum(1 for x in final_answers if x)}개 ({sum(1 for x in final_answers if x)/len(df)*100:.1f}%)")
    
    # Exact matching vs MathVerifier 비교
    logger.info(f"\n=== 검증 방법 비교 ===")
    logger.info(f"Exact Matching 정답 비율: {sum(is_corrects_exact)}/{len(is_corrects_exact)} ({sum(is_corrects_exact)/len(is_corrects_exact)*100:.1f}%)")
    logger.info(f"MathVerifier 정답 비율: {sum(is_corrects_math_verifier)}/{len(is_corrects_math_verifier)} ({sum(is_corrects_math_verifier)/len(is_corrects_math_verifier)*100:.1f}%)")
    
    # 일치/불일치 분석
    exact_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and not m)
    math_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and m)
    both_correct = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and m)
    both_wrong = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and not m)
    
    logger.info(f"\n검증 결과 일치 분석:")
    logger.info(f"  둘 다 정답: {both_correct}개 ({both_correct/len(df)*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong}개 ({both_wrong/len(df)*100:.1f}%)")
    logger.info(f"  Exact만 정답: {exact_only}개 ({exact_only/len(df)*100:.1f}%)")
    logger.info(f"  MathVerifier만 정답: {math_only}개 ({math_only/len(df)*100:.1f}%)")
    
    return df


def print_verification_comparison(df: pd.DataFrame) -> None:
    """
    검증 방법 비교 결과 출력 (이미 파싱된 데이터용)
    
    Args:
        df: 데이터프레임
    """
    if 'is_correct_exact' not in df.columns or 'is_correct_math_verifier' not in df.columns:
        logger.warning("검증 컬럼이 없어 비교 분석을 수행할 수 없습니다.")
        return
    
    # ArrowExtensionArray 방지를 위해 명시적 캐스팅
    is_corrects_exact = df['is_correct_exact'].astype(int).to_numpy()
    is_corrects_math_verifier = df['is_correct_math_verifier'].astype(int).to_numpy()
    
    logger.info("\n" + "="*50)
    logger.info("=== 검증 방법 비교 ===")
    logger.info("="*50)
    
    # Exact matching vs MathVerifier 비교
    logger.info(f"Exact Matching 정답 비율: {is_corrects_exact.sum()}/{len(is_corrects_exact)} ({is_corrects_exact.sum()/len(is_corrects_exact)*100:.1f}%)")
    logger.info(f"MathVerifier 정답 비율: {is_corrects_math_verifier.sum()}/{len(is_corrects_math_verifier)} ({is_corrects_math_verifier.sum()/len(is_corrects_math_verifier)*100:.1f}%)")
    
    # 일치/불일치 분석
    exact_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and not m)
    math_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and m)
    both_correct = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and m)
    both_wrong = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and not m)
    
    logger.info(f"\n검증 결과 일치 분석:")
    logger.info(f"  둘 다 정답: {both_correct}개 ({both_correct/len(df)*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong}개 ({both_wrong/len(df)*100:.1f}%)")
    logger.info(f"  Exact만 정답: {exact_only}개 ({exact_only/len(df)*100:.1f}%)")
    logger.info(f"  MathVerifier만 정답: {math_only}개 ({math_only/len(df)*100:.1f}%)")


def verify_logprobs_structure(df: pd.DataFrame) -> None:
    """
    logprobs가 토큰별로 5개씩 있는지 확인 (전체 instance, 각 inference)
    
    Args:
        df: 데이터프레임
    """
    logger.info("\n" + "="*50)
    logger.info("=== Logprobs 구조 검증 ===")
    logger.info("="*50)
    
    if 'logprobs' not in df.columns:
        logger.error("logprobs 컬럼이 없습니다.")
        return
    
    total_inferences = len(df)
    valid_count = 0
    invalid_count = 0
    valid_but_wrong_size = 0
    
    # 각 inference별 검증
    per_inference_results = []
    
    for idx, row in df.iterrows():
        logprobs = row['logprobs']
        is_valid = True
        has_5_per_token = True
        num_tokens = 0
        num_tokens_with_5 = 0
        
        try:
            # None 체크
            if logprobs is None:
                is_valid = False
                invalid_count += 1
            else:
                # numpy array인 경우 리스트로 변환
                if isinstance(logprobs, np.ndarray):
                    logprobs = logprobs.tolist()
                
                # 리스트가 아니거나 비어있으면 무효
                if not isinstance(logprobs, list) or len(logprobs) == 0:
                    is_valid = False
                    invalid_count += 1
                else:
                    # 각 토큰별로 5개씩 있는지 확인
                    for token_idx, token_logprobs in enumerate(logprobs):
                        num_tokens += 1
                        
                        # numpy array인 경우 리스트로 변환
                        if isinstance(token_logprobs, np.ndarray):
                            token_logprobs_list = token_logprobs.tolist()
                        elif isinstance(token_logprobs, list):
                            token_logprobs_list = token_logprobs
                        else:
                            # 리스트나 배열이 아니면 무효
                            has_5_per_token = False
                            continue
                        
                        # 토큰별 logprobs가 정확히 5개인지 확인
                        if len(token_logprobs_list) == 5:
                            num_tokens_with_5 += 1
                        else:
                            has_5_per_token = False
                    
                    # 결과 처리
                    if has_5_per_token and num_tokens > 0:
                        valid_count += 1
                    else:
                        if num_tokens > 0:
                            valid_but_wrong_size += 1
                        invalid_count += 1
                        
        except Exception as e:
            logger.warning(f"logprobs 검증 실패 (idx={idx}): {e}")
            is_valid = False
            invalid_count += 1
        
        per_inference_results.append({
            'is_valid': is_valid,
            'has_5_per_token': has_5_per_token,
            'num_tokens': num_tokens,
            'num_tokens_with_5': num_tokens_with_5
        })
    
    # 전체 통계
    logger.info(f"\n=== 전체 Inference 통계 ===")
    logger.info(f"총 inference 수: {total_inferences}")
    logger.info(f"유효한 logprobs (토큰별 5개): {valid_count}개 ({valid_count/total_inferences*100:.1f}%)")
    logger.info(f"무효한 logprobs: {invalid_count}개 ({invalid_count/total_inferences*100:.1f}%)")
    
    if valid_but_wrong_size > 0:
        logger.warning(f"  - 구조는 있지만 토큰별 5개가 아닌 경우: {valid_but_wrong_size}개")
    
    # 문제(instance)별 통계
    if 'problem_id' in df.columns:
        logger.info(f"\n=== 문제(Instance)별 통계 ===")
        
        problem_stats = {}
        for idx, (problem_id, group) in enumerate(df.groupby('problem_id')):
            problem_valid = 0
            problem_invalid = 0
            
            for row_idx, row in group.iterrows():
                # 원본 데이터프레임에서의 위치 찾기
                original_idx = df.index.get_loc(row_idx) if row_idx in df.index else None
                if original_idx is not None and original_idx < len(per_inference_results):
                    result = per_inference_results[original_idx]
                    if result['has_5_per_token']:
                        problem_valid += 1
                    else:
                        problem_invalid += 1
                else:
                    problem_invalid += 1
            
            total_problem_inferences = len(group)
            problem_stats[problem_id] = {
                'total': total_problem_inferences,
                'valid': problem_valid,
                'invalid': problem_invalid,
                'valid_ratio': problem_valid / total_problem_inferences if total_problem_inferences > 0 else 0
            }
        
        # 문제별 정리 통계
        all_valid_problems = sum(1 for s in problem_stats.values() if s['valid'] == s['total'])
        partially_valid_problems = sum(1 for s in problem_stats.values() if 0 < s['valid'] < s['total'])
        all_invalid_problems = sum(1 for s in problem_stats.values() if s['valid'] == 0)
        
        logger.info(f"총 문제 수: {len(problem_stats)}")
        logger.info(f"모든 inference가 유효한 문제: {all_valid_problems}개 ({all_valid_problems/len(problem_stats)*100:.1f}%)")
        logger.info(f"일부 inference만 유효한 문제: {partially_valid_problems}개 ({partially_valid_problems/len(problem_stats)*100:.1f}%)")
        logger.info(f"모든 inference가 무효한 문제: {all_invalid_problems}개 ({all_invalid_problems/len(problem_stats)*100:.1f}%)")
        
        # 유효하지 않은 문제들 샘플 출력
        if partially_valid_problems > 0 or all_invalid_problems > 0:
            logger.info(f"\n문제별 유효성 샘플 (상위 10개):")
            sorted_problems = sorted(problem_stats.items(), key=lambda x: x[1]['valid_ratio'])
            for problem_id, stats in sorted_problems[:10]:
                logger.info(f"  {problem_id}: {stats['valid']}/{stats['total']} 유효 ({stats['valid_ratio']*100:.1f}%)")


def analyze_confidence_correlation(df: pd.DataFrame, output_dir: str = None) -> None:
    """
    confidence score와 정답률 간의 상관관계 분석 및 시각화
    
    Args:
        df: 데이터프레임
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
    """
    logger.info("\n" + "="*50)
    logger.info("=== Confidence Score와 정답률 상관관계 분석 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # confidence score 컬럼들 확인
    confidence_columns = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence'
    ]
    
    available_confidence_cols = [col for col in confidence_columns if col in df.columns]
    
    if not available_confidence_cols:
        logger.error("confidence score 컬럼이 없습니다.")
        return
    
    # 정답 컬럼 확인 (MathVerifier 사용)
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.error("정답 컬럼(is_correct 또는 is_correct_math_verifier)이 없습니다.")
        return
    
    logger.info(f"사용 가능한 confidence score 컬럼: {available_confidence_cols}")
    logger.info(f"사용하는 정답 컬럼: {correct_col}")
    
    # 유효한 데이터만 필터링 (NaN 제외)
    valid_mask = df[correct_col].notna()
    for col in available_confidence_cols:
        valid_mask = valid_mask & df[col].notna()
    
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        logger.error("분석 가능한 데이터가 없습니다.")
        return
    
    logger.info(f"분석 가능한 데이터 수: {len(valid_df)}/{len(df)} ({len(valid_df)/len(df)*100:.1f}%)")
    
    # 각 confidence score별 상관관계 분석 및 시각화
    from scipy.stats import pearsonr, spearmanr
    
    for conf_col in available_confidence_cols:
        logger.info(f"\n--- {conf_col} 분석 ---")
        
        # ArrowExtensionArray 방지를 위해 numpy로 명시 변환
        confidence_scores = valid_df[conf_col].to_numpy(dtype=float)
        is_correct = valid_df[correct_col].astype(int).values
        
        # 피어슨 상관계수
        pearson_corr, pearson_p = pearsonr(confidence_scores, is_correct)
        logger.info(f"  피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        
        # 스피어만 상관계수
        spearman_corr, spearman_p = spearmanr(confidence_scores, is_correct)
        logger.info(f"  스피어만 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        
        # 구간별 정답률 분석
        logger.info(f"\n  구간별 정답률:")
        # confidence score 구간별로 나누기
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        percentile_values = [np.percentile(confidence_scores, p) for p in percentiles]
        
        for i in range(len(percentiles) - 1):
            pct_start = percentiles[i]
            pct_end = percentiles[i+1]
            val_start = percentile_values[i]
            val_end = percentile_values[i+1] if i+1 < len(percentile_values) else np.inf
            
            mask = (confidence_scores >= val_start) & (confidence_scores < val_end) if i+1 < len(percentile_values) else (confidence_scores >= val_start) & (confidence_scores <= val_end)
            
            if mask.sum() > 0:
                interval_correct_rate = is_correct[mask].mean()
                interval_count = mask.sum()
                logger.info(f"    {pct_start}%-{pct_end}% 구간 ({val_start:.4f} ~ {val_end:.4f}): 정답률 {interval_correct_rate:.3f} ({interval_count}개)")
        
        # 전체 통계
        logger.info(f"\n  전체 통계:")
        logger.info(f"    Confidence score 평균: {confidence_scores.mean():.4f}")
        logger.info(f"    Confidence score 표준편차: {confidence_scores.std():.4f}")
        logger.info(f"    전체 정답률: {is_correct.mean():.3f}")
        logger.info(f"    정답인 경우 평균 confidence: {confidence_scores[is_correct == 1].mean():.4f}")
        logger.info(f"    오답인 경우 평균 confidence: {confidence_scores[is_correct == 0].mean():.4f}")
        
        # 시각화: 각 confidence score별로 정답/오답을 색으로 구분하여 히스토그램 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 정답인 경우와 오답인 경우 분리
        correct_mask = is_correct == 1
        incorrect_mask = is_correct == 0
        
        correct_scores = confidence_scores[correct_mask]
        incorrect_scores = confidence_scores[incorrect_mask]
        
        # 히스토그램 bins 설정 (전체 데이터 범위 사용)
        bins = np.linspace(confidence_scores.min(), confidence_scores.max(), 50)
        
        # 정답인 경우 히스토그램 (녹색)
        ax.hist(correct_scores, bins=bins, alpha=0.6, color='green', label='Correct', edgecolor='black', linewidth=0.5)
        
        # 오답인 경우 히스토그램 (빨간색)
        ax.hist(incorrect_scores, bins=bins, alpha=0.6, color='red', label='Incorrect', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{conf_col} Distribution (Pearson Correlation: {pearson_corr:.4f})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 저장
        plot_filename = f'confidence_correlation_{conf_col}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  시각화 저장: {plot_path}")


def load_single_file(file_path: str) -> pd.DataFrame:
    """
    단일 Parquet 파일을 로드
    
    Args:
        file_path: Parquet 파일 경로
    
    Returns:
        로드된 데이터프레임
    """
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    
    if not file_path.endswith('.parquet'):
        logger.error(f"Parquet 파일이 아닙니다: {file_path}")
        return None
    
    try:
        logger.info(f"파일 로드 중: {file_path}")
        if HAS_PYARROW:
            df = _fast_read_single_parquet(file_path)
        else:
            # pandas 경로에서도 pyarrow 엔진 우선 사용 시 약간 유리
            df = pd.read_parquet(file_path, engine="pyarrow") if 'pyarrow' in sys.modules else pd.read_parquet(file_path)
        logger.info(f"로드 완료: {len(df)}개 행")
        try:
            size_mb = os.path.getsize(file_path) / (1024 ** 2)
            logger.info(f"로드한 Parquet 파일 크기: {size_mb:.2f} MB ({file_path})")
        except Exception as e:
            logger.warning(f"파일 크기 확인 실패: {file_path}, 오류: {e}")
        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
        return None


def merge_shard_files(data_dir: str, output_filename: str = "raw_generated.parquet") -> pd.DataFrame:
    """
    디렉토리 내 모든 Parquet 파일들을 하나로 병합
    
    Args:
        data_dir: 데이터 디렉토리 경로
        output_filename: 출력 파일명
    
    Returns:
        병합된 데이터프레임
    """
    if not os.path.exists(data_dir):
        logger.error(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return None
    
    # 이미 병합된 파일이 있는지 확인
    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path):
        logger.info(f"병합된 파일이 이미 존재합니다: {output_path}")
        logger.info("기존 파일을 사용합니다.")
        try:
            df = pd.read_parquet(output_path)
            logger.info(f"기존 파일 로드 완료: {len(df)}개 행")
            try:
                size_mb = os.path.getsize(output_path) / (1024 ** 2)
                logger.info(f"로드한 Parquet 파일 크기: {size_mb:.2f} MB ({output_path})")
            except Exception as e:
                logger.warning(f"파일 크기 확인 실패: {output_path}, 오류: {e}")
            return df
        except Exception as e:
            logger.warning(f"기존 파일 로드 실패: {e}")
            logger.info("새로 병합을 진행합니다...")
    
    # 디렉토리 내 모든 Parquet 파일 찾기 (출력 파일 제외, 정보 로그용)
    parquet_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.parquet') and file != output_filename:
            parquet_path = os.path.join(data_dir, file)
            parquet_files.append(parquet_path)
            logger.info(f"Parquet 파일 발견: {parquet_path}")
    
    if not parquet_files:
        logger.error("병합할 Parquet 파일이 없습니다.")
        return None
    
    logger.info(f"{len(parquet_files)}개 Parquet 파일 병합 시작...")
    
    # 빠른 경로: PyArrow Dataset 사용 (병합 전에는 출력 파일이 없으므로 포함 문제 없음)
    if HAS_PYARROW:
        merged_df = _fast_read_dataset_dir(data_dir)
    else:
        # 폴백: pandas로 순차 로드 후 concat
        dataframes = []
        for shard_file in parquet_files:
            try:
                df = pd.read_parquet(shard_file, engine="pyarrow") if 'pyarrow' in sys.modules else pd.read_parquet(shard_file)
                dataframes.append(df)
                logger.info(f"로드 완료: {shard_file} ({len(df)}개 행)")
                try:
                    size_mb = os.path.getsize(shard_file) / (1024 ** 2)
                    logger.info(f"로드한 Parquet 파일 크기: {size_mb:.2f} MB ({shard_file})")
                except Exception as e:
                    logger.warning(f"파일 크기 확인 실패: {shard_file}, 오류: {e}")
            except Exception as e:
                logger.error(f"파일 로드 실패: {shard_file}, 오류: {e}")
        
        if not dataframes:
            logger.error("로드된 데이터프레임이 없습니다.")
            return None
        merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 결과 저장
    output_path = os.path.join(data_dir, output_filename)
    merged_df.to_parquet(output_path, index=False, compression="zstd")
    
    logger.info(f"✅ 병합 완료: {len(merged_df)}개 결과 저장")
    logger.info(f"저장 위치: {output_path}")
    
    # 통계 정보
    logger.info(f"총 응답 수: {len(merged_df)}")
    logger.info(f"문제 수: {merged_df['problem_id'].nunique()}")
    logger.info(f"문제당 평균 응답 수: {len(merged_df) / merged_df['problem_id'].nunique():.1f}")
    
    # 샤드별 통계
    if 'worker_replica' in merged_df.columns:
        shard_stats = merged_df['worker_replica'].value_counts()
        logger.info("샤드별 응답 수:")
        for shard, count in shard_stats.items():
            logger.info(f"  {shard}: {count}개")
    
    return merged_df


def analyze_token_distribution(df: pd.DataFrame, output_dir: str = None) -> None:
    """
    output_token_count 분포 분석
    
    Args:
        df: 데이터프레임
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
    """
    logger.info("\n" + "="*50)
    logger.info("=== output_token_count 분포 분석 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'output_token_count' not in df.columns:
        logger.error("output_token_count 컬럼이 없습니다.")
        return
    
    token_counts = df['output_token_count']
    
    # 기본 통계
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"평균 토큰 수: {token_counts.mean():.2f}")
    logger.info(f"중앙값 토큰 수: {token_counts.median():.2f}")
    logger.info(f"표준편차: {token_counts.std():.2f}")
    logger.info(f"최소값: {token_counts.min()}")
    logger.info(f"최대값: {token_counts.max()}")
    
    # 분위수 (75% 이후 5%씩 세밀하게)
    percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
    logger.info("\n분위수:")
    for p in percentiles:
        logger.info(f"  {p}%: {token_counts.quantile(p/100):.2f}")
    
    # 토큰 수 구간별 분포 (분위수 기반으로 설정)
    logger.info("\n토큰 수 구간별 분포:")
    # 데이터 특성에 맞게 5000씩 증가하는 구간 + 극값 구간
    bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, float('inf')]
    labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K-30K', '30K-35K', '35K+']
    
    for i in range(len(bins)-1):
        count = ((token_counts >= bins[i]) & (token_counts < bins[i+1])).sum()
        percentage = count / len(token_counts) * 100
        logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")


def analyze_accuracy_by_total_tokens(df: pd.DataFrame, output_dir: str = None, num_bins: int = 10) -> None:
    """
    total token count 구간별 정답률 분석 및 시각화
    
    Args:
        df: 데이터프레임
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
        num_bins: 구간 개수 (분위수 기반)
    """
    logger.info("\n" + "="*50)
    logger.info("=== Total Token Count 분포별 정답률 분석 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 정답 컬럼 확인 (MathVerifier 우선)
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.error("정답 컬럼(is_correct 또는 is_correct_math_verifier)이 없습니다.")
        return
    
    # total token count 컬럼 결정
    if 'total_token_count' in df.columns:
        token_col = 'total_token_count'
    elif 'output_token_count' in df.columns and 'prompt_token_count' in df.columns:
        df = df.copy()
        df['total_token_count'] = df['output_token_count'].fillna(0) + df['prompt_token_count'].fillna(0)
        token_col = 'total_token_count'
    elif 'output_token_count' in df.columns:
        logger.warning("total_token_count나 prompt_token_count가 없어 output_token_count만 사용합니다.")
        token_col = 'output_token_count'
    else:
        logger.error("토큰 카운트 컬럼이 없습니다.")
        return
    
    # confidence score 컬럼들 확인
    confidence_columns = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence'
    ]
    available_confidence_cols = [col for col in confidence_columns if col in df.columns]
    
    # 유효 행 필터 (confidence 컬럼도 포함)
    valid_mask = df[token_col].notna() & df[correct_col].notna()
    for col in available_confidence_cols:
        valid_mask = valid_mask & df[col].notna()
    
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        logger.error("분석 가능한 데이터가 없습니다.")
        return
    
    # ArrowExtensionArray 방지를 위해 numpy로 명시 변환
    token_values = valid_df[token_col].to_numpy(dtype=float)
    correct_values = valid_df[correct_col].astype(int).values
    
    # confidence 값들 준비
    confidence_dict = {}
    for col in available_confidence_cols:
        confidence_dict[col] = valid_df[col].to_numpy(dtype=float)
    
    # 분위수 기반 구간 경계 생성 (중복 제거)
    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.unique(np.percentile(token_values, percentiles))
    if len(edges) < 2:
        logger.error("구간을 생성할 수 없습니다 (토큰 값 다양성 부족).")
        return
    
    # 각 구간별 정답률 및 confidence 계산
    bin_accs = []
    bin_counts = []
    bin_labels = []
    bin_confidences = {col: [] for col in available_confidence_cols}
    
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        # 마지막 구간은 right 포함
        if i < len(edges) - 2:
            mask = (token_values >= left) & (token_values < right)
        else:
            mask = (token_values >= left) & (token_values <= right)
        count = mask.sum()
        if count > 0:
            acc = correct_values[mask].mean()
            bin_accs.append(float(acc))
            bin_counts.append(int(count))
            bin_labels.append(f"{int(left)}-{int(right)}")
            
            # 각 confidence 점수 평균 계산
            for col in available_confidence_cols:
                conf_mean = confidence_dict[col][mask].mean()
                bin_confidences[col].append(float(conf_mean))
        else:
            # count가 0인 경우에도 빈 리스트에 추가하지 않음 (skip)
            pass
    
    if not bin_accs:
        logger.error("구간별 결과가 비어 있습니다.")
        return
    
    # 로그 출력
    logger.info(f"총 유효 샘플: {len(valid_df)}")
    if available_confidence_cols:
        logger.info(f"Confidence 컬럼: {', '.join(available_confidence_cols)}")
        logger.info("\n구간별 정답률 및 Confidence 점수:")
        for idx, (lbl, acc, cnt) in enumerate(zip(bin_labels, bin_accs, bin_counts)):
            conf_strs = []
            for col in available_confidence_cols:
                if col in bin_confidences and idx < len(bin_confidences[col]):
                    conf_val = bin_confidences[col][idx]
                    conf_strs.append(f"{col}={conf_val:.4f}")
            conf_str = ", ".join(conf_strs) if conf_strs else ""
            logger.info(f"  [{lbl}] 정답률: {acc:.3f} ({cnt}개), {conf_str}")
    else:
        logger.info("\n구간별 정답률:")
        for lbl, acc, cnt in zip(bin_labels, bin_accs, bin_counts):
            logger.info(f"  [{lbl}] 정답률: {acc:.3f} ({cnt}개)")
    
    # 시각화 저장 (보조축 count 제거)
    try:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        x = np.arange(len(bin_labels))
        ax1.bar(x, bin_accs, color='#4C78A8', alpha=0.85, label='Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Total Token Count Bins')
        ax1.set_title('Accuracy by Total Token Count (quantile bins)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, rotation=30, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)

        # 범례 (Accuracy만)
        ax1.legend(loc='upper right')

        plot_path = os.path.join(output_dir, 'accuracy_by_total_tokens.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {plot_path}")
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")

def extract_post_redaction_tokens(df: pd.DataFrame) -> np.ndarray:
    """
    </think> 토큰 이후 생성된 토큰 수 추출
    """
    logger.info("\n" + "="*50)
    logger.info("=== </think> 이후 토큰 수 분석 ===")
    logger.info("="*50)
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return np.array([])
    
    post_redaction_tokens = []
    
    for idx, text in enumerate(df['generated_text']):
        if pd.isna(text):
            post_redaction_tokens.append(0)
            continue
        
        text_str = str(text)
        
        # </think> 위치 찾기
        marker = "</think>"
        marker_pos = text_str.find(marker)
        
        if marker_pos == -1:
            # 마커가 없으면 전체 텍스트 길이를 사용
            # 간단한 토큰 추정: 공백으로 분리 (대략적)
            post_redaction_tokens.append(0)
            continue
        
        # 마커 이후 텍스트 추출
        post_text = text_str[marker_pos + len(marker):]
        
        # 간단한 토큰 수 추정: 단어 수로 근사치 계산
        # 실제로는 tokenizer를 사용해야 하지만, 여기서는 단순화
        # 단어 + 특수문자 개수로 대략적인 토큰 수 추정
        words = post_text.split()
        estimated_tokens = len(words) * 1.3  # 한국어/영어 혼합 고려
        
        post_redaction_tokens.append(int(estimated_tokens))
    
    post_redaction_tokens = np.array(post_redaction_tokens)
    non_zero_count = (post_redaction_tokens > 0).sum()
    
    logger.info(f"\n총 텍스트 수: {len(post_redaction_tokens)}")
    logger.info(f"</think> 마커가 있는 텍스트 수: {non_zero_count}")
    logger.info(f"비율: {non_zero_count/len(post_redaction_tokens)*100:.2f}%")
    
    return post_redaction_tokens


def analyze_post_redaction_tokens(post_redaction_tokens: np.ndarray, output_dir: str = None) -> None:
    """
    </think> 이후 토큰 수 분포 분석
    
    Args:
        post_redaction_tokens: </think> 이후 토큰 수 배열
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    non_zero_tokens = post_redaction_tokens[post_redaction_tokens > 0]
    
    if len(non_zero_tokens) == 0:
        logger.info("</think> 이후 토큰이 발견되지 않았습니다.")
        return
    
    logger.info(f"\n=== </think> 이후 토큰 수 분석 ===")
    
    # 기본 통계
    logger.info(f"총 데이터 수: {len(post_redaction_tokens)}")
    logger.info(f"마커가 있는 데이터 수: {len(non_zero_tokens)}")
    logger.info(f"평균 토큰 수 (전체): {post_redaction_tokens.mean():.2f}")
    logger.info(f"평균 토큰 수 (마커 있음): {non_zero_tokens.mean():.2f}")
    logger.info(f"중앙값 (마커 있음): {np.median(non_zero_tokens):.2f}")
    logger.info(f"표준편차 (마커 있음): {non_zero_tokens.std():.2f}")
    logger.info(f"최소값 (마커 있음): {non_zero_tokens.min():.2f}")
    logger.info(f"최대값 (마커 있음): {non_zero_tokens.max():.2f}")
    
    # 분위수 (75% 이후 5%씩 세밀하게)
    percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
    logger.info("\n분위수 (마커 있음):")
    for p in percentiles:
        logger.info(f"  {p}%: {np.percentile(non_zero_tokens, p):.2f}")
    
    # 구간별 분포
    logger.info("\n토큰 수 구간별 분포:")
    # </think> 이후 토큰은 전체보다 훨씬 작을 가능성이 높으므로 더 작은 구간 사용
    max_val = non_zero_tokens.max()
    if max_val < 500:
        bins = [0, 50, 100, 150, 200, 250, 300, 350, float('inf')]
        labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
    elif max_val < 2000:
        bins = [0, 200, 400, 600, 800, 1000, 1200, 1500, float('inf')]
        labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1500', '1500+']
    else:
        bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, float('inf')]
        labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K-6K', '6K-7K', '7K+']
    
    for i in range(len(bins)-1):
        count = ((non_zero_tokens >= bins[i]) & (non_zero_tokens < bins[i+1])).sum()
        percentage = count / len(non_zero_tokens) * 100
        logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")


def analyze_instance_level_distributions(df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Instance(문제)당 분포 분석
    1. max_tokens에 도달한 inference 수 분포
    2. </think> 마커가 없는 inference 수 분포
    
    Args:
        df: 데이터프레임
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("\n" + "="*50)
    logger.info("=== Instance별 분포 분석 ===")
    logger.info("="*50)
    
    if 'problem_id' not in df.columns:
        logger.error("problem_id 컬럼이 없습니다.")
        return
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return
    
    # total_token_count (prompt + output) 기준으로 max_tokens 찾기
    # total_token_count가 없으면 output_token_count + prompt_token_count 사용
    if 'total_token_count' in df.columns:
        token_col = 'total_token_count'
        logger.info("total_token_count 컬럼을 사용합니다.")
    elif 'output_token_count' in df.columns and 'prompt_token_count' in df.columns:
        # total_token_count가 없으면 계산
        df['total_token_count'] = df['output_token_count'] + df['prompt_token_count']
        token_col = 'total_token_count'
        logger.info("output_token_count + prompt_token_count를 계산하여 사용합니다.")
    elif 'output_token_count' in df.columns:
        logger.warning("total_token_count나 prompt_token_count가 없어 output_token_count만 사용합니다.")
        token_col = 'output_token_count'
    else:
        logger.error("토큰 카운트 컬럼이 없습니다.")
        return
    
    # 전체 최댓값 찾기 (total_token_count 기준)
    max_tokens = df[token_col].max()
    logger.info(f"전체 최댓값 (max_tokens): {max_tokens}")
    
    # Instance별 분석
    max_token_counts = []
    no_marker_counts = []
    inference_counts = []  # 문제당 inference 수 저장
    # problem_text 길이 저장용 딕셔너리
    max_token_count_to_prompts = {}
    no_marker_count_to_prompts = {}
    
    marker = "</think>"
    
    for problem_id, group in df.groupby('problem_id'):
        # 문제당 inference 수 저장
        inference_counts.append(len(group))
        
        # 1. max_tokens에 도달한 inference 수 (total_token_count 기준)
        max_token_count = (group[token_col] == max_tokens).sum()
        max_token_counts.append(max_token_count)
        
        # problem_text 길이 저장 (문제당 하나의 problem_text이므로)
        if 'problem_text' in group.columns and len(group) > 0:
            prompt_len = len(str(group['problem_text'].iloc[0]))
            if max_token_count not in max_token_count_to_prompts:
                max_token_count_to_prompts[max_token_count] = []
            max_token_count_to_prompts[max_token_count].append(prompt_len)
        
        # 2. </think> 마커가 없는 inference 수
        no_marker_count = 0
        for text in group['generated_text']:
            if pd.isna(text):
                no_marker_count += 1
            else:
                text_str = str(text)
                if marker not in text_str:
                    no_marker_count += 1
        no_marker_counts.append(no_marker_count)
        
        # problem_text 길이 저장
        if 'problem_text' in group.columns and len(group) > 0:
            prompt_len = len(str(group['problem_text'].iloc[0]))
            if no_marker_count not in no_marker_count_to_prompts:
                no_marker_count_to_prompts[no_marker_count] = []
            no_marker_count_to_prompts[no_marker_count].append(prompt_len)
    
    max_token_counts = np.array(max_token_counts)
    no_marker_counts = np.array(no_marker_counts)
    inference_counts = np.array(inference_counts)
    
    # 문제당 inference 수 통계 추가
    logger.info(f"\n=== 문제당 inference 수 분포 ===")
    logger.info(f"총 문제 수: {len(inference_counts)}")
    logger.info(f"평균: {inference_counts.mean():.2f}")
    logger.info(f"중앙값: {np.median(inference_counts):.2f}")
    logger.info(f"표준편차: {inference_counts.std():.2f}")
    logger.info(f"최소값: {inference_counts.min()}")
    logger.info(f"최대값: {inference_counts.max()}")
    
    # 문제당 inference 수가 다른 것들의 통계
    unique_inference_counts = np.unique(inference_counts)
    if len(unique_inference_counts) > 1:
        logger.info(f"\n문제당 inference 수가 다른 문제들:")
        logger.info(f"  서로 다른 inference 수 종류: {len(unique_inference_counts)}가지")
        for count in sorted(unique_inference_counts):
            problem_count = (inference_counts == count).sum()
            percentage = problem_count / len(inference_counts) * 100
            logger.info(f"  {count}개 inference: {problem_count}문제 ({percentage:.1f}%)")
    
    # 결과 출력
    logger.info(f"\n=== 문제별 max_tokens 도달 inference 수 분포 ===")
    logger.info(f"총 문제 수: {len(max_token_counts)}")
    logger.info(f"평균: {max_token_counts.mean():.2f}")
    logger.info(f"중앙값: {np.median(max_token_counts):.2f}")
    logger.info(f"표준편차: {max_token_counts.std():.2f}")
    logger.info(f"최소값: {max_token_counts.min()}")
    logger.info(f"최대값: {max_token_counts.max()}")
    
    # 개별 분포 출력
    logger.info("\n개별 분포:")
    max_count = max_token_counts.max()
    max_token_distribution = {}  # count -> 통계 정보
    
    for count in range(0, int(max_count) + 1):
        problem_count = (max_token_counts == count).sum()
        percentage = problem_count / len(max_token_counts) * 100
        
        # 평균 problem_text 길이 계산
        avg_prompt_len = 0.0
        if count in max_token_count_to_prompts and len(max_token_count_to_prompts[count]) > 0:
            avg_prompt_len = np.mean(max_token_count_to_prompts[count])
        
        logger.info(f"  {count}개: {problem_count}문제 ({percentage:.1f}%), 평균 prompt 길이: {avg_prompt_len:.0f}")
        
        # 통계 정보만 저장
        if problem_count > 0:
            max_token_distribution[count] = {
                'problem_count': int(problem_count),
                'percentage': float(percentage),
                'avg_prompt_len': float(avg_prompt_len)
            }
    
    logger.info(f"\n=== 문제별 </think> 마커 없는 inference 수 분포 ===")
    logger.info(f"총 문제 수: {len(no_marker_counts)}")
    logger.info(f"평균: {no_marker_counts.mean():.2f}")
    logger.info(f"중앙값: {np.median(no_marker_counts):.2f}")
    logger.info(f"표준편차: {no_marker_counts.std():.2f}")
    logger.info(f"최소값: {no_marker_counts.min()}")
    logger.info(f"최대값: {no_marker_counts.max()}")
    
    # 개별 분포 출력
    logger.info("\n개별 분포:")
    max_no_marker = no_marker_counts.max()
    no_marker_distribution = {}  # count -> 통계 정보
    
    for count in range(0, int(max_no_marker) + 1):
        problem_count = (no_marker_counts == count).sum()
        percentage = problem_count / len(no_marker_counts) * 100
        
        # 평균 problem_text 길이 계산
        avg_prompt_len = 0.0
        if count in no_marker_count_to_prompts and len(no_marker_count_to_prompts[count]) > 0:
            avg_prompt_len = np.mean(no_marker_count_to_prompts[count])
        
        logger.info(f"  {count}개: {problem_count}문제 ({percentage:.1f}%), 평균 prompt 길이: {avg_prompt_len:.0f}")
        
        # 통계 정보만 저장
        if problem_count > 0:
            no_marker_distribution[count] = {
                'problem_count': int(problem_count),
                'percentage': float(percentage),
                'avg_prompt_len': float(avg_prompt_len)
            }
    
    # 분석 결과 통계 저장
    analysis_results = {
        'max_tokens': float(max_tokens),
        'inference_count_stats': {
            'total_problems': int(len(inference_counts)),
            'mean': float(inference_counts.mean()),
            'median': float(np.median(inference_counts)),
            'std': float(inference_counts.std()),
            'min': int(inference_counts.min()),
            'max': int(inference_counts.max()),
            'unique_counts': {int(k): int(v) for k, v in zip(*np.unique(inference_counts, return_counts=True))}
        },
        'max_token_distribution': {
            'total_problems': int(len(max_token_counts)),
            'mean': float(max_token_counts.mean()),
            'median': float(np.median(max_token_counts)),
            'std': float(max_token_counts.std()),
            'min': int(max_token_counts.min()),
            'max': int(max_token_counts.max()),
            'distribution': max_token_distribution
        },
        'no_marker_distribution': {
            'total_problems': int(len(no_marker_counts)),
            'mean': float(no_marker_counts.mean()),
            'median': float(np.median(no_marker_counts)),
            'std': float(no_marker_counts.std()),
            'min': int(no_marker_counts.min()),
            'max': int(no_marker_counts.max()),
            'distribution': no_marker_distribution
        }
    }
    
    results_path = os.path.join(output_dir, 'instance_analysis_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n분석 결과 JSON 저장: {results_path}")
    
    return analysis_results


def analyze_single_file(file_path: str, output_dir: str = None) -> None:
    """
    단일 Parquet 파일 분석 (병합 없이)
    
    Args:
        file_path: 분석할 Parquet 파일 경로
        output_dir: 출력 디렉토리 (기본값: 파일이 있는 디렉토리)
    """
    logger.info("\n" + "="*50)
    logger.info("=== 단일 파일 분석 모드 ===")
    logger.info("="*50)
    
    # 1. 파일 로드
    df = load_single_file(file_path)
    if df is None:
        logger.error("파일 로드 실패")
        return
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))
    
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # 2. generated_text 파싱 및 검증
    if 'reasoning_content' not in df.columns or 'content' not in df.columns or 'final_answer' not in df.columns or 'is_correct_math_verifier' not in df.columns:
        logger.info("\n파싱 컬럼이 없어 파싱을 진행합니다...")
        df = parse_and_verify_dataframe(df)
    else:
        logger.info("이미 파싱된 데이터입니다.")
    
    # 3. Exact matching vs MathVerifier 비교
    print_verification_comparison(df)
    
    # 4. confidence score와 정답률 상관관계 분석
    analyze_confidence_correlation(df, output_dir)
    
    # 5. output_token_count 분포 분석
    analyze_token_distribution(df, output_dir)
    
    # 5.1 total token 구간별 정답률 분석
    analyze_accuracy_by_total_tokens(df, output_dir)
    
    # 6. </think> 이후 토큰 수 분석
    post_redaction_tokens = extract_post_redaction_tokens(df)
    analyze_post_redaction_tokens(post_redaction_tokens, output_dir)
    
    # 7. Instance별 분포 분석
    analyze_instance_level_distributions(df, output_dir)
    
    logger.info("\n" + "="*50)
    logger.info("✅ 단일 파일 분석 완료!")
    logger.info("="*50)


def main(data_dir: str = None, output_filename: str = "raw_generated.parquet", single_file: str = None) -> None:
    """
    메인 함수: 샤드 병합 및 분석 또는 단일 파일 분석
    
    Args:
        data_dir: 데이터 디렉토리 경로 (단일 파일 모드에서는 None)
        output_filename: 출력 파일명 (병합 모드용)
        single_file: 단일 파일 경로 (단일 파일 모드용)
    """
    # 단일 파일 모드
    if single_file:
        analyze_single_file(single_file)
        return
    
    # 병합 모드
    if data_dir is None:
        logger.error("data_dir 또는 single_file 인자가 필요합니다.")
        return
    
    # 1. 샤드 파일 병합 (기존 파일이 있으면 사용)
    df = merge_shard_files(data_dir, output_filename)
    if df is None:
        logger.error("병합 실패")
        return
    
    # 2. generated_text 파싱 및 검증 (새 기능!)
    # 이미 파싱된 컬럼이 있는지 확인
    if 'reasoning_content' not in df.columns or 'content' not in df.columns or 'final_answer' not in df.columns or 'is_correct_math_verifier' not in df.columns:
        logger.info("\n파싱 컬럼이 없어 파싱을 진행합니다...")
        df = parse_and_verify_dataframe(df)
        
        # 파싱된 결과 저장
        output_path = os.path.join(data_dir, output_filename)
        df.to_parquet(output_path, index=False, compression="zstd")
        logger.info(f"파싱된 데이터 저장: {output_path}")
    else:
        logger.info("이미 파싱된 데이터입니다.")
    
    # 3. Exact matching vs MathVerifier 비교
    print_verification_comparison(df)
    
    # 4. confidence score와 정답률 상관관계 분석
    analyze_confidence_correlation(df, data_dir)
    
    # 6. output_token_count 분포 분석
    analyze_token_distribution(df, data_dir)
    
    # 6.1 total token 구간별 정답률 분석
    analyze_accuracy_by_total_tokens(df, data_dir)
    
    # 7. </think> 이후 토큰 수 분석
    post_redaction_tokens = extract_post_redaction_tokens(df)
    analyze_post_redaction_tokens(post_redaction_tokens, data_dir)
    
    # 8. Instance별 분포 분석 (추가)
    samples = analyze_instance_level_distributions(df, data_dir)
    
    logger.info("\n" + "="*50)
    logger.info("✅ 분석 완료!")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="샤드 파일 병합 및 토큰 분포 분석, 또는 단일 파일 분석")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="데이터 디렉토리 경로 (병합 모드)")
    parser.add_argument("--output", type=str, default="raw_generated_merged.parquet", 
                       help="출력 파일명 (병합 모드)")
    parser.add_argument("--single-file", type=str, default=None,
                       help="분석할 단일 Parquet 파일 경로 (단일 파일 분석 모드)")
    
    args = parser.parse_args()
    
    # 단일 파일 모드와 병합 모드 중 하나는 반드시 지정되어야 함
    if not args.single_file and not args.data_dir:
        parser.error("--single-file 또는 --data-dir 중 하나는 반드시 지정해야 합니다.")
    
    if args.single_file and args.data_dir:
        parser.error("--single-file과 --data-dir은 동시에 사용할 수 없습니다.")
    
    main(data_dir=args.data_dir, output_filename=args.output, single_file=args.single_file)
