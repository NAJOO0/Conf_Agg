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

# 프로젝트 루트를 sys.path에 추가 (절대 경로 사용)
# 스크립트 위치: scripts/merge_analysis/merge_and_analyze_0.py
# 프로젝트 루트: scripts의 부모 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/merge_analysis
project_root = os.path.dirname(os.path.dirname(script_dir))  # scripts의 부모 = 프로젝트 루트
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.math_verifier import MathVerifier

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로그 파일 핸들러 관리
_log_file_handler = None


def setup_log_file_handler(output_dir: str, log_filename: str = "analysis_log.txt") -> None:
    """
    분석 결과를 파일로 저장하기 위한 로그 핸들러 설정
    
    Args:
        output_dir: 로그 파일을 저장할 디렉토리
        log_filename: 로그 파일명
    """
    global _log_file_handler
    
    # 기존 핸들러 제거 (중복 방지)
    if _log_file_handler is not None:
        logger.removeHandler(_log_file_handler)
        _log_file_handler.close()
    
    # 파일 핸들러 생성
    log_path = os.path.join(output_dir, log_filename)
    _log_file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    _log_file_handler.setLevel(logging.INFO)
    
    # 포맷 설정 (콘솔과 동일하게)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    _log_file_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.addHandler(_log_file_handler)
    logger.info(f"분석 로그 파일 저장 시작: {log_path}")


def remove_log_file_handler() -> None:
    """
    로그 파일 핸들러 제거
    """
    global _log_file_handler
    
    if _log_file_handler is not None:
        logger.removeHandler(_log_file_handler)
        _log_file_handler.close()
        _log_file_handler = None


# 빠른 Parquet 로드를 위한 PyArrow 가용성 확인
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


def convert_string_to_large_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
    
    Args:
        df: 변환할 데이터프레임
        
    Returns:
        변환된 데이터프레임
    """
    if not HAS_PYARROW:
        return df
    
    # Arrow dtype이 있는 문자열 컬럼 찾기
    string_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        # ArrowDtype이고 string 타입인 경우
        if isinstance(dtype, pd.ArrowDtype) and dtype.pyarrow_dtype == pa.string():
            string_cols.append(col)
    
    if not string_cols:
        return df
    
    # 각 문자열 컬럼을 large_string으로 변환
    for col in string_cols:
        try:
            # PyArrow 배열로 변환 후 large_string으로 캐스팅
            pa_array = df[col].array._pa_array
            large_string_array = pa.compute.cast(pa_array, pa.large_string())
            # pandas ArrowExtensionArray로 변환
            if hasattr(pd.arrays, 'ArrowExtensionArray'):
                df[col] = pd.arrays.ArrowExtensionArray(large_string_array)
            else:
                # pandas 버전에 따라 다를 수 있음
                df[col] = pd.array(large_string_array, dtype=pd.ArrowDtype(large_string_array.type))
        except Exception as e:
            logger.warning(f"컬럼 {col}을 large_string으로 변환 실패: {e}, 일반 pandas dtype으로 변환합니다.")
            # 변환 실패 시 일반 pandas dtype으로 변환
            try:
                df[col] = df[col].astype('string')
            except Exception:
                # string dtype도 실패하면 object로 변환
                df[col] = df[col].astype('object')
    
    return df


def _fast_read_single_parquet(file_path: str):
    """
    PyArrow로 단일 Parquet를 빠르게 로드하여 pandas DataFrame으로 반환.
    """
    t0 = time.time()
    # ParquetFile.read는 memory_map 인자를 지원하지 않으므로 read_table 사용
    table = pq.read_table(file_path, memory_map=True)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    # 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
    df = convert_string_to_large_string(df)
    dt = time.time() - t0
    logger.info(f"PyArrow fast read 완료: {len(df)}행, {dt:.2f}s")
    return df


def _fast_read_dataset_dir(data_dir: str, exclude_files: list = None):
    """
    PyArrow Dataset로 디렉토리 내 모든 Parquet를 빠르게 로드하여 pandas DataFrame으로 반환.
    출력 파일이 아직 없다는 전제에서 병합 전에 사용.
    .parquet 확장자만 필터링하여 읽음.
    
    Args:
        data_dir: 데이터 디렉토리
        exclude_files: 제외할 파일명 리스트 (예: 출력 파일명)
    """
    t0 = time.time()
    # .parquet 파일만 필터링하여 읽기
    parquet_files = []
    exclude_set = set(exclude_files) if exclude_files else set()
    
    for file in os.listdir(data_dir):
        if file.endswith('.parquet') and file not in exclude_set:
            parquet_files.append(os.path.join(data_dir, file))
    
    if not parquet_files:
        logger.error("병합할 Parquet 파일이 없습니다.")
        return None
    
    # PyArrow로 각 파일을 읽어서 pandas DataFrame으로 변환 후 병합
    # (스키마가 다른 경우를 대비해 pandas concat 사용)
    dataframes = []
    for file_path in parquet_files:
        try:
            table = pq.read_table(file_path, memory_map=True)
            df_part = table.to_pandas(types_mapper=pd.ArrowDtype)
            # 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
            df_part = convert_string_to_large_string(df_part)
            dataframes.append(df_part)
        except Exception as e:
            logger.warning(f"파일 로드 실패 (건너뜀): {file_path}, 오류: {e}")
    
    if not dataframes:
        logger.error("로드된 데이터프레임이 없습니다.")
        return None
    
    # pandas concat으로 병합 (스키마가 다른 경우 자동으로 처리)
    df = pd.concat(dataframes, ignore_index=True)
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
    """content에서 최종 답안 추출"""
    if pd.isna(content) or not content:
        return ""
    
    content_str = str(content).strip()
    
    # \boxed{} 찾기
    boxed_matches = list(re.finditer(r'\\boxed\{', content_str))
    if boxed_matches:
        last_start = boxed_matches[-1].end()
        brace_count = 1
        end_pos = last_start
        
        while end_pos < len(content_str) and brace_count > 0:
            if content_str[end_pos] == '{' and (end_pos == 0 or content_str[end_pos-1] != '\\'):
                brace_count += 1
            elif content_str[end_pos] == '}' and (end_pos == 0 or content_str[end_pos-1] != '\\'):
                brace_count -= 1
            end_pos += 1
        
        if brace_count == 0:
            return content_str[last_start:end_pos-1].strip()
    
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


def compute_instance_voting_results(df: pd.DataFrame) -> dict:
    """
    Instance별 majority voting과 weighted majority voting 결과 계산
    bottom_10_percent_confidence, lowest_group_confidence, tail_confidence 세 가지를 각각 사용
    
    Args:
        df: 데이터프레임 (final_answer, problem_id, ground_truth, bottom_10_percent_confidence, lowest_group_confidence, tail_confidence 컬럼 필요)
        
    Returns:
        각 instance별 voting 결과 딕셔너리
        {
            problem_id: {
                'majority_answer': str,
                'weighted_majority_answer_bottom10': str,
                'weighted_majority_answer_lowest': str,
                'weighted_majority_answer_tail': str,
                'is_correct_majority': bool,
                'is_correct_weighted_majority_bottom10': bool,
                'is_correct_weighted_majority_lowest': bool,
                'is_correct_weighted_majority_tail': bool,
                'ground_truth': str
            }
        }
    """
    # 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
    df = convert_string_to_large_string(df.copy())
    
    if 'problem_id' not in df.columns:
        logger.error("problem_id 컬럼이 없습니다.")
        return {}
    
    if 'final_answer' not in df.columns:
        logger.error("final_answer 컬럼이 없습니다.")
        return {}
    
    if 'ground_truth' not in df.columns:
        logger.error("ground_truth 컬럼이 없습니다.")
        return {}
    
    if 'bottom_10_percent_confidence' not in df.columns:
        logger.error("bottom_10_percent_confidence 컬럼이 없습니다.")
        return {}
    
    if 'lowest_group_confidence' not in df.columns:
        logger.error("lowest_group_confidence 컬럼이 없습니다.")
        return {}
    
    if 'tail_confidence' not in df.columns:
        logger.error("tail_confidence 컬럼이 없습니다.")
        return {}
        
    # MathVerifier 초기화
    math_verifier = MathVerifier()
    
    results = {}
    
    logger.info("Instance별 voting 결과 계산 중... (bottom_10_percent_confidence, lowest_group_confidence, tail_confidence 사용)")
    
    for problem_id, group in df.groupby('problem_id'):
        # 유효한 답안들만 수집 (빈 답안 제외)
        valid_answers = []
        valid_confidences_bottom10 = []
        valid_confidences_lowest = []
        valid_confidences_tail = []
        ground_truth = None
        
        for idx, row in group.iterrows():
            final_answer = row['final_answer']
            bottom10_conf = row['bottom_10_percent_confidence']
            lowest_conf = row['lowest_group_confidence']
            tail_conf = row['tail_confidence']
            
            # ground_truth는 모든 row에서 동일하므로 첫 번째 것만 사용
            if ground_truth is None:
                ground_truth = str(row['ground_truth']).strip()
            
            # 유효한 답안만 추가
            if pd.notna(final_answer) and str(final_answer).strip():
                valid_answers.append(str(final_answer).strip())
                # confidence가 NaN이면 0으로 처리
                if pd.notna(bottom10_conf):
                    valid_confidences_bottom10.append(float(bottom10_conf))
                else:
                    valid_confidences_bottom10.append(0.0)
                
                if pd.notna(lowest_conf):
                    valid_confidences_lowest.append(float(lowest_conf))
                else:
                    valid_confidences_lowest.append(0.0)
                
                if pd.notna(tail_conf):
                    valid_confidences_tail.append(float(tail_conf))
                else:
                    valid_confidences_tail.append(0.0)
        
        if not valid_answers:
            # 유효한 답안이 없으면 정답 아님
            results[problem_id] = {
                'majority_answer': '',
                'weighted_majority_answer_bottom10': '',
                'weighted_majority_answer_lowest': '',
                'weighted_majority_answer_tail': '',
                'is_correct_majority': False,
                'is_correct_weighted_majority_bottom10': False,
                'is_correct_weighted_majority_lowest': False,
                'is_correct_weighted_majority_tail': False,
                'ground_truth': ground_truth
            }
            continue
        
        # 1. Majority Voting
        from collections import Counter
        answer_counts = Counter(valid_answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        
        # 2. Weighted Majority Voting (bottom_10_percent_confidence 사용)
        answer_weights_bottom10 = {}
        for answer, conf in zip(valid_answers, valid_confidences_bottom10):
            if answer not in answer_weights_bottom10:
                answer_weights_bottom10[answer] = 0.0
            answer_weights_bottom10[answer] += conf
        
        weighted_majority_answer_bottom10 = max(answer_weights_bottom10.items(), key=lambda x: x[1])[0]
        
        # 3. Weighted Majority Voting (lowest_group_confidence 사용)
        answer_weights_lowest = {}
        for answer, conf in zip(valid_answers, valid_confidences_lowest):
            if answer not in answer_weights_lowest:
                answer_weights_lowest[answer] = 0.0
            answer_weights_lowest[answer] += conf
        
        weighted_majority_answer_lowest = max(answer_weights_lowest.items(), key=lambda x: x[1])[0]
        
        # 4. Weighted Majority Voting (tail_confidence 사용)
        answer_weights_tail = {}
        for answer, conf in zip(valid_answers, valid_confidences_tail):
            if answer not in answer_weights_tail:
                answer_weights_tail[answer] = 0.0
            answer_weights_tail[answer] += conf
        
        weighted_majority_answer_tail = max(answer_weights_tail.items(), key=lambda x: x[1])[0]
        
        # 5. 검증 (MathVerifier 사용)
        is_correct_majority = False
        is_correct_weighted_majority_bottom10 = False
        is_correct_weighted_majority_lowest = False
        is_correct_weighted_majority_tail = False
        
        if majority_answer:
            try:
                is_correct_majority = math_verifier.verify_answer(majority_answer, ground_truth)
            except Exception as e:
                logger.warning(f"Majority voting 검증 실패 (problem_id={problem_id}): {e}")
                is_correct_majority = False
        
        if weighted_majority_answer_bottom10:
            try:
                is_correct_weighted_majority_bottom10 = math_verifier.verify_answer(weighted_majority_answer_bottom10, ground_truth)
            except Exception as e:
                logger.warning(f"Weighted majority voting (bottom10) 검증 실패 (problem_id={problem_id}): {e}")
                is_correct_weighted_majority_bottom10 = False
        
        if weighted_majority_answer_lowest:
            try:
                is_correct_weighted_majority_lowest = math_verifier.verify_answer(weighted_majority_answer_lowest, ground_truth)
            except Exception as e:
                logger.warning(f"Weighted majority voting (lowest) 검증 실패 (problem_id={problem_id}): {e}")
                is_correct_weighted_majority_lowest = False
        
        if weighted_majority_answer_tail:
            try:
                is_correct_weighted_majority_tail = math_verifier.verify_answer(weighted_majority_answer_tail, ground_truth)
            except Exception as e:
                logger.warning(f"Weighted majority voting (tail) 검증 실패 (problem_id={problem_id}): {e}")
                is_correct_weighted_majority_tail = False
        
        results[problem_id] = {
            'majority_answer': majority_answer,
            'weighted_majority_answer_bottom10': weighted_majority_answer_bottom10,
            'weighted_majority_answer_lowest': weighted_majority_answer_lowest,
            'weighted_majority_answer_tail': weighted_majority_answer_tail,
            'is_correct_majority': is_correct_majority,
            'is_correct_weighted_majority_bottom10': is_correct_weighted_majority_bottom10,
            'is_correct_weighted_majority_lowest': is_correct_weighted_majority_lowest,
            'is_correct_weighted_majority_tail': is_correct_weighted_majority_tail,
            'ground_truth': ground_truth
        }
    
    return results


def add_voting_results_to_dataframe(df: pd.DataFrame, voting_results: dict) -> pd.DataFrame:
    """
    Voting 결과를 데이터프레임에 추가
    
    Args:
        df: 데이터프레임
        voting_results: compute_instance_voting_results의 결과
        
    Returns:
        voting 결과 컬럼이 추가된 데이터프레임
    """
    if 'problem_id' not in df.columns:
        logger.error("problem_id 컬럼이 없습니다.")
        return df
    
    # 각 inference에 대해 해당 instance의 voting 결과 추가
    is_correct_majority_list = []
    is_correct_weighted_majority_bottom10_list = []
    is_correct_weighted_majority_lowest_list = []
    is_correct_weighted_majority_tail_list = []
    
    for idx, row in df.iterrows():
        problem_id = row['problem_id']
        
        if problem_id in voting_results:
            result = voting_results[problem_id]
            is_correct_majority_list.append(result['is_correct_majority'])
            is_correct_weighted_majority_bottom10_list.append(result['is_correct_weighted_majority_bottom10'])
            is_correct_weighted_majority_lowest_list.append(result['is_correct_weighted_majority_lowest'])
            is_correct_weighted_majority_tail_list.append(result['is_correct_weighted_majority_tail'])
        else:
            is_correct_majority_list.append(False)
            is_correct_weighted_majority_bottom10_list.append(False)
            is_correct_weighted_majority_lowest_list.append(False)
            is_correct_weighted_majority_tail_list.append(False)
    
    df['is_correct_majority_voting'] = pd.Series(is_correct_majority_list, dtype='bool')
    df['is_correct_weighted_majority_voting_bottom10'] = pd.Series(is_correct_weighted_majority_bottom10_list, dtype='bool')
    df['is_correct_weighted_majority_voting_lowest'] = pd.Series(is_correct_weighted_majority_lowest_list, dtype='bool')
    df['is_correct_weighted_majority_voting_tail'] = pd.Series(is_correct_weighted_majority_tail_list, dtype='bool')
    
    return df


def print_verification_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    검증 방법 비교 결과 출력 (이미 파싱된 데이터용)
    Exact matching, MathVerifier, Majority Voting, Weighted Majority Voting 비교
    
    Args:
        df: 데이터프레임
    """
    if 'is_correct_exact' not in df.columns or 'is_correct_math_verifier' not in df.columns:
        logger.warning("검증 컬럼이 없어 비교 분석을 수행할 수 없습니다.")
        return
    
    # Voting 결과가 없으면 계산
    if 'is_correct_majority_voting' not in df.columns or 'is_correct_weighted_majority_voting_bottom10' not in df.columns or 'is_correct_weighted_majority_voting_lowest' not in df.columns or 'is_correct_weighted_majority_voting_tail' not in df.columns:
        logger.info("Voting 결과가 없어 계산합니다...")
        voting_results = compute_instance_voting_results(df)
        df = add_voting_results_to_dataframe(df, voting_results)
    
    # ArrowExtensionArray 방지를 위해 명시적 캐스팅
    is_corrects_exact = df['is_correct_exact'].astype(int).to_numpy()
    is_corrects_math_verifier = df['is_correct_math_verifier'].astype(int).to_numpy()
    is_corrects_majority = df['is_correct_majority_voting'].astype(int).to_numpy()
    is_corrects_weighted_majority_bottom10 = df['is_correct_weighted_majority_voting_bottom10'].astype(int).to_numpy()
    is_corrects_weighted_majority_lowest = df['is_correct_weighted_majority_voting_lowest'].astype(int).to_numpy()
    is_corrects_weighted_majority_tail = df['is_correct_weighted_majority_voting_tail'].astype(int).to_numpy()
    
    logger.info("\n" + "="*50)
    logger.info("=== 검증 방법 비교 (6가지 방법) ===")
    logger.info("="*50)
    
    # 각 방법별 정답률
    total_count = len(is_corrects_exact)
    logger.info(f"\n=== 개별 방법별 정답률 ===")
    
    # Inference 기준 방법들 (Exact Matching, MathVerifier)
    logger.info(f"[Inference 기준]")
    logger.info(f"Exact Matching 정답 비율: {is_corrects_exact.sum()}/{total_count} ({is_corrects_exact.sum()/total_count*100:.1f}%)")
    logger.info(f"MathVerifier 정답 비율: {is_corrects_math_verifier.sum()}/{total_count} ({is_corrects_math_verifier.sum()/total_count*100:.1f}%)")
    
    # Instance 기준 방법들 (Majority Voting, Weighted Majority Voting)
    if 'problem_id' in df.columns:
        unique_problems = df['problem_id'].nunique()
        
        # 각 instance별로 voting 결과는 하나만 카운트
        instance_majority_correct = 0
        instance_weighted_majority_bottom10_correct = 0
        instance_weighted_majority_lowest_correct = 0
        instance_weighted_majority_tail_correct = 0
        
        for problem_id, group in df.groupby('problem_id'):
            if len(group) > 0:
                if group['is_correct_majority_voting'].iloc[0]:
                    instance_majority_correct += 1
                if group['is_correct_weighted_majority_voting_bottom10'].iloc[0]:
                    instance_weighted_majority_bottom10_correct += 1
                if group['is_correct_weighted_majority_voting_lowest'].iloc[0]:
                    instance_weighted_majority_lowest_correct += 1
                if group['is_correct_weighted_majority_voting_tail'].iloc[0]:
                    instance_weighted_majority_tail_correct += 1
        
        logger.info(f"\n[Instance 기준 - 총 {unique_problems}개 instance]")
        logger.info(f"Majority Voting 정답 비율: {instance_majority_correct}/{unique_problems} ({instance_majority_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (bottom_10_percent_confidence) 정답 비율: {instance_weighted_majority_bottom10_correct}/{unique_problems} ({instance_weighted_majority_bottom10_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (lowest_group_confidence) 정답 비율: {instance_weighted_majority_lowest_correct}/{unique_problems} ({instance_weighted_majority_lowest_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (tail_confidence) 정답 비율: {instance_weighted_majority_tail_correct}/{unique_problems} ({instance_weighted_majority_tail_correct/unique_problems*100:.1f}%)")
        
        # 전체 비교 요약
        logger.info(f"\n=== 전체 방법 비교 요약 ===")
        logger.info(f"Exact Matching (inference 기준): {is_corrects_exact.sum()}/{total_count} ({is_corrects_exact.sum()/total_count*100:.1f}%)")
        logger.info(f"MathVerifier (inference 기준): {is_corrects_math_verifier.sum()}/{total_count} ({is_corrects_math_verifier.sum()/total_count*100:.1f}%)")
        logger.info(f"Majority Voting (instance 기준): {instance_majority_correct}/{unique_problems} ({instance_majority_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (bottom_10_percent_confidence, instance 기준): {instance_weighted_majority_bottom10_correct}/{unique_problems} ({instance_weighted_majority_bottom10_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (lowest_group_confidence, instance 기준): {instance_weighted_majority_lowest_correct}/{unique_problems} ({instance_weighted_majority_lowest_correct/unique_problems*100:.1f}%)")
        logger.info(f"Weighted Majority Voting (tail_confidence, instance 기준): {instance_weighted_majority_tail_correct}/{unique_problems} ({instance_weighted_majority_tail_correct/unique_problems*100:.1f}%)")
    
    # 6가지 방법 간 일치 분석
    logger.info(f"\n=== 6가지 방법 간 일치 분석 ===")
    
    # 모든 방법이 정답인 경우
    all_correct = sum(1 for e, m, maj, wmaj_b10, wmaj_low, wmaj_tail in zip(
        is_corrects_exact, is_corrects_math_verifier, is_corrects_majority, 
        is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail
    ) if e and m and maj and wmaj_b10 and wmaj_low and wmaj_tail)
    
    # 모든 방법이 오답인 경우
    all_wrong = sum(1 for e, m, maj, wmaj_b10, wmaj_low, wmaj_tail in zip(
        is_corrects_exact, is_corrects_math_verifier, is_corrects_majority,
        is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail
    ) if not e and not m and not maj and not wmaj_b10 and not wmaj_low and not wmaj_tail)
    
    logger.info(f"모든 방법이 정답: {all_correct}개 ({all_correct/total_count*100:.1f}%)")
    logger.info(f"모든 방법이 오답: {all_wrong}개 ({all_wrong/total_count*100:.1f}%)")
    
    # 두 방법씩 비교
    logger.info(f"\n=== 방법별 쌍 비교 ===")
    
    # Exact vs MathVerifier
    exact_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and not m)
    math_only = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and m)
    both_correct_em = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if e and m)
    both_wrong_em = sum(1 for e, m in zip(is_corrects_exact, is_corrects_math_verifier) if not e and not m)
    
    logger.info(f"[Exact vs MathVerifier]")
    logger.info(f"  둘 다 정답: {both_correct_em}개 ({both_correct_em/total_count*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong_em}개 ({both_wrong_em/total_count*100:.1f}%)")
    logger.info(f"  Exact만 정답: {exact_only}개 ({exact_only/total_count*100:.1f}%)")
    logger.info(f"  MathVerifier만 정답: {math_only}개 ({math_only/total_count*100:.1f}%)")
    
    # Majority vs Weighted Majority (bottom10)
    majority_only_b10 = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_bottom10) if maj and not wmaj)
    weighted_only_b10 = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_bottom10) if not maj and wmaj)
    both_correct_mw_b10 = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_bottom10) if maj and wmaj)
    both_wrong_mw_b10 = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_bottom10) if not maj and not wmaj)
    
    logger.info(f"\n[Majority vs Weighted Majority (bottom_10_percent_confidence)]")
    logger.info(f"  둘 다 정답: {both_correct_mw_b10}개 ({both_correct_mw_b10/total_count*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong_mw_b10}개 ({both_wrong_mw_b10/total_count*100:.1f}%)")
    logger.info(f"  Majority만 정답: {majority_only_b10}개 ({majority_only_b10/total_count*100:.1f}%)")
    logger.info(f"  Weighted Majority (bottom10)만 정답: {weighted_only_b10}개 ({weighted_only_b10/total_count*100:.1f}%)")
    
    # Majority vs Weighted Majority (lowest)
    majority_only_low = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_lowest) if maj and not wmaj)
    weighted_only_low = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_lowest) if not maj and wmaj)
    both_correct_mw_low = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_lowest) if maj and wmaj)
    both_wrong_mw_low = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_lowest) if not maj and not wmaj)
    
    logger.info(f"\n[Majority vs Weighted Majority (lowest_group_confidence)]")
    logger.info(f"  둘 다 정답: {both_correct_mw_low}개 ({both_correct_mw_low/total_count*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong_mw_low}개 ({both_wrong_mw_low/total_count*100:.1f}%)")
    logger.info(f"  Majority만 정답: {majority_only_low}개 ({majority_only_low/total_count*100:.1f}%)")
    logger.info(f"  Weighted Majority (lowest)만 정답: {weighted_only_low}개 ({weighted_only_low/total_count*100:.1f}%)")
    
    # Majority vs Weighted Majority (tail)
    majority_only_tail = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_tail) if maj and not wmaj)
    weighted_only_tail = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_tail) if not maj and wmaj)
    both_correct_mw_tail = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_tail) if maj and wmaj)
    both_wrong_mw_tail = sum(1 for maj, wmaj in zip(is_corrects_majority, is_corrects_weighted_majority_tail) if not maj and not wmaj)
    
    logger.info(f"\n[Majority vs Weighted Majority (tail_confidence)]")
    logger.info(f"  둘 다 정답: {both_correct_mw_tail}개 ({both_correct_mw_tail/total_count*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong_mw_tail}개 ({both_wrong_mw_tail/total_count*100:.1f}%)")
    logger.info(f"  Majority만 정답: {majority_only_tail}개 ({majority_only_tail/total_count*100:.1f}%)")
    logger.info(f"  Weighted Majority (tail)만 정답: {weighted_only_tail}개 ({weighted_only_tail/total_count*100:.1f}%)")
    
    # Weighted Majority 세 방법 비교
    bottom10_only = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if b10 and not low and not tail)
    lowest_only = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if not b10 and low and not tail)
    tail_only = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if not b10 and not low and tail)
    both_correct_b10_low = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if b10 and low and not tail)
    both_correct_b10_tail = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if b10 and not low and tail)
    both_correct_low_tail = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if not b10 and low and tail)
    all_three_correct_wm = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if b10 and low and tail)
    all_three_wrong_wm = sum(1 for b10, low, tail in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest, is_corrects_weighted_majority_tail) if not b10 and not low and not tail)
    
    logger.info(f"\n[Weighted Majority 세 방법 비교 (bottom_10_percent_confidence vs lowest_group_confidence vs tail_confidence)]")
    logger.info(f"  세 방법 모두 정답: {all_three_correct_wm}개 ({all_three_correct_wm/total_count*100:.1f}%)")
    logger.info(f"  세 방법 모두 오답: {all_three_wrong_wm}개 ({all_three_wrong_wm/total_count*100:.1f}%)")
    logger.info(f"  Bottom10만 정답: {bottom10_only}개 ({bottom10_only/total_count*100:.1f}%)")
    logger.info(f"  Lowest만 정답: {lowest_only}개 ({lowest_only/total_count*100:.1f}%)")
    logger.info(f"  Tail만 정답: {tail_only}개 ({tail_only/total_count*100:.1f}%)")
    logger.info(f"  Bottom10+Lowest 정답: {both_correct_b10_low}개 ({both_correct_b10_low/total_count*100:.1f}%)")
    logger.info(f"  Bottom10+Tail 정답: {both_correct_b10_tail}개 ({both_correct_b10_tail/total_count*100:.1f}%)")
    logger.info(f"  Lowest+Tail 정답: {both_correct_low_tail}개 ({both_correct_low_tail/total_count*100:.1f}%)")
    
    # 기존 두 방법 비교 (하위 호환성)
    bottom10_only_old = sum(1 for b10, low in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest) if b10 and not low)
    lowest_only_old = sum(1 for b10, low in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest) if not b10 and low)
    both_correct_wm_old = sum(1 for b10, low in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest) if b10 and low)
    both_wrong_wm_old = sum(1 for b10, low in zip(is_corrects_weighted_majority_bottom10, is_corrects_weighted_majority_lowest) if not b10 and not low)
    
    logger.info(f"\n[Weighted Majority (bottom_10_percent_confidence) vs Weighted Majority (lowest_group_confidence)]")
    logger.info(f"  둘 다 정답: {both_correct_wm_old}개 ({both_correct_wm_old/total_count*100:.1f}%)")
    logger.info(f"  둘 다 오답: {both_wrong_wm_old}개 ({both_wrong_wm_old/total_count*100:.1f}%)")
    logger.info(f"  Bottom10만 정답: {bottom10_only_old}개 ({bottom10_only_old/total_count*100:.1f}%)")
    logger.info(f"  Lowest만 정답: {lowest_only_old}개 ({lowest_only_old/total_count*100:.1f}%)")
    
    # Inference 방법 (Exact, MathVerifier) vs Instance 방법 (Majority, Weighted Majority) 비교
    # 각 instance별로 하나의 결과만 카운트
    if 'problem_id' in df.columns:
        logger.info(f"\n=== Inference 방법 vs Instance 방법 비교 ===")
        
        instance_stats = {
            'exact_correct': 0,
            'math_correct': 0,
            'majority_correct': 0,
            'weighted_bottom10_correct': 0,
            'weighted_lowest_correct': 0,
            'weighted_tail_correct': 0,
            'all_correct': 0,
            'all_wrong': 0
        }
        
        for problem_id, group in df.groupby('problem_id'):
            if len(group) == 0:
                continue
            
            # 각 instance에서 최소 하나라도 정답인지 확인
            has_exact_correct = group['is_correct_exact'].any()
            has_math_correct = group['is_correct_math_verifier'].any()
            has_majority_correct = group['is_correct_majority_voting'].iloc[0] if len(group) > 0 else False
            has_weighted_bottom10_correct = group['is_correct_weighted_majority_voting_bottom10'].iloc[0] if len(group) > 0 else False
            has_weighted_lowest_correct = group['is_correct_weighted_majority_voting_lowest'].iloc[0] if len(group) > 0 else False
            has_weighted_tail_correct = group['is_correct_weighted_majority_voting_tail'].iloc[0] if len(group) > 0 else False
            
            if has_exact_correct:
                instance_stats['exact_correct'] += 1
            if has_math_correct:
                instance_stats['math_correct'] += 1
            if has_majority_correct:
                instance_stats['majority_correct'] += 1
            if has_weighted_bottom10_correct:
                instance_stats['weighted_bottom10_correct'] += 1
            if has_weighted_lowest_correct:
                instance_stats['weighted_lowest_correct'] += 1
            if has_weighted_tail_correct:
                instance_stats['weighted_tail_correct'] += 1
            
            if has_exact_correct and has_math_correct and has_majority_correct and has_weighted_bottom10_correct and has_weighted_lowest_correct and has_weighted_tail_correct:
                instance_stats['all_correct'] += 1
            if not has_exact_correct and not has_math_correct and not has_majority_correct and not has_weighted_bottom10_correct and not has_weighted_lowest_correct and not has_weighted_tail_correct:
                instance_stats['all_wrong'] += 1
        
        unique_problems = df['problem_id'].nunique()
        logger.info(f"Instance 기준 정답률:")
        logger.info(f"  Exact (최소 1개 inference 정답): {instance_stats['exact_correct']}/{unique_problems} ({instance_stats['exact_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  MathVerifier (최소 1개 inference 정답): {instance_stats['math_correct']}/{unique_problems} ({instance_stats['math_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  Majority Voting: {instance_stats['majority_correct']}/{unique_problems} ({instance_stats['majority_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  Weighted Majority Voting (bottom_10_percent_confidence): {instance_stats['weighted_bottom10_correct']}/{unique_problems} ({instance_stats['weighted_bottom10_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  Weighted Majority Voting (lowest_group_confidence): {instance_stats['weighted_lowest_correct']}/{unique_problems} ({instance_stats['weighted_lowest_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  Weighted Majority Voting (tail_confidence): {instance_stats['weighted_tail_correct']}/{unique_problems} ({instance_stats['weighted_tail_correct']/unique_problems*100:.1f}%)")
        
        logger.info(f"\nInstance 기준 일치 분석:")
        logger.info(f"  모든 방법이 정답: {instance_stats['all_correct']}개 ({instance_stats['all_correct']/unique_problems*100:.1f}%)")
        logger.info(f"  모든 방법이 오답: {instance_stats['all_wrong']}개 ({instance_stats['all_wrong']/unique_problems*100:.1f}%)")
    
    return df


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
        'tail_confidence',
        'bottom_10_percent_confidence',
        'lowest_group_confidence',
        'top_10_percent_confidence',
        'highest_group_confidence',
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
        
        # 시각화: 각 confidence score별로 정답/오답을 색으로 구분하여 히스토그램 생성 (정규화)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 정답인 경우와 오답인 경우 분리
        correct_mask = is_correct == 1
        incorrect_mask = is_correct == 0
        
        correct_scores = confidence_scores[correct_mask]
        incorrect_scores = confidence_scores[incorrect_mask]
        
        # 히스토그램 bins 설정 (전체 데이터 범위 사용)
        bins = np.linspace(confidence_scores.min(), confidence_scores.max(), 100)
        
        # 정답인 경우 히스토그램 (녹색) - 정규화하여 비율로 표시
        correct_weights = np.ones_like(correct_scores) / len(correct_scores) if len(correct_scores) > 0 else None
        if correct_weights is not None:
            ax.hist(correct_scores, bins=bins, weights=correct_weights, alpha=0.6, color='green', 
                   label=f'Correct (n={len(correct_scores)})', edgecolor='black', linewidth=0.5)
        
        # 오답인 경우 히스토그램 (빨간색) - 정규화하여 비율로 표시
        incorrect_weights = np.ones_like(incorrect_scores) / len(incorrect_scores) if len(incorrect_scores) > 0 else None
        if incorrect_weights is not None:
            ax.hist(incorrect_scores, bins=bins, weights=incorrect_weights, alpha=0.6, color='red', 
                   label=f'Incorrect (n={len(incorrect_scores)})', edgecolor='black', linewidth=0.5)
        
        # 정답/오답 평균을 세로 점선으로 표시
        if len(correct_scores) > 0:
            correct_mean = correct_scores.mean()
            ax.axvline(correct_mean, color='green', linestyle='--', linewidth=2, 
                      label=f'Correct Mean ({correct_mean:.4f})', alpha=0.8)
        
        if len(incorrect_scores) > 0:
            incorrect_mean = incorrect_scores.mean()
            ax.axvline(incorrect_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Incorrect Mean ({incorrect_mean:.4f})', alpha=0.8)
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
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
            # 문자열 컬럼을 large_string으로 변환 (PyArrow가 있는 경우)
            if HAS_PYARROW:
                df = convert_string_to_large_string(df)
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
            # 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
            if HAS_PYARROW:
                df = convert_string_to_large_string(df)
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
    
    # 빠른 경로: PyArrow Dataset 사용 (출력 파일 제외)
    if HAS_PYARROW:
        merged_df = _fast_read_dataset_dir(data_dir, exclude_files=[output_filename])
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
        'tail_confidence',
        'bottom_10_percent_confidence',
        'lowest_group_confidence',
        'top_10_percent_confidence',
        'highest_group_confidence',
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


def analyze_instance_accuracy_correlation(df: pd.DataFrame, output_dir: str = None) -> None:
    """
    같은 instance 내에서 추론 간 차이 분석 (Within-instance analysis)
    - 문제 난이도 효과를 제거하고, 같은 문제 내에서 output_token_count, confidence, 정확도 간의 관계 분석
    - 각 instance의 평균값을 기준으로 상대적 차이를 계산하여 경향성 파악
    
    Args:
        df: 데이터프레임
        output_dir: 출력 디렉토리 (기본값: 스크립트 디렉토리)
    """
    logger.info("\n" + "="*50)
    logger.info("=== Within-Instance 분석: 같은 문제 내 추론 간 차이 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'problem_id' not in df.columns:
        logger.error("problem_id 컬럼이 없습니다.")
        return
    
    # 정답 컬럼 확인 (MathVerifier 우선)
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.error("정답 컬럼(is_correct 또는 is_correct_math_verifier)이 없습니다.")
        return
    
    # confidence 컬럼 (bottom_10_percent_confidence, lowest_group_confidence, tail_confidence 세 개 사용)
    conf_cols = ['bottom_10_percent_confidence', 'lowest_group_confidence', 'tail_confidence']
    missing_conf_cols = [col for col in conf_cols if col not in df.columns]
    if missing_conf_cols:
        logger.error(f"필요한 confidence 컬럼이 없습니다: {missing_conf_cols}")
        return
    
    # 필요한 컬럼 확인
    required_cols = [correct_col] + conf_cols + ['output_token_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"필요한 컬럼이 없습니다: {missing_cols}")
        return
    
    logger.info(f"사용하는 정답 컬럼: {correct_col}")
    logger.info(f"사용하는 confidence 컬럼: {', '.join(conf_cols)}")
    
    # 각 confidence 컬럼별로 분석 수행
    for conf_col in conf_cols:
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Within-Instance 분석: {conf_col} ===")
        logger.info(f"{'='*50}")
        
        # 유효한 데이터만 필터링 (NaN 제외)
        valid_mask = df[correct_col].notna() & df[conf_col].notna() & df['output_token_count'].notna()
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            logger.error(f"분석 가능한 데이터가 없습니다 ({conf_col}).")
            continue
        
        logger.info(f"분석 가능한 데이터 수: {len(valid_df)}/{len(df)} ({len(valid_df)/len(df)*100:.1f}%)")
        
        # Within-instance 분석: 각 instance의 평균값을 기준으로 상대적 차이 계산
        from scipy.stats import pearsonr, spearmanr
        
        # 모든 inference 데이터를 저장할 리스트
        all_within_data = []
        
        # 각 instance별로 분석
        instance_count = 0
        for problem_id, group in valid_df.groupby('problem_id'):
            if len(group) < 2:  # 최소 2개 inference 필요
                continue
            
            instance_count += 1
            
            # Instance별 평균값 계산
            instance_mean_conf = group[conf_col].mean()
            instance_mean_tokens = group['output_token_count'].mean()
            instance_mean_accuracy = group[correct_col].astype(int).mean()
            
            # 각 inference의 상대적 차이 계산
            for idx, row in group.iterrows():
                rel_conf_diff = row[conf_col] - instance_mean_conf  # instance 평균 대비 차이
                rel_token_diff = row['output_token_count'] - instance_mean_tokens
                is_correct = int(row[correct_col])
                
                all_within_data.append({
                    'problem_id': problem_id,
                    'instance_mean_conf': instance_mean_conf,
                    'instance_mean_tokens': instance_mean_tokens,
                    'instance_mean_accuracy': instance_mean_accuracy,
                    'rel_conf_diff': rel_conf_diff,
                    'rel_token_diff': rel_token_diff,
                    'absolute_conf': row[conf_col],
                    'absolute_tokens': row['output_token_count'],
                    'is_correct': is_correct,
                })
        
        if len(all_within_data) == 0:
            logger.error(f"분석 가능한 instance가 없습니다 ({conf_col}).")
            continue
        
        within_df = pd.DataFrame(all_within_data)
        
        logger.info(f"\n=== Within-Instance 분석 결과 ({conf_col}) ===")
        logger.info(f"총 {instance_count}개 instance 분석")
        logger.info(f"총 {len(within_df)}개 inference 분석")
        logger.info(f"Instance당 평균 inference 수: {len(within_df)/instance_count:.2f}")
        
        # 1. 상대적 confidence 차이와 정확도 관계
        logger.info(f"\n=== 상대적 Confidence 차이와 정확도 관계 ({conf_col}) ===")
        rel_conf = within_df['rel_conf_diff'].to_numpy()
        is_correct = within_df['is_correct'].to_numpy()
        
        # 상대적 confidence 구간별 정확도
        percentiles = [0, 25, 50, 75, 100]
        percentile_values = [np.percentile(rel_conf, p) for p in percentiles]
        
        logger.info("상대적 Confidence 차이 구간별 정확도:")
        for i in range(len(percentiles) - 1):
            pct_start = percentiles[i]
            pct_end = percentiles[i+1]
            val_start = percentile_values[i]
            val_end = percentile_values[i+1]
            
            if i < len(percentiles) - 2:
                mask = (rel_conf >= val_start) & (rel_conf < val_end)
            else:
                mask = (rel_conf >= val_start) & (rel_conf <= val_end)
            
            if mask.sum() > 0:
                interval_accuracy = is_correct[mask].mean()
                interval_count = mask.sum()
                logger.info(f"  {pct_start}%-{pct_end}% 구간 (상대 차이 {val_start:.4f} ~ {val_end:.4f}): 정답률 {interval_accuracy:.3f} ({interval_count}개)")
        
        # 상관관계
        pearson_corr, pearson_p = pearsonr(rel_conf, is_correct)
        spearman_corr, spearman_p = spearmanr(rel_conf, is_correct)
        logger.info(f"피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        logger.info(f"스피어만 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        
        # 2. 상대적 output_token_count 차이와 정확도 관계 (작을수록 좋음 - 반대로 해석)
        logger.info(f"\n=== 상대적 Output Token Count 차이와 정확도 관계 (작을수록 좋음, {conf_col}) ===")
        rel_tokens = within_df['rel_token_diff'].to_numpy()
        
        # 상대적 token 차이 구간별 정확도 (작을수록 좋으므로 반대로 해석)
        percentiles = [0, 25, 50, 75, 100]
        percentile_values = [np.percentile(rel_tokens, p) for p in percentiles]
        
        logger.info("상대적 Output Token Count 차이 구간별 정답률 (작을수록 좋음):")
        for i in range(len(percentiles) - 1):
            pct_start = percentiles[i]
            pct_end = percentiles[i+1]
            val_start = percentile_values[i]
            val_end = percentile_values[i+1]
            
            if i < len(percentiles) - 2:
                mask = (rel_tokens >= val_start) & (rel_tokens < val_end)
            else:
                mask = (rel_tokens >= val_start) & (rel_tokens <= val_end)
            
            if mask.sum() > 0:
                interval_accuracy = is_correct[mask].mean()
                interval_count = mask.sum()
                # 작을수록 좋으므로 반대로 해석: 상대 차이가 작은(음수) 구간이 더 좋음
                logger.info(f"  {pct_start}%-{pct_end}% 구간 (상대 차이 {val_start:.0f} ~ {val_end:.0f}, 작을수록 좋음): 정답률 {interval_accuracy:.3f} ({interval_count}개)")
        
        # 상관관계 (작을수록 좋으므로 음수 상관관계가 좋은 것)
        pearson_corr, pearson_p = pearsonr(rel_tokens, is_correct)
        spearman_corr, spearman_p = spearmanr(rel_tokens, is_correct)
        logger.info(f"피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.4e}) [음수일수록 좋음: 토큰이 적을수록 정답률 높음]")
        logger.info(f"스피어만 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.4e}) [음수일수록 좋음: 토큰이 적을수록 정답률 높음]")
        
        # 3. 상대적 confidence와 상대적 token 차이의 관계
        logger.info(f"\n=== 상대적 Confidence 차이와 상대적 Output Token Count 차이 관계 ({conf_col}) ===")
        pearson_corr, pearson_p = pearsonr(rel_conf, rel_tokens)
        spearman_corr, spearman_p = spearmanr(rel_conf, rel_tokens)
        logger.info(f"피어슨 상관계수: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        logger.info(f"스피어만 상관계수: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        
        # 4. Instance 내 percentile 기반 분석
        logger.info(f"\n=== Instance 내 Percentile 기반 분석 ({conf_col}) ===")
        
        # 각 instance에서 상위/하위 50% 비교
        high_conf_correct = []
        low_conf_correct = []
        high_token_correct = []
        low_token_correct = []
        
        # 10% 단위 분석을 위한 리스트
        conf_percentile_accuracies = {p: [] for p in range(0, 101, 10)}  # 0-10%, 10-20%, ..., 90-100%
        token_percentile_accuracies = {p: [] for p in range(0, 101, 10)}
        
        for problem_id, group in within_df.groupby('problem_id'):
            if len(group) < 2:
                continue
            
            # Confidence 기준 상위/하위 50%
            median_conf = group['absolute_conf'].median()
            high_conf_group = group[group['absolute_conf'] >= median_conf]
            low_conf_group = group[group['absolute_conf'] < median_conf]
            
            if len(high_conf_group) > 0:
                high_conf_correct.append(high_conf_group['is_correct'].mean())
            if len(low_conf_group) > 0:
                low_conf_correct.append(low_conf_group['is_correct'].mean())
            
            # Confidence 기준 10% 단위 분석
            sorted_conf = group.sort_values('absolute_conf')
            for p in range(0, 101, 10):
                if p == 0:
                    percentile_group = sorted_conf.iloc[:max(1, int(len(sorted_conf) * 0.1))]
                elif p == 100:
                    percentile_group = sorted_conf.iloc[int(len(sorted_conf) * 0.9):]
                else:
                    start_idx = int(len(sorted_conf) * (p / 100))
                    end_idx = int(len(sorted_conf) * ((p + 10) / 100))
                    if end_idx > start_idx:
                        percentile_group = sorted_conf.iloc[start_idx:end_idx]
                    else:
                        continue
                
                if len(percentile_group) > 0:
                    conf_percentile_accuracies[p].append(percentile_group['is_correct'].mean())
            
            # Token 기준 상위/하위 50% (작을수록 좋으므로 반대로 해석)
            median_tokens = group['absolute_tokens'].median()
            high_token_group = group[group['absolute_tokens'] >= median_tokens]  # 많은 토큰 = 나쁨
            low_token_group = group[group['absolute_tokens'] < median_tokens]  # 적은 토큰 = 좋음
            
            if len(high_token_group) > 0:
                high_token_correct.append(high_token_group['is_correct'].mean())
            if len(low_token_group) > 0:
                low_token_correct.append(low_token_group['is_correct'].mean())
            
            # Token 기준 10% 단위 분석 (작을수록 좋으므로 작은 순서대로)
            sorted_tokens = group.sort_values('absolute_tokens')
            for p in range(0, 101, 10):
                if p == 0:
                    percentile_group = sorted_tokens.iloc[:max(1, int(len(sorted_tokens) * 0.1))]  # 가장 작은 10%
                elif p == 100:
                    percentile_group = sorted_tokens.iloc[int(len(sorted_tokens) * 0.9):]  # 가장 큰 10%
                else:
                    start_idx = int(len(sorted_tokens) * (p / 100))
                    end_idx = int(len(sorted_tokens) * ((p + 10) / 100))
                    if end_idx > start_idx:
                        percentile_group = sorted_tokens.iloc[start_idx:end_idx]
                    else:
                        continue
                
                if len(percentile_group) > 0:
                    token_percentile_accuracies[p].append(percentile_group['is_correct'].mean())
        
        # 50% 비교 결과 출력
        if high_conf_correct and low_conf_correct:
            logger.info(f"Instance 내 Confidence 상위 50% 평균 정답률: {np.mean(high_conf_correct):.3f} (n={len(high_conf_correct)})")
            logger.info(f"Instance 내 Confidence 하위 50% 평균 정답률: {np.mean(low_conf_correct):.3f} (n={len(low_conf_correct)})")
            logger.info(f"차이: {np.mean(high_conf_correct) - np.mean(low_conf_correct):.3f}")
        
        if high_token_correct and low_token_correct:
            logger.info(f"Instance 내 Output Token 상위 50% 평균 정답률 (많은 토큰): {np.mean(high_token_correct):.3f} (n={len(high_token_correct)})")
            logger.info(f"Instance 내 Output Token 하위 50% 평균 정답률 (적은 토큰): {np.mean(low_token_correct):.3f} (n={len(low_token_correct)})")
            logger.info(f"차이: {np.mean(low_token_correct) - np.mean(high_token_correct):.3f} (적은 토큰이 더 좋음)")
        
        # 10% 단위 분석 결과 출력
        logger.info(f"\n=== Instance 내 Confidence 10% 단위 분석 ({conf_col}) ===")
        for p in range(0, 101, 10):
            if conf_percentile_accuracies[p]:
                avg_acc = np.mean(conf_percentile_accuracies[p])
                logger.info(f"  {p}%-{p+10}% 구간 평균 정답률: {avg_acc:.3f} (n={len(conf_percentile_accuracies[p])} instances)")
        
        logger.info(f"\n=== Instance 내 Output Token 10% 단위 분석 (작을수록 좋음, {conf_col}) ===")
        for p in range(0, 101, 10):
            if token_percentile_accuracies[p]:
                avg_acc = np.mean(token_percentile_accuracies[p])
                if p == 0:
                    label = "가장 작은 10%"
                elif p == 100:
                    label = "가장 큰 10%"
                else:
                    label = f"{p}%-{p+10}%"
                logger.info(f"  {label} 평균 정답률: {avg_acc:.3f} (n={len(token_percentile_accuracies[p])} instances)")
    
    logger.info("\n" + "="*50)
    logger.info("Within-Instance 분석 완료!")
    logger.info("="*50)


def analyze_instance_level_distributions(df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Instance(문제)당 분포 분석
    1. max_tokens에 도달한 inference 수 분포
    2. </think> 마커가 없는 inference 수 분포
    3. Instance별 정확도, prompt 길이, output_token_count 평균 분포
    
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
    
    # 정답 컬럼 확인 (MathVerifier 우선)
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.warning("정답 컬럼이 없어 instance별 정확도 분석을 건너뜁니다.")
        correct_col = None
    
    # Instance별 분석
    max_token_counts = []
    no_marker_counts = []
    inference_counts = []  # 문제당 inference 수 저장
    # problem_text 길이 저장용 딕셔너리
    max_token_count_to_prompts = {}
    no_marker_count_to_prompts = {}
    
    # 새로운 분석용: instance별 정확도, prompt 길이, output_token_count 평균
    instance_accuracies = []  # instance별 정확도 (%)
    instance_prompt_lengths = []  # instance별 prompt 길이
    instance_avg_output_tokens = []  # instance별 output_token_count 평균
    
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
        
        # 3. Instance별 정확도, prompt 길이, output_token_count 평균 계산
        if correct_col and correct_col in group.columns:
            # 정확도 계산 (정답인 inference 수 / 전체 inference 수)
            correct_count = group[correct_col].astype(int).sum()
            total_count = len(group)
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
            instance_accuracies.append(accuracy)
        else:
            instance_accuracies.append(None)
        
        # Prompt 길이 계산
        if 'problem_text' in group.columns and len(group) > 0:
            prompt_len = len(str(group['problem_text'].iloc[0]))
            instance_prompt_lengths.append(prompt_len)
        else:
            instance_prompt_lengths.append(None)
        
        # output_token_count 평균 계산
        if 'output_token_count' in group.columns:
            avg_output_tokens = group['output_token_count'].mean()
            instance_avg_output_tokens.append(float(avg_output_tokens))
        else:
            instance_avg_output_tokens.append(None)
    
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
    
    # 3. Instance별 정확도, prompt 길이, output_token_count 평균 분포 분석
    logger.info(f"\n=== Instance별 정확도, Prompt 길이, Output Token 평균 분포 ===")
    
    # 정확도 분포 분석
    if correct_col and any(acc is not None for acc in instance_accuracies):
        valid_accuracies = [acc for acc in instance_accuracies if acc is not None]
        if valid_accuracies:
            acc_array = np.array(valid_accuracies)
            logger.info(f"\n[Instance별 정확도 분포]")
            logger.info(f"  총 instance 수: {len(valid_accuracies)}")
            logger.info(f"  평균 정확도: {acc_array.mean():.2f}%")
            logger.info(f"  중앙값: {np.median(acc_array):.2f}%")
            logger.info(f"  표준편차: {acc_array.std():.2f}%")
            logger.info(f"  최소값: {acc_array.min():.2f}%")
            logger.info(f"  최대값: {acc_array.max():.2f}%")
            
            # 구간별 분포 (0-20%, 20-40%, ..., 80-100%)
            bins = [0, 20, 40, 60, 80, 100]
            labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            logger.info(f"  구간별 분포:")
            for i in range(len(bins) - 1):
                count = ((acc_array >= bins[i]) & (acc_array < bins[i+1])).sum() if i < len(bins)-2 else ((acc_array >= bins[i]) & (acc_array <= bins[i+1])).sum()
                percentage = count / len(acc_array) * 100
                logger.info(f"    {labels[i]}: {count}개 instance ({percentage:.1f}%)")
    
    # Prompt 길이 분포 분석
    if any(pl is not None for pl in instance_prompt_lengths):
        valid_prompt_lengths = [pl for pl in instance_prompt_lengths if pl is not None]
        if valid_prompt_lengths:
            pl_array = np.array(valid_prompt_lengths)
            logger.info(f"\n[Instance별 Prompt 길이 분포]")
            logger.info(f"  총 instance 수: {len(valid_prompt_lengths)}")
            logger.info(f"  평균 길이: {pl_array.mean():.0f} 문자")
            logger.info(f"  중앙값: {np.median(pl_array):.0f} 문자")
            logger.info(f"  표준편차: {pl_array.std():.0f} 문자")
            logger.info(f"  최소값: {pl_array.min():.0f} 문자")
            logger.info(f"  최대값: {pl_array.max():.0f} 문자")
            
            # 분위수 출력
            percentiles = [25, 50, 75, 90, 95, 99]
            logger.info(f"  분위수:")
            for p in percentiles:
                logger.info(f"    {p}%: {np.percentile(pl_array, p):.0f} 문자")
    
    # Output Token Count 평균 분포 분석
    if any(ot is not None for ot in instance_avg_output_tokens):
        valid_output_tokens = [ot for ot in instance_avg_output_tokens if ot is not None]
        if valid_output_tokens:
            ot_array = np.array(valid_output_tokens)
            logger.info(f"\n[Instance별 Output Token Count 평균 분포]")
            logger.info(f"  총 instance 수: {len(valid_output_tokens)}")
            logger.info(f"  평균: {ot_array.mean():.2f} 토큰")
            logger.info(f"  중앙값: {np.median(ot_array):.2f} 토큰")
            logger.info(f"  표준편차: {ot_array.std():.2f} 토큰")
            logger.info(f"  최소값: {ot_array.min():.2f} 토큰")
            logger.info(f"  최대값: {ot_array.max():.2f} 토큰")
            
            # 분위수 출력
            percentiles = [25, 50, 75, 90, 95, 99]
            logger.info(f"  분위수:")
            for p in percentiles:
                logger.info(f"    {p}%: {np.percentile(ot_array, p):.2f} 토큰")
            
            # 구간별 분포 (데이터 특성에 맞게 조정)
            max_val = ot_array.max()
            if max_val < 5000:
                bins = [0, 1000, 2000, 3000, 4000, float('inf')]
                labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K+']
            elif max_val < 15000:
                bins = [0, 5000, 10000, 15000, float('inf')]
                labels = ['0-5K', '5K-10K', '10K-15K', '15K+']
            else:
                bins = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
                labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']
            
            logger.info(f"  구간별 분포:")
            for i in range(len(bins) - 1):
                count = ((ot_array >= bins[i]) & (ot_array < bins[i+1])).sum() if i < len(bins)-2 else (ot_array >= bins[i]).sum()
                percentage = count / len(ot_array) * 100
                logger.info(f"    {labels[i]}: {count}개 instance ({percentage:.1f}%)")
    
    # 분석 결과에 추가
    instance_stats = {}
    if correct_col and any(acc is not None for acc in instance_accuracies):
        valid_accuracies = [acc for acc in instance_accuracies if acc is not None]
        if valid_accuracies:
            acc_array = np.array(valid_accuracies)
            instance_stats['accuracy'] = {
                'total_instances': int(len(valid_accuracies)),
                'mean': float(acc_array.mean()),
                'median': float(np.median(acc_array)),
                'std': float(acc_array.std()),
                'min': float(acc_array.min()),
                'max': float(acc_array.max())
            }
    
    if any(pl is not None for pl in instance_prompt_lengths):
        valid_prompt_lengths = [pl for pl in instance_prompt_lengths if pl is not None]
        if valid_prompt_lengths:
            pl_array = np.array(valid_prompt_lengths)
            instance_stats['prompt_length'] = {
                'total_instances': int(len(valid_prompt_lengths)),
                'mean': float(pl_array.mean()),
                'median': float(np.median(pl_array)),
                'std': float(pl_array.std()),
                'min': float(pl_array.min()),
                'max': float(pl_array.max()),
                'percentiles': {int(p): float(np.percentile(pl_array, p)) for p in [25, 50, 75, 90, 95, 99]}
            }
    
    if any(ot is not None for ot in instance_avg_output_tokens):
        valid_output_tokens = [ot for ot in instance_avg_output_tokens if ot is not None]
        if valid_output_tokens:
            ot_array = np.array(valid_output_tokens)
            instance_stats['avg_output_token_count'] = {
                'total_instances': int(len(valid_output_tokens)),
                'mean': float(ot_array.mean()),
                'median': float(np.median(ot_array)),
                'std': float(ot_array.std()),
                'min': float(ot_array.min()),
                'max': float(ot_array.max()),
                'percentiles': {int(p): float(np.percentile(ot_array, p)) for p in [25, 50, 75, 90, 95, 99]}
            }
    
    if instance_stats:
        analysis_results['instance_level_stats'] = instance_stats
    
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
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))
    
    # 로그 파일 핸들러 설정
    setup_log_file_handler(output_dir, "analysis_log.txt")
    
    try:
        logger.info("\n" + "="*50)
        logger.info("=== 단일 파일 분석 모드 ===")
        logger.info("="*50)
        
        # 1. 파일 로드
        df = load_single_file(file_path)
        if df is None:
            logger.error("파일 로드 실패")
            return
        
        # 1.1. logprobs 컬럼 제거 (메모리 절약을 위해 초기에 제거)
        if 'logprobs' in df.columns:
            logger.info("\nlogprobs 컬럼 제거 중 (메모리 절약)...")
            df = df.drop(columns=['logprobs'])
            logger.info("logprobs 컬럼 제거 완료")
        
        logger.info(f"출력 디렉토리: {output_dir}")
        
        # 2. generated_text 파싱 및 검증
        if 'reasoning_content' not in df.columns or 'content' not in df.columns or 'final_answer' not in df.columns or 'is_correct_math_verifier' not in df.columns:
            logger.info("\n파싱 컬럼이 없어 파싱을 진행합니다...")
            df = parse_and_verify_dataframe(df)
            output_filename = "raw_generated_parsed.parquet"
            output_path = os.path.join(output_dir, output_filename)
            df.to_parquet(output_path, index=False, compression="zstd")
            logger.info(f"파싱된 데이터 저장: {output_path}")
        else:
            logger.info("이미 파싱된 데이터입니다.")
        
        # 3. Exact matching vs MathVerifier 비교 (voting 결과 포함)
        df = print_verification_comparison(df)
        
        # 4. confidence score와 정답률 상관관계 분석
        analyze_confidence_correlation(df, output_dir)
        
        # # 5. output_token_count 분포 분석
        analyze_token_distribution(df, output_dir)
        
        # # 5.1 total token 구간별 정답률 분석
        analyze_accuracy_by_total_tokens(df, output_dir)
        
        # # 6. </think> 이후 토큰 수 분석
        # post_redaction_tokens = extract_post_redaction_tokens(df)
        # analyze_post_redaction_tokens(post_redaction_tokens, output_dir)
        
        # # 7. Instance별 분포 분석
        # analyze_instance_level_distributions(df, output_dir)
        
        # # 8. Instance별 정확도와 Confidence/Output Token 상관관계 분석
        # analyze_instance_accuracy_correlation(df, output_dir)
        
        # 9. cleaned 파일 저장 (logprobs는 이미 제거됨)
        logger.info("\n" + "="*50)
        logger.info("=== cleaned 파일 저장 ===")
        logger.info("="*50)
        cleaned_filename = "raw_generated_cleaned.parquet"
        cleaned_path = os.path.join(output_dir, cleaned_filename)
        df.to_parquet(cleaned_path, index=False, compression="zstd")
        logger.info(f"Cleaned 파일 저장 완료: {cleaned_path}")
        logger.info(f"Cleaned 파일 크기: {len(df)}개 행, {len(df.columns)}개 컬럼")
        
        # output_dir에 raw_generated_parsed.parquet 파일이 있으면 삭제
        parsed_file_path = os.path.join(output_dir, "raw_generated_parsed.parquet")
        if os.path.exists(parsed_file_path):
            try:
                os.remove(parsed_file_path)
                logger.info(f"raw_generated_parsed.parquet 파일 삭제 완료: {parsed_file_path}")
            except Exception as e:
                logger.warning(f"raw_generated_parsed.parquet 삭제 중 오류: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("✅ 단일 파일 분석 완료!")
        logger.info("="*50)
    finally:
        # 로그 파일 핸들러 제거
        remove_log_file_handler()
        logger.info(f"분석 로그 파일 저장 완료: {os.path.join(output_dir, 'analysis_log.txt')}")


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
    
    try:
        # 샤드 파일 병합만 수행 (분석 없이)
        df = merge_shard_files(data_dir, output_filename)
        if df is None:
            logger.error("병합 실패")
            return
        
        logger.info("\n" + "="*50)
        logger.info("✅ 병합 완료!")
        logger.info("="*50)
        logger.info(f"병합된 파일: {os.path.join(data_dir, output_filename)}")
        logger.info(f"총 {len(df)}개 행, {df['problem_id'].nunique() if 'problem_id' in df.columns else 'N/A'}개 문제")
    except Exception as e:
        logger.error(f"병합 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())


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
