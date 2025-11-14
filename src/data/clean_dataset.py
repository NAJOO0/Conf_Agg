#!/usr/bin/env python3
"""
데이터셋 정리 스크립트

주요 기능:
1. final_answer가 없는 응답 제거
2. output_token_count 컬럼을 사용한 토큰 길이 필터링
3. 문제별 응답 개수 필터링
4. logprobs 컬럼 제거하여 파일 크기 줄이기
"""
import os
import sys
import pandas as pd
import argparse
import logging
from typing import Optional

# PyArrow 가용성 확인
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_dataset(
    input_path: str,
    output_path: str,
    token_threshold: int = 2000,
    target_num_responses: Optional[int] = None,
    remove_logprobs: bool = True
):
    """
    데이터셋 정리: final_answer가 없는 경우, 토큰 길이, 응답 개수 필터링
    
    Args:
        input_path: 입력 parquet 파일 경로
        output_path: 출력 parquet 파일 경로
        token_threshold: 토큰 길이 필터링 임계값 (이 값 이상인 응답 제거)
        target_num_responses: 목표 응답 개수 (None이면 가장 많이 나온 개수 사용)
        remove_logprobs: logprobs 컬럼 제거 여부
    """
    logger.info("데이터셋 정리 시작")
    logger.info(f"입력 파일: {input_path}")
    logger.info(f"출력 파일: {output_path}")
    
    # 데이터 로드
    logger.info("데이터 로드 중...")
    if HAS_PYARROW:
        try:
            table = pq.read_table(input_path, memory_map=False)
            generated_data = table.to_pandas(types_mapper=pd.ArrowDtype)
        except Exception as e:
            logger.warning(f"PyArrow types_mapper 사용 실패: {e}, 기본 변환으로 재시도...")
            table = pq.read_table(input_path, memory_map=False)
            generated_data = table.to_pandas()
    else:
        generated_data = pd.read_parquet(input_path)
    
    original_count = len(generated_data)
    original_problem_count = generated_data['problem_id'].nunique()
    logger.info(f"로드 완료: {original_count}개 응답, {original_problem_count}개 문제")
    
    # logprobs 컬럼 제거
    if remove_logprobs:
        logprobs_cols = [col for col in generated_data.columns if 'logprobs' in col.lower() or 'log_prob' in col.lower()]
        if logprobs_cols:
            logger.info(f"logprobs 컬럼 제거 중: {logprobs_cols}")
            generated_data = generated_data.drop(columns=logprobs_cols)
            logger.info(f"제거 완료: {len(logprobs_cols)}개 컬럼 제거됨")
        else:
            logger.info("logprobs 컬럼이 없습니다.")
    
    # string 타입을 large_string으로 변환 (offset overflow 방지)
    logger.info("string 타입을 large_string으로 변환 중...")
    string_cols = generated_data.select_dtypes(include=['string[pyarrow]']).columns
    if not string_cols.empty:
        logger.info(f"변환 대상 컬럼: {string_cols.to_list()}")
        for col in string_cols:
            try:
                generated_data[col] = generated_data[col].astype('large_string[pyarrow]')
            except Exception as e:
                logger.warning(f"'{col}' 컬럼 large_string 변환 실패: {e}")
    else:
        object_cols = generated_data.select_dtypes(include=['object']).columns
        for col in object_cols:
            try:
                if not generated_data[col].empty and isinstance(generated_data[col].dropna().iloc[0], str):
                    generated_data[col] = generated_data[col].astype('large_string[pyarrow]')
                    logger.info(f"'{col}' (object) 컬럼을 large_string으로 변환.")
            except Exception as e:
                logger.warning(f"'{col}' (object) 컬럼 변환 중 오류 발생 (무시): {e}")
    
    logger.info("large_string 변환 완료.")
    
    # Step 1: final_answer가 없는 경우 제거
    logger.info("\nStep 1: final_answer가 없는 응답 제거 중...")
    before_final_answer = len(generated_data)
    
    if 'final_answer' in generated_data.columns:
        generated_data = generated_data[
            generated_data['final_answer'].notna() & 
            (generated_data['final_answer'] != '')
        ].copy()
    else:
        logger.warning("'final_answer' 컬럼이 없습니다. 이 단계를 건너뜁니다.")
    
    after_final_answer = len(generated_data)
    removed_final_answer = before_final_answer - after_final_answer
    logger.info(f"final_answer가 비어있는 응답 제거: {removed_final_answer}개 응답 제거됨")
    
    # Step 2: output_token_count 컬럼을 사용한 토큰 길이 필터링
    if 'output_token_count' in generated_data.columns:
        logger.info("\nStep 2: output_token_count 컬럼을 사용한 토큰 길이 필터링 중...")
        logger.info(f"토큰 길이 필터링: {token_threshold}개 이상인 응답 제거")
        before_token_filter = len(generated_data)
        
        # output_token_count가 NaN이거나 threshold 이상인 경우 제거
        generated_data = generated_data[
            generated_data['output_token_count'].notna() & 
            (generated_data['output_token_count'] < token_threshold)
        ].copy()
        
        after_token_filter = len(generated_data)
        removed_token_filter = before_token_filter - after_token_filter
        logger.info(f"토큰 길이 기준 제거: {removed_token_filter}개 응답 제거됨")
    else:
        logger.warning("'output_token_count' 컬럼이 없습니다. 토큰 길이 필터 단계를 건너뜁니다.")
    
    # Step 3: 문제별 응답 개수 확인 및 필터링
    logger.info("\nStep 3: 문제별 응답 개수 필터링 중...")
    problem_counts = generated_data.groupby('problem_id').size()
    
    if target_num_responses is None:
        # 가장 많이 나온 응답 개수를 최대값으로 설정
        target_num_responses = problem_counts.max()
        logger.info(f"최대 응답 개수: {target_num_responses}개")
    
    # 최대 개수보다 2개를 초과해서 부족한 문제만 제거
    # 예: 최대 16개면 14개 이하인 문제만 제거 (15개, 16개는 유지)
    threshold = 7
    valid_problem_ids = problem_counts[problem_counts > threshold].index
    problems_removed = problem_counts[problem_counts <= threshold]
    num_problems_removed = len(problems_removed)
    num_responses_removed = problems_removed.sum() if num_problems_removed > 0 else 0
    
    logger.info(f"응답 개수 필터링: {threshold}개 이하인 문제 제거")
    logger.info(f"제거될 문제: {num_problems_removed}개")
    logger.info(f"응답 개수 분포:")
    count_dist = problem_counts.value_counts().sort_index()
    for count, num_problems in count_dist.items():
        status = "제거" if count <= threshold else "유지"
        logger.info(f"  {count}개 응답: {num_problems}개 문제 ({status})")
    
    # 필터링 적용
    generated_data = generated_data[generated_data['problem_id'].isin(valid_problem_ids)].copy()
    
    final_count = len(generated_data)
    final_problem_count = generated_data['problem_id'].nunique()
    
    # 최종 요약
    logger.info(f"\n{'='*60}")
    logger.info("데이터셋 정리 완료:")
    logger.info(f"  원본 응답 수: {original_count}개")
    logger.info(f"  최종 응답 수: {final_count}개 ({original_count - final_count}개 제거됨)")
    logger.info(f"  원본 문제 수: {original_problem_count}개")
    logger.info(f"  최종 문제 수: {final_problem_count}개 ({original_problem_count - final_problem_count}개 문제 제거됨)")
    logger.info(f"{'='*60}\n")
    
    # 저장
    logger.info(f"정리된 데이터 저장 중: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    generated_data.to_parquet(output_path, index=False)
    logger.info(f"저장 완료: {output_path}")
    
    # 파일 크기 비교
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"\n파일 크기:")
    logger.info(f"  입력 파일: {input_size:.2f} MB")
    logger.info(f"  출력 파일: {output_size:.2f} MB")
    logger.info(f"  크기 감소: {input_size - output_size:.2f} MB ({(1 - output_size/input_size)*100:.2f}% 감소)")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 정리 스크립트")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 parquet 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 parquet 파일 경로 (지정하지 않으면 입력 파일명에 _cleaned를 붙여서 자동 생성)"
    )
    parser.add_argument(
        "--token-threshold",
        type=int,
        default=2500,
        help="토큰 길이 필터링 임계값 (output_token_count 컬럼 사용, 기본값: 1000)"
    )
    parser.add_argument(
        "--target-num-responses",
        type=int,
        default=None,
        help="목표 응답 개수 (None이면 가장 많이 나온 개수 사용)"
    )
    parser.add_argument(
        "--keep-logprobs",
        action="store_true",
        help="logprobs 컬럼을 유지합니다 (기본값: 제거)"
    )
    
    args = parser.parse_args()
    
    # 출력 파일 경로가 지정되지 않으면 입력 파일명에 _cleaned를 붙여서 자동 생성
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_basename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_basename)
        output_basename = f"{input_name}_cleaned.parquet"
        args.output = os.path.join(input_dir, output_basename) if input_dir else output_basename
    
    clean_dataset(
        input_path=args.input,
        output_path=args.output,
        token_threshold=args.token_threshold,
        target_num_responses=args.target_num_responses,
        remove_logprobs=not args.keep_logprobs
    )


if __name__ == "__main__":
    main()

