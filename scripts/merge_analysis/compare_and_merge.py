#!/usr/bin/env python3
"""
폴더 내 두 개의 Parquet 파일을 찾아서 problem_id를 비교하고, 
모두 동일하면 병합하는 스크립트
"""
import os
import pandas as pd
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging

# PyArrow 가용성 확인
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_read_parquet(file_path: str) -> pd.DataFrame:
    """
    Parquet 파일을 안전하게 읽기 (중첩된 리스트 타입 처리)
    merge_and_analyze.py와 동일한 방식 사용
    """
    if HAS_PYARROW:
        # PyArrow로 직접 읽기 (merge_and_analyze.py와 동일한 방식)
        table = pq.read_table(file_path, memory_map=True)
        # types_mapper=pd.ArrowDtype를 사용하여 변환
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        return df
    else:
        # PyArrow가 없으면 기본 pandas 방법 사용
        return pd.read_parquet(file_path)


def find_parquet_files(directory: str) -> list:
    """
    디렉토리에서 Parquet 파일 찾기
    
    Args:
        directory: 디렉토리 경로
    
    Returns:
        Parquet 파일 경로 리스트
    """
    if not os.path.exists(directory):
        logger.error(f"디렉토리를 찾을 수 없습니다: {directory}")
        return []
    
    parquet_files = []
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            parquet_path = os.path.join(directory, file)
            parquet_files.append(parquet_path)
    
    return sorted(parquet_files)


def compare_problem_ids(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    두 DataFrame의 problem_id를 비교
    
    Args:
        df1: 첫 번째 DataFrame
        df2: 두 번째 DataFrame
    
    Returns:
        problem_id가 모두 동일하면 True, 그렇지 않으면 False
    """
    logger.info("=" * 80)
    logger.info("problem_id 비교 시작")
    logger.info("=" * 80)
    
    # problem_id 컬럼 확인
    if 'problem_id' not in df1.columns:
        logger.error("파일 1에 'problem_id' 컬럼이 없습니다.")
        return False
    
    if 'problem_id' not in df2.columns:
        logger.error("파일 2에 'problem_id' 컬럼이 없습니다.")
        return False
    
    # problem_id 추출 (고유값)
    problems1 = set(df1['problem_id'].unique())
    problems2 = set(df2['problem_id'].unique())
    
    logger.info(f"\n파일 1 고유 problem_id 수: {len(problems1)}")
    logger.info(f"파일 2 고유 problem_id 수: {len(problems2)}")
    
    # 비교
    only_in_file1 = problems1 - problems2
    only_in_file2 = problems2 - problems1
    common_problems = problems1 & problems2
    
    logger.info(f"\n공통 problem_id 수: {len(common_problems)}")
    logger.info(f"파일 1에만 있는 problem_id 수: {len(only_in_file1)}")
    logger.info(f"파일 2에만 있는 problem_id 수: {len(only_in_file2)}")
    
    # 결과 출력
    if len(only_in_file1) == 0 and len(only_in_file2) == 0:
        logger.info("\n✅ 모든 problem_id가 동일합니다!")
        return True
    else:
        logger.error("\n❌ problem_id가 일치하지 않습니다!")
        
        if len(only_in_file1) > 0:
            logger.error(f"\n파일 1에만 있는 problem_id ({len(only_in_file1)}개):")
            for problem_id in sorted(list(only_in_file1))[:20]:  # 최대 20개만 출력
                logger.error(f"  {problem_id}")
            if len(only_in_file1) > 20:
                logger.error(f"  ... 외 {len(only_in_file1) - 20}개 더")
        
        if len(only_in_file2) > 0:
            logger.error(f"\n파일 2에만 있는 problem_id ({len(only_in_file2)}개):")
            for problem_id in sorted(list(only_in_file2))[:20]:  # 최대 20개만 출력
                logger.error(f"  {problem_id}")
            if len(only_in_file2) > 20:
                logger.error(f"  ... 외 {len(only_in_file2) - 20}개 더")
        
        return False


def merge_parquet_dfs(df1: pd.DataFrame, df2: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    두 DataFrame을 병합하고 저장
    
    Args:
        df1: 첫 번째 DataFrame
        df2: 두 번째 DataFrame
        output_path: 출력 파일 경로
    
    Returns:
        병합된 데이터프레임
    """
    logger.info("\n" + "=" * 80)
    logger.info("DataFrame 병합 시작")
    logger.info("=" * 80)
    
    # 병합
    logger.info("\n파일 병합 중...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 결과 저장
    logger.info(f"\n병합된 데이터 저장 중: {output_path}")
    merged_df.to_parquet(output_path, index=False, compression="zstd")
    
    logger.info(f"\n✅ 병합 완료: {len(merged_df)}개 결과 저장")
    logger.info(f"저장 위치: {output_path}")
    
    # 통계 정보
    logger.info(f"\n통계 정보:")
    logger.info(f"  총 응답 수: {len(merged_df)}")
    logger.info(f"  문제 수: {merged_df['problem_id'].nunique()}")
    if merged_df['problem_id'].nunique() > 0:
        logger.info(f"  문제당 평균 응답 수: {len(merged_df) / merged_df['problem_id'].nunique():.1f}")
    
    return merged_df


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="폴더 내 두 개의 Parquet 파일을 찾아서 problem_id를 비교하고 병합",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python compare_parquet_problems.py --dir outputs/generated/merge_folder
  
  폴더 내에 정확히 2개의 .parquet 파일이 있어야 합니다.
        """
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Parquet 파일들이 있는 디렉토리 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged.parquet",
        help="병합된 출력 파일명 (기본값: merged.parquet)"
    )
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    directory = os.path.abspath(args.dir)
    
    # Parquet 파일 찾기
    logger.info(f"\n디렉토리에서 Parquet 파일 검색 중: {directory}")
    parquet_files = find_parquet_files(directory)
    
    if len(parquet_files) == 0:
        logger.error("Parquet 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    if len(parquet_files) != 2:
        logger.error(f"Parquet 파일이 정확히 2개여야 합니다. 현재 {len(parquet_files)}개 발견:")
        for f in parquet_files:
            logger.error(f"  {f}")
        sys.exit(1)
    
    file1_path = parquet_files[0]
    file2_path = parquet_files[1]
    
    logger.info(f"\n발견된 Parquet 파일:")
    logger.info(f"  파일 1: {file1_path}")
    logger.info(f"  파일 2: {file2_path}")
    
    # 파일 로드 (한 번만)
    logger.info(f"\n파일 1 로드 중: {file1_path}")
    try:
        df1 = safe_read_parquet(file1_path)
        logger.info(f"파일 1 로드 완료: {len(df1)}개 행")
    except Exception as e:
        logger.error(f"파일 1 로드 실패: {e}")
        sys.exit(1)

    logger.info(f"\n파일 2 로드 중: {file2_path}")
    try:
        df2 = safe_read_parquet(file2_path)
        logger.info(f"파일 2 로드 완료: {len(df2)}개 행")
    except Exception as e:
        logger.error(f"파일 2 로드 실패: {e}")
        sys.exit(1)
    
    # problem_id 비교 (이미 로드한 DF 사용)
    is_identical = compare_problem_ids(df1, df2)
    
    if not is_identical:
        logger.error("\n❌ problem_id가 일치하지 않아 병합을 수행하지 않습니다.")
        sys.exit(1)
    
    # 병합 수행
    output_path = os.path.join(directory, args.output)
    merged_df = merge_parquet_dfs(df1, df2, output_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 모든 작업 완료!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

