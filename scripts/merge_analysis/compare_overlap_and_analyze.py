#!/usr/bin/env python3
"""
폴더 내 두 개의 Parquet 파일을 찾아서
1) 공통 problem_id만 필터링하고
2) 각 파일의 정답 비율(Exact, MathVerifier)을 비교 출력

compare_and_merge.py 의 안전한 Parquet 로딩(safe_read_parquet) 방식을 참고하여 구현.
merge_and_analyze.py 의 검증 컬럼 명(`is_correct_exact`, `is_correct_math_verifier`)을 따름.
"""
import os
import sys
import argparse
import logging
from typing import Tuple, Set

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# PyArrow 가용성 확인 (compare_and_merge.py와 동일한 방식)
try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_read_parquet(file_path: str) -> pd.DataFrame:
    """
    Parquet 파일을 안전하게 읽기 (중첩 리스트 타입 등 Arrow 확장 고려)
    compare_and_merge.py 와 동일한 접근.
    """
    if HAS_PYARROW:
        table = pq.read_table(file_path, memory_map=True)
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        return df
    return pd.read_parquet(file_path)


def find_parquet_files(directory: str) -> list:
    """디렉토리에서 .parquet 파일 2개를 찾아 정렬해 반환."""
    if not os.path.exists(directory):
        logger.error(f"디렉토리를 찾을 수 없습니다: {directory}")
        return []
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    return sorted(parquet_files)


def get_problem_sets(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[Set, Set, Set]:
    """두 DF의 problem_id 집합 및 교집합 계산."""
    if 'problem_id' not in df1.columns:
        raise ValueError("파일 1에 'problem_id' 컬럼이 없습니다.")
    if 'problem_id' not in df2.columns:
        raise ValueError("파일 2에 'problem_id' 컬럼이 없습니다.")

    problems1 = set(pd.unique(df1['problem_id']))
    problems2 = set(pd.unique(df2['problem_id']))
    common = problems1 & problems2
    return problems1, problems2, common


def filter_by_problems(df: pd.DataFrame, problems: Set) -> pd.DataFrame:
    """지정된 problem_id 집합만 남기도록 필터링."""
    return df[df['problem_id'].isin(problems)].copy()


def compute_accuracy(df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    DF에서 정답 비율(Exact, MathVerifier)을 계산.
    검증 컬럼이 없으면 NaN을 반환.
    반환: (exact_acc, mv_acc, n_rows)
    """
    n = len(df)
    if n == 0:
        return float('nan'), float('nan'), 0

    exact_acc = float('nan')
    mv_acc = float('nan')

    if 'is_correct_exact' in df.columns:
        try:
            exact = df['is_correct_exact'].astype(int).to_numpy()
            exact_acc = exact.sum() / len(exact) if len(exact) > 0 else float('nan')
        except Exception:
            logger.warning("is_correct_exact을 정수로 변환하지 못했습니다.")

    if 'is_correct_math_verifier' in df.columns:
        try:
            mv = df['is_correct_math_verifier'].astype(int).to_numpy()
            mv_acc = mv.sum() / len(mv) if len(mv) > 0 else float('nan')
        except Exception:
            logger.warning("is_correct_math_verifier를 정수로 변환하지 못했습니다.")

    return exact_acc, mv_acc, n


def print_rate(label: str, acc: float, n: int) -> None:
    if pd.isna(acc):
        logger.info(f"  {label}: N/A (표시할 컬럼 없음), N={n}")
    else:
        logger.info(f"  {label}: {acc*100:.2f}% (N={n})")


def main():
    parser = argparse.ArgumentParser(
        description="두 Parquet의 공통 problem_id만 남겨 각 파일의 정답 비율을 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python scripts/compare_overlap_and_analyze.py --dir outputs/generated/some_folder \
      --save_filtered_prefix filtered

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
        "--save_filtered_prefix",
        type=str,
        default=None,
        help="공통 문제만 필터링한 파일을 저장할 접두사(미지정 시 저장 안 함)"
    )

    args = parser.parse_args()

    directory = os.path.abspath(args.dir)
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

    file1_path, file2_path = parquet_files[0], parquet_files[1]
    logger.info(f"\n발견된 Parquet 파일:")
    logger.info(f"  파일 1: {file1_path}")
    logger.info(f"  파일 2: {file2_path}")

    # 로드
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

    # 공통 problem_id 계산
    try:
        problems1, problems2, common = get_problem_sets(df1, df2)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("\n" + "="*80)
    logger.info("problem_id 교집합 분석")
    logger.info("="*80)
    logger.info(f"파일 1 고유 문제 수: {len(problems1)}")
    logger.info(f"파일 2 고유 문제 수: {len(problems2)}")
    logger.info(f"공통 문제 수: {len(common)}")

    if len(common) == 0:
        logger.error("공통 problem_id가 없어 비교를 수행할 수 없습니다.")
        sys.exit(1)

    # 공통 문제로 필터링
    df1_common = filter_by_problems(df1, common)
    df2_common = filter_by_problems(df2, common)
    logger.info(f"\n공통 문제 기준 필터링 후 행 수: 파일1={len(df1_common)}, 파일2={len(df2_common)}")

    # 정답 비율 계산 및 비교
    logger.info("\n" + "="*80)
    logger.info("정답 비율 비교 (공통 problem_id 기준)")
    logger.info("="*80)

    f1_exact, f1_mv, n1 = compute_accuracy(df1_common)
    f2_exact, f2_mv, n2 = compute_accuracy(df2_common)

    logger.info("\n파일 1 (공통 문제) 정답 비율:")
    print_rate("Exact", f1_exact, n1)
    print_rate("MathVerifier", f1_mv, n1)

    logger.info("\n파일 2 (공통 문제) 정답 비율:")
    print_rate("Exact", f2_exact, n2)
    print_rate("MathVerifier", f2_mv, n2)

    # 차이 출력
    logger.info("\n차이(파일2 - 파일1):")
    if not pd.isna(f1_exact) and not pd.isna(f2_exact):
        logger.info(f"  Exact 차이: {(f2_exact - f1_exact)*100:.2f}pp")
    else:
        logger.info("  Exact 차이: N/A")
    if not pd.isna(f1_mv) and not pd.isna(f2_mv):
        logger.info(f"  MathVerifier 차이: {(f2_mv - f1_mv)*100:.2f}pp")
    else:
        logger.info("  MathVerifier 차이: N/A")

    # 선택적으로 필터링 파일 저장
    if args.save_filtered_prefix:
        prefix = args.save_filtered_prefix
        out1 = os.path.join(directory, f"{prefix}_file1_common.parquet")
        out2 = os.path.join(directory, f"{prefix}_file2_common.parquet")
        logger.info(f"\n필터링된 데이터 저장 중...")
        try:
            df1_common.to_parquet(out1, index=False, compression="zstd")
            df2_common.to_parquet(out2, index=False, compression="zstd")
            logger.info(f"저장 완료:\n  {out1}\n  {out2}")
        except Exception as e:
            logger.warning(f"필터링된 데이터 저장 실패: {e}")

    logger.info("\n" + "="*80)
    logger.info("✅ 비교 완료")
    logger.info("="*80)


if __name__ == "__main__":
    main()


