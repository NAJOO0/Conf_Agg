"""
Diverse Confidence Sampling 테스트 스크립트

이 스크립트는 diverse_confidence_sampling 기능이 제대로 작동하는지 검증합니다.
생성된 세트들의 confidence 분포를 분석하여 diverse sampling이 작동하는지 확인합니다.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_confidence_distribution(curated_data_path: str, confidence_key: str = "bottom_10_percent_confidence"):
    """
    큐레이션된 데이터의 confidence 분포를 분석합니다.

    Args:
        curated_data_path: 큐레이션된 데이터 파일 경로
        confidence_key: 사용된 confidence 키
    """
    logger.info(f"데이터 로드: {curated_data_path}")

    # 필요한 컬럼만 읽기
    import pyarrow.parquet as pq
    columns_to_read = ['set_id', 'problem_id', 'selected_confidence']

    try:
        table = pq.read_table(curated_data_path, columns=columns_to_read)
        df = table.to_pandas()
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise

    logger.info(f"총 세트 수: {len(df)}")

    # 각 세트의 confidence 분포 분석
    set_conf_stats = []

    for idx, row in df.iterrows():
        # selected_confidence 파싱 (리스트 형태로 저장되어 있음)
        selected_conf = row.get('selected_confidence', [])

        # numpy array나 list인지 확인
        if selected_conf is None:
            logger.warning(f"세트 {idx}: confidence 정보 없음")
            continue

        # confidence 통계 계산
        conf_array = np.array(selected_conf)

        if len(conf_array) == 0:
            logger.warning(f"세트 {idx}: confidence 정보 없음")
            continue
        stats = {
            'set_id': row.get('set_id', idx),
            'problem_id': row.get('problem_id', 'unknown'),
            'min': float(np.min(conf_array)),
            'max': float(np.max(conf_array)),
            'mean': float(np.mean(conf_array)),
            'std': float(np.std(conf_array)),
            'range': float(np.max(conf_array) - np.min(conf_array)),
            'num_solutions': len(selected_conf)
        }
        set_conf_stats.append(stats)

    # 전체 통계 계산
    if set_conf_stats:
        ranges = [s['range'] for s in set_conf_stats]
        stds = [s['std'] for s in set_conf_stats]
        means = [s['mean'] for s in set_conf_stats]

        logger.info("\n=== 전체 세트의 Confidence 분포 통계 ===")
        logger.info(f"평균 Range: {np.mean(ranges):.6f} (±{np.std(ranges):.6f})")
        logger.info(f"평균 Std: {np.mean(stds):.6f} (±{np.std(stds):.6f})")
        logger.info(f"평균 Mean: {np.mean(means):.6f} (±{np.std(means):.6f})")

        # Range가 큰 순서대로 상위 5개 출력
        logger.info("\n=== Confidence Range가 큰 상위 5개 세트 ===")
        sorted_stats = sorted(set_conf_stats, key=lambda x: x['range'], reverse=True)
        for i, s in enumerate(sorted_stats[:5], 1):
            logger.info(f"{i}. Set: {s['set_id']}, Problem: {s['problem_id']}")
            logger.info(f"   Range: {s['range']:.6f}, Mean: {s['mean']:.6f}, Std: {s['std']:.6f}")
            logger.info(f"   Min: {s['min']:.6f}, Max: {s['max']:.6f}")

        # Range가 작은 순서대로 하위 5개 출력
        logger.info("\n=== Confidence Range가 작은 하위 5개 세트 ===")
        for i, s in enumerate(sorted_stats[-5:], 1):
            logger.info(f"{i}. Set: {s['set_id']}, Problem: {s['problem_id']}")
            logger.info(f"   Range: {s['range']:.6f}, Mean: {s['mean']:.6f}, Std: {s['std']:.6f}")
            logger.info(f"   Min: {s['min']:.6f}, Max: {s['max']:.6f}")

        return set_conf_stats
    else:
        logger.error("분석할 데이터가 없습니다.")
        return []


def compare_distributions(diverse_stats: list, random_stats: list):
    """
    Diverse sampling과 random sampling의 분포를 비교합니다.

    Args:
        diverse_stats: Diverse sampling 통계
        random_stats: Random sampling 통계
    """
    logger.info("\n=== Diverse vs Random Sampling 비교 ===")

    if diverse_stats and random_stats:
        diverse_ranges = [s['range'] for s in diverse_stats]
        random_ranges = [s['range'] for s in random_stats]

        diverse_stds = [s['std'] for s in diverse_stats]
        random_stds = [s['std'] for s in random_stats]

        logger.info(f"\nDiverse Sampling:")
        logger.info(f"  평균 Range: {np.mean(diverse_ranges):.6f}")
        logger.info(f"  평균 Std: {np.mean(diverse_stds):.6f}")

        logger.info(f"\nRandom Sampling:")
        logger.info(f"  평균 Range: {np.mean(random_ranges):.6f}")
        logger.info(f"  평균 Std: {np.mean(random_stds):.6f}")

        logger.info(f"\n개선율:")
        range_improvement = (np.mean(diverse_ranges) - np.mean(random_ranges)) / np.mean(random_ranges) * 100
        std_improvement = (np.mean(diverse_stds) - np.mean(random_stds)) / np.mean(random_stds) * 100
        logger.info(f"  Range: {range_improvement:+.2f}%")
        logger.info(f"  Std: {std_improvement:+.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diverse Confidence Sampling 테스트")
    parser.add_argument(
        "--diverse-data",
        type=str,
        help="Diverse sampling으로 생성된 데이터 경로"
    )
    parser.add_argument(
        "--random-data",
        type=str,
        help="Random sampling으로 생성된 데이터 경로 (비교용, 선택사항)"
    )
    parser.add_argument(
        "--confidence-key",
        type=str,
        default="bottom_10_percent_confidence",
        help="사용된 confidence 키"
    )

    args = parser.parse_args()

    if not args.diverse_data:
        logger.error("--diverse-data 경로를 지정해주세요")
        sys.exit(1)

    if not os.path.exists(args.diverse_data):
        logger.error(f"파일을 찾을 수 없습니다: {args.diverse_data}")
        sys.exit(1)

    # Diverse sampling 분석
    logger.info("=" * 80)
    logger.info("Diverse Confidence Sampling 분석")
    logger.info("=" * 80)
    diverse_stats = analyze_confidence_distribution(args.diverse_data, args.confidence_key)

    # Random sampling과 비교 (선택사항)
    if args.random_data and os.path.exists(args.random_data):
        logger.info("\n" + "=" * 80)
        logger.info("Random Sampling 분석 (비교용)")
        logger.info("=" * 80)
        random_stats = analyze_confidence_distribution(args.random_data, args.confidence_key)

        # 비교 분석
        compare_distributions(diverse_stats, random_stats)
    else:
        logger.info("\n비교용 random sampling 데이터가 제공되지 않았습니다.")

    logger.info("\n분석 완료!")
