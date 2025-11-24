#!/usr/bin/env python3
"""
문제를 맞춘 inference 개수로 분류하고 각 카테고리에서 10% 샘플링하는 스크립트

처리 과정:
1. 48개 이상의 inference를 가진 문제는 48개로 필터링
2. 각 문제를 맞춘 inference 개수로 분류
3. 각 카테고리에서 10% 랜덤 샘플링
4. 샘플링된 문제의 inference만 필터링하여 저장
5. 통계 출력 및 시각화
"""
import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings
import random
from collections import defaultdict
from typing import Dict, List, Tuple

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_log_file_handler(output_dir: str, log_filename: str = "sampling_log.txt") -> logging.FileHandler:
    """
    로그 파일 핸들러 설정

    Args:
        output_dir: 로그 파일을 저장할 디렉토리
        log_filename: 로그 파일명

    Returns:
        로그 파일 핸들러
    """
    log_path = os.path.join(output_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info(f"로그 파일 저장 시작: {log_path}")

    return file_handler


def filter_inferences_to_max(df: pd.DataFrame, max_inferences: int = 48,
                              problem_id_col: str = 'problem_id') -> pd.DataFrame:
    """
    각 문제의 inference 개수를 최대값으로 제한

    Args:
        df: 원본 데이터프레임
        max_inferences: 최대 inference 개수
        problem_id_col: 문제 ID 컬럼명

    Returns:
        필터링된 데이터프레임
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 1: Inference 개수 필터링 (최대 {max_inferences}개) ===")
    logger.info(f"{'='*50}")

    original_size = len(df)
    logger.info(f"원본 데이터 크기: {original_size:,} rows")

    # 각 문제별로 inference 개수 확인
    problem_inference_counts = df.groupby(problem_id_col).size()
    problems_over_max = (problem_inference_counts > max_inferences).sum()

    logger.info(f"\n{max_inferences}개 초과 문제 수: {problems_over_max} / {len(problem_inference_counts)}")

    if problems_over_max > 0:
        # 각 문제별로 최대 max_inferences개만 랜덤 샘플링
        filtered_dfs = []
        for problem_id, group in df.groupby(problem_id_col):
            if len(group) > max_inferences:
                # 랜덤 샘플링
                sampled_group = group.sample(n=max_inferences, random_state=42)
                filtered_dfs.append(sampled_group)
            else:
                filtered_dfs.append(group)

        filtered_df = pd.concat(filtered_dfs, ignore_index=True)

        logger.info(f"\n필터링 후 데이터 크기: {len(filtered_df):,} rows")
        logger.info(f"제거된 행 수: {original_size - len(filtered_df):,}")

        # 필터링 후 통계
        new_inference_counts = filtered_df.groupby(problem_id_col).size()
        logger.info(f"\n필터링 후 inference 통계:")
        logger.info(f"  평균: {new_inference_counts.mean():.2f}")
        logger.info(f"  중앙값: {new_inference_counts.median():.2f}")
        logger.info(f"  최소: {new_inference_counts.min()}")
        logger.info(f"  최대: {new_inference_counts.max()}")

        return filtered_df
    else:
        logger.info(f"\n모든 문제의 inference 개수가 {max_inferences} 이하입니다. 필터링 불필요.")
        return df.copy()


def classify_by_correct_count(df: pd.DataFrame, problem_id_col: str = 'problem_id',
                                correct_col: str = None) -> Tuple[Dict[int, List[str]], pd.DataFrame]:
    """
    각 문제를 맞춘 inference 개수로 분류

    Args:
        df: 데이터프레임
        problem_id_col: 문제 ID 컬럼명
        correct_col: 정답 여부 컬럼명 (None이면 자동 탐지)

    Returns:
        (카테고리별 문제 ID 딕셔너리, 문제별 통계 데이터프레임)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 2: 맞춘 Inference 개수로 문제 분류 ===")
    logger.info(f"{'='*50}")

    # 정답 컬럼 자동 탐지
    if correct_col is None:
        if 'is_correct_math_verifier' in df.columns:
            correct_col = 'is_correct_math_verifier'
        elif 'is_correct' in df.columns:
            correct_col = 'is_correct'
        else:
            raise ValueError("정답 컬럼(is_correct 또는 is_correct_math_verifier)을 찾을 수 없습니다.")

    logger.info(f"사용하는 정답 컬럼: {correct_col}")

    # 각 문제별로 맞춘 inference 개수 계산
    problem_stats = []

    for problem_id, group in df.groupby(problem_id_col):
        total_inferences = len(group)
        correct_count = group[correct_col].astype(int).sum()

        problem_stats.append({
            'problem_id': problem_id,
            'total_inferences': total_inferences,
            'correct_count': correct_count,
            'accuracy': correct_count / total_inferences if total_inferences > 0 else 0.0
        })

    problem_stats_df = pd.DataFrame(problem_stats)

    # 맞춘 개수별로 문제 분류
    correct_count_categories = defaultdict(list)

    for _, row in problem_stats_df.iterrows():
        correct_count = row['correct_count']
        problem_id = row['problem_id']
        correct_count_categories[correct_count].append(problem_id)

    # 통계 출력
    logger.info(f"\n총 문제 수: {len(problem_stats_df):,}")
    logger.info(f"서로 다른 정답 개수 종류: {len(correct_count_categories)}")

    logger.info(f"\n맞춘 inference 개수 통계:")
    logger.info(f"  평균: {problem_stats_df['correct_count'].mean():.2f}")
    logger.info(f"  중앙값: {problem_stats_df['correct_count'].median():.2f}")
    logger.info(f"  표준편차: {problem_stats_df['correct_count'].std():.2f}")
    logger.info(f"  최소: {problem_stats_df['correct_count'].min()}")
    logger.info(f"  최대: {problem_stats_df['correct_count'].max()}")

    logger.info(f"\n평균 정답률: {problem_stats_df['accuracy'].mean():.3f} ({problem_stats_df['accuracy'].mean()*100:.1f}%)")

    # 카테고리별 문제 개수 출력 (정답 개수 오름차순)
    logger.info(f"\n=== 맞춘 Inference 개수별 문제 분포 ===")
    sorted_categories = sorted(correct_count_categories.items(), key=lambda x: x[0])

    for correct_count, problem_ids in sorted_categories:
        count = len(problem_ids)
        percentage = count / len(problem_stats_df) * 100
        logger.info(f"  {correct_count}개 맞춤: {count:3d}개 문제 ({percentage:5.1f}%)")

    return dict(correct_count_categories), problem_stats_df


def sample_problems_by_category(categories: Dict[int, List[str]],
                                  sample_ratio: float = 0.1,
                                  random_seed: int = 42) -> Dict[int, List[str]]:
    """
    각 카테고리에서 일정 비율만큼 랜덤 샘플링

    Args:
        categories: 카테고리별 문제 ID 딕셔너리
        sample_ratio: 샘플링 비율 (0.1 = 10%)
        random_seed: 랜덤 시드

    Returns:
        샘플링된 카테고리별 문제 ID 딕셔너리
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 3: 각 카테고리에서 {sample_ratio*100:.0f}% 랜덤 샘플링 ===")
    logger.info(f"{'='*50}")

    random.seed(random_seed)
    np.random.seed(random_seed)

    sampled_categories = {}
    total_original = 0
    total_sampled = 0

    logger.info(f"\n샘플링 결과 (random_seed={random_seed}):")

    sorted_categories = sorted(categories.items(), key=lambda x: x[0])

    for correct_count, problem_ids in sorted_categories:
        original_count = len(problem_ids)
        sample_size = max(1, int(original_count * sample_ratio))  # 최소 1개

        # 랜덤 샘플링
        sampled_ids = random.sample(problem_ids, sample_size)
        sampled_categories[correct_count] = sampled_ids

        total_original += original_count
        total_sampled += sample_size

        logger.info(f"  {correct_count}개 맞춤: {original_count:3d} → {sample_size:3d} ({sample_size/original_count*100:5.1f}%)")

    logger.info(f"\n전체 샘플링 결과:")
    logger.info(f"  원본 문제 수: {total_original:,}")
    logger.info(f"  샘플링된 문제 수: {total_sampled:,}")
    logger.info(f"  실제 샘플링 비율: {total_sampled/total_original*100:.2f}%")

    return sampled_categories


def filter_by_sampled_problems(df: pd.DataFrame,
                                 sampled_categories: Dict[int, List[str]],
                                 problem_id_col: str = 'problem_id') -> pd.DataFrame:
    """
    샘플링된 문제의 inference만 필터링

    Args:
        df: 원본 데이터프레임
        sampled_categories: 샘플링된 카테고리별 문제 ID 딕셔너리
        problem_id_col: 문제 ID 컬럼명

    Returns:
        필터링된 데이터프레임
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 4: 샘플링된 문제의 Inference 필터링 ===")
    logger.info(f"{'='*50}")

    # 샘플링된 모든 문제 ID 수집
    all_sampled_ids = []
    for problem_ids in sampled_categories.values():
        all_sampled_ids.extend(problem_ids)

    logger.info(f"샘플링된 문제 수: {len(all_sampled_ids):,}")

    # 필터링
    original_size = len(df)
    filtered_df = df[df[problem_id_col].isin(all_sampled_ids)].copy()

    logger.info(f"\n필터링 결과:")
    logger.info(f"  원본 inference 수: {original_size:,}")
    logger.info(f"  필터링된 inference 수: {len(filtered_df):,}")
    logger.info(f"  비율: {len(filtered_df)/original_size*100:.2f}%")

    # 문제당 평균 inference 수
    avg_inferences = len(filtered_df) / len(all_sampled_ids)
    logger.info(f"  문제당 평균 inference 수: {avg_inferences:.2f}")

    return filtered_df


def visualize_sampling_results(original_stats_df: pd.DataFrame,
                                 sampled_categories: Dict[int, List[str]],
                                 output_dir: str) -> None:
    """
    샘플링 결과 시각화

    Args:
        original_stats_df: 원본 문제 통계 데이터프레임
        sampled_categories: 샘플링된 카테고리별 문제 ID 딕셔너리
        output_dir: 출력 디렉토리
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 5: 샘플링 결과 시각화 ===")
    logger.info(f"{'='*50}")

    # 샘플링 전후 비교를 위한 데이터 준비
    original_dist = original_stats_df['correct_count'].value_counts().sort_index()

    sampled_dist = {}
    for correct_count, problem_ids in sampled_categories.items():
        sampled_dist[correct_count] = len(problem_ids)
    sampled_dist = pd.Series(sampled_dist).sort_index()

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 샘플링 전후 분포 비교 (막대 그래프)
    ax1 = axes[0, 0]
    x = np.arange(len(original_dist))
    width = 0.35

    ax1.bar(x - width/2, original_dist.values, width, label='Original', alpha=0.7, color='steelblue')
    ax1.bar(x + width/2, sampled_dist.reindex(original_dist.index, fill_value=0).values,
            width, label='Sampled (10%)', alpha=0.7, color='orange')

    ax1.set_xlabel('Number of Correct Inferences', fontsize=12)
    ax1.set_ylabel('Number of Problems', fontsize=12)
    ax1.set_title('Distribution Comparison: Original vs Sampled', fontsize=14)
    ax1.set_xticks(x[::max(1, len(x)//20)])  # 최대 20개 tick만 표시
    ax1.set_xticklabels(original_dist.index[::max(1, len(x)//20)], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 샘플링 비율 (선 그래프)
    ax2 = axes[0, 1]
    sampling_ratios = (sampled_dist.reindex(original_dist.index, fill_value=0) / original_dist * 100)

    ax2.plot(original_dist.index, sampling_ratios.values, marker='o', linewidth=2, markersize=4, color='green')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: 10%')

    ax2.set_xlabel('Number of Correct Inferences', fontsize=12)
    ax2.set_ylabel('Sampling Ratio (%)', fontsize=12)
    ax2.set_title('Sampling Ratio by Category', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 누적 분포 비교
    ax3 = axes[1, 0]
    original_cumsum = original_dist.cumsum()
    sampled_cumsum = sampled_dist.reindex(original_dist.index, fill_value=0).cumsum()

    ax3.plot(original_dist.index, original_cumsum.values, label='Original', linewidth=2, color='steelblue')
    ax3.plot(original_dist.index, sampled_cumsum.values, label='Sampled (10%)', linewidth=2, color='orange')

    ax3.set_xlabel('Number of Correct Inferences', fontsize=12)
    ax3.set_ylabel('Cumulative Number of Problems', fontsize=12)
    ax3.set_title('Cumulative Distribution Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 정답률 분포 (샘플링된 문제)
    ax4 = axes[1, 1]
    sampled_problem_ids = []
    for problem_ids in sampled_categories.values():
        sampled_problem_ids.extend(problem_ids)

    sampled_stats = original_stats_df[original_stats_df['problem_id'].isin(sampled_problem_ids)]

    ax4.hist(sampled_stats['accuracy'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(sampled_stats['accuracy'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean: {sampled_stats['accuracy'].mean():.3f}")
    ax4.axvline(sampled_stats['accuracy'].median(), color='orange', linestyle='--',
                linewidth=2, label=f"Median: {sampled_stats['accuracy'].median():.3f}")

    ax4.set_xlabel('Accuracy (Correct / Total)', fontsize=12)
    ax4.set_ylabel('Number of Problems', fontsize=12)
    ax4.set_title('Accuracy Distribution of Sampled Problems', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    plot_path = os.path.join(output_dir, 'sampling_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"시각화 저장 완료: {plot_path}")


def save_filtered_data(filtered_df: pd.DataFrame,
                        output_path: str,
                        sampled_categories: Dict[int, List[str]]) -> None:
    """
    필터링된 데이터 저장

    Args:
        filtered_df: 필터링된 데이터프레임
        output_path: 출력 파일 경로
        sampled_categories: 샘플링된 카테고리 정보 (메타데이터용)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"=== Step 6: 필터링된 데이터 저장 ===")
    logger.info(f"{'='*50}")

    # Parquet으로 저장
    filtered_df.to_parquet(output_path, index=False, engine='pyarrow')
    logger.info(f"필터링된 데이터 저장 완료: {output_path}")
    logger.info(f"  저장된 행 수: {len(filtered_df):,}")
    logger.info(f"  파일 크기: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    # 메타데이터 저장 (JSON)
    metadata = {
        'total_problems': sum(len(ids) for ids in sampled_categories.values()),
        'total_inferences': len(filtered_df),
        'categories': {
            str(correct_count): len(problem_ids)
            for correct_count, problem_ids in sorted(sampled_categories.items())
        },
        'sampling_info': {
            'method': 'random_sampling',
            'ratio': 0.1,
            'random_seed': 42,
            'max_inferences_per_problem': 48
        }
    }

    metadata_path = output_path.replace('.parquet', '_metadata.json')
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"메타데이터 저장 완료: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='문제를 맞춘 inference 개수로 분류하고 10% 샘플링하는 스크립트'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 Parquet 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 Parquet 파일 경로 (기본값: 입력 파일과 같은 디렉토리에 _sampled_10pct.parquet 추가)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='출력 디렉토리 (로그, 시각화 등) (기본값: 입력 파일 디렉토리)'
    )
    parser.add_argument(
        '--max-inferences',
        type=int,
        default=48,
        help='문제당 최대 inference 개수 (기본값: 48)'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=0.1,
        help='샘플링 비율 (기본값: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='랜덤 시드 (기본값: 42)'
    )
    parser.add_argument(
        '--problem-id-col',
        type=str,
        default='problem_id',
        help='문제 ID 컬럼명 (기본값: problem_id)'
    )
    parser.add_argument(
        '--correct-col',
        type=str,
        default=None,
        help='정답 여부 컬럼명 (기본값: 자동 탐지)'
    )

    args = parser.parse_args()

    # 출력 경로 설정
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_sampled_10pct.parquet")

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.output)

    os.makedirs(args.output_dir, exist_ok=True)

    # 로그 파일 핸들러 설정
    log_handler = setup_log_file_handler(args.output_dir)

    try:
        logger.info("="*80)
        logger.info("=== 문제별 Correct Inference Count 샘플링 스크립트 시작 ===")
        logger.info("="*80)
        logger.info(f"\n설정:")
        logger.info(f"  입력 파일: {args.input}")
        logger.info(f"  출력 파일: {args.output}")
        logger.info(f"  출력 디렉토리: {args.output_dir}")
        logger.info(f"  최대 inference 개수: {args.max_inferences}")
        logger.info(f"  샘플링 비율: {args.sample_ratio*100:.0f}%")
        logger.info(f"  랜덤 시드: {args.random_seed}")
        logger.info(f"  문제 ID 컬럼: {args.problem_id_col}")
        logger.info(f"  정답 컬럼: {args.correct_col if args.correct_col else '자동 탐지'}")

        # 데이터 로드
        logger.info(f"\n데이터 로딩 중...")
        df = pd.read_parquet(args.input)
        logger.info(f"로드 완료: {len(df):,} rows, {len(df.columns)} columns")

        # Step 1: Inference 개수 필터링
        filtered_df = filter_inferences_to_max(df, args.max_inferences, args.problem_id_col)

        # Step 2: 맞춘 inference 개수로 분류
        categories, problem_stats_df = classify_by_correct_count(
            filtered_df, args.problem_id_col, args.correct_col
        )

        # Step 3: 각 카테고리에서 샘플링
        sampled_categories = sample_problems_by_category(
            categories, args.sample_ratio, args.random_seed
        )

        # Step 4: 샘플링된 문제의 inference만 필터링
        final_df = filter_by_sampled_problems(
            filtered_df, sampled_categories, args.problem_id_col
        )

        # Step 5: 시각화
        visualize_sampling_results(problem_stats_df, sampled_categories, args.output_dir)

        # Step 6: 저장
        save_filtered_data(final_df, args.output, sampled_categories)

        logger.info("\n" + "="*80)
        logger.info("=== 모든 작업 완료 ✅ ===")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise
    finally:
        # 로그 핸들러 정리
        if log_handler:
            logger.removeHandler(log_handler)
            log_handler.close()


if __name__ == '__main__':
    main()
