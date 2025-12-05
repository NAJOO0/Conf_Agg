"""
테스트용 큐레이션 스크립트
"""
import os
import sys
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.curation import DataCurator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_curation(use_diverse_sampling: bool, input_path: str = None):
    """
    큐레이션 실행

    Args:
        use_diverse_sampling: diverse confidence sampling 사용 여부
        input_path: 입력 파일 경로 (None이면 기본 테스트 데이터 사용)
    """
    # 입력 파일 경로
    if input_path is None:
        generated_data_path = "/mnt/data1/datasets/nlp/conf_agg/generated/test_diverse_sampling/dataset_100_test.parquet"
        suffix = "diverse" if use_diverse_sampling else "random"
        output_dir = f"/mnt/data1/datasets/nlp/conf_agg/curated_test_{suffix}"
    else:
        generated_data_path = input_path
        suffix = "diverse" if use_diverse_sampling else "random"
        output_dir = f"/mnt/data1/datasets/nlp/conf_agg/curated_full_{suffix}"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting curation with diverse_confidence_sampling={use_diverse_sampling}")
    logger.info(f"Input: {generated_data_path}")
    logger.info(f"Output: {output_dir}")

    # 데이터 큐레이션 실행
    curator = DataCurator(
        strategy="naive",
        easy_sample_percentage=50,
        num_sets_per_problem=4,
        set_size=8,
        enable_thinking=False,
        timeout=30,
        confidence_key="bottom_10_percent_confidence",
        fill_insufficient_with_sampling=True,
        order_strategy="shuffle",
        diverse_confidence_sampling=use_diverse_sampling,
    )

    result_paths = curator.curate_data(
        generated_data_path,
        output_dir,
        train_split=1.0  # 테스트용이므로 전체를 train으로
    )

    logger.info(f"Curation completed!")
    logger.info(f"Output: {result_paths['train']}")

    return result_paths['train']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--diverse", action="store_true", help="Use diverse confidence sampling")
    parser.add_argument("--input", type=str, default=None, help="Input parquet file path")
    args = parser.parse_args()

    run_curation(use_diverse_sampling=args.diverse, input_path=args.input)
