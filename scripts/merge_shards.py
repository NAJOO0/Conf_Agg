#!/usr/bin/env python3
"""
샤드별로 생성된 Parquet 파일들을 하나로 병합하는 스크립트
"""
import os
import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_shard_files(data_dir: str, output_filename: str = "raw_generated.parquet") -> None:
    """
    샤드별 Parquet 파일들을 하나로 병합
    
    Args:
        data_dir: 데이터 디렉토리 경로
        output_filename: 출력 파일명
    """
    # generated_dir = os.path.join(data_dir, "generated")
    generated_dir = data_dir
    
    if not os.path.exists(generated_dir):
        logger.error(f"Generated 디렉토리를 찾을 수 없습니다: {generated_dir}")
        return
    
    # 샤드 파일들 찾기
    shard_files = []
    for i in range(4):  # 0, 1, 2, 3 샤드
        shard_file = os.path.join(generated_dir, f"raw_generated_shard_{i}.parquet")
        if os.path.exists(shard_file):
            shard_files.append(shard_file)
            logger.info(f"샤드 파일 발견: {shard_file}")
        else:
            logger.warning(f"샤드 파일 없음: {shard_file}")
    
    if not shard_files:
        logger.error("병합할 샤드 파일이 없습니다.")
        return
    
    # 샤드 파일들 로드 및 병합
    logger.info(f"{len(shard_files)}개 샤드 파일 병합 시작...")
    dataframes = []
    
    for shard_file in shard_files:
        try:
            df = pd.read_parquet(shard_file)
            dataframes.append(df)
            logger.info(f"로드 완료: {shard_file} ({len(df)}개 행)")
        except Exception as e:
            logger.error(f"파일 로드 실패: {shard_file}, 오류: {e}")
    
    if not dataframes:
        logger.error("로드된 데이터프레임이 없습니다.")
        return
    
    # 병합
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 결과 저장
    output_path = os.path.join(generated_dir, output_filename)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="샤드별 Parquet 파일들을 병합")
    parser.add_argument("--data-dir", type=str, required=True, help="데이터 디렉토리 경로")
    parser.add_argument("--output", type=str, default="raw_generated.parquet", help="출력 파일명")
    
    args = parser.parse_args()
    
    merge_shard_files(args.data_dir, args.output)


