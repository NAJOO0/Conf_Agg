#!/usr/bin/env python3
"""
Parquet 파일의 logprobs를 사용하여 confidence 점수를 재계산하는 스크립트 (최적화 버전 v2)
"""
import os
import sys
import pandas as pd
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.confidence import ConfidenceCalculator

# PyArrow 가용성 확인
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    pa = None
    pq = None
    HAS_PYARROW = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_logprobs_to_list(logprobs_value) -> Optional[List[List[float]]]:
    """logprobs 값을 리스트 형태로 변환 (최적화)"""
    if logprobs_value is None:
        return None
    
    # numpy array 우선 처리
    if isinstance(logprobs_value, np.ndarray):
        if logprobs_value.size == 0:
            return None
        return logprobs_value.tolist()
    
    # 리스트 처리
    if isinstance(logprobs_value, list):
        if len(logprobs_value) == 0:
            return None
        return logprobs_value
    
    # NaN 체크 (스칼라만)
    try:
        if pd.isna(logprobs_value):
            return None
    except (ValueError, TypeError):
        pass
    
    return None


def process_batch(batch_data: List[tuple], group_size: int) -> List[Optional[Dict]]:
    """
    배치 단위로 여러 row를 처리 (프로세스당 한 번만 Calculator 생성)
    """
    calculator = ConfidenceCalculator(group_size=group_size)
    results = []
    
    for idx, logprobs_value in batch_data:
        logprobs_list = convert_logprobs_to_list(logprobs_value)
        
        if logprobs_list is None:
            results.append((idx, None))
            continue
        
        try:
            confidence_scores = calculator.calculate_all_confidence_scores(logprobs_list)
            results.append((idx, confidence_scores))
        except Exception as e:
            logger.warning(f"Row {idx} confidence 계산 실패: {e}")
            results.append((idx, None))
    
    return results


def recalculate_confidence_scores_vectorized(
    df: pd.DataFrame,
    group_size: int = 512,
    chunk_offset: int = 0,
    num_workers: int = 4,
    batch_size: int = 100
) -> tuple:
    """
    벡터화 + 배치 멀티프로세싱으로 confidence 점수 계산 (최적화)
    """
    if 'logprobs' not in df.columns:
        logger.error("logprobs 컬럼이 없습니다.")
        return df, 0, len(df)
    
    column_names = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence',
        'head_confidence',
        'highest_group_confidence',
        'lowest_group_confidence',
        'top_10_percent_confidence',
    ]
    
    n_rows = len(df)
    
    # logprobs를 numpy array로 변환 (한 번만)
    logprobs_values = df['logprobs'].values
    
    # 인덱스와 데이터를 튜플로 준비
    data_with_idx = [(i, logprobs_values[i]) for i in range(n_rows)]
    
    # 배치로 나누기
    batches = []
    for i in range(0, n_rows, batch_size):
        batches.append(data_with_idx[i:i + batch_size])
    
    logger.info(f"  → {n_rows:,}개 row를 {len(batches):,}개 배치로 분할")
    
    # 결과 저장용 딕셔너리 (인덱스 → confidence_scores)
    results_dict = {}
    valid_count = 0
    invalid_count = 0
    
    # 멀티프로세싱
    if num_workers > 1:
        process_func = partial(process_batch, group_size=group_size)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_func, batch): batch_idx 
                      for batch_idx, batch in enumerate(batches)}
            
            completed_batches = 0
            for future in as_completed(futures):
                batch_results = future.result()
                
                for idx, confidence_scores in batch_results:
                    results_dict[idx] = confidence_scores
                    if confidence_scores is not None:
                        valid_count += 1
                    else:
                        invalid_count += 1
                
                completed_batches += 1
                if completed_batches % 50 == 0 or completed_batches == len(batches):
                    logger.info(
                        f"  → 진행: {completed_batches}/{len(batches)} 배치 "
                        f"(유효: {valid_count:,}, 무효: {invalid_count:,})"
                    )
        
        # ⭐ 계산 완료 후 즉시 logprobs 관련 메모리 해제
        del logprobs_values
        del data_with_idx
        
        # DataFrame에서도 logprobs 컬럼 제거
        if 'logprobs' in df.columns:
            df = df.drop(columns=['logprobs'])
            logger.info(f"  → logprobs 컬럼 제거 완료 (메모리 절약)")
    
    else:
        # 단일 프로세스 처리
        calculator = ConfidenceCalculator(group_size=group_size)
        for idx, logprobs_value in data_with_idx:
            logprobs_list = convert_logprobs_to_list(logprobs_value)
            
            if logprobs_list is None:
                results_dict[idx] = None
                invalid_count += 1
            else:
                try:
                    confidence_scores = calculator.calculate_all_confidence_scores(logprobs_list)
                    results_dict[idx] = confidence_scores
                    valid_count += 1
                except Exception as e:
                    logger.warning(f"Row {idx} 계산 실패: {e}")
                    results_dict[idx] = None
                    invalid_count += 1
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"  → 진행: {idx + 1}/{n_rows} (유효: {valid_count:,}, 무효: {invalid_count:,})")
        
        # ⭐ 계산 완료 후 즉시 logprobs 관련 메모리 해제
        del logprobs_values
        del data_with_idx
        
        # DataFrame에서도 logprobs 컬럼 제거
        if 'logprobs' in df.columns:
            df = df.drop(columns=['logprobs'])
            logger.info(f"  → logprobs 컬럼 제거 완료 (메모리 절약)")
    
    # 결과를 데이터프레임 컬럼으로 변환 (벡터화)
    for col_name in column_names:
        values = [
            results_dict[i].get(col_name, np.nan) if results_dict[i] is not None else np.nan
            for i in range(n_rows)
        ]
        df[col_name] = values
    
    # 메모리 명시적 해제
    del results_dict
    
    return df, valid_count, invalid_count


def read_parquet_chunk_by_rowgroups(
    parquet_file: pq.ParquetFile,
    start_row: int,
    chunk_size: int
) -> pd.DataFrame:
    """
    Row group 단위로 정확히 chunk_size만큼만 읽기
    """
    num_row_groups = parquet_file.num_row_groups
    dfs = []
    current_global_row = 0
    rows_read = 0
    
    for rg_idx in range(num_row_groups):
        rg_metadata = parquet_file.metadata.row_group(rg_idx)
        rg_num_rows = rg_metadata.num_rows
        rg_start = current_global_row
        rg_end = current_global_row + rg_num_rows
        
        # 이 row group이 우리가 원하는 범위에 포함되는지 확인
        if rg_end <= start_row:
            # 아직 시작 전
            current_global_row = rg_end
            continue
        
        if rg_start >= start_row + chunk_size:
            # 이미 충분히 읽음
            break
        
        # 이 row group 읽기
        table = parquet_file.read_row_group(rg_idx)
        df_rg = table.to_pandas()
        
        # 필요한 부분만 슬라이싱
        if rg_start < start_row:
            # row group 시작이 우리 범위 이전 → 앞부분 잘라내기
            skip_rows = start_row - rg_start
            df_rg = df_rg.iloc[skip_rows:]
        
        if rg_end > start_row + chunk_size:
            # row group 끝이 우리 범위 이후 → 뒷부분 잘라내기
            take_rows = (start_row + chunk_size) - max(rg_start, start_row)
            df_rg = df_rg.iloc[:take_rows]
        
        dfs.append(df_rg)
        rows_read += len(df_rg)
        current_global_row = rg_end
        
        # 충분히 읽었으면 중단
        if rows_read >= chunk_size:
            break
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def process_parquet_in_chunks_v2(
    input_path: str,
    output_path: str,
    group_size: int = 512,
    chunk_size: int = 20000,
    num_workers: int = 4,
    batch_size: int = 100
) -> None:
    """
    Parquet 파일을 정확히 chunk_size 단위로 처리 (v2)
    """
    if os.path.exists(output_path):
        os.remove(output_path)
        logger.info(f"기존 출력 파일 삭제: {output_path}")
    
    if not HAS_PYARROW:
        logger.error("PyArrow가 필요합니다. pip install pyarrow")
        return
    
    # 파일 정보 조회
    try:
        parquet_file = pq.ParquetFile(input_path, memory_map=False)
    except Exception as e:
        logger.error(f"파일 열기 실패: {e}")
        return
    
    num_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups
    size_mb = os.path.getsize(input_path) / (1024 ** 2)
    
    logger.info(f"\n{'='*70}")
    logger.info("📊 Parquet 파일 정보")
    logger.info(f"{'='*70}")
    logger.info(f"  파일 크기: {size_mb:.2f} MB")
    logger.info(f"  총 행 수: {num_rows:,}")
    logger.info(f"  Row groups: {num_row_groups}")
    logger.info(f"\n⚙️  처리 설정")
    logger.info(f"  청크 크기: {chunk_size:,} rows")
    logger.info(f"  워커 수: {num_workers}")
    logger.info(f"  배치 크기: {batch_size}")
    logger.info(f"  예상 청크 수: {(num_rows + chunk_size - 1) // chunk_size}")
    logger.info(f"{'='*70}\n")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 스키마 준비
    first_table = parquet_file.read_row_group(0)
    first_df = first_table.to_pandas()
    
    if 'logprobs' in first_df.columns:
        first_df = first_df.drop(columns=['logprobs'])
    
    # confidence 컬럼 추가
    column_names = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence',
        'head_confidence',
        'highest_group_confidence',
        'lowest_group_confidence',
        'top_10_percent_confidence',
    ]
    for col_name in column_names:
        first_df[col_name] = 0.0
    
    schema = pa.Schema.from_pandas(first_df)
    
    # Writer 생성
    writer = pq.ParquetWriter(
        output_path,
        schema=schema,
        compression='zstd',
        write_statistics=True
    )
    
    total_valid = 0
    total_invalid = 0
    processed_rows = 0
    
    try:
        # chunk_size 단위로 정확히 나눠서 처리
        chunk_idx = 0
        
        for start_row in range(0, num_rows, chunk_size):
            end_row = min(start_row + chunk_size, num_rows)
            expected_rows = end_row - start_row
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 청크 {chunk_idx + 1} / {(num_rows + chunk_size - 1) // chunk_size}")
            logger.info(f"{'='*70}")
            logger.info(f"  범위: row {start_row:,} ~ {end_row:,} (총 {expected_rows:,}개)")
            
            # 청크 읽기
            df_chunk = read_parquet_chunk_by_rowgroups(
                parquet_file,
                start_row,
                chunk_size
            )
            
            if len(df_chunk) == 0:
                logger.warning(f"  ⚠️  청크 {chunk_idx + 1}에서 데이터를 읽지 못했습니다.")
                chunk_idx += 1
                continue
            
            logger.info(f"  ✓ {len(df_chunk):,}개 row 로드 완료")
            
            # Confidence 계산
            df_chunk, valid, invalid = recalculate_confidence_scores_vectorized(
                df_chunk,
                group_size=group_size,
                chunk_offset=start_row,
                num_workers=num_workers,
                batch_size=batch_size
            )
            
            total_valid += valid
            total_invalid += invalid
            processed_rows += len(df_chunk)
            
            # 테이블로 변환 후 저장 (logprobs는 이미 제거됨)
            table = pa.Table.from_pandas(df_chunk, schema=schema)
            writer.write_table(table)
            
            logger.info(f"  ✅ 저장 완료 (유효: {valid:,}, 무효: {invalid:,})")
            logger.info(f"  📈 전체 진행률: {processed_rows:,}/{num_rows:,} ({processed_rows/num_rows*100:.1f}%)")
            
            del df_chunk, table
            chunk_idx += 1
    
    finally:
        writer.close()
    
    output_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ 처리 완료!")
    logger.info(f"{'='*70}")
    logger.info(f"  총 처리: {processed_rows:,} / {num_rows:,} rows")
    logger.info(f"  유효: {total_valid:,} ({total_valid/processed_rows*100:.1f}%)")
    logger.info(f"  무효: {total_invalid:,} ({total_invalid/processed_rows*100:.1f}%)")
    logger.info(f"  출력 파일: {output_path}")
    logger.info(f"  출력 크기: {output_size_mb:.2f} MB")
    logger.info(f"{'='*70}\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Parquet 파일의 logprobs를 사용하여 confidence 점수를 재계산 (최적화 버전 v2)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 Parquet 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 Parquet 파일 경로"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=512,
        help="토큰 그룹 크기 (기본값: 512)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40000,
        help="청크 크기 (기본값: 20000)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="병렬 처리 워커 수 (기본값: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="각 워커의 배치 크기 (기본값: 100)"
    )
    
    args = parser.parse_args()
    
    # 출력 파일 경로 자동 생성
    if args.output is None:
        input_path = Path(args.input)
        output_filename = input_path.stem + "_recalculated" + input_path.suffix
        args.output = str(input_path.parent / output_filename)
        logger.info(f"출력 파일: {args.output}\n")
    
    # 처리 시작
    process_parquet_in_chunks_v2(
        input_path=args.input,
        output_path=args.output,
        group_size=args.group_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    
    logger.info("🎉 모든 작업 완료!")


if __name__ == "__main__":
    main()