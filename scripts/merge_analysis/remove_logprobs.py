#!/usr/bin/env python3
"""
Parquet 파일에서 logprobs 컬럼을 제거하는 스크립트
메모리 효율적인 청크 단위 처리 방식 사용
"""
import os
import pandas as pd
import argparse
import logging
import sys
import shutil

# 프로젝트 루트를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyArrow 가용성 확인
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False
    logger.error("PyArrow가 필요합니다. pip install pyarrow로 설치해주세요.")
    sys.exit(1)

# 청크 크기 설정 (row group 단위로 읽기)
CHUNK_SIZE = 50000  # row group당 읽을 행 수 (필요에 따라 조정 가능)


def convert_string_to_large_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 문자열 컬럼을 large_string으로 변환하여 오프셋 오버플로우 방지
    """
    if not HAS_PYARROW:
        return df
    
    string_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.ArrowDtype) and dtype.pyarrow_dtype == pa.string():
            string_cols.append(col)
    
    if not string_cols:
        return df
    
    for col in string_cols:
        try:
            pa_array = df[col].array._pa_array
            large_string_array = pa.compute.cast(pa_array, pa.large_string())
            if hasattr(pd.arrays, 'ArrowExtensionArray'):
                df[col] = pd.arrays.ArrowExtensionArray(large_string_array)
            else:
                df[col] = pd.array(large_string_array, dtype=pd.ArrowDtype(large_string_array.type))
        except Exception as e:
            logger.warning(f"컬럼 {col}을 large_string으로 변환 실패: {e}, 일반 pandas dtype으로 변환합니다.")
            try:
                df[col] = df[col].astype('string')
            except Exception:
                df[col] = df[col].astype('object')
    
    return df


def remove_logprobs(input_file: str, output_file: str = None, chunk_size: int = CHUNK_SIZE) -> None:
    """
    Parquet 파일에서 logprobs 컬럼을 청크 단위로 제거하고 저장
    메모리 효율적인 방식으로 대용량 파일 처리
    
    Args:
        input_file: 입력 Parquet 파일 경로
        output_file: 출력 Parquet 파일 경로 (None이면 입력 파일과 같은 디렉토리에 저장)
        chunk_size: 한 번에 읽을 행 수 (기본값: 50000)
    """
    # 출력 파일 경로 설정
    if output_file is None:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        input_basename = os.path.basename(input_file)
        name, ext = os.path.splitext(input_basename)
        output_file = os.path.join(input_dir, f"{name}_no_logprobs{ext}")
    
    if not os.path.exists(input_file):
        logger.error(f"파일을 찾을 수 없습니다: {input_file}")
        return
    
    if not input_file.endswith('.parquet'):
        logger.error(f"Parquet 파일이 아닙니다: {input_file}")
        return
    
    logger.info("\n" + "="*50)
    logger.info("=== logprobs 컬럼 제거 (청크 단위 처리) ===")
    logger.info("="*50)
    logger.info(f"입력 파일: {input_file}")
    logger.info(f"출력 파일: {output_file}")
    logger.info(f"청크 크기: {chunk_size:,} 행")
    
    try:
        # Parquet 파일 열기
        parquet_file = pq.ParquetFile(input_file)
        num_row_groups = parquet_file.num_row_groups
        
        # 첫 번째 row group을 읽어서 스키마 확인
        first_table = parquet_file.read_row_group(0)
        first_df = first_table.to_pandas()
        
        # logprobs 컬럼 존재 여부 확인
        has_logprobs = 'logprobs' in first_df.columns
        if not has_logprobs:
            logger.warning("logprobs 컬럼이 없습니다. 파일을 그대로 저장합니다.")
            # logprobs가 없으면 파일을 그대로 복사
            shutil.copy2(input_file, output_file)
            logger.info(f"파일 복사 완료: {output_file}")
            return
        
        logger.info(f"logprobs 컬럼 발견. 제거 시작...")
        logger.info(f"총 {num_row_groups}개의 row group 처리 예정")
        
        # logprobs를 제외한 스키마 생성
        schema = first_table.schema
        columns_to_keep = [col for col in schema.names if col != 'logprobs']
        new_schema = pa.schema([schema.field(col) for col in columns_to_keep])
        
        # Parquet writer 초기화
        parquet_writer = pq.ParquetWriter(
            output_file,
            new_schema,
            compression='zstd',
            use_dictionary=True
        )
        
        total_rows = 0
        
        # 각 row group을 청크 단위로 처리
        for rg_idx in range(num_row_groups):
            # Row group 읽기
            table = parquet_file.read_row_group(rg_idx, columns=columns_to_keep)
            
            # Parquet에 쓰기
            parquet_writer.write_table(table)
            
            num_rows = len(table)
            total_rows += num_rows
            
            if (rg_idx + 1) % 10 == 0 or (rg_idx + 1) == num_row_groups:
                logger.info(f"처리된 row group: {rg_idx + 1}/{num_row_groups}, 누적 행: {total_rows:,}")
        
        # Writer 닫기
        parquet_writer.close()
        
        # 파일 크기 비교
        try:
            input_size_mb = os.path.getsize(input_file) / (1024 ** 2)
            output_size_mb = os.path.getsize(output_file) / (1024 ** 2)
            size_reduction = input_size_mb - output_size_mb
            reduction_percent = (size_reduction / input_size_mb * 100) if input_size_mb > 0 else 0
            
            logger.info(f"\n저장 완료!")
            logger.info(f"입력 파일 크기: {input_size_mb:.2f} MB")
            logger.info(f"출력 파일 크기: {output_size_mb:.2f} MB")
            logger.info(f"크기 감소: {size_reduction:.2f} MB ({reduction_percent:.1f}%)")
            logger.info(f"총 처리된 행: {total_rows:,}개")
            logger.info(f"출력 컬럼 수: {len(columns_to_keep)}개")
        except Exception as e:
            logger.warning(f"파일 크기 비교 실패: {e}")
            logger.info(f"저장 완료: {output_file}")
        
    except Exception as e:
        logger.error(f"처리 실패: {input_file}, 오류: {e}", exc_info=True)
        return
    
    logger.info("\n" + "="*50)
    logger.info("✅ logprobs 제거 완료!")
    logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description="Parquet 파일에서 logprobs 컬럼 제거")
    parser.add_argument("input_file", type=str, help="입력 Parquet 파일 경로")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="출력 Parquet 파일 경로 (지정하지 않으면 입력 파일과 같은 디렉토리에 _no_logprobs 접미사로 저장)")
    
    args = parser.parse_args()
    
    remove_logprobs(args.input_file, args.output)


if __name__ == "__main__":
    main()

