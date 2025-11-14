#!/usr/bin/env python3
"""
Parquet 파일에서 logprobs 컬럼을 제거하는 스크립트
"""
import os
import pandas as pd
import argparse
import logging
import sys

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


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """
    Parquet 파일을 로드
    
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
            table = pq.read_table(file_path, memory_map=True)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            df = convert_string_to_large_string(df)
        else:
            df = pd.read_parquet(file_path, engine="pyarrow") if 'pyarrow' in sys.modules else pd.read_parquet(file_path)
            if HAS_PYARROW:
                df = convert_string_to_large_string(df)
        
        logger.info(f"로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
        try:
            size_mb = os.path.getsize(file_path) / (1024 ** 2)
            logger.info(f"로드한 Parquet 파일 크기: {size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"파일 크기 확인 실패: {e}")
        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
        return None


def remove_logprobs(input_file: str, output_file: str = None) -> None:
    """
    Parquet 파일에서 logprobs 컬럼을 제거하고 저장
    
    Args:
        input_file: 입력 Parquet 파일 경로
        output_file: 출력 Parquet 파일 경로 (None이면 입력 파일과 같은 디렉토리에 저장)
    """
    # 출력 파일 경로 설정
    if output_file is None:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        input_basename = os.path.basename(input_file)
        name, ext = os.path.splitext(input_basename)
        output_file = os.path.join(input_dir, f"{name}_no_logprobs{ext}")
    
    logger.info("\n" + "="*50)
    logger.info("=== logprobs 컬럼 제거 ===")
    logger.info("="*50)
    logger.info(f"입력 파일: {input_file}")
    logger.info(f"출력 파일: {output_file}")
    
    # 파일 로드
    df = load_parquet_file(input_file)
    if df is None:
        logger.error("파일 로드 실패")
        return
    
    # logprobs 컬럼 확인 및 제거
    if 'logprobs' not in df.columns:
        logger.warning("logprobs 컬럼이 없습니다. 파일을 그대로 저장합니다.")
    else:
        logger.info("logprobs 컬럼 제거 중...")
        df = df.drop(columns=['logprobs'])
        logger.info("logprobs 컬럼 제거 완료")
    
    # 저장
    try:
        logger.info(f"파일 저장 중: {output_file}")
        df.to_parquet(output_file, index=False, compression="zstd")
        
        # 파일 크기 비교
        try:
            input_size_mb = os.path.getsize(input_file) / (1024 ** 2)
            output_size_mb = os.path.getsize(output_file) / (1024 ** 2)
            size_reduction = input_size_mb - output_size_mb
            reduction_percent = (size_reduction / input_size_mb * 100) if input_size_mb > 0 else 0
            
            logger.info(f"저장 완료!")
            logger.info(f"입력 파일 크기: {input_size_mb:.2f} MB")
            logger.info(f"출력 파일 크기: {output_size_mb:.2f} MB")
            logger.info(f"크기 감소: {size_reduction:.2f} MB ({reduction_percent:.1f}%)")
            logger.info(f"출력 파일: {len(df)}개 행, {len(df.columns)}개 컬럼")
        except Exception as e:
            logger.warning(f"파일 크기 비교 실패: {e}")
            logger.info(f"저장 완료: {output_file}")
    except Exception as e:
        logger.error(f"파일 저장 실패: {output_file}, 오류: {e}")
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

