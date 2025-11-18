#!/usr/bin/env python3
"""
하나 또는 두 개의 JSONL 파일을 합쳐서 Parquet 파일로 저장하는 스크립트
메모리 효율적인 청크 단위 처리 방식 사용
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

# 청크 크기 설정 (한 번에 읽을 레코드 수)
CHUNK_SIZE = 20000  # 필요에 따라 조정 가능 (메모리에 따라 5000~50000 사이 권장)

def merge_jsonl_to_parquet(file1_path, file2_path, output_path, chunk_size=CHUNK_SIZE):
    """
    하나 또는 두 개의 JSONL 파일을 청크 단위로 읽어서 합치고 Parquet 파일로 저장
    메모리 효율적인 방식으로 대용량 파일 처리
    
    Args:
        file1_path: 첫 번째 JSONL 파일 경로
        file2_path: 두 번째 JSONL 파일 경로 (없으면 None 또는 경로 없음)
        output_path: 출력 Parquet 파일 경로
        chunk_size: 한 번에 읽을 레코드 수 (기본값: 10000)
    """
    total_records = 0
    schema = None
    parquet_writer = None
    first_chunk = True
    
    def process_file(file_path, file_label):
        """파일을 청크 단위로 읽어서 Parquet에 추가"""
        nonlocal total_records, schema, parquet_writer, first_chunk
        
        if not file_path or not os.path.exists(file_path):
            if file_path:
                print(f"경고: 파일이 존재하지 않습니다: {file_path}")
            return
        
        print(f"{file_label} 파일 읽는 중: {file_path}")
        file_records = 0
        
        # 청크 단위로 읽기
        chunk_reader = pd.read_json(file_path, lines=True, chunksize=chunk_size)
        
        for chunk_idx, chunk_df in enumerate(chunk_reader):
            file_records += len(chunk_df)
            total_records += len(chunk_df)
            
            if chunk_idx == 0 and first_chunk:
                # 첫 번째 청크에서 스키마 추출 및 Parquet writer 초기화
                print(f"컬럼: {list(chunk_df.columns)}")
                table = pa.Table.from_pandas(chunk_df)
                schema = table.schema
                parquet_writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression='zstd',
                    use_dictionary=True
                )
                first_chunk = False
                print(f"Parquet 파일 초기화 완료: {output_path}")
            
            # 청크를 Parquet에 추가
            table = pa.Table.from_pandas(chunk_df, schema=schema)
            parquet_writer.write_table(table)
            
            if (chunk_idx + 1) % 10 == 0:
                print(f"  처리된 청크: {chunk_idx + 1}, 누적 레코드: {file_records:,}")
        
        print(f"{file_label} 파일에서 총 {file_records:,}개의 레코드 읽음")
    
    # 첫 번째 파일 처리
    process_file(file1_path, "첫 번째")
    
    # 두 번째 파일 처리
    if file2_path:
        process_file(file2_path, "두 번째")
    
    # Parquet writer 닫기
    if parquet_writer:
        parquet_writer.close()
        print(f"\n총 {total_records:,}개의 레코드 처리 완료")
        print(f"완료! Parquet 파일 저장됨: {output_path}")
        print(f"파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    else:
        print("오류: 적어도 하나의 JSONL 파일이 필요합니다.")
        sys.exit(1)

if __name__ == "__main__":
    # 기본 경로 설정
    base_dir = "/mnt/data1/projects/Conf_Agg/output_s/generated/sample_0"
    
    file1 = os.path.join(base_dir, "raw_generated_shard_0_temp.jsonl")
    # file1 = os.path.join(base_dir, "raw_generated_shard_1_temp.jsonl")
    # file2 = os.path.join(base_dir, "raw_generated_shard_1_temp.jsonl")
    output = os.path.join(base_dir, "raw_generated_shard_0.parquet")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # 병합 및 변환 실행
    merge_jsonl_to_parquet(file1, None, output)

