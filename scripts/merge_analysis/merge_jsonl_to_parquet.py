#!/usr/bin/env python3
"""
하나 또는 두 개의 JSONL 파일을 합쳐서 Parquet 파일로 저장하는 스크립트
"""
import pandas as pd
import os
import sys

def merge_jsonl_to_parquet(file1_path, file2_path, output_path):
    """
    하나 또는 두 개의 JSONL 파일을 읽어서 합치고 Parquet 파일로 저장
    
    Args:
        file1_path: 첫 번째 JSONL 파일 경로
        file2_path: 두 번째 JSONL 파일 경로 (없으면 None 또는 경로 없음)
        output_path: 출력 Parquet 파일 경로
    """
    dfs = []
    if file1_path and os.path.exists(file1_path):
        print(f"첫 번째 파일 읽는 중: {file1_path}")
        df1 = pd.read_json(file1_path, lines=True)
        dfs.append(df1)
        print(f"첫 번째 파일에서 {len(df1)}개의 레코드 읽음")
    else:
        print(f"경고: 파일이 존재하지 않습니다: {file1_path}")

    if file2_path and os.path.exists(file2_path):
        print(f"두 번째 파일 읽는 중: {file2_path}")
        df2 = pd.read_json(file2_path, lines=True)
        dfs.append(df2)
        print(f"두 번째 파일에서 {len(df2)}개의 레코드 읽음")
    else:
        if file2_path:
            print(f"경고: 파일이 존재하지 않습니다: {file2_path}")

    if not dfs:
        print("오류: 적어도 하나의 JSONL 파일이 필요합니다.")
        sys.exit(1)

    # DataFrame합치기 또는 단일 DataFrame 사용
    if len(dfs) == 1:
        df = dfs[0]
    else:
        print("두 파일을 합치는 중...")
        df = pd.concat(dfs, ignore_index=True)
    print(f"총 {len(df)}개의 레코드")
    print(f"DataFrame 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    
    # Parquet 파일로 저장
    print(f"Parquet 파일로 저장 중: {output_path}")
    df.to_parquet(output_path, index=False, compression="zstd")
    print(f"완료! Parquet 파일 저장됨: {output_path}")
    print(f"파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # 기본 경로 설정
    base_dir = "/mnt/data1/projects/Conf_Agg/output_s/generated/sample_0"
    
    # file1 = os.path.join(base_dir, "raw_generated_shard_0_temp.jsonl")
    file1 = os.path.join(base_dir, "raw_generated_shard_1_temp.jsonl")
    # file2 = os.path.join(base_dir, "raw_generated_shard_1_temp.jsonl")
    output = os.path.join(base_dir, "raw_generated_shard_1.parquet")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # 병합 및 변환 실행
    merge_jsonl_to_parquet(file1, None, output)

