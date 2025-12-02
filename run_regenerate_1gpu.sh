#!/bin/bash
# 단일 GPU 재생성 스크립트

set -e

# 기본 파일 경로
INPUT_FILE="output_s/generated/sample_3200_offset_0/raw_generated_merged.parquet"
OUTPUT_FILE="output_s/generated/sample_3200_offset_0/raw_generated_merged_fixed.parquet"

# GPU ID (기본값: 0)
GPU_ID=${1:-0}

echo "🚀 /think 응답 재생성 시작 (Single GPU: ${GPU_ID})"
echo "입력 파일: ${INPUT_FILE}"
echo "출력 파일: ${OUTPUT_FILE}"
echo ""

# 로그 디렉토리 생성
mkdir -p logs

# 실행
CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python scripts/regenerate_think_responses.py \
    --gpu-id ${GPU_ID} \
    --shard-id 0 \
    --total-shards 1 \
    --input-file ${INPUT_FILE} \
    --output-file ${OUTPUT_FILE}

echo ""
echo "✅ 완료!"
echo ""

# 통계 출력
uv run python -c "
import pandas as pd

df = pd.read_parquet('${OUTPUT_FILE}')
has_think = df['generated_text'].str.contains('/think', na=False, regex=False)

print('📊 최종 통계:')
print(f'  총 행: {len(df)}')
print(f'  /think 있음: {has_think.sum()}')
print(f'  /think 없음: {(~has_think).sum()}')

problem_counts = df.groupby('problem_id').size()
print(f'\n  문제별 응답 수:')
print(f'    평균: {problem_counts.mean():.2f}')
print(f'    최소: {problem_counts.min()}')
print(f'    최대: {problem_counts.max()}')
print(f'    32개인 문제: {(problem_counts == 32).sum()}')
print(f'    32개 미만: {(problem_counts < 32).sum()}')
"
