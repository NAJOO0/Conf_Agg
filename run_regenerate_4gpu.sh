#!/bin/bash
# 4개 GPU로 병렬 재생성 스크립트

set -e

# 기본 파일 경로
INPUT_FILE="output_s/generated/sample_3200_offset_0/raw_generated_merged.parquet"
OUTPUT_FILE="output_s/generated/sample_3200_offset_0/raw_generated_merged_fixed.parquet"

# GPU 개수
TOTAL_SHARDS=4

echo "🚀 /think 응답 재생성 시작 (4 GPUs)"
echo "입력 파일: ${INPUT_FILE}"
echo "출력 파일: ${OUTPUT_FILE}"
echo ""

# 로그 디렉토리 생성
mkdir -p logs

# 각 GPU에서 비동기 실행
for SHARD_ID in 0 1 2 3; do
    GPU_ID=$SHARD_ID
    LOG_FILE="logs/regenerate_shard_${SHARD_ID}.log"

    echo "Starting Shard ${SHARD_ID} on GPU ${GPU_ID}..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python scripts/regenerate_think_responses.py \
        --gpu-id ${GPU_ID} \
        --shard-id ${SHARD_ID} \
        --total-shards ${TOTAL_SHARDS} \
        --input-file ${INPUT_FILE} \
        --output-file ${OUTPUT_FILE} \
        > ${LOG_FILE} 2>&1 &

    PIDS[$SHARD_ID]=$!
    echo "  PID: ${PIDS[$SHARD_ID]}, Log: ${LOG_FILE}"
done

echo ""
echo "모든 샤드 시작 완료. 대기 중..."
echo "실시간 로그 확인: tail -f logs/regenerate_shard_*.log"
echo ""

# 모든 프로세스 대기
FAILED=0
for SHARD_ID in 0 1 2 3; do
    PID=${PIDS[$SHARD_ID]}
    echo "Waiting for Shard ${SHARD_ID} (PID: ${PID})..."

    if wait $PID; then
        echo "✅ Shard ${SHARD_ID} 완료"
    else
        echo "❌ Shard ${SHARD_ID} 실패"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "✅ 모든 샤드 완료!"
    echo ""
    echo "이제 샤드 파일들을 병합합니다..."

    # Python으로 샤드 파일 병합
    uv run python -c "
import pandas as pd
import glob
import os

# 샤드 파일 찾기
shard_pattern = '${OUTPUT_FILE}'.replace('.parquet', '_shard_*.parquet')
shard_files = sorted(glob.glob(shard_pattern))

if not shard_files:
    print('❌ 샤드 파일을 찾을 수 없습니다.')
    exit(1)

print(f'병합할 샤드 파일: {len(shard_files)}개')
for f in shard_files:
    print(f'  - {f}')

# 샤드 파일 병합
dfs = []
for f in shard_files:
    df = pd.read_parquet(f)
    print(f'{f}: {len(df)}행')
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f'\\n병합 후 총 {len(combined)}행')

# 중복 제거 (response_id 기준)
original_len = len(combined)
combined = combined.drop_duplicates(subset=['response_id'], keep='first')
if len(combined) < original_len:
    print(f'중복 제거: {original_len - len(combined)}개')

# 저장
combined.to_parquet('${OUTPUT_FILE}', index=False)
print(f'\\n✅ 최종 병합 파일 저장: ${OUTPUT_FILE}')

# 통계 출력
has_think = combined['generated_text'].str.contains('/think', na=False, regex=False)
print(f'\\n📊 최종 통계:')
print(f'  총 행: {len(combined)}')
print(f'  /think 있음: {has_think.sum()}')
print(f'  /think 없음: {(~has_think).sum()}')

problem_counts = combined.groupby('problem_id').size()
print(f'\\n  문제별 응답 수:')
print(f'    평균: {problem_counts.mean():.2f}')
print(f'    최소: {problem_counts.min()}')
print(f'    최대: {problem_counts.max()}')
print(f'    32개인 문제: {(problem_counts == 32).sum()}')
print(f'    32개 미만: {(problem_counts < 32).sum()}')

# 샤드 파일 삭제
print(f'\\n🗑️  샤드 파일 삭제 중...')
for f in shard_files:
    os.remove(f)
    print(f'  삭제: {f}')

print('\\n✅ 완료!')
"

else
    echo "❌ ${FAILED}개 샤드 실패. 로그를 확인하세요."
    exit 1
fi
