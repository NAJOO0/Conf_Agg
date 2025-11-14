#!/bin/bash
set -euo pipefail

# Stage 1 - 클라이언트 실행 전용 스크립트
# - /workspace/vllm_servers.json을 사용해 shard별 클라이언트를 실행
# - 결과를 /workspace/outputs/data/generated/sample_${SAMPLE_LIMIT}에 저장 및 병합

WORKSPACE="/mnt/data1/projects/Conf_Agg"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=${HF_HOME:-$WORKSPACE/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WORKSPACE/.cache/hf_datasets}
export VLLM_USE_FLASHINFER=1
export PYTHONPATH=$WORKSPACE

# 파라미터 (ENV 우선)
SAMPLE_LIMIT=${SAMPLE_LIMIT:-100}
CONFIG_PATH=${CONFIG_PATH:-"$WORKSPACE/config"}
CONFIG_NAME=${CONFIG_NAME:-"config"}
LOG_DIR=${LOG_DIR:-"$WORKSPACE/outputs/logs/api_sample_${SAMPLE_LIMIT}"}
DATA_DIR=${DATA_DIR:-"$WORKSPACE/output_s"}
GEN_DIR="$DATA_DIR/generated/sample_${SAMPLE_LIMIT}"
SERVERS_JSON=${VLLM_SERVERS_JSON:-"$WORKSPACE/vllm_servers.json"}

# 마스터 nohup 실행/로그/PID
MASTER_LOG="$LOG_DIR/client_master.out"
MASTER_PID_FILE="$LOG_DIR/client_master.pid"
mkdir -p "$LOG_DIR"
if [ -z "${SELF_NOHUPPED:-}" ] && [ "${RUN_NOHUP:-1}" = "1" ]; then
  echo "클라이언트 스크립트를 nohup 백그라운드로 재실행합니다... (로그: $MASTER_LOG)"
  export SELF_NOHUPPED=1
  nohup bash "$0" >> "$MASTER_LOG" 2>&1 &
  echo $! > "$MASTER_PID_FILE"
  echo "마스터 PID: $(cat "$MASTER_PID_FILE")"
  echo "종료: kill -9 $(cat $MASTER_PID_FILE)"
  exit 0
fi

mkdir -p "$LOG_DIR" "$GEN_DIR"
export SERVERS_JSON
export GEN_DIR

echo "=========================================="
echo "Stage 1 클라이언트 실행"
echo "샘플 수: $SAMPLE_LIMIT"
echo "설정: $CONFIG_PATH/$CONFIG_NAME"
echo "서버 정보: $SERVERS_JSON"
echo "로그: $LOG_DIR"
echo "출력: $GEN_DIR"
echo "=========================================="

# 서버 정보 존재 확인
if [ ! -f "$SERVERS_JSON" ]; then
  echo "❌ 서버 정보 파일이 없습니다: $SERVERS_JSON"
  echo "먼저 서버를 실행하세요: bash $WORKSPACE/scripts/run_stage1_server.sh"
  exit 1
fi

# 서버 개수 파악 → total shards로 사용
NUM_SERVERS=$(uv run python - <<'PY'
import json, sys
from pathlib import Path
import os
p = Path(os.environ.get('SERVERS_JSON'))
try:
    d = json.loads(p.read_text())
    print(len(d.get('servers', [])))
except Exception:
    print(0)
PY
)

if [ -z "$NUM_SERVERS" ] || [ "$NUM_SERVERS" -le 0 ]; then
  echo "❌ 서버 정보 파싱 실패 또는 서버 수 0"
  exit 1
fi

echo "감지된 서버 수: $NUM_SERVERS"

export SAMPLE_LIMIT=$SAMPLE_LIMIT

echo "클라이언트 샤드들을 실행합니다..."
for shard_id in $(seq 0 $((NUM_SERVERS-1))); do
  echo "샤드 $shard_id 시작..."
  LOG_DIR="$LOG_DIR" DATA_DIR="$DATA_DIR" SERVERS_JSON="$SERVERS_JSON" \
  uv run python $WORKSPACE/scripts/stage1_generate_api.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --shard-id $shard_id \
    --total-shards $NUM_SERVERS \
    >> "$LOG_DIR/api_shard_${shard_id}.log" 2>&1 &
  echo $! > "$LOG_DIR/api_shard_${shard_id}.pid"
  echo "샤드 $shard_id PID: $(cat \"$LOG_DIR/api_shard_${shard_id}.pid\") (종료: kill -9 $(cat $LOG_DIR/api_shard_${shard_id}.pid))"
done

echo "모든 클라이언트 완료 대기 중..."
wait

echo "결과 병합 중..."
uv run python - <<PY
import pandas as pd
import glob
import os
gen_dir = os.environ.get('GEN_DIR')
if not gen_dir:
    raise SystemExit('GEN_DIR 환경변수가 설정되지 않았습니다.')
shard_files = glob.glob(os.path.join(gen_dir, 'raw_generated_shard_*.parquet'))
print(f'찾은 샤드 파일: {len(shard_files)}개')
if shard_files:
    dfs = [pd.read_parquet(f) for f in shard_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(gen_dir, 'raw_generated_merged.parquet')
    merged_df.to_parquet(output_path, index=False, compression='zstd')
    print(f'병합 완료: {len(merged_df)}개 레코드')
    print(f'저장 위치: {output_path}')
    if len(merged_df) > 0 and 'problem_id' in merged_df.columns:
        print(f'총 응답 수: {len(merged_df)}')
        print(f'고유 문제 수: {merged_df["problem_id"].nunique()}')
        print(f'문제당 평균 응답: {len(merged_df) / merged_df["problem_id"].nunique():.1f}')
    else:
        print('병합 결과가 비어 있거나 problem_id 열이 없습니다. 통계를 건너뜁니다.')
else:
    print('샤드 파일을 찾을 수 없습니다.')
PY

echo "✅ 클라이언트 실행 및 병합 완료"

