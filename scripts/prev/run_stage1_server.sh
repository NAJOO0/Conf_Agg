#!/bin/bash
set -euo pipefail

# """
# Stage 1 - 서버 준비 전용 스크립트
#  - Hydra 설정을 JSON으로 변환하여 /workspace/config.json 저장
#  - vLLM 서버를 기동하고 /workspace/vllm_servers.json 생성까지 대기
#  - 서버 프로세스를 포그라운드로 유지 (종료: Ctrl+C)
# """

# 절대 경로 및 환경 변수 설정
WORKSPACE="/mnt/data1/projects/Conf_Agg"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=${HF_HOME:-$WORKSPACE/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WORKSPACE/.cache/hf_datasets}
export VLLM_USE_FLASHINFER=1
export PYTHONPATH=$WORKSPACE

# 파라미터 (ENV 우선)
NUM_GPUS=${NUM_GPUS:-2}
BASE_PORT=${BASE_PORT:-8000}
CONFIG_PATH=${CONFIG_PATH:-"$WORKSPACE/config"}
CONFIG_NAME=${CONFIG_NAME:-"config"}

echo "=========================================="
echo "Stage 1 서버 준비 시작"
echo "GPU 수: $NUM_GPUS"
echo "기준 포트: $BASE_PORT"
echo "설정 디렉토리: $CONFIG_PATH, 설정 이름: $CONFIG_NAME"
echo "=========================================="

# 1) Hydra 설정 → JSON 변환
echo "설정 파일 변환 중..."
uv run python -c "
import hydra
from omegaconf import OmegaConf
import json
from pathlib import Path

config_dir = Path('$CONFIG_PATH').resolve()
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
cfg = hydra.compose(config_name='$CONFIG_NAME')

config_dict = OmegaConf.to_container(cfg, resolve=True)
out_path = Path('$WORKSPACE') / 'config.json'
with open(out_path, 'w') as f:
    json.dump(config_dict, f, indent=2)
print(f'설정 파일 변환 완료: {out_path}')
"

# 2) vLLM 서버 기동 (nohup 백그라운드)
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"$WORKSPACE/outputs/logs/vllm"}
mkdir -p "$SERVER_LOG_DIR"
SUPERVISOR_LOG="$SERVER_LOG_DIR/supervisor.out"

echo "vLLM API 서버를 nohup 백그라운드로 시작합니다..."
nohup uv run python $WORKSPACE/scripts/vllm_launcher.py \
    --num-gpus $NUM_GPUS \
    --base-port $BASE_PORT \
    --config-path $WORKSPACE/config.json \
    >> "$SUPERVISOR_LOG" 2>&1 &

SERVER_LAUNCHER_PID=$!
echo $SERVER_LAUNCHER_PID > "$SERVER_LOG_DIR/server_launcher.pid"
echo "서버 런처 PID: $SERVER_LAUNCHER_PID"
echo "로그: $SUPERVISOR_LOG"
echo "종료: kill -9 $(cat $SERVER_LOG_DIR/server_launcher.pid)"

