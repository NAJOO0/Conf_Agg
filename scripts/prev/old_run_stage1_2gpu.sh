#!/bin/bash
# Stage 1 백그라운드 실행 스크립트 (2개 GPU 최적화)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/mnt/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data1/models/nlp/huggingface_cache
export HF_DATASETS_CACHE=/mnt/data1/datasets/nlp/cache
export VLLM_USE_FLASHINFER=1
export PYTHONPATH=/mnt/data1/projects/Conf_Agg
export SAMPLE_LIMIT=4000
export UV_CACHE_DIR=/mnt/data1/.uv-cache

LOG_DIR="outputs/logs/sample_${SAMPLE_LIMIT}"
mkdir -p "$LOG_DIR"

echo "Stage 1 시작 (2개 GPU): $(date)" >> "$LOG_DIR/stage1_background.log"

pkill -f "stage1_generate.py" || echo 'Python 프로세스 없음'
sleep 3

CONFIG_PATH="./config" 
CONFIG_NAME="config"
TOTAL_SHARDS=2  # GPU 2개로 변경

echo "Starting 2 parallel vLLM worker processes..."

# GPU 0 (Shard 0, 2)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_0.log" 2>&1 &

# GPU 1 (Shard 1, 3)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "1" \
    --shard-id 1 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_1.log" 2>&1 &

echo $! > "$LOG_DIR/stage1_shard_0_pid.txt"
echo $! > "$LOG_DIR/stage1_shard_1_pid.txt"

echo "Stage 1 백그라운드 실행 시작됨 (2개 GPU)"
echo "로그 파일들:"
echo "  Shard 0: $LOG_DIR/stage1_shard_0.log"
echo "  Shard 1: $LOG_DIR/stage1_shard_1.log"
echo ""
echo "모니터링:"
echo "  tail -f $LOG_DIR/stage1_shard_*.log"
echo "  nvidia-smi -l 1"
