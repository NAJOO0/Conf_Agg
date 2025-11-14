#!/bin/bash

# Stage 1 백그라운드 실행 스크립트 (단순 데이터 병렬 최적화)
# 4개 GPU에서 독립적인 프로세스로 데이터 샤딩 처리

# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/data1/models/nlp/huggingface_cache
export HF_DATASETS_CACHE=/data2/datasets/nlp/cache
export VLLM_USE_FLASHINFER=1
export PYTHONPATH=/workspace
export SAMPLE_LIMIT=400

# 로그 디렉토리 생성
LOG_DIR="/workspace/outputs/logs/sample_${SAMPLE_LIMIT}"
mkdir -p "$LOG_DIR"

# 실행 시간 기록
echo "Stage 1 시작 (단순 데이터 병렬 처리): $(date)" >> "$LOG_DIR/stage1_background.log"

# GPU 메모리 정리 (이전 프로세스 정리)
echo "GPU 메모리 정리 중..."
pkill -f python || echo 'Python 프로세스 없음'
sleep 3

# Hydra 설정 파일의 디렉토리 경로와 이름
CONFIG_PATH="./config" 
CONFIG_NAME="config"
TOTAL_SHARDS=4

echo "Starting 4 parallel vLLM worker processes..."

# GPU 0 (Shard 0)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_0.log" 2>&1 &

# GPU 1 (Shard 1)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "1" \
    --shard-id 1 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_1.log" 2>&1 &

# GPU 2 (Shard 2)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "2" \
    --shard-id 2 \
    --total-shards $TOTAL_SHARDS \
 > "$LOG_DIR/stage1_shard_2.log" 2>&1 &

# GPU 3 (Shard 3)
nohup uv run python scripts/stage1_generate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "3" \
    --shard-id 3 \
    --total-shards $TOTAL_SHARDS \
 > "$LOG_DIR/stage1_shard_3.log" 2>&1 &

# 프로세스 ID들을 저장
echo $! > "$LOG_DIR/stage1_shard_0_pid.txt"
echo $! > "$LOG_DIR/stage1_shard_1_pid.txt"
echo $! > "$LOG_DIR/stage1_shard_2_pid.txt"
echo $! > "$LOG_DIR/stage1_shard_3_pid.txt"

echo "Stage 1 백그라운드 실행 시작됨 (단순 데이터 병렬 처리)"
echo "로그 파일들:"
echo "  Shard 0: $LOG_DIR/stage1_shard_0.log"
echo "  Shard 1: $LOG_DIR/stage1_shard_1.log"
echo "  Shard 2: $LOG_DIR/stage1_shard_2.log"
echo "  Shard 3: $LOG_DIR/stage1_shard_3.log"
echo ""
echo "모니터링 명령어:"
echo "  모든 로그 확인: tail -f $LOG_DIR/stage1_shard_*.log"
echo "  특정 샤드 로그: tail -f $LOG_DIR/stage1_shard_0.log"
echo "  GPU 상태: nvidia-smi"
echo "  프로세스 확인: ps aux | grep stage1_generate"
echo ""
echo "예상 성능: Ray Serve 대비 더 안정적이고 빠른 처리"
echo ""
echo "모든 워커가 완료되면 다음 명령어로 결과 병합:"
echo "  python scripts/merge_shards.py --data-dir /workspace/outputs/data"