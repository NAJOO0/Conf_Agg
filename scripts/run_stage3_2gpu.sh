#!/bin/bash
# GPU 2개로 GRPO 훈련 백그라운드 실행 스크립트
# 사용법: ./run_stage3_2gpu.sh [GPU_IDs]
# 예시: ./run_stage3_2gpu.sh 0,1 (기본값)

# 에러 처리: 중요한 명령만 에러 체크
set -o pipefail  # 파이프라인에서 에러 발생 시 종료

# 작업 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 설정 (인자가 없으면 기본값 0,1 사용)
GPU_IDS=${1:-"0,1"}
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

# 기본 로그 디렉토리 (config.yaml의 log_dir과 다를 수 있음)
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# 로그 파일명 (GPU 개수 포함)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage3_${NUM_GPUS}gpu_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/stage3_${NUM_GPUS}gpu.pid"

echo "=========================================="
echo "🚀 GPU ${NUM_GPUS}개로 GRPO 훈련을 백그라운드에서 시작합니다"
echo "=========================================="
echo "   GPU IDs: ${GPU_IDS}"
echo "   로그 파일: ${LOG_FILE}"
echo "   PID 파일: ${PID_FILE}"
echo ""

# 사전 조건 확인
echo "📋 사전 조건 확인 중..."

# 1. Python 환경 확인
if ! command -v uv &> /dev/null; then
    echo "❌ 오류: 'uv' 명령을 찾을 수 없습니다."
    echo "   uv를 설치하거나 다른 Python 실행기를 사용하세요."
    exit 1
fi

# 2. GPU 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  경고: nvidia-smi를 찾을 수 없습니다. GPU가 제대로 설정되어 있는지 확인하세요."
else
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    echo "   사용 가능한 GPU: ${AVAILABLE_GPUS}개"
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
        echo "⚠️  경고: 요청한 GPU 개수(${NUM_GPUS})가 사용 가능한 GPU(${AVAILABLE_GPUS})보다 많습니다."
    fi
fi

# 3. 데이터 파일 확인
DATA_DIR="${DATA_DIR:-/mnt/data1/datasets/nlp/conf_agg}"
TRAIN_DATA="$DATA_DIR/curated/train_curated.parquet"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 오류: 훈련 데이터 파일을 찾을 수 없습니다: $TRAIN_DATA"
    echo "   먼저 Stage 2를 실행해주세요: python scripts/stage2_curate.py"
    exit 1
fi
echo "   ✅ 훈련 데이터 파일 확인: $TRAIN_DATA"

# 4. 이전 프로세스 정리
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  기존 프로세스 발견 (PID: $OLD_PID). 종료합니다..."
        kill $OLD_PID 2>/dev/null || true
        sleep 2
        # 강제 종료 확인
        if ps -p $OLD_PID > /dev/null 2>&1; then
            echo "⚠️  프로세스가 종료되지 않았습니다. 강제 종료합니다..."
            kill -9 $OLD_PID 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$PID_FILE"
fi

# 5. GPU 메모리 확인 (선택적)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "📊 현재 GPU 메모리 상태:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -n "$NUM_GPUS"
    echo ""
fi

# ===== 환경 변수 설정 =====
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=29500  # ⬅️ 추가!
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1  # ⬅️ InfiniBand 비활성화
export NCCL_DEBUG=WARN  # ⬅️ 디버그 로그
export TORCH_DISTRIBUTED_TIMEOUT=300  # ⬅️ 타임아웃 30초
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_API_KEY=cef6d541e9983fb4a433b2e72a63997ed465e0ac

# ===== 디버그: 환경 변수 확인 =====
echo "🔍 DDP 환경 변수 확인:" | tee -a "$LOG_FILE"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "   MASTER_ADDR=$MASTER_ADDR" | tee -a "$LOG_FILE"
echo "   MASTER_PORT=$MASTER_PORT" | tee -a "$LOG_FILE"
echo "   NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" | tee -a "$LOG_FILE"
echo "   TORCH_DISTRIBUTED_TIMEOUT=$TORCH_DISTRIBUTED_TIMEOUT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 실행 시작 시간 기록
echo "시작 시간: $(date)" >> "$LOG_FILE"
echo "GPU IDs: $GPU_IDS" >> "$LOG_FILE"
echo "명령어: torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/stage3_train.py" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ===== 백그라운드 실행 (torchrun 사용) =====
echo "🚀 훈련 시작 중..."
nohup uv run torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    scripts/stage3_train.py \
    >> "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장 및 확인
MAIN_PID=$!
sleep 1  # 프로세스 시작 대기

# 프로세스가 실제로 실행 중인지 확인
if ! ps -p $MAIN_PID > /dev/null 2>&1; then
    echo "❌ 오류: 프로세스가 시작되지 않았습니다."
    echo "   로그 파일을 확인하세요: $LOG_FILE"
    tail -n 20 "$LOG_FILE"
    exit 1
fi

echo $MAIN_PID > "$PID_FILE"

echo ""
echo "=========================================="
echo "✅ 훈련이 백그라운드에서 시작되었습니다!"
echo "=========================================="
echo "📝 PID: $MAIN_PID"
echo "📋 로그 파일: $LOG_FILE"
echo "📁 PID 파일: $PID_FILE"
echo ""
echo "모니터링 명령어:"
echo "  📊 로그 확인: tail -f $LOG_FILE"
echo "  📊 실시간 로그 (INFO/WARNING/ERROR만): tail -f $LOG_FILE | grep -E '(INFO|WARNING|ERROR|✅|⚠️|❌)'"
echo "  🖥️  GPU 상태: watch -n 1 nvidia-smi"
echo "  📋 프로세스 확인: ps aux | grep stage3_train"
echo "  🔍 Python 프로세스: ps aux | grep python | grep torchrun"
echo ""
echo "중지 방법:"
echo "  🛑 종료: kill \$(cat $PID_FILE)"
echo "  🛑 강제 종료: kill -9 \$(cat $PID_FILE)"
echo "  🛑 모든 stage3 프로세스 종료: pkill -f stage3_train.py"
echo ""
echo "💡 팁: 훈련 상태를 실시간으로 확인하려면"
echo "   tail -f $LOG_FILE | grep -E '(INFO|WARNING|ERROR|✅|⚠️|❌|📊|🎯)'"
echo "=========================================="