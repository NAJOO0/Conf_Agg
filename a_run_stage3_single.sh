#!/bin/bash
# DDP 없이 단일 프로세스로 GPU 2개를 사용하여 vLLM Colocate 모드로 실행
# 사용법: ./run_stage3_single_colocate.sh [GPU_IDs]
# 예시: ./run_stage3_single_colocate.sh 0,1 (기본값)
export WANDB_API_KEY=cef6d541e9983fb4a433b2e72a63997ed465e0ac
# 에러 처리
set -o pipefail

# 작업 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 설정 (인자가 없으면 기본값 0,1 사용)
GPU_IDS=${1:-"1"}
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

# 기본 로그 디렉토리
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

# 로그 파일명
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage3_single_colocate_${NUM_GPUS}gpu_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/stage3_single_colocate_${NUM_GPUS}gpu.pid"

echo "=========================================="
echo "🚀 단일 프로세스 vLLM Colocate 모드로 훈련 시작"
echo "=========================================="
echo "   GPU IDs: ${GPU_IDS}"
echo "   모드: 단일 프로세스 (DDP 없음)"
echo "   vLLM: Colocate (GPU ${NUM_GPUS}개 활용)"
echo "   로그 파일: ${LOG_FILE}"
echo ""

# 사전 조건 확인
echo "📋 사전 조건 확인 중..."

# 1. Python 환경 확인
if ! command -v uv &> /dev/null; then
    echo "❌ 오류: 'uv' 명령을 찾을 수 없습니다."
    exit 1
fi

# 2. GPU 확인
if command -v nvidia-smi &> /dev/null; then
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
    exit 1
fi
echo "   ✅ 훈련 데이터 파일 확인"

# 4. 이전 프로세스 정리
# if [ -f "$PID_FILE" ]; then
#     OLD_PID=$(cat "$PID_FILE")
#     if ps -p $OLD_PID > /dev/null 2>&1; then
#         echo "⚠️  기존 프로세스 발견 (PID: $OLD_PID). 종료합니다..."
#         kill $OLD_PID 2>/dev/null || true
#         sleep 2
#         if ps -p $OLD_PID > /dev/null 2>&1; then
#             kill -9 $OLD_PID 2>/dev/null || true
#             sleep 1
#         fi
#     fi
#     rm -f "$PID_FILE"
# fi

# 5. GPU 메모리 확인
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "📊 현재 GPU 메모리 상태:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -n "$NUM_GPUS"
    echo ""
fi

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# DDP 환경 변수 제거 (단일 프로세스 모드)
unset RANK
unset LOCAL_RANK
unset WORLD_SIZE

# 실행 시작 시간 기록
echo "시작 시간: $(date)" >> "$LOG_FILE"
echo "GPU IDs: $GPU_IDS" >> "$LOG_FILE"
echo "모드: 단일 프로세스 (DDP 없음)" >> "$LOG_FILE"
echo "vLLM 모드: colocate" >> "$LOG_FILE"
echo "명령어: uv run python scripts/stage3_train_2.py" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 백그라운드 실행 (torchrun 없이)
echo "🚀 훈련 시작 중..."
nohup uv run python scripts/stage3_train.py \
    >> "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장 및 확인
MAIN_PID=$!
sleep 1

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
echo "💡 실행 모드:"
echo "   - DDP: 비활성화 (단일 프로세스)"
echo "   - GPU: ${NUM_GPUS}개 (${GPU_IDS})"
echo "   - vLLM: Colocate 모드 (모든 GPU 활용)"
echo ""
echo "모니터링 명령어:"
echo "  📊 로그 확인: tail -f $LOG_FILE"
echo "  📊 실시간 로그: tail -f $LOG_FILE | grep -E '(INFO|WARNING|ERROR|✅|⚠️|❌|📊|🎯)'"
echo "  🖥️  GPU 상태: watch -n 1 nvidia-smi"
echo "  📋 프로세스 확인: ps aux | grep stage3_train"
echo ""
echo "중지 방법:"
echo "  🛑 종료: kill \$(cat $PID_FILE)"
echo "  🛑 강제 종료: kill -9 \$(cat $PID_FILE)"
echo "=========================================="

