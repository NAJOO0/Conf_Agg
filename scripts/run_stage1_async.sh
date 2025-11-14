#!/bin/bash
# Stage 1 백그라운드 실행 스크립트 (2개 GPU 최적화)
# 개선판: 모든 성능/안정성 환경변수 포함

# =============================================================================
# 기본 환경 설정
# =============================================================================
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/mnt/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data1/models/nlp/huggingface_cache
export HF_DATASETS_CACHE=/mnt/data1/datasets/nlp/cache
export VLLM_USE_FLASHINFER=1
export PYTHONPATH=/mnt/data1/projects/Conf_Agg
export UV_CACHE_DIR=/mnt/data1/.uv-cache

# =============================================================================
# Stage 1 전용 설정
# =============================================================================

# 샘플링 제한 (0 = 전체 데이터 사용)
# - 0: 전체 데이터셋 사용
# - 100: 처음 100개 문제만 처리 (테스트용)
# - 1000: 처음 1000개 문제만 처리
export SAMPLE_LIMIT=0

# 백프레셔 제어 (기본: config의 max_num_seqs * 1.5)
# - vLLM 엔진에 동시 등록 가능한 최대 요청 수
# - 너무 높으면: 메모리 부족 (OOM) 위험
# - 너무 낮으면: GPU 활용률 저하
# - 권장값: max_num_seqs의 1.5~2배
# - config의 max_num_seqs=112 기준 → 168 (1.5배)
# - V1 API에서는 각 completion이 별도 카운트되므로 낮게 설정
export MAX_INFLIGHT=168

# 진행 상황 로깅 주기 (기본: 10)
# - N개 completion 처리마다 진행률 출력
# - V1 API: 각 문제당 n개 completion 생성 (예: n=8, 5문제 → 40 completions)
# - 로그 파일 크기와 가시성의 균형
# - 너무 낮으면: 로그 파일 비대화
# - 너무 높으면: 진행 상황 파악 어려움
# - 권장값: 10~50
export SNAPSHOT_EVERY=10

# JSONL flush 주기 (기본: 100)
# - N개 행 쓰기마다 디스크에 강제 동기화 (fsync)
# - 너무 낮으면: I/O 병목 (성능 저하)
# - 너무 높으면: 비정상 종료 시 데이터 손실 위험
# - 권장값: 100~500
export FLUSH_EVERY=100

# 재시작 기능 활성화 (기본: true)
# - true: 기존 결과 파일 로드, 중복 응답 자동 스킵
# - false: 기존 결과 무시, 처음부터 재생성
# - 비정상 종료 후 재실행 시 반드시 true 유지
export RESUME=true

# vLLM 로깅 레벨 (기본: INFO)
# - ERROR: 에러만 출력
# - WARNING: 경고 이상 출력
# - INFO: 일반 정보 출력 (권장)
# - DEBUG: 상세 디버그 정보 (성능 저하 가능)
export VLLM_LOGGING_LEVEL=INFO

# =============================================================================
# 실행 설정
# =============================================================================
# 타임스탬프 생성 (로그 디렉토리 구분용)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="outputs/logs/sample_${SAMPLE_LIMIT}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

CONFIG_PATH="./config" 
CONFIG_NAME="config"
TOTAL_SHARDS=2  # GPU 2개 사용

# =============================================================================
# 시작 로그
# =============================================================================
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "Stage 1 Data Generation (Async Improved Version)" | tee -a "$LOG_DIR/stage1_background.log"
echo "시작 시각: $(date)" | tee -a "$LOG_DIR/stage1_background.log"
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

# 환경변수 출력
echo "환경 설정:" | tee -a "$LOG_DIR/stage1_background.log"
echo "  SAMPLE_LIMIT: $SAMPLE_LIMIT" | tee -a "$LOG_DIR/stage1_background.log"
echo "  MAX_INFLIGHT: $MAX_INFLIGHT" | tee -a "$LOG_DIR/stage1_background.log"
echo "  SNAPSHOT_EVERY: $SNAPSHOT_EVERY" | tee -a "$LOG_DIR/stage1_background.log"
echo "  FLUSH_EVERY: $FLUSH_EVERY" | tee -a "$LOG_DIR/stage1_background.log"
echo "  RESUME: $RESUME" | tee -a "$LOG_DIR/stage1_background.log"
echo "  TOTAL_SHARDS: $TOTAL_SHARDS" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

# =============================================================================
# 기존 프로세스 정리
# =============================================================================
echo "기존 프로세스 확인 및 정리..." | tee -a "$LOG_DIR/stage1_background.log"
pkill -f "stage1_generate_async.py" && echo "기존 프로세스 종료됨" || echo "실행 중인 프로세스 없음"
sleep 3

# =============================================================================
# GPU 워커 프로세스 시작
# =============================================================================
# 각 GPU에 독립적인 torch compile 캐시 디렉토리 할당
TORCH_COMPILE_GPU0="/tmp/torch_compile_gpu0_$$"
TORCH_COMPILE_GPU1="/tmp/torch_compile_gpu1_$$"

# 디렉토리 생성 및 권한 설정
mkdir -p "$TORCH_COMPILE_GPU0" "$TORCH_COMPILE_GPU1"
chmod 755 "$TORCH_COMPILE_GPU0" "$TORCH_COMPILE_GPU1"

echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "Starting $TOTAL_SHARDS parallel vLLM worker processes..." | tee -a "$LOG_DIR/stage1_background.log"
echo "Torch compile directories:" | tee -a "$LOG_DIR/stage1_background.log"
echo "  GPU 0: $TORCH_COMPILE_GPU0" | tee -a "$LOG_DIR/stage1_background.log"
echo "  GPU 1: $TORCH_COMPILE_GPU1" | tee -a "$LOG_DIR/stage1_background.log"
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

# GPU 0 (Shard 0)
echo "[GPU 0] Shard 0 시작..." | tee -a "$LOG_DIR/stage1_background.log"
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
TORCH_COMPILE_DIR="$TORCH_COMPILE_GPU0" \
nohup uv run python scripts/stage1_generate_async.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_0.log" 2>&1 &
PID_0=$!
echo $PID_0 > "$LOG_DIR/stage1_shard_0_pid.txt"
echo "  PID: $PID_0" | tee -a "$LOG_DIR/stage1_background.log"

sleep 5

# GPU 1 (Shard 1)
echo "[GPU 1] Shard 1 시작..." | tee -a "$LOG_DIR/stage1_background.log"
CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
TORCH_COMPILE_DIR="$TORCH_COMPILE_GPU1" \
nohup uv run python scripts/stage1_generate_async.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    --gpu-id "1" \
    --shard-id 1 \
    --total-shards $TOTAL_SHARDS \
> "$LOG_DIR/stage1_shard_1.log" 2>&1 &
PID_1=$!
echo $PID_1 > "$LOG_DIR/stage1_shard_1_pid.txt"
echo "  PID: $PID_1" | tee -a "$LOG_DIR/stage1_background.log"

echo "" | tee -a "$LOG_DIR/stage1_background.log"

# =============================================================================
# 실행 완료 안내
# =============================================================================
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "✅ Stage 1 백그라운드 실행 시작 완료" | tee -a "$LOG_DIR/stage1_background.log"
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "📁 로그 파일:" | tee -a "$LOG_DIR/stage1_background.log"
echo "  Shard 0: $LOG_DIR/stage1_shard_0.log" | tee -a "$LOG_DIR/stage1_background.log"
echo "  Shard 1: $LOG_DIR/stage1_shard_1.log" | tee -a "$LOG_DIR/stage1_background.log"
echo "  Background: $LOG_DIR/stage1_background.log" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "🔍 모니터링 명령어:" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 전체 로그 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  tail -f $LOG_DIR/stage1_shard_*.log" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # Shard 0만 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  tail -f $LOG_DIR/stage1_shard_0.log" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 진행률만 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  tail -f $LOG_DIR/stage1_shard_*.log | grep '진행:'" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 메모리 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  tail -f $LOG_DIR/stage1_shard_*.log | grep '메모리:'" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # GPU 상태 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  nvidia-smi -l 1" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "⚙️  프로세스 관리:" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 프로세스 확인" | tee -a "$LOG_DIR/stage1_background.log"
echo "  ps aux | grep stage1_generate_async.py" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 특정 샤드 종료" | tee -a "$LOG_DIR/stage1_background.log"
echo "  kill \$(cat $LOG_DIR/stage1_shard_0_pid.txt)" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"
echo "  # 모든 샤드 종료" | tee -a "$LOG_DIR/stage1_background.log"
echo "  pkill -f stage1_generate_async.py" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "📊 출력 파일 위치:" | tee -a "$LOG_DIR/stage1_background.log"
if [ "$SAMPLE_LIMIT" -eq 0 ]; then
    OUTPUT_PATH="/mnt/data1/datasets/nlp/conf_agg/generated"
else
    OUTPUT_PATH="/mnt/data1/datasets/nlp/conf_agg/generated/sample_${SAMPLE_LIMIT}"
fi
echo "  $OUTPUT_PATH/" | tee -a "$LOG_DIR/stage1_background.log"
echo "  ├── raw_generated_shard_0.parquet" | tee -a "$LOG_DIR/stage1_background.log"
echo "  ├── raw_generated_shard_0.jsonl" | tee -a "$LOG_DIR/stage1_background.log"
echo "  ├── raw_generated_shard_1.parquet" | tee -a "$LOG_DIR/stage1_background.log"
echo "  └── raw_generated_shard_1.jsonl" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "🧹 Cleanup (작업 완료 후 수동 실행):" | tee -a "$LOG_DIR/stage1_background.log"
echo "  rm -rf $TORCH_COMPILE_GPU0 $TORCH_COMPILE_GPU1" | tee -a "$LOG_DIR/stage1_background.log"
echo "" | tee -a "$LOG_DIR/stage1_background.log"

echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"
echo "실행이 시작되었습니다. 로그를 확인하세요." | tee -a "$LOG_DIR/stage1_background.log"
echo "=============================================================================" | tee -a "$LOG_DIR/stage1_background.log"