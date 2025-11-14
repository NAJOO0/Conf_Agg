#!/bin/bash
# 프로젝트 실행 스크립트 (v4 - Unsloth Standby 적용)

# 1. .bashrc 로드 (CUDA, HF_HOME, PYTHONNOUSERSITE 등)
source ~/.bashrc 2>/dev/null || true

# CUDA 경로 재확인 및 환경변수 적용 (Persistent Storage 우선)
if [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# 2. 가상 환경 활성화
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ .venv (Python $(python -V)) 활성화됨"
else
    echo "❌ 오류: .venv/bin/activate 파일을 찾을 수 없습니다."
    echo "   /mnt/data1/projects/Conf_Agg 에서 ./restart_setup.sh 를 실행했는지 확인하세요."
    exit 1
fi

# 3. (변경) Unsloth Standby 모드 설정 (vLLM colocate 메모리 최적화)
# Unsloth import 이전에 설정해야 합니다.
export UNSLOTH_VLLM_STANDBY=1
echo "✅ UNSLOTH_VLLM_STANDBY=1 (메모리 최적화) 설정됨"

# 4. 프로젝트 코드 Python Path 설정
export PYTHONPATH=/root/projects/Conf_Agg

# 5. 메인 스크립트 실행
echo "🚀 프로젝트 실행: ./scripts/run_stage1_2gpu.sh"
./scripts/run_stage1_2gpu.sh
