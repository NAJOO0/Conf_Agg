#!/bin/bash
# CUDA 환경변수를 현재 셸에 즉시 적용하는 스크립트

# CUDA 경로 확인 및 환경변수 설정
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
    echo "✅ CUDA 환경변수 설정 완료:"
    echo "   CUDA_HOME=$CUDA_HOME"
    echo "   nvcc: $(which nvcc 2>/dev/null || echo 'not found')"
    
    # nvcc 버전 확인
    if command -v nvcc &> /dev/null; then
        echo ""
        echo "nvcc 버전:"
        nvcc --version
    fi
else
    echo "❌ CUDA Toolkit을 찾을 수 없습니다"
    echo "   다음 경로를 확인하세요:"
    echo "   - /mnt/data1/cuda-12.8"
    echo "   - /mnt/data1/cuda"
    echo "   - /usr/local/cuda"
fi
