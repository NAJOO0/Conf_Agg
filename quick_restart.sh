#!/bin/bash
# 빠른 복구 스크립트 (이미 설치된 패키지 스킵)

set -e

echo "=========================================="
echo "🚀 빠른 환경 복구"
echo "=========================================="

# PATH 업데이트
export PATH="$HOME/.local/bin:$PATH"

# 환경 변수 설정
echo "1️⃣  환경 변수 설정..."
if ! grep -q "# UV Persistent Storage 설정" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'BASHRC_EOF'

# UV Persistent Storage 설정
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy

# CUDA 설정 (Persistent Storage 우선)
# Persistent Storage에 CUDA가 있으면 사용, 없으면 /usr/local/cuda 사용
if [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    # 심볼릭 링크 생성 (없으면)
    if [ ! -e "/usr/local/cuda" ] && [ "$CUDA_HOME" != "/usr/local/cuda" ]; then
        ln -sf "$CUDA_HOME" /usr/local/cuda 2>/dev/null || true
    fi
fi

# Python 경로 우선순위 설정
export PYTHONNOUSERSITE=1
BASHRC_EOF
    echo "   ✓ 환경 변수 추가됨"
else
    echo "   ✓ 환경 변수 이미 설정됨"
fi

# 즉시 적용
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
# CUDA_HOME은 아래에서 설정됨

# uv 설치 (없을 때만)
echo ""
echo "2️⃣  uv 확인..."
if ! command -v uv &> /dev/null; then
    echo "   uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "   ✓ uv 준비됨"

# CUDA 경로 확인 및 심볼릭 링크 설정
echo ""
echo "3️⃣  CUDA 경로 확인..."
if [ -d "/mnt/data1/cuda-12.8" ]; then
    CUDA_HOME="/mnt/data1/cuda-12.8"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/mnt/data1/cuda" ]; then
    CUDA_HOME="/mnt/data1/cuda"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    echo "   ✓ 시스템 CUDA 발견: $CUDA_HOME"
else
    CUDA_HOME=""
    echo "   ⚠️  CUDA Toolkit을 찾을 수 없습니다"
fi

if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    # 심볼릭 링크 생성 (Persistent Storage의 CUDA를 사용하는 경우)
    if [[ "$CUDA_HOME" == /mnt/data1/* ]] && [ ! -e "/usr/local/cuda" ]; then
        ln -sf "$CUDA_HOME" /usr/local/cuda 2>/dev/null || true
        echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda -> $CUDA_HOME"
    fi
fi

# 디렉토리 생성
echo ""
echo "4️⃣  디렉토리 생성..."
mkdir -p /mnt/data1/.uv-cache
mkdir -p /mnt/data1/models/nlp/{huggingface_cache,conf_agg}
mkdir -p /mnt/data1/datasets/nlp/{cache,conf_agg/{outputs,logs,generated,curated,benchmarks}}
echo "   ✓ 디렉토리 생성 완료"

# 하이브리드 아키텍처
echo ""
echo "5️⃣  프로젝트 설정..."
mkdir -p /root/projects
if [ ! -d /root/projects/Conf_Agg ]; then
    cp -r /mnt/data1/projects/Conf_Agg /root/projects/
    echo "   ✓ 코드 복사됨"
fi

cd /root/projects/Conf_Agg
if [ ! -e .venv ]; then
    ln -sf /mnt/data1/projects/Conf_Agg/.venv .venv 2>/dev/null || true
    echo "   ✓ venv 링크 생성"
fi

# Python 확인
echo ""
echo "6️⃣  Python 환경 확인..."
if [ -f "/mnt/data1/projects/Conf_Agg/.venv/bin/python" ]; then
    python_ver=$(/mnt/data1/projects/Conf_Agg/.venv/bin/python --version)
    echo "   ✓ Python: $python_ver"
else
    echo "   ⚠️  .venv가 없습니다"
    echo "   실행: cd /mnt/data1/projects/Conf_Agg && export UV_CACHE_DIR=/mnt/data1/.uv-cache && uv sync"
fi

# 실행 스크립트 생성
echo ""
echo "7️⃣  실행 스크립트 준비..."
cat > /root/projects/Conf_Agg/run.sh << 'RUN_EOF'
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export HF_HOME=/mnt/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data1/models/nlp/huggingface_cache
export PYTHONPATH=/root/projects/Conf_Agg
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_USE_FLASHINFER=1
export SAMPLE_LIMIT=400
export PYTHONNOUSERSITE=1
# CUDA 경로 자동 감지 (Persistent Storage 우선)
if [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "$CUDA_HOME" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi
cd /root/projects/Conf_Agg
./scripts/run_stage1_2gpu.sh
RUN_EOF
chmod +x /root/projects/Conf_Agg/run.sh
echo "   ✓ run.sh 생성"

echo ""
echo "=========================================="
echo "✅ 준비 완료!"
echo "=========================================="
echo ""
echo "실행:"
echo "  cd /root/projects/Conf_Agg"
echo "  ./run.sh"
echo ""

