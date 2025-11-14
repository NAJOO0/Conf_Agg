#!/bin/bash
# GRPO Training 환경 설정 스크립트
# torch 2.5.1 + unsloth[cu124-torch250] + trl + vllm + flash-attn 스택

set -e

echo "=========================================="
echo "GRPO Training 환경 설정 스크립트"
echo "torch 2.5.0 + unsloth[cu124-torch250] + trl + vllm"
echo "=========================================="
echo ""

# 0. 사용할 Python 버전 및 가상환경 이름
PYTHON_VERSION="3.10"
VENV_NAME=".venv-grpo"
PROJECT_DIR="/mnt/data1/projects/Conf_Agg"

# 1. 필수 패키지 설치
echo "1️⃣  필수 패키지 설치 중..."
# IPv4 강제 설정 및 카카오 미러 사용
if [ -f /etc/apt/sources.list ]; then
    echo 'Acquire::ForceIPv4 "true";' | tee /etc/apt/apt.conf.d/99force-ipv4 > /dev/null 2>&1 || true
    # http와 https 모두 처리 (tw.archive.ubuntu.com, kr.archive.ubuntu.com 등 모든 변형 포함)
    sed -i.bak \
        -e "s|https\?://[^/]*archive\.ubuntu\.com|http://mirror.kakao.com|g" \
        -e "s|https\?://[^/]*security\.ubuntu\.com|http://mirror.kakao.com|g" \
        -e "s|https\?://[^/]*\.archive\.ubuntu\.com|http://mirror.kakao.com|g" \
        /etc/apt/sources.list 2>/dev/null || true
    echo "   ✓ 미러 설정: http://mirror.kakao.com"
fi

apt-get update -qq 2>/dev/null || echo "   ⚠️  apt-get update 실패 (계속 진행)"
apt-get install -y build-essential curl wget git python3 htop > /dev/null 2>&1 || echo "   ⚠️  일부 패키지는 이미 설치되어 있습니다"
echo "   ✓ 패키지 설치 완료"

# 2. uv 설치
echo ""
echo "2️⃣  uv 설치 중..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
    echo "   ✓ uv 설치 완료"
else
    export PATH="$HOME/.local/bin:$PATH"
    echo "   ✓ uv가 이미 설치되어 있습니다"
fi

# 3. 영구 환경 변수 설정 (.bashrc)
echo ""
echo "3️⃣  영구 환경 변수 설정..."
if grep -q "UV_CACHE_DIR=/mnt/data1/.uv-cache" ~/.bashrc 2>/dev/null; then
    echo "   ✓ 환경 변수는 이미 .bashrc에 설정되어 있습니다"
else
    cat >> ~/.bashrc << 'BASHRC_EOF'

# --- Project & UV Persistent Settings ---
# UV
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy

# Python (시스템 패키지 충돌 방지)
export PYTHONNOUSERSITE=1

# CUDA 설정 (Persistent Storage 우선)
# Persistent Storage에 CUDA가 있으면 사용, 없으면 /usr/local/cuda 사용
if [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda-12.4" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.4
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# HuggingFace Cache
export HF_HOME=/mnt/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data1/models/nlp/huggingface_cache

# Python Path
export PYTHONPATH=/mnt/data1/projects/Conf_Agg
# --- End Settings ---
BASHRC_EOF
    echo "   ✓ ~/.bashrc에 환경 변수 추가"
fi
# .bashrc 로드 시도 (실패해도 계속 진행)
source ~/.bashrc 2>/dev/null || true

# 3-1. CUDA 경로 확인 및 심볼릭 링크 설정
echo ""
echo "3-1️⃣  CUDA 경로 확인 및 심볼릭 링크 설정..."
CUDA_HOME=""
# 여러 CUDA 버전 경로 확인 (12.8, 12.4, 일반 cuda 순서)
if [ -d "/mnt/data1/cuda-12.8" ]; then
    CUDA_HOME="/mnt/data1/cuda-12.8"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/mnt/data1/cuda-12.4" ]; then
    CUDA_HOME="/mnt/data1/cuda-12.4"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/mnt/data1/cuda" ]; then
    CUDA_HOME="/mnt/data1/cuda"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    echo "   ✓ 시스템 CUDA 발견: $CUDA_HOME"
else
    echo "   ⚠️  CUDA Toolkit을 찾을 수 없습니다"
    echo "   💡 CUDA Toolkit이 필요하면 설치하세요"
fi

if [ -n "$CUDA_HOME" ]; then
    # 현재 셸에 즉시 적용 (확실하게)
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    
    # 심볼릭 링크 생성 (Persistent Storage의 CUDA를 사용하는 경우)
    if [[ "$CUDA_HOME" == /mnt/data1/* ]]; then
        # 기존 링크 제거 (있다면)
        if [ -L "/usr/local/cuda" ]; then
            rm /usr/local/cuda 2>/dev/null || true
        fi
        # 새 링크 생성
        if [ ! -e "/usr/local/cuda" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda -> $CUDA_HOME"
        else
            echo "   ✓ 심볼릭 링크 이미 존재: /usr/local/cuda"
        fi
        
        # 버전별 링크도 생성
        if [[ "$CUDA_HOME" == *cuda-12.8* ]] && [ ! -e "/usr/local/cuda-12.8" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda-12.8 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda-12.8 -> $CUDA_HOME"
        elif [[ "$CUDA_HOME" == *cuda-12.4* ]] && [ ! -e "/usr/local/cuda-12.4" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda-12.4 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda-12.4 -> $CUDA_HOME"
        fi
    fi
    echo "   ✓ CUDA 환경 변수 설정 완료: CUDA_HOME=$CUDA_HOME"
    echo "   ✓ 현재 셸에 환경 변수 적용됨 (PATH에 $CUDA_HOME/bin 추가됨)"
    
    # nvcc 확인 및 CUDA 버전 감지
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "   ✓ nvcc 확인: $CUDA_HOME/bin/nvcc"
        # CUDA 버전 감지 (예: 12.8, 12.4 등)
        if [[ "$CUDA_HOME" == *cuda-12.8* ]]; then
            CUDA_VERSION="12.8"
            CUDA_INDEX="cu128"
        elif [[ "$CUDA_HOME" == *cuda-12.4* ]]; then
            CUDA_VERSION="12.4"
            CUDA_INDEX="cu124"
        else
            # nvcc로 버전 확인 시도
            NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' | cut -d. -f1,2 || echo "")
            if [[ "$NVCC_VERSION" == "12.8" ]]; then
                CUDA_VERSION="12.8"
                CUDA_INDEX="cu128"
            elif [[ "$NVCC_VERSION" == "12.4" ]]; then
                CUDA_VERSION="12.4"
                CUDA_INDEX="cu124"
            else
                # 기본값으로 12.4 사용
                CUDA_VERSION="12.4"
                CUDA_INDEX="cu124"
                echo "   ⚠️  CUDA 버전을 정확히 감지하지 못했습니다. 기본값(12.4)을 사용합니다."
            fi
        fi
        echo "   ✓ CUDA 버전 감지: $CUDA_VERSION (인덱스: $CUDA_INDEX)"
    else
        echo "   ⚠️  경고: $CUDA_HOME/bin/nvcc 파일이 없습니다"
        # 기본값 설정
        CUDA_VERSION="12.4"
        CUDA_INDEX="cu124"
    fi
else
    # CUDA가 없을 때 기본값
    CUDA_VERSION="12.4"
    CUDA_INDEX="cu124"
fi

# 4. 디렉토리 생성
echo ""
echo "4️⃣  필수 디렉토리 생성..."
mkdir -p /mnt/data1/{.uv-cache,models/nlp/huggingface_cache}
echo "   ✓ 디렉토리 생성 완료"

# 5. 가상환경 생성 및 설정
echo ""
echo "5️⃣  가상환경 생성 및 설정..."
cd "$PROJECT_DIR"

# 기존 .venv-grpo가 있으면 확인
if [ -d "$VENV_NAME" ]; then
    echo "   ⚠️  기존 가상환경 발견: $VENV_NAME"
    read -p "   삭제하고 재생성하시겠습니까? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   - 기존 가상환경 삭제 중..."
        rm -rf "$VENV_NAME"
        echo "   ✓ 기존 가상환경 삭제 완료"
    else
        echo "   ✓ 기존 가상환경 유지"
        NEED_RECREATE=false
    fi
else
    NEED_RECREATE=true
fi

# 가상환경 생성
if [ "$NEED_RECREATE" != false ]; then
    echo "   - Python $PYTHON_VERSION 으로 새 가상환경 생성 중..."
    uv venv --python $PYTHON_VERSION "$VENV_NAME"
    echo "   ✓ 가상환경 생성 완료"
fi

# 가상환경 활성화
source "$VENV_NAME/bin/activate"
echo "   ✓ 가상환경 활성화됨"

# .venv 심볼릭 링크 생성 (uv sync가 .venv를 찾을 수 있도록)
if [ -L ".venv" ]; then
    rm .venv 2>/dev/null || true
fi
if [ ! -e ".venv" ]; then
    ln -sf "$VENV_NAME" .venv
    echo "   ✓ .venv 심볼릭 링크 생성: .venv -> $VENV_NAME"
fi

# 6. pyproject.toml 생성 및 의존성 설치
echo ""
echo "6️⃣  pyproject.toml 생성 및 의존성 설치..."
echo "   - pyproject.toml 설정 중..."
cat > pyproject.toml << 'PYPROJECT_EOF'
[project]
name = "grpo-training"
version = "0.1.0"
requires-python = ">=3.10,<3.12"
dependencies = [
    "torch==2.5.0",
    "xformers; sys_platform == 'linux'",
    "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git",
    "trl>=0.11.0",
    "vllm>=0.6.0",
    "transformers>=4.45.0",
    "datasets>=2.14.0",
    "accelerate>=0.33.0",
    "bitsandbytes>=0.43.0",
]

[project.optional-dependencies]
flash = [
    "flash-attn>=2.7.1,<=2.8.2",
    "flashinfer",
]
PYPROJECT_EOF
echo "   ✓ pyproject.toml 생성 완료"

# PyTorch 인덱스 URL 설정 (CUDA 버전에 맞춰서)
if [ "$CUDA_INDEX" = "cu128" ]; then
    echo "   💡 참고: PyTorch 2.5.1은 cu124만 제공됩니다. CUDA 12.8과 호환되므로 cu124 버전을 사용합니다."
fi

# uv.toml 생성 (PyTorch 인덱스 URL 설정)
echo "   - uv.toml 생성 (PyTorch 인덱스 URL 설정)..."
cat > uv.toml << 'UV_EOF'
[[index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
UV_EOF
echo "   ✓ uv.toml 생성 완료"

# uv sync로 의존성 설치
echo "   - uv sync로 의존성 설치 중 (시간이 걸릴 수 있습니다)..."
uv sync --python $PYTHON_VERSION || {
    echo "   ⚠️  uv sync 실패, 수동 설치 시도..."
    # PyTorch 먼저 설치
    uv pip install --index-url https://download.pytorch.org/whl/cu124 \
        "torch==2.5.1" "torchvision" "torchaudio" || true
    # xformers Linux용 설치
    uv pip install --index-url https://download.pytorch.org/whl/${CUDA_INDEX} \
        "xformers; sys_platform == 'linux'" || true
    # 나머지 패키지 설치
    uv pip install -e .
}
echo "   ✓ 기본 의존성 설치 완료"

# # Flash Attention 별도 설치 (빌드 필요, Unsloth 호환 버전)
# echo ""
# echo "7️⃣  Flash Attention 설치 (Unsloth 호환 버전: 2.7.1~2.8.2)..."
# echo "   - Flash Attention 빌드 및 설치 (시간이 걸릴 수 있습니다)..."
# uv pip install "flash-attn>=2.7.1,<=2.8.2" --no-build-isolation || {
#     echo "   ⚠️  Flash Attention 설치 실패 (계속 진행)"
#     echo "   💡 CUDA toolkit과 build-essential이 필요할 수 있습니다"
# }
# echo "   ✓ Flash Attention 설치 완료"

# # FlashInfer 설치
# echo ""
# echo "8️⃣  FlashInfer 설치..."
# echo "   - FlashInfer 설치 (cu124/torch2.4용)..."
# uv pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/ || {
#     echo "   ⚠️  FlashInfer 설치 실패 (계속 진행)"
#     echo "   💡 CUDA 12.4와 torch 2.4+ 호환 버전이 필요합니다"
# }
# echo "   ✓ FlashInfer 설치 완료"

# 9. GPU 및 nvcc 확인
echo ""
echo "9️⃣  GPU 및 nvcc 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | grep "Driver Version" || true
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
    echo "   ✓ GPU 개수: $gpu_count"
else
    echo "   ⚠️  nvidia-smi를 찾을 수 없습니다"
fi

# CUDA 환경변수 다시 적용 (확실하게)
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

if command -v nvcc &> /dev/null; then
    nvcc_path=$(which nvcc)
    nvcc_version=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' || echo "unknown")
    echo "   ✓ nvcc 확인 완료"
    echo "      경로: $nvcc_path"
    echo "      버전: $nvcc_version"
else
    echo "   ⚠️  nvcc를 찾을 수 없습니다."
    if [ -n "$CUDA_HOME" ]; then
        echo "   💡 CUDA_HOME=$CUDA_HOME 이 설정되어 있지만 nvcc를 찾을 수 없습니다."
    else
        echo "   💡 CUDA Toolkit이 설치되지 않았습니다."
    fi
fi

# 10. 설치된 패키지 버전 확인
echo ""
echo "🔟  설치된 패키지 버전 확인..."
if [ -f "$VENV_NAME/bin/activate" ]; then
    source "$VENV_NAME/bin/activate"
    echo "   - Python 버전: $(python --version 2>/dev/null || echo 'N/A')"
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
    echo "   - PyTorch: $TORCH_VERSION"
    
    # 선택적 패키지 확인
    if python -c "import unsloth" 2>/dev/null; then
        echo "   - Unsloth: 설치됨"
    else
        echo "   - Unsloth: 설치 안 됨"
    fi
    
    if python -c "import trl" 2>/dev/null; then
        TRL_VERSION=$(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "N/A")
        echo "   - TRL: $TRL_VERSION"
    else
        echo "   - TRL: 설치 안 됨"
    fi
    
    if python -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "N/A")
        echo "   - vLLM: $VLLM_VERSION"
    else
        echo "   - vLLM: 설치 안 됨"
    fi
    
    if python -c "import flash_attn" 2>/dev/null; then
        echo "   - Flash Attention: 설치됨"
    else
        echo "   - Flash Attention: 설치 안 됨"
    fi
    
    if python -c "import flashinfer" 2>/dev/null; then
        echo "   - FlashInfer: 설치됨"
    else
        echo "   - FlashInfer: 설치 안 됨"
    fi
else
    echo "   ⚠️  .venv를 찾을 수 없습니다"
fi

# 완료 메시지
echo ""
echo "=========================================="
echo "✅ 모든 설정 완료!"
echo "=========================================="
echo ""
echo "📋 프로젝트 위치:"
echo "  $PROJECT_DIR"
echo ""
echo "📋 사용 방법:"
echo ""
echo "  cd $PROJECT_DIR"
echo "  source $VENV_NAME/bin/activate"
echo "  # 이제 Python 스크립트를 실행할 수 있습니다"
echo ""
echo "🔧 무결성 체크 (수동):"
echo "  cd $PROJECT_DIR"
echo "  source $VENV_NAME/bin/activate"
echo "  python -c 'import torch; print(f\"Torch: {torch.__version__}\")'"
echo "  python -c 'import unsloth; print(\"Unsloth: OK\")'"
echo "  python -c 'import trl; print(f\"TRL: {trl.__version__}\")'"
echo "  python -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'"
echo ""
echo "🔧 CUDA 환경변수 적용 (현재 셸에서 nvcc를 사용하려면):"
echo "  source ~/.bashrc"
echo "  nvcc --version"
echo ""
echo ""

