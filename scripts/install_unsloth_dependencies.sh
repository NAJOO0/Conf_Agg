#!/bin/bash
# Unsloth 누락된 의존성 설치 스크립트
# CUDA Toolkit 설치 후 flash-attn, flashinfer 등 누락된 의존성 설치

set -e

echo "=========================================="
echo "🔧 Unsloth 누락된 의존성 설치"
echo "=========================================="
echo ""

# CUDA 환경변수 확인
if [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "❌ CUDA_HOME을 찾을 수 없습니다."
    echo "   먼저 CUDA Toolkit을 설치하세요."
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "✓ CUDA_HOME: $CUDA_HOME"
echo ""

# nvcc 확인
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc를 찾을 수 없습니다."
    echo "   CUDA Toolkit이 제대로 설치되었는지 확인하세요."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
echo "✓ nvcc 버전: $NVCC_VERSION"
echo ""

# Python 환경 확인
cd /mnt/data1/projects/Conf_Agg

if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ .venv를 찾을 수 없습니다."
    echo "   먼저 ./restart_setup.sh를 실행하세요."
    exit 1
fi

source .venv/bin/activate
echo "✓ Python 환경 활성화됨"
echo ""

# Unsloth 설치 확인
echo "1️⃣  Unsloth 설치 확인..."
if python -c "import unsloth" 2>/dev/null; then
    UNSLOTH_VERSION=$(python -c "import unsloth; print(getattr(unsloth, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "   ✓ Unsloth 설치됨: $UNSLOTH_VERSION"
else
    echo "   ⚠️  Unsloth가 설치되어 있지 않습니다."
    echo "   먼저 unsloth를 설치하세요: uv pip install unsloth"
    exit 1
fi
echo ""

# 누락된 의존성 확인
echo "2️⃣  누락된 의존성 확인..."

MISSING_PACKAGES=()

# flash-attn 확인
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "   ⚠️  flash-attn 누락"
    MISSING_PACKAGES+=("flash-attn==2.7.4")
else
    FLASH_ATTN_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
    echo "   ✓ flash-attn 설치됨: $FLASH_ATTN_VERSION"
fi

# flashinfer 확인
if ! python -c "import flashinfer" 2>/dev/null; then
    echo "   ⚠️  flashinfer 누락"
    MISSING_PACKAGES+=("flashinfer")
else
    FLASHINFER_VERSION=$(python -c "import flashinfer; print(getattr(flashinfer, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "   ✓ flashinfer 설치됨: $FLASHINFER_VERSION"
fi

echo ""

# 누락된 패키지가 있으면 설치
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "3️⃣  누락된 패키지 설치 중..."
    echo ""
    
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "   📦 $pkg 설치 중..."
        
        if [[ "$pkg" == "flash-attn==2.7.4" ]]; then
            # flash-attn은 특별 처리 (빌드 필요)
            echo "      ⚠️  flash-attn 빌드는 시간이 걸릴 수 있습니다 (10-30분)..."
            export MAX_JOBS=4
            uv pip install flash-attn==2.7.4 --no-build-isolation || {
                echo "      ❌ flash-attn 설치 실패"
                echo "      💡 수동 설치: uv pip install flash-attn==2.7.4 --no-build-isolation"
            }
        else
            # 일반 패키지
            uv pip install "$pkg" || {
                echo "      ❌ $pkg 설치 실패"
            }
        fi
        
        echo "      ✓ $pkg 설치 완료"
        echo ""
    done
    
    echo "✅ 누락된 패키지 설치 완료!"
else
    echo "✅ 모든 의존성이 이미 설치되어 있습니다!"
fi

echo ""
echo "4️⃣  설치 확인..."
echo ""

# 최종 확인
if python -c "import flash_attn" 2>/dev/null; then
    FLASH_ATTN_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
    echo "   ✓ flash-attn: $FLASH_ATTN_VERSION"
else
    echo "   ⚠️  flash-attn: 설치되지 않음"
fi

if python -c "import flashinfer" 2>/dev/null; then
    FLASHINFER_VERSION=$(python -c "import flashinfer; print(getattr(flashinfer, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "   ✓ flashinfer: $FLASHINFER_VERSION"
else
    echo "   ⚠️  flashinfer: 설치되지 않음"
fi

echo ""
echo "=========================================="
echo "✅ 완료!"
echo "=========================================="
echo ""
echo "💡 참고:"
echo "   - flash-attn은 CUDA Toolkit이 필요하며 빌드 시간이 오래 걸립니다"
echo "   - 설치 후 Python을 재시작하면 import가 가능합니다"
echo ""

