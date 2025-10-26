#!/bin/bash

# uv 기반 프로젝트 설정 스크립트
# 로컬 개발 환경 설정

echo "🚀 Conf_Agg 프로젝트 설정 시작 (uv 기반)..."

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "📦 uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv가 이미 설치되어 있습니다."
fi

# 프로젝트 의존성 설치
echo "📦 프로젝트 의존성 설치 중..."
uv sync

# FlashInfer 설치 (선택사항)
echo "🔧 FlashInfer 설치 중..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA 버전: $CUDA_VERSION"
    uv add flashinfer --index-url https://flashinfer.ai/whl/cu${CUDA_VERSION//.}/torch2.4/
else
    echo "⚠️  nvcc를 찾을 수 없습니다. FlashInfer 설치를 건너뜁니다."
fi

# 환경 변수 설정
echo "🔧 환경 변수 설정..."
export PYTHONPATH="$(pwd)"
export VLLM_USE_FLASHINFER=1

# 설정 확인
echo "✅ 설정 확인 중..."
uv run python -c "import torch; print(f'PyTorch 버전: {torch.__version__}')"
uv run python -c "import vllm; print(f'vLLM 버전: {vllm.__version__}')"

echo "🎉 설정 완료!"
echo ""
echo "📋 사용법:"
echo "1. 스크립트 실행:"
echo "   uv run python scripts/stage1_generate.py"
echo ""
echo "2. 개발 도구 실행:"
echo "   uv run jupyter lab"
echo ""
echo "3. 패키지 추가:"
echo "   uv add <package-name>"
echo ""
echo "4. 패키지 제거:"
echo "   uv remove <package-name>"
