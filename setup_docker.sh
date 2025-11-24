#!/bin/bash

echo "🚀 Conf-AggLLM 환경 설정 시작..."

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker가 설치되지 않았습니다. Docker를 먼저 설치해주세요."
    exit 1
fi

# NVIDIA Docker 지원 확인
if ! docker run --rm --runtime=nvidia nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi; then
    echo "❌ NVIDIA Docker 지원이 설정되지 않았습니다."
    echo "다음 명령어로 NVIDIA Container Toolkit을 설치해주세요:"
    echo "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    exit 1
fi

# Docker 이미지 빌드
echo "📦 Docker 이미지 빌드 중..."
docker compose build

# 데이터 디렉토리 생성
echo "📁 데이터 디렉토리 생성 중..."
mkdir -p data/{raw,generated,curated,benchmarks}
mkdir -p outputs/{models,logs,results}

# 환경 변수 파일 생성
if [ ! -f .env ]; then
    echo "🔑 환경 변수 파일 생성 중..."
    cat > .env << EOF
# WandB API Key (선택사항)
WANDB_API_KEY=your_wandb_api_key_here

# CUDA 설정 (GPU 4개 사용)
CUDA_VISIBLE_DEVICES=0,1,2,3
EOF
    echo "⚠️  .env 파일을 생성했습니다. WandB API Key를 설정해주세요."
fi

echo "✅ 환경 설정 완료!"
echo ""
echo "다음 명령어로 컨테이너를 시작하세요:"
echo "  docker compose up -d"
echo ""
echo "컨테이너에 접속하려면:"
echo "  docker compose exec conf-agg-llm bash"
echo ""
echo "GPU 4개가 모두 사용 가능한지 확인하려면:"
echo "  docker compose exec conf-agg-llm nvidia-smi"
