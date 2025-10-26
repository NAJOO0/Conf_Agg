#!/bin/bash
#
# 빠른 배포 스크립트 - 새 서버에서 실행
# 
# 사용법:
#   bash scripts/quick_deploy.sh
#

set -e  # 에러 발생 시 중단

echo "🚀 Conf_Agg 빠른 배포 스크립트 시작"
echo "======================================"

# 1. 현재 디렉토리 확인
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 오류: Conf_Agg 프로젝트 루트에서 실행해주세요"
    exit 1
fi

# 2. .env 파일 확인 및 생성
if [ ! -f ".env" ]; then
    echo "📝 .env 파일을 생성합니다..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "⚠️  .env 파일을 생성했습니다. WANDB_API_KEY를 설정해주세요:"
        echo "   nano .env"
    else
        echo "WANDB_API_KEY=your_key_here" > .env
        echo "⚠️  .env 파일을 생성했습니다. WANDB_API_KEY를 설정해주세요:"
        echo "   nano .env"
    fi
fi

# 3. Docker 컨테이너 빌드
echo ""
echo "🔨 Docker 이미지 빌드 중..."
docker-compose build

# 4. 컨테이너 시작
echo ""
echo "🚀 Docker 컨테이너 시작 중..."
docker-compose up -d

# 5. 컨테이너가 준비될 때까지 대기
echo ""
echo "⏳ 컨테이너가 준비될 때까지 대기 중..."
sleep 10

# 6. GPU 확인
echo ""
echo "🎮 GPU 상태 확인 중..."
docker-compose exec -T conf-agg-llm nvidia-smi

# 7. uv sync (의존성 설치)
echo ""
echo "📦 uv sync 실행 중 (의존성 설치)..."
echo "   이 과정은 시간이 걸릴 수 있습니다..."
docker-compose exec -T conf-agg-llm uv sync

# 8. 완료 메시지
echo ""
echo "✅ 배포가 완료되었습니다!"
echo ""
echo "다음 명령어로 컨테이너에 접속하세요:"
echo "   docker-compose exec conf-agg-llm bash"
echo ""
echo "컨테이너 안에서 실행할 수 있는 명령어:"
echo "   # Stage 1 실행 (단일 GPU)"
echo "   SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \\"
echo "       --config-path config \\"
echo "       --config-name config \\"
echo "       --gpu-id \"0\" \\"
echo "       --shard-id 0 \\"
echo "       --total-shards 1"
echo ""
echo "   # Stage 1 실행 (4개 GPU - 백그라운드)"
echo "   uv run bash scripts/run_stage1_background.sh"
echo ""
echo "   # 로그 확인"
echo "   tail -f outputs/logs/sample_400/stage1_shard_0.log"
echo ""
echo "   # GPU 모니터링"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "======================================"
