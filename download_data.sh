#!/bin/bash
# DeepScaler 데이터 다운로드 및 준비 스크립트

set -e

echo "=========================================="
echo "DeepScaler 데이터 다운로드"
echo "=========================================="

# 디렉토리 생성
mkdir -p /mnt/data1/datasets/nlp/conf_agg/raw

# 환경 변수 설정
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export PYTHONPATH=/mnt/data1/projects/Conf_Agg

# 샘플 데이터 생성 (빠른 테스트용)
echo ""
echo "🔧 샘플 데이터 생성 중..."
cd /mnt/data1/projects/Conf_Agg

# uv 환경에서 실행
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/download_and_prepare_data.py \
        --output-dir /mnt/data1/datasets/nlp/conf_agg/raw \
        --sample-only
else
    echo "⚠️  .venv가 없습니다. uv sync를 먼저 실행하세요."
    exit 1
fi

echo ""
echo "✅ 데이터 준비 완료!"
echo "📁 위치: /mnt/data1/datasets/nlp/conf_agg/raw/deepscaler.jsonl"
echo ""
echo "진짜 데이터를 다운로드하려면:"
echo "  uv run python scripts/download_and_prepare_data.py"


