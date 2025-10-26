#!/bin/bash
#
# vLLM 간단 테스트 실행 스크립트
#

echo "🚀 vLLM 간단 테스트 시작"
echo "========================"

# 컨테이너 안에서 실행하는지 확인
if [ -f "/.dockerenv" ]; then
    echo "🐳 Docker 컨테이너 안에서 실행 중"
    echo ""
    
    # uv로 실행
    uv run python test_vllm_simple.py
    
else
    echo "🖥️  호스트에서 실행 중"
    echo "⚠️  Docker 컨테이너 안에서 실행하는 것을 권장합니다"
    echo ""
    echo "컨테이너 접속:"
    echo "  docker-compose exec conf-agg-llm bash"
    echo "  bash run_test.sh"
fi
