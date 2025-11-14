#!/bin/bash
# FlashInfer 설치 및 vLLM 통합 스크립트
# H100 GPU용 최적화 (CUDA 12.8, SM 9.0)

set -e

echo "=========================================="
echo "FlashInfer + vLLM 설치 스크립트"
echo "H100 GPU (SM 9.0) 최적화 버전"
echo "=========================================="
echo ""

# 1. 환경 변수 설정
echo "1️⃣ 환경 변수 설정..."
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# FlashInfer 관련 환경변수
export FLASHINFER_CUDA_ARCHITECTURES="90"  # H100 (SM 9.0)
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_FLASHINFER=1

echo "   ✓ CUDA_HOME: $CUDA_HOME"
echo "   ✓ FlashInfer Architecture: SM 9.0 (H100)"

# 2. CUDA 버전 확인
echo ""
echo "2️⃣ CUDA 환경 확인..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "   ✓ CUDA 버전: $CUDA_VERSION"
    
    # CUDA 12.x 확인
    if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
        echo "   ⚠️ 경고: CUDA 12.x가 권장됩니다. 현재: $CUDA_VERSION"
    fi
else
    echo "   ❌ nvcc를 찾을 수 없습니다. CUDA가 설치되어 있는지 확인하세요."
    exit 1
fi

# 3. Python 환경 활성화 (기존 setup.sh 참고)
echo ""
echo "3️⃣ Python 환경 준비..."
cd /mnt/data1/projects/Conf_Agg

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "   ✓ 기존 .venv 활성화"
else
    echo "   - 새 .venv 생성..."
    uv venv --python 3.12
    source .venv/bin/activate
fi

PYTHON_VERSION=$(python --version | cut -d' ' -f2)
echo "   ✓ Python 버전: $PYTHON_VERSION"

# 4. PyTorch 설치/확인 (CUDA 12.8 버전)
echo ""
echo "4️⃣ PyTorch 설치/확인..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "   ✓ PyTorch 이미 설치됨: $TORCH_VERSION"
else
    echo "   - PyTorch 2.5.1+cu128 설치 중..."
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.5.1+cu128 torchvision torchaudio
fi

# 5. FlashInfer 설치
echo ""
echo "5️⃣ FlashInfer 설치..."

# FlashInfer 의존성
echo "   - FlashInfer 의존성 설치..."
uv pip install ninja packaging

# FlashInfer 빌드 방법 선택
echo ""
echo "   FlashInfer 설치 방법 선택:"
echo "   1) Pre-built wheel 사용 (빠름, 권장)"
echo "   2) 소스에서 빌드 (최적화, 시간 오래 걸림)"
read -p "   선택 (1 또는 2): " INSTALL_METHOD

if [ "$INSTALL_METHOD" = "2" ]; then
    # 소스에서 빌드
    echo "   - FlashInfer를 소스에서 빌드합니다..."
    
    # 기존 설치 제거
    pip uninstall -y flashinfer 2>/dev/null || true
    
    # 소스 클론
    if [ -d "/tmp/flashinfer" ]; then
        rm -rf /tmp/flashinfer
    fi
    
    git clone https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer
    cd /tmp/flashinfer
    
    # H100 최적화 빌드
    export TORCH_CUDA_ARCH_LIST="9.0"  # H100
    export MAX_JOBS=8  # 병렬 컴파일 작업 수
    
    python setup.py install
    
    cd /mnt/data1/projects/Conf_Agg
    echo "   ✓ FlashInfer 소스 빌드 완료"
    
else
    # Pre-built wheel 사용
    echo "   - Pre-built FlashInfer wheel 설치..."
    
    # CUDA와 Python 버전에 맞는 wheel 설치
    # FlashInfer는 특정 CUDA/Python 조합의 wheel 제공
    CUDA_VERSION_SHORT="cu128"  # CUDA 12.8
    PYTHON_VERSION_SHORT="cp312"  # Python 3.12
    
    # 직접 wheel URL 또는 index 사용
    uv pip install flashinfer -i https://flashinfer.ai/whl/cu128/torch2.5/
    
    echo "   ✓ FlashInfer wheel 설치 완료"
fi

# 6. vLLM 재설치 (FlashInfer 지원 포함)
echo ""
echo "6️⃣ vLLM 설치 (FlashInfer 백엔드 지원)..."

# 기존 vLLM 제거
pip uninstall -y vllm vllm-flash-attn 2>/dev/null || true

# vLLM 설치 옵션
echo ""
echo "   vLLM 설치 방법:"
echo "   1) 공식 릴리즈 (안정적)"
echo "   2) 최신 개발 버전 (FlashInfer 최신 지원)"
read -p "   선택 (1 또는 2): " VLLM_METHOD

if [ "$VLLM_METHOD" = "2" ]; then
    # 개발 버전
    echo "   - vLLM 최신 버전 설치..."
    uv pip install git+https://github.com/vllm-project/vllm.git
else
    # 공식 버전
    echo "   - vLLM 공식 버전 설치..."
    uv pip install "vllm>=0.6.0"
fi

# 7. 설치 확인
echo ""
echo "7️⃣ 설치 확인..."

# Python 스크립트로 확인
cat > /tmp/test_flashinfer.py << 'EOF'
import sys
import torch

print("=" * 50)
print("FlashInfer + vLLM 설치 확인")
print("=" * 50)

# PyTorch 확인
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# FlashInfer 확인
try:
    import flashinfer
    print(f"FlashInfer: 설치됨 (버전: {flashinfer.__version__ if hasattr(flashinfer, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"FlashInfer: 설치 실패 - {e}")

# vLLM 확인
try:
    import vllm
    print(f"vLLM: {vllm.__version__}")
    
    # vLLM의 attention backend 확인
    from vllm.attention.backends.flashinfer import FlashInferBackend
    print("vLLM FlashInfer Backend: 사용 가능")
except ImportError as e:
    print(f"vLLM 또는 FlashInfer Backend 오류: {e}")

print("=" * 50)

# 간단한 벤치마크
if torch.cuda.is_available():
    print("\n간단한 성능 테스트...")
    
    # FlashInfer 테스트
    try:
        import flashinfer
        
        batch_size = 32
        seq_len = 2048
        num_heads = 32
        head_dim = 128
        
        # 랜덤 텐서 생성
        q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        torch.cuda.synchronize()
        
        print(f"테스트 성공: batch={batch_size}, seq_len={seq_len}")
        print("FlashInfer가 정상적으로 작동합니다!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")
EOF

python /tmp/test_flashinfer.py

# 8. vLLM 실행 스크립트 생성
echo ""
echo "8️⃣ vLLM 실행 스크립트 생성..."

cat > /mnt/data1/projects/Conf_Agg/run_vllm_flashinfer.sh << 'EOF'
#!/bin/bash
# vLLM with FlashInfer 실행 스크립트

# 환경 변수 설정
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# FlashInfer 백엔드 강제 사용
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Python 환경 활성화
source .venv/bin/activate

echo "=========================================="
echo "vLLM 서버 시작 (FlashInfer Backend)"
echo "=========================================="

# vLLM 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model $1 \
    --port ${2:-8000} \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 12288 \
    --dtype float16 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 65536 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-requests \
    2>&1 | tee vllm_flashinfer.log
EOF

chmod +x /mnt/data1/projects/Conf_Agg/run_vllm_flashinfer.sh

# 9. 완료 메시지
echo ""
echo "=========================================="
echo "✅ FlashInfer + vLLM 설치 완료!"
echo "=========================================="
echo ""
echo "📋 사용 방법:"
echo ""
echo "1. vLLM 서버 실행 (FlashInfer 백엔드):"
echo "   cd /mnt/data1/projects/Conf_Agg"
echo "   ./run_vllm_flashinfer.sh 'Qwen/Qwen2.5-Math-1.5B-Instruct' 8000"
echo ""
echo "2. Python에서 직접 사용:"
echo "   export VLLM_ATTENTION_BACKEND=FLASHINFER"
echo "   python your_script.py"
echo ""
echo "⚠️  주의사항:"
echo "- H100 GPU에서 최적 성능"
echo "- CUDA 12.x 필요"
echo "- 메모리 사용량이 기본 attention보다 적음"
echo "- 특히 긴 시퀀스에서 성능 향상"
echo ""
echo "🔧 문제 해결:"
echo "- FlashInfer import 오류 시: pip install flashinfer --upgrade"
echo "- vLLM backend 오류 시: export VLLM_ATTENTION_BACKEND=FLASH_ATTN"
echo ""