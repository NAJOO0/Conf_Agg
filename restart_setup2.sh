#!/bin/bash
# 서버 재시작 후 환경 복구 스크립트 (v2.1 - PyTorch 2.5.1 + CUDA 12.4)
# 기존 .venv가 정상이면 재사용하여 무거운 패키지(CUDA toolkit 등) 재설치를 방지합니다.
# Python 버전 및 핵심 패키지 버전을 확인하여 필요시에만 재생성합니다.

set -e

echo "=========================================="
echo "서버 재시작 후 환경 복구 스크립트 (v2.1)"
echo "PyTorch 2.5.1 + CUDA 12.4"
echo "Unsloth + TRL(GRPO) + vLLM(colocate) 스택"
echo "=========================================="
echo ""

# 0. 사용할 Python 버전 (스택 B 기준)
PYTHON_VERSION="3.12"

# 1. 필수 패키지 설치
echo "1️⃣  필수 패키지 설치 중..."
# (변경) IPv4 강제 설정 및 카카오 미러 사용
echo 'Acquire::ForceIPv4 "true";' | tee /etc/apt/apt.conf.d/99force-ipv4 > /dev/null
sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list

apt-get update -qq
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

# CUDA 설정 (CUDA 12.4 우선)
# Persistent Storage에 CUDA가 있으면 사용, 없으면 /usr/local/cuda 사용
if [ -d "/mnt/data1/cuda-12.4" ]; then
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
export PYTHONPATH=/root/projects/Conf_Agg
# --- End Settings ---
BASHRC_EOF
    echo "   ✓ ~/.bashrc에 환경 변수 추가"
fi
source ~/.bashrc 2>/dev/null || true
echo "   ✓ 현재 세션에 환경 변수 적용됨"

# 3-1. CUDA 경로 확인 및 심볼릭 링크 설정
echo ""
echo "3-1️⃣  CUDA 경로 확인 및 심볼릭 링크 설정..."
CUDA_HOME=""
# CUDA 12.4 우선 검색 (Unsloth 호환성)
if [ -d "/mnt/data1/cuda-12.4" ]; then
    CUDA_HOME="/mnt/data1/cuda-12.4"
    echo "   ✓ Persistent Storage에서 CUDA 12.4 발견: $CUDA_HOME"
elif [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_HOME="/usr/local/cuda-12.4"
    echo "   ✓ 시스템에서 CUDA 12.4 발견: $CUDA_HOME"
elif [ -d "/mnt/data1/cuda-12.8" ]; then
    CUDA_HOME="/mnt/data1/cuda-12.8"
    echo "   ⚠️  CUDA 12.8 발견 (12.4 권장): $CUDA_HOME"
    echo "   💡 CUDA 12.8 드라이버는 12.4 빌드와 호환되지만, 12.4 설치를 권장합니다"
elif [ -d "/mnt/data1/cuda" ]; then
    CUDA_HOME="/mnt/data1/cuda"
    echo "   ✓ Persistent Storage에서 CUDA 발견: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    echo "   ✓ 시스템 CUDA 발견: $CUDA_HOME"
else
    echo "   ⚠️  CUDA Toolkit을 찾을 수 없습니다"
    echo "   💡 CUDA Toolkit이 필요하면 다음을 실행하세요:"
    echo "      ./scripts/install_cuda_toolkit.sh"
fi

if [ -n "$CUDA_HOME" ]; then
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
        if [[ "$CUDA_HOME" == *cuda-12.4* ]] && [ ! -e "/usr/local/cuda-12.4" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda-12.4 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda-12.4 -> $CUDA_HOME"
        elif [[ "$CUDA_HOME" == *cuda-12.8* ]] && [ ! -e "/usr/local/cuda-12.8" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda-12.8 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda-12.8 -> $CUDA_HOME"
        fi
    fi
    echo "   ✓ CUDA 환경 변수 설정 완료: CUDA_HOME=$CUDA_HOME"
fi

# 4. 디렉토리 생성
echo ""
echo "4️⃣  필수 디렉토리 생성..."
mkdir -p /mnt/data1/{.uv-cache,models/nlp/{huggingface_cache,conf_agg},datasets/nlp/{cache,conf_agg/{outputs,logs,generated,curated,benchmarks}}}
echo "   ✓ 디렉토리 생성 완료"

# 5. .venv 재생성 또는 재사용
echo ""
echo "5️⃣  Persistent .venv 확인 및 설정 (PyTorch 2.5.1 + CUDA 12.4)..."
cd /mnt/data1/projects/Conf_Agg

NEED_RECREATE=false

# 기존 .venv가 있는지 확인
if [ -d ".venv" ]; then
    echo "   - 기존 .venv 발견, 무결성 검사 중..."
    
    # .venv 활성화하여 Python 버전 확인
    source .venv/bin/activate 2>/dev/null || NEED_RECREATE=true
    
    if [ "$NEED_RECREATE" = false ]; then
        # Python 버전 확인
        VENV_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
        
        if [ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
            echo "   ⚠️  Python 버전 불일치 (기존: $VENV_PYTHON_VERSION, 필요: $PYTHON_VERSION)"
            NEED_RECREATE=true
        else
            # 핵심 패키지 버전 확인
            echo "   - 핵심 패키지 버전 확인 중..."
            TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
            TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
            VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
            TRL_VERSION=$(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "")
            
            # PyTorch 2.5.1 + CUDA 12.4 체크
            if [[ -z "$TORCH_VERSION" ]] || [[ ! "$TORCH_VERSION" =~ ^2\.5\.1 ]]; then
                echo "   ⚠️  PyTorch 버전 불일치 또는 미설치 (현재: ${TORCH_VERSION:-없음}, 필요: 2.5.1)"
                NEED_RECREATE=true
            elif [[ "$TORCH_CUDA" != "12.4" ]]; then
                echo "   ⚠️  PyTorch CUDA 버전 불일치 (현재: ${TORCH_CUDA:-없음}, 필요: 12.4)"
                NEED_RECREATE=true
            elif [[ -z "$VLLM_VERSION" ]] || [[ "$VLLM_VERSION" != "0.10.2" ]]; then
                echo "   ⚠️  vLLM 버전 불일치 (현재: ${VLLM_VERSION:-없음}, 필요: 0.10.2)"
                NEED_RECREATE=true
            elif [[ -z "$TRL_VERSION" ]] || [[ ! "$TRL_VERSION" =~ ^0\.24\. ]]; then
                echo "   ⚠️  TRL 버전 불일치 (현재: ${TRL_VERSION:-없음}, 필요: 0.24.0)"
                NEED_RECREATE=true
            else
                echo "   ✓ 기존 .venv가 정상입니다 (Python $VENV_PYTHON_VERSION)"
                echo "   ✓ PyTorch: $TORCH_VERSION (CUDA $TORCH_CUDA), vLLM: $VLLM_VERSION, TRL: $TRL_VERSION"
                echo "   ✓ 기존 .venv 재사용 (무거운 패키지 재설치 생략)"
            fi
        fi
    fi
    
    if [ "$NEED_RECREATE" = true ]; then
        echo "   - 기존 .venv 삭제 중..."
        deactivate 2>/dev/null || true
        rm -rf .venv
    fi
else
    echo "   - 기존 .venv 없음, 새로 생성 필요"
    NEED_RECREATE=true
fi

# 필요시 새로 생성
if [ "$NEED_RECREATE" = true ]; then
    echo "   - Python $PYTHON_VERSION 으로 새 .venv 생성 중..."
    uv venv --python $PYTHON_VERSION
    
    # .venv 활성화 (uv pip install이 venv를 인식하도록)
    source .venv/bin/activate
    
    echo "   - PyTorch 2.5.1 + CUDA 12.4 스택 설치 중..."
    
    # 1) PyTorch 2.5.1 + CUDA 12.4
    echo "   - (1/5) PyTorch 2.5.1 (CUDA 12.4) 설치..."
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    
    # 2) TRL + vLLM (핀 고정)
    echo "   - (2/5) TRL, vLLM, Transformers 설치..."
    uv pip install "trl[vllm]==0.24.0" "vllm==0.10.2" \
      "transformers>=4.56.1,<5" "accelerate>=1.9.0" "peft>=0.17.1" "datasets>=2.20.0"
    
    # 3) Unsloth (CUDA 12.4 + PyTorch 2.5.1 + Ampere)
    echo "   - (3/5) Unsloth (cu124-ampere-torch250) 설치..."
    uv pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
    
    # 4) bitsandbytes (QLoRA용)
    echo "   - (4/5) bitsandbytes 설치..."
    uv pip install "bitsandbytes>=0.46.0"
    
    echo "   - 핵심 스택 설치 완료 (PyTorch 2.5.1 + CUDA 12.4)."
else
    # 기존 .venv 재사용 시에도 추가 의존성은 동기화
    source .venv/bin/activate
fi

# 5) [선택적] pyproject.toml/requirements.txt의 나머지 의존성 설치
if [ -f "pyproject.toml" ]; then
    echo "   - (5/5) pyproject.toml의 나머지 패키지 동기화..."
    uv sync || true
elif [ -f "requirements.txt" ]; then
    echo "   - (5/5) requirements.txt 설치 중..."
    uv pip install -r requirements.txt || true
else
    echo "   - (5/5) 추가 의존성 파일 없음 (pyproject.toml, requirements.txt 모두 없음)"
fi
echo "   ✓ .venv 설정 완료!"


# 6. 하이브리드 아키텍처 설정
echo ""
echo "6️⃣  하이브리드 아키텍처 설정..."
mkdir -p /root/projects

# 기존 코드 디렉토리가 있으면 삭제 (심볼릭 링크로 교체하기 위해)
if [ -d /root/projects/Conf_Agg ] && [ ! -L /root/projects/Conf_Agg ]; then
    echo "   - 기존 복사본 제거 중..."
    rm -rf /root/projects/Conf_Agg
fi

# 심볼릭 링크 생성 또는 확인
if [ ! -L /root/projects/Conf_Agg ]; then
    ln -sf /mnt/data1/projects/Conf_Agg /root/projects/Conf_Agg
    echo "   ✓ 코드 디렉토리 심볼릭 링크 생성"
else
    echo "   ✓ 코드 디렉토리 심볼릭 링크 이미 존재"
fi


# 7. GPU 및 nvcc 확인
echo ""
echo "7️⃣  GPU 및 nvcc 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | grep "Driver Version"
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "   ✓ GPU 개수: $gpu_count (H100 Ampere)"
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
    nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "   ✓ nvcc 확인 완료"
    echo "      경로: $nvcc_path"
    echo "      버전: $nvcc_version"
else
    echo "   ⚠️  nvcc를 찾을 수 없습니다."
    if [ -n "$CUDA_HOME" ]; then
        echo "   💡 CUDA_HOME=$CUDA_HOME 이 설정되어 있지만 nvcc를 찾을 수 없습니다."
        echo "   💡 다음을 확인하세요:"
        echo "      ls -la $CUDA_HOME/bin/nvcc"
        echo "      ls -la /usr/local/cuda/bin/nvcc"
    else
        echo "   💡 CUDA Toolkit이 설치되지 않았습니다."
        echo "   💡 설치: ./scripts/install_cuda_toolkit.sh"
    fi
fi

# 8. 실행 스크립트 생성 (run.sh)
echo ""
echo "8️⃣  실행 스크립트 준비 (run.sh)..."
cat > /mnt/data1/projects/Conf_Agg/run.sh << 'RUN_EOF'
#!/bin/bash
# 프로젝트 실행 스크립트 (v4.1 - PyTorch 2.5.1 + CUDA 12.4)

# 1. .bashrc 로드 (CUDA, HF_HOME, PYTHONNOUSERSITE 등)
source ~/.bashrc 2>/dev/null || true

# CUDA 경로 재확인 및 환경변수 적용 (CUDA 12.4 우선)
if [ -d "/mnt/data1/cuda-12.4" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.4
elif [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
elif [ -d "/mnt/data1/cuda-12.8" ]; then
    export CUDA_HOME=/mnt/data1/cuda-12.8
elif [ -d "/mnt/data1/cuda" ]; then
    export CUDA_HOME=/mnt/data1/cuda
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# 2. 가상 환경 활성화
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ .venv (Python $(python -V)) 활성화됨"
else
    echo "❌ 오류: .venv/bin/activate 파일을 찾을 수 없습니다."
    echo "   /mnt/data1/projects/Conf_Agg 에서 ./restart_setup.sh 를 실행했는지 확인하세요."
    exit 1
fi

# 3. Unsloth Standby 모드 설정 (vLLM colocate 메모리 최적화)
export UNSLOTH_VLLM_STANDBY=1
echo "✅ UNSLOTH_VLLM_STANDBY=1 (메모리 최적화) 설정됨"

# 4. 프로젝트 코드 Python Path 설정
export PYTHONPATH=/root/projects/Conf_Agg

# 5. 메인 스크립트 실행
echo "🚀 프로젝트 실행: ./scripts/run_stage1_2gpu.sh"
./scripts/run_stage1_2gpu.sh
RUN_EOF
chmod +x /mnt/data1/projects/Conf_Agg/run.sh
echo "   ✓ run.sh 스크립트 생성 (PyTorch 2.5.1 + CUDA 12.4)"

# 완료 메시지
echo ""
echo "=========================================="
echo "✅ 모든 설정 완료!"
echo "PyTorch 2.5.1 + CUDA 12.4 + Unsloth"
echo "=========================================="
echo ""
echo "📋 실행 방법:"
echo ""
echo "  cd /root/projects/Conf_Agg"
echo "  ./run.sh"
echo ""
echo "🔧 무결성 체크 (수동):"
echo "  cd /root/projects/Conf_Agg"
echo "  source .venv/bin/activate"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")'"
echo "  python -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'"
echo "  python -c 'import trl; print(f\"TRL: {trl.__version__}\")'"
echo "  python -c 'import unsloth; print(f\"Unsloth: OK\")'"
echo ""