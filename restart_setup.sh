#!/bin/bash
# 서버 재시작 후 환경 복구 스크립트 (v4 - Unsloth/TRL/vLLM 스택 B 적용)
# 기존 .venv가 정상이면 재사용하여 무거운 패키지(CUDA toolkit 등) 재설치를 방지합니다.
# Python 버전 및 핵심 패키지 버전을 확인하여 필요시에만 재생성합니다.

set -e

echo "=========================================="
echo "서버 재시작 후 환경 복구 스크립트 (v4)"
echo "Unsloth + TRL(GRPO) + vLLM(colocate) 스택"
echo "=========================================="
echo ""

# 0. 사용할 Python 버전 (스택 B 기준)
PYTHON_VERSION="3.12"

# 1. 필수 패키지 설치
echo "1️⃣  필수 패키지 설치 중..."

# 빠른 미러 설정 (지역별 최적 미러 자동 선택)
echo "   - 빠른 미러 설정 중..."
if [ -f /etc/apt/sources.list ]; then
    # 백업
    cp /etc/apt/sources.list /etc/apt/sources.list.backup 2>/dev/null || true
    
    # 지역별 미러 목록 (asia-east-1 대만 지역에 최적화)
    # 대만/일본/홍콩 미러가 한국 미러보다 더 가까움
    MIRRORS=(
        "mirror.kakao.com"                # 한국 미러 (백업)
        "kr.archive.ubuntu.com"           # 한국 미러 (백업)
    )
    
    # 첫 번째 미러(카카오) 사용
    MIRROR="${MIRRORS[0]}"
    # http와 https 모두 처리 (archive.ubuntu.com, security.ubuntu.com, tw.archive.ubuntu.com 등 모든 변형 포함)
    sed -i.bak \
        -e "s|https\?://[^/]*archive\.ubuntu\.com|http://${MIRROR}|g" \
        -e "s|https\?://[^/]*security\.ubuntu\.com|http://${MIRROR}|g" \
        -e "s|https\?://[^/]*\.archive\.ubuntu\.com|http://${MIRROR}|g" \
        /etc/apt/sources.list 2>/dev/null && \
        echo "   ✓ 미러 설정: http://${MIRROR}" || \
        echo "   ⚠️  미러 설정 실패, 기본 미러 사용"
fi

# IPv4 강제 설정 (IPv6 문제 방지)
echo 'Acquire::ForceIPv4 "true";' | tee /etc/apt/apt.conf.d/99force-ipv4 > /dev/null 2>&1 || true

# 병렬 다운로드 및 속도 최적화 설정
echo 'Acquire::http::MaxParallelDownloads "20";' | tee /etc/apt/apt.conf.d/99parallel > /dev/null 2>&1 || true
echo 'Acquire::Queue-Mode "access";' | tee -a /etc/apt/apt.conf.d/99parallel > /dev/null 2>&1 || true
echo 'Acquire::http::Timeout "30";' | tee -a /etc/apt/apt.conf.d/99parallel > /dev/null 2>&1 || true
echo 'Acquire::http::Pipeline-Depth "5";' | tee -a /etc/apt/apt.conf.d/99parallel > /dev/null 2>&1 || true

apt-get update -qq
apt-get install -y build-essential curl wget git python3 htop || {
    echo "   ⚠️  패키지 설치 중 오류 발생, 재시도 중..."
    apt-get install -y build-essential curl wget git python3 htop
}
echo "   ✓ 패키지 설치 완료"

# curl 또는 wget 사용 가능 여부 확인
if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
    echo "   ❌ 오류: curl과 wget이 모두 설치되지 않았습니다"
    echo "   💡 수동 설치: apt-get install -y curl wget"
    exit 1
fi

# 2. uv 설치
echo ""
echo "2️⃣  uv 설치 중..."
export PATH="$HOME/.local/bin:$PATH"

# uv가 없으면 설치
if ! command -v uv &> /dev/null; then
    echo "   - uv 설치 중..."
    
    # curl 또는 wget을 사용하여 uv 설치
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &> /dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "   ❌ 오류: curl 또는 wget이 필요합니다"
        exit 1
    fi
    
    export PATH="$HOME/.local/bin:$PATH"
    
    # 설치 후 다시 확인
    if ! command -v uv &> /dev/null; then
        echo "   ❌ 오류: uv 설치 후에도 명령어를 찾을 수 없습니다"
        if command -v curl &> /dev/null; then
            echo "   💡 수동 설치 시도: curl -LsSf https://astral.sh/uv/install.sh | sh"
        elif command -v wget &> /dev/null; then
            echo "   💡 수동 설치 시도: wget -qO- https://astral.sh/uv/install.sh | sh"
        fi
        exit 1
    fi
    echo "   ✓ uv 설치 완료"
else
    echo "   ✓ uv가 이미 설치되어 있습니다"
fi

# uv 버전 확인 (설치 확인용)
UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "   ✓ uv 버전: $UV_VERSION"

# 3. 영구 환경 변수 설정 (.bashrc)
# (변경 없음, 기존 CUDA 설정 등 그대로 사용)
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
# .bashrc 로드 시도 (실패해도 계속 진행)
source ~/.bashrc 2>/dev/null || true

# 3-1. CUDA 경로 확인 및 심볼릭 링크 설정
echo ""
echo "3-1️⃣  CUDA 경로 확인 및 심볼릭 링크 설정..."
CUDA_HOME=""
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
    echo "   ⚠️  CUDA Toolkit을 찾을 수 없습니다"
    echo "   💡 CUDA Toolkit이 필요하면 다음을 실행하세요:"
    echo "      ./scripts/install_cuda_toolkit.sh"
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
        
        # 버전별 링크도 생성 (cuda-12.8)
        if [[ "$CUDA_HOME" == *cuda-12.8* ]] && [ ! -e "/usr/local/cuda-12.8" ]; then
            ln -sf "$CUDA_HOME" /usr/local/cuda-12.8 2>/dev/null || true
            echo "   ✓ 심볼릭 링크 생성: /usr/local/cuda-12.8 -> $CUDA_HOME"
        fi
    fi
    echo "   ✓ CUDA 환경 변수 설정 완료: CUDA_HOME=$CUDA_HOME"
    echo "   ✓ 현재 셸에 환경 변수 적용됨 (PATH에 $CUDA_HOME/bin 추가됨)"
    
    # nvcc 확인
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "   ✓ nvcc 확인: $CUDA_HOME/bin/nvcc"
    else
        echo "   ⚠️  경고: $CUDA_HOME/bin/nvcc 파일이 없습니다"
    fi
fi

# 4. 디렉토리 생성
# (변경 없음)
echo ""
echo "4️⃣  필수 디렉토리 생성..."
mkdir -p /mnt/data1/{.uv-cache,models/nlp/{huggingface_cache,conf_agg},datasets/nlp/{cache,conf_agg/{outputs,logs,generated,curated,benchmarks}}}
echo "   ✓ 디렉토리 생성 완료"

# 5. .venv 재생성 또는 재사용 (변경 - 스택 B 적용, 효율적 재사용)
echo ""
echo "5️⃣  Persistent .venv 확인 및 설정 (Stack B 적용)..."
cd /mnt/data1/projects/Conf_Agg

# 기존 .venv가 있는지 확인
if [ -d ".venv" ]; then
    echo "   ✓ 기존 .venv 발견, 재사용합니다"
    
    # Python 인터프리터가 실제로 존재하는지만 확인 (최소 검증)
    if [ -f ".venv/bin/python" ] && .venv/bin/python --version >/dev/null 2>&1; then
        source .venv/bin/activate 2>/dev/null || true
        echo "   ✓ 기존 .venv 재사용 (무거운 패키지 재설치 생략)"
        NEED_RECREATE=false
    else
        echo "   ⚠️  .venv의 Python 인터프리터가 작동하지 않음, Python만 재설치 중..."
        # Python만 재설치 (.venv는 그대로 사용)
        export PATH="$HOME/.local/bin:$PATH"
        if command -v uv &> /dev/null; then
            uv python install $PYTHON_VERSION 2>&1 | grep -E "(Installed|already)" || true
        fi
        # Python 재설치 후 .venv 활성화
        source .venv/bin/activate 2>/dev/null || true
        echo "   ✓ Python 재설치 완료, 기존 .venv 재사용"
        NEED_RECREATE=false
    fi
else
    echo "   - 기존 .venv 없음, 새로 생성 필요"
    NEED_RECREATE=true
fi

# 필요시 새로 생성
if [ "$NEED_RECREATE" = true ]; then
    echo "   - Python $PYTHON_VERSION 으로 새 .venv 생성 중..."
    
    # uv 명령어 사용 가능 여부 확인
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "   ❌ 오류: uv 명령어를 찾을 수 없습니다"
        echo "   💡 uv를 다시 설치하세요: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    # Python 3.12가 없으면 uv가 자동으로 다운로드하도록 보장
    # uv python install을 먼저 시도하여 Python 3.12를 확보
    if ! uv python find $PYTHON_VERSION &>/dev/null; then
        echo "   - Python $PYTHON_VERSION이 없어서 uv를 통해 다운로드 중..."
        uv python install $PYTHON_VERSION || {
            echo "   ⚠️  Python $PYTHON_VERSION 다운로드 실패, 시스템 Python 사용 시도..."
        }
    fi
    
    # .venv 생성 (Python 버전 명시)
    uv venv --python $PYTHON_VERSION || {
        echo "   ❌ 오류: Python $PYTHON_VERSION로 .venv 생성 실패"
        echo "   💡 Python 3.12가 필요합니다. 다음을 실행하세요:"
        echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "      uv python install 3.12"
        exit 1
    }
    
    # .venv 활성화 (uv pip install이 venv를 인식하도록)
    source .venv/bin/activate
    
    # Python 버전 재확인
    ACTUAL_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    if [ "$ACTUAL_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
        echo "   ❌ 오류: .venv의 Python 버전이 예상과 다릅니다 (실제: $ACTUAL_PYTHON_VERSION, 예상: $PYTHON_VERSION)"
        echo "   💡 .venv를 삭제하고 다시 실행하세요: rm -rf .venv && ./restart_setup.sh"
        exit 1
    fi
    echo "   ✓ Python $ACTUAL_PYTHON_VERSION 확인됨"
    
    echo "   - (Stack B) 핵심 스택 설치 중..."
    
    # uv 명령어 사용 전 PATH 재확인
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "   ❌ 오류: uv 명령어를 찾을 수 없습니다"
        exit 1
    fi
    
    # 1) PyTorch 2.8.0 + CUDA 12.8
    echo "   - (1/5) PyTorch 2.8.0+cu128 설치..."
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
    
    # 3) Unsloth
    echo "   - (3/5) Unsloth 설치..."
    uv pip install unsloth

    # 2) TRL + vLLM (핀 고정)
    echo "   - (2/5) TRL, vLLM, Transformers 설치..."
    uv pip install "trl[vllm]==0.23.0" "vllm==0.10.2" \
      "transformers>=4.56.1,<5" "accelerate>=1.9.0" "peft>=0.17.1" "datasets>=2.20.0"
    
    # 4) bitsandbytes (QLoRA용)
    echo "   - (4/5) bitsandbytes 설치..."
    uv pip install "bitsandbytes>=0.46.0"
    
    echo "   - (Stack B) 핵심 스택 설치 완료."
else
    # 기존 .venv 재사용 시에도 추가 의존성은 동기화
    source .venv/bin/activate
fi

# 5) [선택적] pyproject.toml/requirements.txt의 나머지 의존성 설치
# uv 명령어 사용 전 PATH 재확인
export PATH="$HOME/.local/bin:$PATH"
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
# (변경) 코드를 복사가 아닌 심볼릭 링크로 사용하여 데이터 동기화 보장
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
# (변경) CUDA 환경변수 적용 후 확인
echo ""
echo "7️⃣  GPU 및 nvcc 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | grep "Driver Version"
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "   ✓ GPU 개수: $gpu_count (Driver R570+ 권장)"
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

# 8. 실행 스크립트 생성 (run.sh) (변경 - Standby 모드 추가)
echo ""
echo "8️⃣  실행 스크립트 준비 (run.sh)..."
cat > /mnt/data1/projects/Conf_Agg/run.sh << 'RUN_EOF'
#!/bin/bash
# 프로젝트 실행 스크립트 (v4 - Unsloth Standby 적용)

# 1. .bashrc 로드 (CUDA, HF_HOME, PYTHONNOUSERSITE 등)
source ~/.bashrc 2>/dev/null || true

# CUDA 경로 재확인 및 환경변수 적용 (Persistent Storage 우선)
if [ -d "/mnt/data1/cuda-12.8" ]; then
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

# 3. (변경) Unsloth Standby 모드 설정 (vLLM colocate 메모리 최적화)
# Unsloth import 이전에 설정해야 합니다.
export UNSLOTH_VLLM_STANDBY=1
echo "✅ UNSLOTH_VLLM_STANDBY=1 (메모리 최적화) 설정됨"

# 4. 프로젝트 코드 Python Path 설정
export PYTHONPATH=/root/projects/Conf_Agg

# 5. 메인 스크립트 실행
echo "🚀 프로젝트 실행: ./scripts/run_stage1_2gpu.sh"
./scripts/run_stage1_2gpu.sh
RUN_EOF
chmod +x /mnt/data1/projects/Conf_Agg/run.sh
echo "   ✓ run.sh 스크립트 생성 (Unsloth Standby 모드 적용)"

# 완료 메시지
echo ""
echo "=========================================="
echo "✅ 모든 설정 완료! (Stack B)"
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
echo "  python -c 'import torch; print(f\"Torch: {torch.__version__}\")'"
echo "  python -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'"
echo "  python -c 'import trl; print(f\"TRL: {trl.__version__}\")'"
echo ""
echo "🔧 CUDA 환경변수 적용 (현재 셸에서 nvcc를 사용하려면):"
echo "  source ~/.bashrc"
echo "  # 또는"
echo "  source ./setup_cuda_env.sh"
echo "  nvcc --version"
echo ""
