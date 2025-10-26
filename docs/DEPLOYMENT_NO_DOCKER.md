# Docker 없이 서버에 배포하기

Docker 없이 직접 Python 환경에서 실행하는 완전한 가이드입니다.

---

## 1. 환경 확인

```bash
# 현재 사용자
whoami

# Sudo 권한 확인
sudo -v 2>/dev/null && echo "✅ Sudo 권한 있음" || echo "❌ Sudo 권한 없음"

# 필수 도구 확인
python3 --version
git --version
curl --version
```

---

## 2. sudo 권한이 있는 경우 (권장: uv 사용)

### 2-1. 빌드 도구 설치

```bash
# 시스템 업데이트
sudo apt-get update

# 필수 빌드 도구 설치
sudo apt-get install -y build-essential curl wget git
sudo apt-get install -y python3-dev python3.10-dev python3.12-dev

# 확인
gcc --version
python3 --version
```

### 2-2. uv 설치

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 환경 변수 설정
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 확인
uv --version
```

### 2-3. 코드 다운로드

```bash
git clone https://github.com/NAJOO0/Conf_Agg.git
cd Conf_Agg
```

### 2-4. 의존성 설치

```bash
# uv sync로 모든 의존성 설치
uv sync

# 설치 확인
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import vllm; print(vllm.__version__)"
```

### 2-5. 실행

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
export SAMPLE_LIMIT=400

uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1
```

---

## 3. sudo 권한이 없는 경우 (conda 사용)

### 3-1. Miniconda 설치

```bash
# Miniconda 다운로드 및 설치 (사용자 홈 디렉토리)
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 환경 변수 설정
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 확인
conda --version
python --version
```

### 3-2. Python 3.12 환경 생성

```bash
# Python 3.12 환경 생성
conda create -n conf_agg python=3.12 -y

# 환경 활성화
conda activate conf_agg

# 확인
python --version
```

### 3-3. 코드 다운로드

```bash
git clone https://github.com/NAJOO0/Conf_Agg.git
cd Conf_Agg
```

### 3-4. 패키지 설치

```bash
# conda 환경 활성화
conda activate conf_agg

# 필수 패키지 설치
pip install torch torchvision transformers vllm
pip install pandas numpy scipy hydra-core wandb tqdm
pip install accelerate peft trl datasets tokenizers safetensors

# 설치 확인
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```

### 3-5. 실행

```bash
# conda 환경 활성화
conda activate conf_agg

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
export SAMPLE_LIMIT=400

# 실행
python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1
```

---

## 4. 모니터링

### 실행 상태 확인

```bash
# GPU 사용률
nvidia-smi
watch -n 1 nvidia-smi

# 로그 확인 (uv 사용 시)
tail -f outputs/logs/sample_400/stage1_shard_0.log

# conda 사용 시
tail -f outputs/logs/sample_400/stage1_shard_0.log

# 프로세스 확인
ps aux | grep stage1_generate
```

### 결과 확인

```bash
# 생성된 파일 확인
ls -lh data/generated/

# Parquet 파일 확인 (uv)
uv run python -c "import pandas as pd; df = pd.read_parquet('data/generated/sample_400/raw_generated_shard_0.parquet'); print(f'총 {len(df)}개')"

# Parquet 파일 확인 (conda)
python -c "import pandas as pd; df = pd.read_parquet('data/generated/sample_400/raw_generated_shard_0.parquet'); print(f'총 {len(df)}개')"
```

---

## 5. 문제 해결

### GPU 인식 안 됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 확인
nvcc --version

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### ModuleNotFoundError

**uv 사용 시:**
```bash
uv sync
```

**conda 사용 시:**
```bash
pip install <missing_package>
```

### 메모리 부족

```bash
# config/data/raw_dataset.yaml 수정
nano config/data/raw_dataset.yaml

# gpu_memory_utilization 값 낮추기: 0.9 → 0.7
```

### Python 버전 문제

```bash
# Python 3.12 강제 사용 (conda)
conda install python=3.12 -y

# 또는 (uv)
uv run python3.12 scripts/stage1_generate.py ...
```

---

## 6. 전체 프로세스 요약

### A. sudo 권한 있음 (권장)

```bash
# 1. 빌드 도구 설치
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3.12-dev

# 2. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. 코드 Clone
git clone https://github.com/NAJOO0/Conf_Agg.git
cd Conf_Agg

# 4. 의존성 설치
uv sync

# 5. 실행
export CUDA_VISIBLE_DEVICES=0
uv run python scripts/stage1_generate.py --config-path config --config-name config --gpu-id "0" --shard-id 0 --total-shards 1
```

### B. sudo 권한 없음 (conda)

```bash
# 1. Miniconda 설치
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# 2. 환경 생성
conda create -n conf_agg python=3.12 -y
conda activate conf_agg

# 3. 코드 Clone
git clone https://github.com/NAJOO0/Conf_Agg.git
cd Conf_Agg

# 4. 패키지 설치
pip install torch transformers vllm

# 5. 실행
export CUDA_VISIBLE_DEVICES=0
python scripts/stage1_generate.py --config-path config --config-name config --gpu-id "0" --shard-id 0 --total-shards 1
```

---

## 7. 체크리스트

- [ ] sudo 권한 확인
- [ ] curl, git, python3 설치 확인
- [ ] 빌드 도구 설치 (sudo 있는 경우)
- [ ] uv 설치 또는 conda 설치
- [ ] 코드 다운로드 (Git clone)
- [ ] 의존성 설치 (uv sync 또는 pip install)
- [ ] GPU 인식 확인
- [ ] 데이터 파일 준비
- [ ] 실행 테스트

---

## 8. 핵심 차이점

| 항목 | uv (sudo 필요) | conda (sudo 불필요) |
|------|----------------|---------------------|
| 빌드 도구 | 필요 ✅ | 불필요 ❌ |
| Python 관리 | uv.lock | conda 환경 |
| 패키지 설치 | uv sync | pip install |
| 실행 명령 | `uv run python` | `python` |
| 재현성 | 높음 ✅ | 낮음 ⚠️ |
| 프로젝트 일관성 | 유지 ✅ | 분기 ⚠️ |

**권장**: sudo 권한이 있다면 uv를 사용하세요!

---

이 가이드로 Docker 없이도 프로젝트를 실행할 수 있습니다!