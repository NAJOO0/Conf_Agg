# Conf_Agg 배포 가이드

다른 서버에서 Docker를 사용하여 Conf_Agg 프로젝트를 실행하는 완전한 가이드입니다.

## 📋 목차
1. [원본 서버에서 코드 업로드](#1-원본-서버에서-코드-업로드)
2. [새 서버 환경 준비](#2-새-서버-환경-준비)
3. [Docker 컨테이너 설정](#3-docker-컨테이너-설정)
4. [프로젝트 실행](#4-프로젝트-실행)
5. [문제 해결](#5-문제-해결)

---

## 1. 원본 서버에서 코드 업로드

### 1.1 Git 저장소에 코드 업로드 (GitHub/GitLab 등)

```bash
# 현재 서버에서
cd /home/najoo0/Conf_Agg

# Git 저장소 초기화 (아직 안 했다면)
git init
git remote add origin <your-repository-url>

# 모든 파일 추가 (단, .gitignore에 있는 파일들은 제외됨)
git add .

# 커밋
git commit -m "Initial commit: Conf_Agg project"

# 원격 저장소에 push
git push -u origin main
```

**⚠️ 중요**: `.gitignore`에 의해 제외되는 파일들:
- `data/raw/*`, `data/generated/*` - 데이터 파일들
- `outputs/*` - 출력 파일들
- `.env` - 환경 변수 파일
- `venv/` - 가상환경

필요한 경우 별도로 업로드하세요:
```bash
# 데이터 파일을 별도로 업로드하려면
rsync -avz data/raw/ server2:/path/to/data/raw/
```

### 1.2 Git에 push하지 않는 경우 (직접 전송)

```bash
# rsync로 직접 복사
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude 'outputs' \
  /home/najoo0/Conf_Agg/ user@new-server:/path/to/Conf_Agg/

# 또는 scp 사용
scp -r Conf_Agg/ user@new-server:/path/to/
```

---

## 2. 새 서버 환경 준비

### 2.1 필요한 소프트웨어 설치

#### Docker 설치
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# NVIDIA Container Toolkit 설치 (GPU 사용 시 필수)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Docker 권한 설정 (sudo 없이 사용하려면)
sudo usermod -aG docker $USER
newgrp docker
```

#### Git 설치
```bash
sudo apt-get update
sudo apt-get install -y git
```

### 2.2 GPU 확인

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# Docker에서 GPU 접근 테스트
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

---

## 3. Docker 컨테이너 설정

### 3.1 코드 클론 및 이동

```bash
# 작업 디렉토리로 이동
cd ~

# Git으로 클론 (방법 1)
git clone <your-repository-url>
cd Conf_Agg

# 또는 직접 복사된 경우 (방법 2)
cd /path/to/Conf_Agg
```

### 3.2 환경 변수 설정

```bash
# .env 파일 생성
cp env.example .env

# .env 파일 편집
nano .env

# 필수 환경 변수 설정
# WANDB_API_KEY=your-wandb-api-key
# SAMPLE_LIMIT=400
```

### 3.3 Docker 컨테이너 빌드

```bash
# docker-compose.yml 수정 (필요한 경우)
# volumes 경로를 새 서버의 경로에 맞게 수정
nano docker-compose.yml

# 주요 수정 사항:
# - /home/najoo0/Conf_Agg -> 현재 서버의 Conf_Agg 경로
# - /data1, /data2 -> 새 서버의 디렉토리 경로

# Docker 이미지 빌드
docker-compose build
```

#### docker-compose.yml 수정 예시

```yaml
services:
  conf-agg-llm:
    build: .
    container_name: conf-agg-llm
    runtime: nvidia
    shm_size: "16g"
    working_dir: /workspace
    stdin_open: true
    tty: true
    volumes:
      - /home/YOUR_USERNAME/Conf_Agg:/workspace  # ← 수정 필요
      - /your/data1:/data1                        # ← 수정 필요
      - /your/data2:/data2                        # ← 수정 필요
      - ~/.cache/huggingface:/root/.cache/huggingface
      - uv-cache:/tmp/uv-cache
    environment:
      - DEBIAN_FRONTEND=noninteractive
      - TZ=Asia/Seoul
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU 개수에 맞게 수정
      - PYTHONPATH=/workspace
      - WANDB_API_KEY=${WANDB_API_KEY}
```

### 3.4 컨테이너 실행

```bash
# 컨테이너 시작 (백그라운드)
docker-compose up -d

# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 컨테이너 안에서 GPU 확인
nvidia-smi
```

---

## 4. 프로젝트 실행

### 4.1 uv 의존성 설치

컨테이너 안에서:
```bash
# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# uv 버전 확인
uv --version

# 의존성 설치 (uv.lock 파일 기반)
uv sync

# 설치 확인
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import vllm; print(vllm.__version__)"
```

### 4.2 데이터 준비

컨테이너 안에서:
```bash
# 데이터 디렉토리 확인
ls -la data/raw/

# 데이터 파일 확인
head data/raw/deepscaler.jsonl

# 만약 데이터 파일이 없다면, 호스트에서 마운트 확인
```

호스트에서 데이터 마운트 또는 복사:
```bash
# 호스트에서 실행
# deepscaler.jsonl 파일을 확인하고 없으면 다운로드하거나 복사
ls -la /path/to/your/data/deepscaler.jsonl

# docker-compose.yml의 volumes 설정 확인
docker-compose down
nano docker-compose.yml  # volumes 경로 수정
docker-compose up -d
```

### 4.3 Stage 1 실행

컨테이너 안에서:
```bash
# uv 환경 활성화
source .venv/bin/activate

# Stage 1 실행 (간단한 방법)
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1

# 또는 백그라운드 스크립트 사용 (4개 GPU 전체)
uv run bash scripts/run_stage1_background.sh
```

### 4.4 실행 중 모니터링

컨테이너 안에서:
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 로그 파일 확인
tail -f outputs/logs/sample_400/stage1_shard_0.log

# 모든 샤드 로그 확인
tail -f outputs/logs/sample_400/stage1_shard_*.log

# 프로세스 확인
ps aux | grep stage1_generate.py
```

호스트에서:
```bash
# 컨테이너 로그 확인
docker-compose logs -f conf-agg-llm

# GPU 상태 확인
nvidia-smi
```

### 4.5 결과 확인

컨테이너 안에서:
```bash
# 생성된 파일 확인
ls -lh data/generated/sample_400/

# Parquet 파일 확인
uv run python -c "import pandas as pd; df = pd.read_parquet('data/generated/sample_400/raw_generated_shard_0.parquet'); print(len(df)); print(df.head())"
```

---

## 5. 문제 해결

### 5.1 GPU 메모리 부족

**증상**: CUDA out of memory 에러

**해결 방법**:
```bash
# config/data/raw_dataset.yaml 수정
nano config/data/raw_dataset.yaml

# gpu_memory_utilization 값 줄이기
# 0.9 -> 0.7 또는 0.8
```

### 5.2 모델 다운로드 실패

**증상**: 403 Forbidden 또는 네트워크 에러

**해결 방법**:
```bash
# Hugging Face 토큰 설정
huggingface-cli login

# 또는 환경 변수로 설정
export HF_TOKEN=your_token_here
```

### 5.3 uv sync 실패

**증상**: uv.lock 충돌 또는 의존성 해결 실패

**해결 방법**:
```bash
# uv.lock 재생성
rm uv.lock
uv sync

# 또는 pyproject.toml만으로 설치
uv pip install -e .
```

### 5.4 컨테이너가 계속 재시작됨

**증상**: 컨테이너가 시작되다가 멈춤

**해결 방법**:
```bash
# 컨테이너 로그 확인
docker-compose logs conf-agg-llm

# 컨테이너 재빌드
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 5.5 데이터 파일을 찾을 수 없음

**증상**: FileNotFoundError: data/raw/deepscaler.jsonl

**해결 방법**:
```bash
# 데이터 파일 위치 확인
find . -name "deepscaler.jsonl"

# 없으면 다운로드하거나 복사
# 또는 docker-compose.yml의 volumes 설정 확인
docker-compose down
nano docker-compose.yml
docker-compose up -d
```

---

## 6. 빠른 참조

### 주요 명령어

```bash
# 컨테이너 시작/중지
docker-compose up -d
docker-compose down

# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 로그 확인
docker-compose logs -f conf-agg-llm

# GPU 사용률 확인
nvidia-smi
watch -n 1 nvidia-smi

# 프로세스 확인
docker ps
ps aux | grep python

# 디스크 사용량 확인
df -h
du -sh data/
du -sh outputs/
```

### 주요 디렉토리

```
/workspace/ (컨테이너 내부)
├── config/          # 설정 파일
├── scripts/         # 실행 스크립트
├── src/            # 소스 코드
├── data/           # 데이터
│   ├── raw/        # 원본 데이터
│   └── generated/  # 생성된 데이터
└── outputs/        # 출력 (로그, 모델, 결과)
```

---

## 7. 추가 최적화

### 7.1 멀티 GPU 설정

4개 GPU를 사용하려면 `docker-compose.yml`에서:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
```

그리고 스크립트 실행 시:
```bash
# 4개 GPU로 분산 실행
uv run bash scripts/run_stage1_background.sh
```

### 7.2 성능 모니터링

```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 컨테이너 리소스 사용량
docker stats conf-agg-llm

# 디스크 I/O 모니터링
iostat -x 1
```

### 7.3 백업 및 복구

```bash
# 결과 데이터 백업
tar -czf conf_agg_backup_$(date +%Y%m%d).tar.gz data/generated/ outputs/logs/

# 모델 백업
tar -czf models_backup_$(date +%Y%m%d).tar.gz outputs/models/

# 복원
tar -xzf conf_agg_backup_20240101.tar.gz
```

---

## 8. 체크리스트

배포 전 확인 사항:

- [ ] Docker 설치 완료
- [ ] NVIDIA Container Toolkit 설치 완료
- [ ] Git으로 코드 클론 완료
- [ ] .env 파일 설정 완료
- [ ] docker-compose.yml volumes 경로 수정 완료
- [ ] 데이터 파일 준비 완료
- [ ] GPU 접근 확인 완료
- [ ] 컨테이너 빌드 및 실행 완료
- [ ] uv sync 완료
- [ ] 모델 로드 테스트 완료
- [ ] 실제 실행 테스트 완료

---

## 문의 및 지원

문제가 발생하면:
1. 로그 파일 확인 (`outputs/logs/`)
2. Docker 로그 확인 (`docker-compose logs`)
3. GPU 상태 확인 (`nvidia-smi`)
4. GitHub Issues에 리포트
