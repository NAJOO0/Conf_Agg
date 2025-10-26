# 🚀 Conf_Agg 배포 가이드 (한국어)

다른 서버에서 Docker를 사용하여 프로젝트를 실행하는 **완전한 단계별 가이드**입니다.

## 📌 전체 프로세스 개요

```
원본 서버 → Git Push → 새 서버 → Git Clone → Docker 빌드 → uv sync → 실행
```

---

## 1️⃣ 원본 서버에서 코드 업로드

### Step 1: Git 저장소에 코드 Push

현재 서버(`/home/najoo0/Conf_Agg`)에서:

```bash
cd /home/najoo0/Conf_Agg

# Git 초기화 (아직 안 했다면)
git init

# 원격 저장소 추가
git remote add origin https://github.com/YOUR_USERNAME/Conf_Agg.git
# 또는 GitLab 등 다른 Git 호스팅 서비스

# 모든 파일 추가 (data/, outputs/ 등은 .gitignore로 제외됨)
git add .

# 커밋
git commit -m "Conf_Agg 프로젝트 초기 커밋"

# Push
git push -u origin main
```

### Step 2: 데이터 파일 별도 전송

`.gitignore`에 의해 다음 파일들은 Git에 포함되지 않습니다:
- `data/raw/deepscaler.jsonl` - 원본 데이터
- `outputs/` - 출력 파일들
- `.env` - 환경 변수

별도로 전송해야 합니다:

```bash
# rsync를 사용한 방법 (새 서버와 직접 연결 시)
rsync -avz data/raw/deepscaler.jsonl user@new-server:/path/to/Conf_Agg/data/raw/

# 또는 scp 사용
scp data/raw/deepscaler.jsonl user@new-server:/path/to/Conf_Agg/data/raw/
```

---

## 2️⃣ 새 서버 준비

### Step 1: Docker 설치

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 사용 가능)
sudo usermod -aG docker $USER

# 그룹 적용
newgrp docker

# Docker 버전 확인
docker --version
```

### Step 2: NVIDIA Container Toolkit 설치 (GPU 사용 시)

```bash
# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 서비스 재시작
sudo systemctl restart docker

# GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Step 3: Git 설치

```bash
sudo apt-get update
sudo apt-get install -y git
```

### Step 4: GPU 확인

```bash
nvidia-smi  # GPU 목록과 상태 확인
```

---

## 3️⃣ 코드 클론 및 이동

### 방법 1: Git으로 클론

```bash
# 클론
git clone https://github.com/YOUR_USERNAME/Conf_Agg.git
cd Conf_Agg
```

### 방법 2: 직접 복사 (Git 없이)

rsync나 scp를 사용하여 직접 복사:

```bash
# rsync 사용
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude 'outputs' \
  user@old-server:/home/najoo0/Conf_Agg/ \
  /path/to/Conf_Agg/

cd /path/to/Conf_Agg
```

---

## 4️⃣ Docker 컨테이너 설정

### Step 1: 환경 변수 설정

```bash
# .env 파일 생성
cp env.example .env
nano .env  # 또는 원하는 에디터 사용

# 필수 설정:
# WANDB_API_KEY=your_wandb_api_key
# SAMPLE_LIMIT=400
```

### Step 2: docker-compose.yml 수정

```bash
nano docker-compose.yml
```

다음 항목을 새 서버 환경에 맞게 수정:

```yaml
services:
  conf-agg-llm:
    volumes:
      # 현재 서버의 Conf_Agg 경로로 변경
      - /home/YOUR_USERNAME/Conf_Agg:/workspace
      # 데이터 디렉토리 경로 변경 (필요시)
      - /data1:/data1
      - /data2:/data2
      # Hugging Face 캐시 경로
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      # GPU 개수에 맞게 수정 (1개 GPU면 0만, 4개면 0,1,2,3)
      - CUDA_VISIBLE_DEVICES=0  # 또는 0,1,2,3
```

### Step 3: 데이터 파일 확인

```bash
# deepscaler.jsonl 파일 확인
ls -lh data/raw/deepscaler.jsonl

# 없으면 별도로 전송
# 또는 docker-compose.yml의 volumes에 데이터 디렉토리 추가
```

---

## 5️⃣ Docker 컨테이너 빌드 및 실행

### 방법 1: 자동 배포 스크립트 사용 (권장)

```bash
# 빠른 배포 스크립트 실행
bash scripts/quick_deploy.sh
```

이 스크립트가 자동으로 Docker 빌드, 실행, uv sync까지 모두 처리합니다.

### 방법 2: 수동 설정

```bash
# Docker 이미지 빌드
docker-compose build

# 컨테이너 시작
docker-compose up -d

# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 접속
docker-compose exec conf-agg-llm bash
```

---

## 6️⃣ 컨테이너 안에서 uv sync 실행

```bash
# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 컨테이너 안에서:

# GPU 확인
nvidia-smi

# uv 버전 확인
uv --version

# 의존성 설치 (이 과정은 시간이 걸립니다)
uv sync

# 설치 확인
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import vllm; print(vllm.__version__)"
```

---

## 7️⃣ Stage 1 실행

### 컨테이너 안에서 실행

```bash
# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 컨테이너 안에서:

# 방법 1: 단일 GPU로 실행
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1

# 방법 2: 4개 GPU로 병렬 실행 (백그라운드)
uv run bash scripts/run_stage1_background.sh
```

### 모니터링

컨테이너 안에서:

```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# 로그 확인
tail -f outputs/logs/sample_400/stage1_shard_0.log

# 모든 샤드 로그
tail -f outputs/logs/sample_400/stage1_shard_*.log

# 프로세스 확인
ps aux | grep stage1_generate
```

호스트에서:

```bash
# 컨테이너 로그
docker-compose logs -f conf-agg-llm

# GPU 상태
nvidia-smi
```

---

## 8️⃣ 결과 확인

```bash
# 컨테이너 안에서
docker-compose exec conf-agg-llm bash

# 생성된 파일 확인
ls -lh data/generated/sample_400/

# Parquet 파일 확인
uv run python -c "import pandas as pd; df = pd.read_parquet('data/generated/sample_400/raw_generated_shard_0.parquet'); print(f'총 {len(df)}개 응답'); print(df.head())"

# 디스크 사용량 확인
du -sh data/generated/
du -sh outputs/logs/
```

---

## 🔧 문제 해결

### 문제 1: Docker 권한 에러

```bash
# 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker

# 또는 sudo 사용
sudo docker-compose up -d
```

### 문제 2: GPU 인식 안 됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# NVIDIA Container Toolkit 재설치
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker

# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 문제 3: uv sync 실패

```bash
# 컨테이너 안에서
docker-compose exec conf-agg-llm bash

# uv.lock 재생성
rm uv.lock
uv sync

# 또는 pip로 직접 설치
uv pip install -e .
```

### 문제 4: GPU 메모리 부족

```bash
# config/data/raw_dataset.yaml 수정
nano config/data/raw_dataset.yaml

# gpu_memory_utilization 값을 낮춤 (0.9 → 0.7)
```

### 문제 5: 데이터 파일을 찾을 수 없음

```bash
# 파일 위치 확인
find . -name "deepscaler.jsonl"

# docker-compose.yml의 volumes 경로 확인
nano docker-compose.yml

# 컨테이너 재시작
docker-compose down
docker-compose up -d
```

---

## 📊 유용한 명령어 모음

```bash
# === Docker 관리 ===
docker-compose up -d          # 컨테이너 시작 (백그라운드)
docker-compose down           # 컨테이너 중지
docker-compose restart        # 컨테이너 재시작
docker-compose ps             # 컨테이너 상태 확인
docker-compose logs -f conf-agg-llm  # 로그 확인

# === 컨테이너 접속 ===
docker-compose exec conf-agg-llm bash  # 컨테이너 접속

# === GPU 모니터링 ===
nvidia-smi                     # GPU 상태
watch -n 1 nvidia-smi         # 실시간 GPU 모니터링
docker stats conf-agg-llm     # 컨테이너 리소스 사용량

# === 파일 확인 ===
ls -lh data/generated/        # 생성된 파일
tail -f outputs/logs/sample_400/stage1_shard_0.log  # 로그 확인

# === 디스크 사용량 ===
df -h                         # 전체 디스크 사용량
du -sh data/ outputs/         # 특정 디렉토리 사용량

# === 프로세스 확인 ===
docker ps                     # 컨테이너 프로세스
ps aux | grep python          # Python 프로세스
```

---

## ✅ 체크리스트

배포 전 확인:

#### 원본 서버에서
- [ ] Git에 코드 Push 완료
- [ ] .env 파일에 민감한 정보 없음
- [ ] data/raw/deepscaler.jsonl 확인

#### 새 서버에서
**사전 준비**
- [ ] Docker 설치
- [ ] NVIDIA Container Toolkit 설치 (GPU)
- [ ] GPU 드라이버 설치
- [ ] Git 설치

**환경 설정**
- [ ] Git 클론 또는 코드 복사
- [ ] .env 파일 생성 및 설정
- [ ] docker-compose.yml 수정
- [ ] 데이터 파일 준비

**실행 준비**
- [ ] Docker 이미지 빌드
- [ ] 컨테이너 시작
- [ ] 컨테이너 접속 확인
- [ ] uv sync 완료
- [ ] GPU 인식 확인

---

## 🎯 다음 단계

Stage 1이 완료되면:

```bash
# 결과 병합 (여러 샤드가 있는 경우)
uv run python scripts/merge_shards.py

# Stage 2: 데이터 큐레이션
uv run python scripts/stage2_curate.py

# Stage 3: 모델 훈련
uv run python scripts/stage3_train.py

# Stage 4: 벤치마크 평가
uv run python scripts/stage4_evaluate.py
```

---

## 📞 추가 지원

- **완전한 가이드**: [docs/DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **빠른 시작**: [docs/QUICKSTART.md](QUICKSTART.md)
- **문제 해결**: GitHub Issues 등록

---

## 💡 요약

1. **원본 서버**: `git push`로 코드 업로드
2. **새 서버**: Docker, NVIDIA Toolkit 설치
3. **코드 클론**: `git clone` 또는 직접 복사
4. **환경 설정**: `.env`, `docker-compose.yml` 수정
5. **빌드 실행**: `docker-compose build && docker-compose up -d`
6. **컨테이너 접속**: `docker-compose exec conf-agg-llm bash`
7. **의존성 설치**: `uv sync`
8. **실행**: Stage 1 스크립트 실행
