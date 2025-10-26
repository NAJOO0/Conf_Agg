# ⚡ 빠른 시작 가이드

다른 서버에서 Conf_Agg를 빠르게 실행하는 방법입니다.

## 📝 전체 프로세스 요약

### 1단계: 원본 서버에서 Git에 Push

```bash
# 원본 서버에서
cd /home/najoo0/Conf_Agg

# Git 저장소 확인
git status

# 변경사항 커밋 및 Push
git add .
git commit -m "Deploy to new server"
git push origin main
```

⚠️ **중요**: `data/raw/*`, `outputs/*` 등 큰 데이터 파일은 Git에 포함되지 않습니다.
별도로 전송해야 합니다.

### 2단계: 새 서버 준비

#### 필수 설치사항

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit (GPU 사용 시)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Git 설치
sudo apt-get install -y git
```

### 3단계: 코드 클론 및 이동

```bash
# Git 저장소 클론
git clone <your-repository-url>
cd Conf_Agg

# 또는 직접 복사된 경우 해당 디렉토리로 이동
cd /path/to/Conf_Agg
```

### 4단계: 환경 설정 및 실행

#### 방법 1: 자동 배포 스크립트 (권장)

```bash
# 빠른 배포 스크립트 실행
bash scripts/quick_deploy.sh
```

이 스크립트가 자동으로:
- ✅ Docker 이미지 빌드
- ✅ 컨테이너 시작
- ✅ uv sync 실행
- ✅ GPU 확인

#### 방법 2: 수동 설정

```bash
# 1. .env 파일 생성
cp env.example .env
nano .env  # WANDB_API_KEY 설정

# 2. docker-compose.yml 수정 (필요시)
nano docker-compose.yml  # volumes 경로 수정

# 3. Docker 빌드 및 실행
docker-compose build
docker-compose up -d

# 4. 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 5. 컨테이너 안에서 uv sync
uv sync
```

### 5단계: Stage 1 실행

```bash
# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 컨테이너 안에서:

# 방법 1: 단일 GPU 실행
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1

# 방법 2: 4개 GPU 병렬 실행 (백그라운드)
uv run bash scripts/run_stage1_background.sh
```

---

## 📋 체크리스트

### 원본 서버에서

- [ ] Git에 코드 업로드 완료
- [ ] .env 파일에 민감한 정보가 없도록 확인
- [ ] data/raw/deepscaler.jsonl 등 데이터 파일 별도 전송

### 새 서버에서

#### 사전 준비
- [ ] Docker 설치 완료
- [ ] NVIDIA Container Toolkit 설치 (GPU 사용 시)
- [ ] Git 설치 완료
- [ ] GPU 드라이버 설치 완료

#### 환경 설정
- [ ] Git 클론 또는 코드 복사 완료
- [ ] .env 파일 생성 및 설정 완료
- [ ] docker-compose.yml volumes 경로 수정 완료

#### 실행 준비
- [ ] Docker 이미지 빌드 완료
- [ ] 컨테이너 시작 완료
- [ ] uv sync 완료
- [ ] GPU 접근 확인 완료

#### 데이터 준비
- [ ] data/raw/deepscaler.jsonl 파일 확인
- [ ] 필요한 디렉토리 구조 확인

---

## 🔍 문제 해결

### Docker 권한 문제

```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker

# 또는 sudo로 실행
sudo docker-compose up -d
```

### GPU 인식 안 됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# NVIDIA Container Toolkit 확인
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### uv sync 실패

```bash
# 컨테이너 내부에서
docker-compose exec conf-agg-llm bash

# uv.lock 재생성
rm uv.lock
uv sync

# 또는 직접 설치
uv pip install -e .
```

### 컨테이너 계속 재시작

```bash
# 로그 확인
docker-compose logs conf-agg-llm

# 재빌드
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 💡 유용한 명령어

```bash
# 컨테이너 시작/중지
docker-compose up -d          # 시작
docker-compose down           # 중지
docker-compose restart        # 재시작

# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 로그 확인
docker-compose logs -f conf-agg-llm

# GPU 모니터링
nvidia-smi
watch -n 1 nvidia-smi

# 디스크 사용량
df -h
du -sh data/ outputs/

# 컨테이너 상태
docker ps
docker stats conf-agg-llm
```

---

## 📞 추가 도움

자세한 내용은 다음 문서를 참조하세요:
- [완전한 배포 가이드](DEPLOYMENT_GUIDE.md)
- [README.md](../README.md)
- [GitHub Issues](https://github.com/your-repo/issues)
