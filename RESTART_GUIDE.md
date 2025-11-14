# 서버 재시작 후 설정 가이드

## 🚨 서버 재시작 시 자동으로 사라지는 것들

- Python 인터프리터 (`/root/.local/share/uv/python`)
- uv 실행 파일 (`/root/.local/bin/uv`)
- 시스템 Python
- UV 캐시 (`/root/.cache/uv`)

## ✅ Persistent Storage에 안전하게 보관되는 것들

- 프로젝트 코드 (`/mnt/data1/projects/Conf_Agg`)
- .venv (9.7GB) - Python 패키지
- 모든 데이터와 결과물

## 🎯 빠른 복구 방법 (3단계)

### 1단계: 필수 패키지 재설치

```bash
apt-get update
apt-get install -y build-essential curl wget git python3 python3-pip
```

### 2단계: uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3단계: 프로젝트 설정 및 실행

```bash
cd /mnt/data1/projects/Conf_Agg
chmod +x restart_setup.sh
./restart_setup.sh
```

## 📝 자세한 수동 설정

### 환경 변수 설정

```bash
# ~/.bashrc에 추가 (한 번만)
cat >> ~/.bashrc << 'EOF'
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy
EOF

source ~/.bashrc
```

### 하이브리드 아키텍처 설정

```bash
# 코드를 메인 스토리지로 복사
cp -r /mnt/data1/projects/Conf_Agg /root/projects/

# .venv 심볼릭 링크
cd /root/projects/Conf_Agg
ln -s /mnt/data1/projects/Conf_Agg/.venv .venv

# 디렉토리 생성
mkdir -p /mnt/data1/{.uv-cache,models/nlp/{huggingface_cache,conf_agg},datasets/nlp/{cache,conf_agg/{outputs,logs}}}

# 설정 확인
ls -lh config/config.yaml
```

### 실행

```bash
cd /root/projects/Conf_Agg
export UV_CACHE_DIR=/mnt/data1/.uv-cache
export HF_HOME=/mnt/data1/models/nlp/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/data1/models/nlp/huggingface_cache
export PYTHONPATH=/root/projects/Conf_Agg
./scripts/run_stage1_2gpu.sh
```

## 🔍 문제 해결

### UV 환경 문제

```bash
export UV_CACHE_DIR=/mnt/data1/.uv-cache
cd /mnt/data1/projects/Conf_Agg
uv sync
```

### Python 경로 문제

```bash
# .venv 삭제 후 재생성
rm -rf .venv
export UV_CACHE_DIR=/mnt/data1/.uv-cache
uv sync
```

## 📊 스토리지 정보

```
메인 스토리지 (/) : 6.0T (Non-Persistent)
Persistent Storage : 200G (/mnt/data1)
GPU : H100 80GB × 2개
```

## ✅ 체크리스트

- [ ] apt-get 필수 패키지 설치
- [ ] uv 재설치
- [ ] UV_CACHE_DIR 환경 변수 설정
- [ ] 코드를 메인 스토리지로 복사
- [ ] .venv 심볼릭 링크 생성
- [ ] 디렉토리 생성
- [ ] 환경 변수 확인
- [ ] GPU 확인 (nvidia-smi)
- [ ] 실행

## 🚀 한 줄 명령으로 빠른 복구

```bash
cd /mnt/data1/projects/Conf_Agg && ./restart_setup.sh
```



