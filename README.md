# Conf-AggLLM: 신뢰도-인식 집계 모델 실험 프레임워크

LLM의 수학 추론 성능과 효율성을 혁신적으로 개선하기 위한 **'신뢰도-인식 집계 모델(Conf-AggLLM)'** 실험 프레임워크입니다.

## 🎯 프로젝트 목표

50~100개의 추론 결과를 다수결 투표해야 얻을 수 있는 성능을, 단 1~2개의 추론 결과와 그에 대한 '신뢰도 점수'만으로 달성하는 것을 목표로 합니다.

## 🏗️ 아키텍처

### 4단계 자동화 파이프라인

1. **Stage 1: 원시 데이터 생성** - 다수의 후보 해답과 신뢰도 점수 추출
2. **Stage 2: 데이터 큐레이션** - Hard/Easy 분류 및 훈련/검증 데이터셋 구성
3. **Stage 3: 모델 훈련** - GRPO 기반 신뢰도-인식 집계 모델 훈련
4. **Stage 4: 벤치마크 평가** - 4개 벤치마크에 대한 성능 측정

### 핵심 기술

- **모델**: Qwen2.5-1.5B (Qwen3-1.7B 호환)
- **추론 엔진**: vLLM + Ray Serve 병렬 처리 (GPU 4개 지원)
- **성능 최적화**: FlashInfer로 추론 속도 향상
- **훈련 알고리즘**: GRPO (Group-Relative Policy Optimization)
- **정답 검증**: math_verify 라이브러리
- **설정 관리**: Hydra
- **패키지 관리**: uv (빠른 의존성 관리)
- **샘플링 파라미터**: TopP=0.95, TopK=20, MinP=0.0
- **메모리 최적화**: float16 logprob 저장으로 50% 메모리 절약

## ⚡ 성능 향상

### Ray Serve 병렬 처리
- **4개 GPU 독립 처리**: 각 GPU가 완전히 독립적으로 작업
- **예상 성능 향상**: 3-4배 속도 향상
- **메모리 효율성**: GPU 간 통신 오버헤드 제거

### FlashInfer 최적화
- **추론 속도 향상**: 샘플링 단계 최적화
- **메모리 사용량 감소**: 효율적인 CUDA 커널 사용

### 메모리 최적화
- **float16 logprob**: 50% 메모리 절약
- **간소화된 추출 함수**: 처리 속도 향상

## 🚀 빠른 시작

### 1. 환경 설정 (uv 기반)

```bash
# 저장소 클론
git clone <repository-url>
cd Conf_Agg

# uv 기반 환경 설정 실행
chmod +x setup_uv.sh
./setup_uv.sh

# .env 파일 편집 (WandB API Key 설정)
nano .env
```

### 2. Docker 컨테이너 실행 (uv 기반)

```bash
# 컨테이너 빌드 및 시작 (uv 기반)
docker-compose up -d

# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# GPU 상태 확인
nvidia-smi
```

### 3. 실험 실행

```bash
# Stage 1: 원시 데이터 생성
python scripts/stage1_generate.py

# Stage 2: 데이터 큐레이션
python scripts/stage2_curate.py

# Stage 3: 모델 훈련
python scripts/stage3_train.py

# Stage 4: 벤치마크 평가
python scripts/stage4_evaluate.py
```

## 🌐 다른 서버에 배포하기

다른 서버에서 이 프로젝트를 실행하려면 다음 가이드를 참조하세요:

- **[🚀 빠른 시작 가이드](docs/QUICKSTART.md)** - 빠른 배포 (5분)
- **[📖 완전한 배포 가이드](docs/DEPLOYMENT_KR.md)** - 상세한 단계별 설명
- **[📚 영문 배포 가이드](docs/DEPLOYMENT_GUIDE.md)** - Complete deployment guide

### 자동 배포 스크립트 사용

```bash
# 새 서버에서 실행
git clone <your-repository-url>
cd Conf_Agg
bash scripts/quick_deploy.sh
```

이 스크립트가 Docker 빌드, 컨테이너 시작, uv sync를 자동으로 처리합니다.

## 📁 프로젝트 구조

```
Conf_Agg/
├── config/                 # Hydra 설정 파일들
│   ├── config.yaml
│   ├── data/
│   ├── training/
│   └── evaluation/
├── src/                    # 소스 코드
│   ├── data/              # 데이터 처리 모듈
│   ├── models/            # 모델 관련 모듈
│   ├── inference/         # 추론 엔진
│   ├── evaluation/        # 평가 모듈
│   └── utils/            # 유틸리티
├── scripts/              # 실행 스크립트
├── data/                 # 데이터 디렉토리
│   ├── raw/              # 원본 데이터셋
│   ├── generated/         # Stage 1 결과
│   ├── curated/          # Stage 2 결과
│   └── benchmarks/       # 벤치마크 데이터셋
├── outputs/               # 출력 디렉토리
│   ├── models/           # 훈련된 모델
│   ├── logs/             # 실험 로그
│   └── results/          # 평가 결과
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## ⚙️ 설정

### 주요 설정 파일

- `config/config.yaml`: 메인 설정 파일
- `config/data/raw_dataset.yaml`: Stage 1 설정
- `config/data/curation.yaml`: Stage 2 설정
- `config/training/lora.yaml`: 훈련 설정
- `config/evaluation/benchmarks.yaml`: 평가 설정

### GPU 설정

기본적으로 GPU 4개를 사용하도록 설정되어 있습니다:

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 4
          capabilities: [gpu]
```

## 🔧 개발

### 개발 환경 접근

```bash
# Jupyter Lab 실행
docker-compose up jupyter
# 브라우저에서 http://localhost:8888 접속

# 개발용 컨테이너 접속
docker-compose exec conf-agg-llm bash
```

### 코드 수정 시

소스 코드는 실시간으로 컨테이너에 동기화되므로, 호스트에서 수정한 내용이 즉시 반영됩니다.

## 📊 실험 추적

WandB를 사용하여 실험을 추적할 수 있습니다:

1. `.env` 파일에 WandB API Key 설정
2. 실험 실행 시 자동으로 로깅
3. WandB 대시보드에서 실험 결과 확인

## 🐛 문제 해결

### 일반적인 문제들

1. **GPU 메모리 부족**: `gpu_memory_utilization` 값을 낮춤
2. **모델 로딩 실패**: `trust_remote_code: true` 확인
3. **Docker 권한 문제**: `sudo` 사용 또는 Docker 그룹 추가

### 로그 확인

```bash
# 컨테이너 로그 확인
docker-compose logs -f conf-agg-llm

# 실험 로그 확인
tail -f outputs/logs/stage1_generate.log
```

## 📈 성능 최적화

### GPU 활용 최적화

- `tensor_parallel_size=4`: GPU 4개 병렬 처리
- `gpu_memory_utilization=0.9`: GPU 메모리 최대 활용
- 배치 크기 조정으로 처리량 최적화

### 메모리 최적화

- `enforce_eager=True`: 메모리 효율성 향상
- 체크포인트 저장으로 메모리 절약
- 불필요한 중간 결과 정리

## 📝 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해 주세요.

## 📚 참고 문헌

- [AggLM 논문] - 다수결 투표 기반 수학 추론
- [GRPO 알고리즘] - 그룹 기반 정책 최적화
- [Qwen 모델] - 기반 언어 모델
- [vLLM] - 고속 추론 엔진
