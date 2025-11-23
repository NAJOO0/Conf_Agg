# CLAUDE.md - AI 어시스턴트를 위한 Conf-AggLLM 가이드

**최종 업데이트:** 2025-11-23
**프로젝트:** Conf-AggLLM - 신뢰도-인식 집계 모델 프레임워크
**언어:** Python 3.10+
**주요 용도:** 수학 추론 연구

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [저장소 구조](#저장소-구조)
3. [핵심 기술](#핵심-기술)
4. [4단계 파이프라인](#4단계-파이프라인)
5. [개발 환경](#개발-환경)
6. [설정 관리](#설정-관리)
7. [코드 규칙](#코드-규칙)
8. [일반 작업](#일반-작업)
9. [주요 소스 모듈](#주요-소스-모듈)
10. [디버깅 및 문제 해결](#디버깅-및-문제-해결)
11. [중요 경로](#중요-경로)

---

## 프로젝트 개요

### 목표
50-100개의 다수결 투표 추론 결과와 동일한 수학 추론 성능을 단 1-2개의 추론 결과와 신뢰도 점수만으로 달성합니다. 이를 통해 높은 정확도를 유지하면서 계산 비용을 크게 줄입니다.

### 핵심 혁신
**신뢰도-인식 집계**: 단순한 다수결 투표 대신, 신뢰도 점수에 가중치를 둔 지능적인 집계를 수행하도록 GRPO로 훈련된 모델입니다.

### 성능 목표
- **벤치마크**: AIME24, AIME25, HMMT24, HMMT25
- **기본 모델**: Qwen2.5-1.5B / Qwen3-1.7B-FP8
- **최적화**: 4-GPU Ray Serve 병렬 처리로 3-4배 속도 향상
- **메모리**: float16 logprobs 사용으로 50% 절감

---

## 저장소 구조

```
Conf_Agg/
├── config/                          # Hydra 설정 파일
│   ├── config.yaml                  # 메인 설정 (모든 단계 통합)
│   ├── data/
│   │   ├── raw_dataset.yaml         # Stage 1: 생성 설정
│   │   └── curation.yaml            # Stage 2: 큐레이션 설정
│   ├── training/
│   │   └── lora.yaml                # Stage 3: GRPO 훈련 설정
│   └── evaluation/
│       └── benchmarks.yaml          # Stage 4: 평가 설정
│
├── src/                             # 소스 코드 모듈
│   ├── data/
│   │   ├── dataset.py               # 데이터셋 로딩 및 관리
│   │   ├── curation.py              # 데이터 큐레이션 로직 (Hard/Easy 분류)
│   │   ├── confidence.py            # 신뢰도 점수 계산기
│   │   ├── training_dataset.py      # GRPO 훈련 데이터셋
│   │   └── clean_dataset.py         # 데이터 정제 유틸리티
│   ├── models/
│   │   └── grpo_trainer.py          # GRPO 트레이너 구현
│   ├── inference/
│   │   ├── vllm_engine.py           # vLLM 추론 엔진 래퍼
│   │   └── local_engine.py          # 로컬 추론 대체
│   ├── evaluation/
│   │   ├── math_verifier.py         # 수학 답안 검증
│   │   ├── benchmark.py             # 벤치마크 평가
│   │   └── comprehensive_benchmark.py # 전체 평가 스위트
│   └── utils/
│       ├── logging.py               # 로깅 유틸리티
│       └── metrics.py               # 메트릭 계산
│
├── scripts/                         # 실행 가능한 스크립트
│   ├── stage1_generate.py           # Stage 1: 단일 GPU 생성
│   ├── stage1_generate_async.py     # Stage 1: 비동기 멀티 GPU 생성
│   ├── run_stage1_async.sh          # Stage 1: 4-GPU 병렬 실행기
│   ├── stage2_curate.py             # Stage 2: 데이터 큐레이션
│   ├── stage3_train.py              # Stage 3: GRPO 훈련
│   ├── stage3_train_2.py            # Stage 3: 대체 트레이너
│   ├── stage4_1_generate.py         # Stage 4: 벤치마크 응답 생성
│   ├── stage4_2_evaluate_metrics.py # Stage 4: 메트릭 계산
│   ├── stage4_3_evaluate_aggregation.py # Stage 4: 집계 평가
│   └── stage4_evaluate.py           # Stage 4: 전체 평가
│
├── data/                            # 데이터 디렉토리 (gitignore됨)
│   ├── raw/                         # 원본 데이터셋 (예: deepscaler.jsonl)
│   ├── generated/                   # Stage 1 출력
│   ├── curated/                     # Stage 2 출력
│   └── benchmarks/                  # AIME/HMMT 벤치마크 데이터셋
│
├── outputs/                         # 출력 디렉토리 (gitignore됨)
│   ├── models/                      # 훈련된 LoRA 모델
│   ├── logs/                        # 실험 로그
│   └── results/                     # 평가 결과
│
├── docs/                            # 문서
│   ├── QUICKSTART.md                # 5분 빠른 시작
│   ├── DEPLOYMENT_GUIDE.md          # 완전한 배포 (영문)
│   ├── DEPLOYMENT_KR.md             # 완전한 배포 (한글)
│   └── DEPLOYMENT_NO_DOCKER.md      # Docker 없는 배포
│
├── Dockerfile                       # Docker 컨테이너 정의
├── docker-compose.yml               # Docker Compose 오케스트레이션
├── requirements.txt                 # Python 의존성
├── uv.toml                          # UV 패키지 매니저 설정
├── uv.lock                          # UV 락 파일
├── config.json                      # 통합 JSON 설정 (레거시)
└── README.md                        # 프로젝트 README (한글)
```

### 데이터 흐름

```
Stage 1: data/raw/*.jsonl
    → data/generated/generated_responses.parquet

Stage 2: data/generated/generated_responses.parquet
    → data/curated/{train,validation}.parquet

Stage 3: data/curated/train.parquet
    → outputs/models/grpo_trainer_lora_model_{checkpoint}/

Stage 4: outputs/models/grpo_trainer_lora_model_final/
    → outputs/results/{benchmark}_results.json
```

---

## 핵심 기술

### ML/DL 프레임워크
- **PyTorch** (≥2.1.0): 핵심 딥러닝 프레임워크
- **Transformers** (≥4.51.0): HuggingFace 모델 로딩
- **PEFT** (≥0.6.0): LoRA 파라미터 효율적 파인튜닝
- **TRL** (≥0.7.0): GRPO 강화학습 트레이너
- **vLLM**: 연속 배칭을 통한 고속 추론 엔진

### 추론 최적화
- **FlashInfer**: 샘플링을 위한 CUDA 커널 최적화
- **Ray Serve**: 멀티 GPU 병렬 처리 (4개 GPU)
- **FP8 KV Cache**: 메모리 효율적인 key-value 캐싱
- **Tensor Parallelism**: GPU 간 모델 병렬화

### 설정 및 로깅
- **Hydra** (≥1.3.0): 계층적 설정 관리
- **WandB** (≥0.15.0): 실험 추적 및 시각화

### 수학 검증
- **math_verify** (≥0.1.0): 수학 답안 검증

### 패키지 관리
- **uv**: 빠른 Python 패키지 매니저 (pip/poetry 대체)

### 샘플링 파라미터 (기본값)
- TopP: 0.95
- TopK: 20
- MinP: 0.0
- Temperature: 0.6 (생성), 1.5 (집계)
- Max Tokens: 32768

---

## 4단계 파이프라인

### Stage 1: 원시 데이터 생성
**스크립트:** `scripts/stage1_generate_async.py`
**설정:** `config/data/raw_dataset.yaml`

**목적:** 신뢰도 점수와 함께 여러 후보 솔루션을 생성합니다.

**프로세스:**
1. `data/raw/deepscaler.jsonl`에서 원시 수학 문제 로드
2. 연속 배칭을 위해 vLLM AsyncLLMEngine 사용
3. 문제당 `num_responses_per_problem`개의 솔루션 생성
4. 신뢰도 계산을 위한 상위 k개 logprobs 추출 (기본 k=5)
5. 신뢰도 점수 계산:
   - `mean_group_confidence`: 그룹별 평균 logprob
   - `bottom_10_percent_confidence`: 낮은 신뢰도 토큰에 대한 견고성
   - `tail_confidence`: 최종 토큰의 신뢰도
6. `data/generated/generated_responses.parquet`에 저장

**주요 파라미터:**
- `num_responses_per_problem`: 2 (기본값)
- `temperature`: 0.6
- `logprobs`: 5 (상위 5개 logprobs)
- `group_size`: 512 토큰/그룹
- `max_tokens`: 32768

**GPU 사용법:**
- 단일 GPU: `SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py --gpu-id 0`
- 4개 GPU 병렬: `bash scripts/run_stage1_async.sh`

**출력 형식 (Parquet):**
```python
{
    "problem_id": str,
    "problem_text": str,
    "ground_truth": str,
    "response_id": str,
    "generated_text": str,
    "output_token_count": int,
    "logprobs": List[List[float16]],  # 메모리 최적화됨
    "mean_group_confidence": float,
    "bottom_10_percent_confidence": float,
    "tail_confidence": float,
    "worker_gpu": str,
    "worker_replica": str
}
```

---

### Stage 2: 데이터 큐레이션
**스크립트:** `scripts/stage2_curate.py`
**설정:** `config/data/curation.yaml`

**목적:** 문제를 분류하고 다양한 솔루션 세트를 생성하여 훈련 세트를 만듭니다.

**프로세스:**
1. Stage 1의 생성된 응답 로드
2. `math_verify` 라이브러리를 사용하여 각 솔루션 검증
3. 문제를 Hard (낮은 정답률) 또는 Easy (높은 정답률)로 분류
4. 문제당 `num_sets_per_problem`개의 솔루션 세트 생성
5. 각 세트는 `set_size`개의 후보 솔루션 포함
6. 훈련용 (80%)과 검증용 (20%)으로 분할

**주요 파라미터:**
- `strategy`: "curriculum" (어려운 문제 우선)
- `easy_sample_percentage`: 50 (hard/easy 균형)
- `num_sets_per_problem`: 문제당 16개 세트
- `set_size`: 세트당 8개 솔루션
- `confidence_key`: "tail_confidence" (사용할 신뢰도)

**큐레이션 전략:**
- **Curriculum**: 더 나은 학습을 위해 어려운 문제에 집중
- **Naive**: 무작위 샘플링
- **Multitask**: 난이도 혼합

**출력:**
- `data/curated/train.parquet`
- `data/curated/validation.parquet`

**데이터 구조:**
```python
{
    "problem_text": str,
    "ground_truth": str,
    "solutions": List[str],  # Parquet을 위해 JSON 직렬화됨
    "confidence_scores": Dict[str, List[float]],  # JSON 직렬화됨
    "aggregator_prompt": str,  # 훈련용 전체 프롬프트
    "num_correct": int,
    "is_hard": bool
}
```

---

### Stage 3: GRPO 훈련
**스크립트:** `scripts/stage3_train.py`
**설정:** `config/training/lora.yaml`

**목적:** Group-Relative Policy Optimization을 사용하여 LoRA 적응 모델을 훈련합니다.

**프로세스:**
1. 기본 모델 로드 (Qwen3-1.7B-FP8)
2. attention 및 MLP 레이어에 LoRA 어댑터 적용
3. 솔루션 그룹으로 GRPO 알고리즘 사용하여 훈련
4. 신뢰도 기반 올바른 솔루션 선택 최적화
5. `outputs/models/`에 체크포인트 저장

**GRPO 세부사항:**
- **Group Size**: 최적화 단계당 8개 솔루션
- **KL Coefficient**: 0.001 (정규화)
- **Aggregator Temperature**: 1.5 (탐색)
- **보상**: 올바른 선택 시 +1, 그 외 0

**LoRA 설정:**
```python
{
    "r": 16,                    # LoRA 랭크
    "lora_alpha": 32,           # 스케일링 인자
    "lora_dropout": 0.1,        # 드롭아웃 비율
    "target_modules": [         # 적응할 모듈
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

**훈련 설정:**
```python
{
    "epochs": 1,
    "batch_size": 1024,
    "max_prompt_length": 16384,
    "max_response_length": 16384,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 50
}
```

**출력:**
- `outputs/models/grpo_trainer_lora_model_0/`
- `outputs/models/grpo_trainer_lora_model_1/`
- `outputs/models/grpo_trainer_lora_model_final/`

---

### Stage 4: 벤치마크 평가
**스크립트:**
- `scripts/stage4_1_generate.py`: 응답 생성
- `scripts/stage4_2_evaluate_metrics.py`: 메트릭 계산
- `scripts/stage4_3_evaluate_aggregation.py`: 집계 평가

**설정:** `config/evaluation/benchmarks.yaml`

**목적:** 벤치마크 데이터셋에서 훈련된 모델을 평가합니다.

**벤치마크:**
- **AIME24** (`data/benchmarks/aime24.jsonl`)
- **AIME25** (`data/benchmarks/aime25.jsonl`)
- **HMMT24** (`data/benchmarks/hmmt24.jsonl`)
- **HMMT25** (`data/benchmarks/hmmt25.jsonl`)

**프로세스:**
1. 벤치마크 문제당 8개의 후보 솔루션 생성
2. 훈련된 모델을 사용하여 신뢰도 기반 최선의 솔루션 선택
3. math_verify를 사용하여 정확성 검증
4. 메트릭 계산: pass@1, pass@k, 신뢰도 상관관계

**메트릭:**
- **pass@1**: 단일 솔루션 정확도
- **pass@k**: k번 시도 중 최소 1개 정답
- **confidence_correlation**: 신뢰도가 정확성을 얼마나 잘 예측하는지

**출력:**
- `outputs/results/{benchmark}_predictions.json`
- `outputs/results/{benchmark}_metrics.json`

---

## 개발 환경

### Docker (권장)

**설정:**
```bash
# 컨테이너 빌드 및 시작
docker-compose up -d

# 컨테이너 접속
docker-compose exec conf-agg-llm bash

# 컨테이너 내부: 의존성 동기화
uv sync

# GPU 접근 확인
nvidia-smi
```

**컨테이너 세부사항:**
- **베이스 이미지**: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
- **Python**: 3.10+
- **GPU**: 0,1,2,3 (docker-compose.yml에서 설정 가능)
- **공유 메모리**: 16GB (`shm_size: "16g"`)
- **작업 디렉토리**: `/workspace`

**볼륨 마운트:**
```yaml
volumes:
  - /home/najoo0/Conf_Agg:/workspace           # 코드
  - /data1:/data1                               # 데이터 저장소
  - /data2:/data2                               # 추가 저장소
  - /root/.cache/huggingface:/root/.cache/huggingface  # 모델 캐시
  - uv-cache:/tmp/uv-cache                      # UV 캐시
```

### 환경 변수

**필수:**
```bash
WANDB_API_KEY=your_wandb_key_here              # WandB 추적
```

**자동 설정:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3                   # GPU 가시성
NVIDIA_VISIBLE_DEVICES=all                     # 모든 GPU 사용 가능
PYTHONPATH=/workspace                          # Python 모듈 경로
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # CUDA 메모리 관리
VLLM_USE_FLASHINFER=1                          # FlashInfer 최적화 활성화
UV_CACHE_DIR=/tmp/uv-cache                     # UV 캐시 위치
```

### Docker 없는 설정

`docs/DEPLOYMENT_NO_DOCKER.md` 참조하여 Python 직접 설치.

---

## 설정 관리

### Hydra 설정 시스템

**메인 설정:** `config/config.yaml`

Hydra는 **구성(composition)**을 사용하여 최종 설정을 빌드합니다:

```yaml
# config/config.yaml
defaults:
  - data: raw_dataset
  - training: lora
  - evaluation: benchmarks
  - _self_

# CLI에서 오버라이드:
# python script.py data.num_responses=4 training.epochs=2
```

### 설정 파일

**1. 데이터 생성** (`config/data/raw_dataset.yaml`)
```yaml
generation:
  num_responses_per_problem: 2
  temperature: 0.6
  max_tokens: 32768
  logprobs: 5

vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_model_len: 32768
  kv_cache_dtype: "fp8"
```

**2. 큐레이션** (`config/data/curation.yaml`)
```yaml
strategy: curriculum
easy_sample_percentage: 50
num_sets_per_problem: 16
set_size: 8
```

**3. 훈련** (`config/training/lora.yaml`)
```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

grpo:
  group_size: 8
  kl_coefficient: 0.001
  aggregator_temperature: 1.5
```

**4. 평가** (`config/evaluation/benchmarks.yaml`)
```yaml
datasets:
  - name: AIME24
    path: data/benchmarks/aime24.jsonl

evaluation:
  num_candidates: 8
  temperature: 1.5
  max_tokens: 16384
```

### 레거시 설정

`config.json`은 모든 설정의 평탄화된 JSON 버전을 포함합니다. 수정 시 Hydra YAML 파일을 선호합니다.

---

## 코드 규칙

### 파일 구성

**스크립트** (`scripts/`):
- 각 단계의 진입점
- Hydra를 사용한 설정 로딩
- 로깅 설정 및 오류 처리 포함
- 명명 규칙: `stage{N}_{action}.py`

**소스 모듈** (`src/`):
- 재사용 가능한 로직 (CLI 인수 없음)
- 타입 힌트 필수
- 모든 공개 함수/클래스에 docstring 필요
- 기능별로 구성 (data, models, inference, evaluation)

### Python 스타일

**임포트:**
```python
# 표준 라이브러리
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 서드파티
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel

# 로컬
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
```

**타입 힌트:**
```python
def process_data(
    input_path: str,
    config: Dict[str, Any],
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """설정에 따라 입력 데이터를 처리합니다."""
    pass
```

**Docstring:**
```python
def calculate_confidence(logprobs: List[List[float]], method: str = "mean") -> float:
    """
    logprobs에서 신뢰도 점수를 계산합니다.

    Args:
        logprobs: 토큰 레벨 logprob 분포 리스트
        method: 계산 방법 ("mean", "tail", "bottom_10")

    Returns:
        신뢰도 점수 (0.0-1.0)

    Raises:
        ValueError: 방법을 알 수 없는 경우
    """
```

### 로깅

**설정:**
```python
import logging
from src.utils.logging import setup_logging

setup_logging(
    log_level="INFO",
    log_file="outputs/logs/stage1.log"
)
logger = logging.getLogger(__name__)
```

**사용법:**
```python
logger.info(f"{len(problems)}개 문제 처리 중")
logger.warning(f"GPU 메모리 사용량 높음: {usage:.2f}%")
logger.error(f"모델 로드 실패: {e}")
```

### 오류 처리

**우아한 실패:**
```python
try:
    result = math_verify.verify(prediction, ground_truth)
except Exception as e:
    logger.warning(f"검증 실패: {e}, 문자열 매치로 대체")
    result = prediction.strip() == ground_truth.strip()
```

**조기 반환:**
```python
if not logprobs:
    logger.warning("빈 logprobs, 기본 신뢰도 반환")
    return 0.0
```

### 데이터 직렬화

**Parquet용 (중첩된 리스트/딕셔너리 불가):**
```python
# 중첩 구조 직렬화
serialized_set = {
    "solutions": json.dumps(solutions_list),
    "confidence_scores": json.dumps(scores_dict)
}
```

**JSON용 (중첩 가능):**
```python
with open("output.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### 메모리 최적화

**logprobs용 float16:**
```python
# float16으로 변환하여 50% 메모리 절약
logprobs_fp16 = np.array(logprobs, dtype=np.float16).tolist()
```

**스트리밍 저장:**
```python
# 점진적으로 저장, 메모리에 축적하지 않음
async with aiofiles.open(output_file, "a") as f:
    await f.write(json.dumps(result) + "\n")
```

---

## 일반 작업

### 1. Stage 1 실행 (데이터 생성)

**단일 GPU:**
```bash
# 컨테이너 내부
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1
```

**4개 GPU 병렬:**
```bash
# 백그라운드에서 4개 워커(GPU 0,1,2,3) 실행
uv run bash scripts/run_stage1_async.sh
```

**모니터링:**
```bash
# GPU 사용 확인
watch -n 1 nvidia-smi

# 로그 확인
tail -f outputs/logs/stage1_generate_shard_0.log
tail -f outputs/logs/stage1_generate_shard_1.log
```

### 2. Stage 2 실행 (큐레이션)

```bash
uv run python scripts/stage2_curate.py
```

**출력:**
- `data/curated/train.parquet`
- `data/curated/validation.parquet`

### 3. Stage 3 실행 (훈련)

```bash
uv run python scripts/stage3_train.py
```

**WandB로 모니터링:**
- 프로젝트: `conf-agg-llm`
- 메트릭: loss, reward, kl_divergence

**체크포인트:**
- 500 스텝마다 `outputs/models/`에 저장

### 4. Stage 4 실행 (평가)

**전체 파이프라인:**
```bash
# 예측 생성
uv run python scripts/stage4_1_generate.py

# 메트릭 계산
uv run python scripts/stage4_2_evaluate_metrics.py

# 집계 평가
uv run python scripts/stage4_3_evaluate_aggregation.py
```

**또는 올인원:**
```bash
uv run python scripts/stage4_comprehensive_evaluate.py
```

### 5. 설정 수정

**CLI 오버라이드:**
```bash
python scripts/stage1_generate.py \
    data.generation.num_responses_per_problem=4 \
    data.vllm.gpu_memory_utilization=0.8
```

**파일 편집:**
```bash
# 설정 편집
nano config/data/raw_dataset.yaml

# 업데이트된 설정으로 실행
python scripts/stage1_generate.py
```

### 6. 새 벤치마크 추가

**1. 데이터셋 파일 추가:**
```bash
# data/benchmarks/new_benchmark.jsonl에 추가
# 형식: {"problem": "...", "answer": "..."}
```

**2. 설정 업데이트:**
```yaml
# config/evaluation/benchmarks.yaml
datasets:
  - name: NEW_BENCHMARK
    path: data/benchmarks/new_benchmark.jsonl
```

**3. 평가 실행:**
```bash
uv run python scripts/stage4_comprehensive_evaluate.py
```

### 7. GPU 문제 디버깅

**GPU 가시성 확인:**
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

**메모리 사용량 줄이기:**
```yaml
# config/data/raw_dataset.yaml
vllm:
  gpu_memory_utilization: 0.7  # 0.95에서 낮춤
  max_num_seqs: 20              # 40에서 낮춤
```

**GPU 메모리 정리:**
```python
import torch
torch.cuda.empty_cache()
```

### 8. 중단된 훈련 재개

```python
# scripts/stage3_train.py는 체크포인트 재개 지원
# 체크포인트에서 로드하도록 스크립트 수정:
model = AutoModelForCausalLM.from_pretrained(
    "outputs/models/grpo_trainer_lora_model_1"
)
```

### 9. HuggingFace에 모델 업로드

```bash
uv run python scripts/upload_to_hf.py \
    --model-path outputs/models/grpo_trainer_lora_model_final \
    --repo-name your-username/conf-agg-model
```

---

## 주요 소스 모듈

### `src/data/dataset.py`
**목적:** 원시 데이터셋 로드 및 관리.

**주요 클래스:**
- `RawDataset`: JSONL 수학 문제 로드
- `GeneratedDataset`: Stage 1 parquet 출력 로드

**사용법:**
```python
from src.data.dataset import RawDataset

dataset = RawDataset(data_path="data/raw/deepscaler.jsonl")
problems = dataset.load()  # List[Dict] 반환
```

### `src/data/curation.py`
**목적:** 생성된 응답에서 훈련 데이터 큐레이션.

**주요 클래스:**
- `DataCurator`: 큐레이션 전략 구현

**주요 메서드:**
- `classify_hard_easy_sets()`: 난이도별 분류
- `create_solution_sets()`: 다양한 훈련 세트 구축
- `curate()`: 전체 큐레이션 파이프라인

**사용법:**
```python
from src.data.curation import DataCurator

curator = DataCurator(
    strategy="curriculum",
    num_sets_per_problem=16,
    set_size=8
)
train_data, val_data = curator.curate(generated_responses)
```

### `src/data/confidence.py`
**목적:** logprobs에서 신뢰도 점수 계산.

**주요 클래스:**
- `ConfidenceCalculator`: 여러 신뢰도 방법 구현

**신뢰도 방법:**
1. **mean_group_confidence**: 토큰 그룹별 평균 logprob
2. **bottom_10_percent_confidence**: 최악의 토큰에 대한 견고성
3. **tail_confidence**: 최종 토큰의 신뢰도

**사용법:**
```python
from src.data.confidence import ConfidenceCalculator

calc = ConfidenceCalculator(group_size=512)
scores = calc.calculate_all_confidence_scores(logprobs)
# 반환: {"mean_group_confidence": 0.85, ...}
```

### `src/models/grpo_trainer.py`
**목적:** GRPO 훈련 구현.

**주요 클래스:**
- `GRPOTrainer`: GRPO 훈련 루프 관리

**주요 메서드:**
- `_initialize_model()`: LoRA로 모델 로드
- `train()`: 훈련 실행
- `compute_rewards()`: GRPO 보상 계산

**주요 개념:**
- KL divergence를 위한 참조 모델 사용
- 그룹 기반 보상 계산
- 파라미터 효율성을 위한 LoRA

### `src/inference/vllm_engine.py`
**목적:** vLLM을 통한 고속 추론.

**주요 클래스:**
- `VLLMInferenceEngine`: vLLM LLM 클래스 래퍼

**주요 메서드:**
- `generate_multiple_responses()`: 배치 생성
- `_initialize_model()`: vLLM 엔진 설정

**최적화 기능:**
- 텐서 병렬화
- FP8 KV 캐시
- 연속 배칭
- 접두사 캐싱

**사용법:**
```python
from src.inference.vllm_engine import VLLMInferenceEngine

engine = VLLMInferenceEngine(
    model_name="Qwen/Qwen3-1.7B-FP8",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95
)
responses = engine.generate_multiple_responses(
    prompt="풀이: 2x + 3 = 7",
    n=8,
    temperature=0.6
)
```

### `src/evaluation/math_verifier.py`
**목적:** 수학 답안 검증.

**주요 클래스:**
- `MathVerifier`: math_verify 라이브러리 래퍼

**주요 메서드:**
- `verify_answer()`: 단일 답안 검증
- `verify_batch()`: 배치 검증
- `extract_final_answer_from_content()`: `\boxed{}` 답안 추출

**사용법:**
```python
from src.evaluation.math_verifier import MathVerifier

verifier = MathVerifier(timeout=30)
is_correct = verifier.verify_answer(
    predicted="42",
    ground_truth="42"
)
```

### `src/evaluation/benchmark.py`
**목적:** 벤치마크 평가 로직.

**주요 기능:**
- 벤치마크 데이터셋 로드
- 벤치마크에서 모델 실행
- pass@k 메트릭 계산

### `src/utils/logging.py`
**목적:** 중앙 집중식 로깅 설정.

**주요 함수:**
- `setup_logging()`: 파일 + 콘솔 로깅 설정

---

## 디버깅 및 문제 해결

### 일반적인 문제

#### 1. GPU 메모리 부족

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결책:**
```yaml
# config/data/raw_dataset.yaml에서 줄이기
vllm:
  gpu_memory_utilization: 0.7  # 0.95에서 낮춤
  max_num_seqs: 20              # 40에서 낮춤
  max_num_batched_tokens: 8192  # 16384에서 낮춤
```

#### 2. 컨테이너 계속 재시작

**로그 확인:**
```bash
docker-compose logs conf-agg-llm
```

**일반적인 원인:**
- .env 파일 누락
- 잘못된 볼륨 경로
- GPU 드라이버 불일치

**해결책:**
```bash
# 캐시 없이 재빌드
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### 3. vLLM 모델 로딩 실패

**증상:**
```
Failed to load model: trust_remote_code
```

**해결책:**
```yaml
# 설정에서 확인
vllm:
  trust_remote_code: true
```

#### 4. math_verify 타임아웃

**증상:**
```
Verification timeout after 30s
```

**해결책:**
```python
# 큐레이션 설정에서 타임아웃 증가
verification:
  timeout: 60  # 30에서 증가
```

#### 5. WandB 로깅 안 됨

**확인:**
```bash
echo $WANDB_API_KEY
```

**해결책:**
```bash
# .env 파일에 추가
WANDB_API_KEY=your_key_here

# 컨테이너 재시작
docker-compose restart
```

#### 6. Hydra 설정을 찾을 수 없음

**증상:**
```
FileNotFoundError: config/config.yaml
```

**해결책:**
```bash
# 프로젝트 루트에서 실행
cd /workspace  # 컨테이너 내부
python scripts/stage1_generate.py
```

#### 7. Parquet 읽기/쓰기 오류

**증상:**
```
ArrowNotImplementedError: Nested types not supported
```

**원인:** Parquet은 중첩된 리스트/딕셔너리를 지원하지 않습니다.

**해결책:**
```python
# 중첩 구조 직렬화
data["solutions"] = json.dumps(solutions_list)
```

### 성능 디버깅

**GPU 사용량 프로파일:**
```bash
# nvitop 설치 (requirements에 이미 포함)
nvitop

# 또는 표준 nvidia-smi 사용
watch -n 1 nvidia-smi
```

**Python 프로파일:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 여기에 코드

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(20)
```

**WandB 모니터링:**
- `conf-agg-llm` 프로젝트 확인
- 메트릭 정체 확인
- GPU 활용도 차트 확인

### 로깅 모범 사례

**적절한 로그 레벨 설정:**
```python
# 개발 중
setup_logging(log_level="DEBUG")

# 프로덕션
setup_logging(log_level="INFO")
```

**로그 확인:**
```bash
# 컨테이너 로그
docker-compose logs -f conf-agg-llm

# 단계별 로그
tail -f outputs/logs/stage1_generate.log
tail -f outputs/logs/stage2_curate.log
tail -f outputs/logs/stage3_train.log
```

---

## 중요 경로

### 데이터 경로

**원시 데이터:**
```
data/raw/deepscaler.jsonl          # 훈련 문제
data/benchmarks/aime24.jsonl       # AIME 2024 벤치마크
data/benchmarks/aime25.jsonl       # AIME 2025 벤치마크
data/benchmarks/hmmt24.jsonl       # HMMT 2024 벤치마크
data/benchmarks/hmmt25.jsonl       # HMMT 2025 벤치마크
```

**생성된 데이터:**
```
data/generated/generated_responses.parquet       # Stage 1 출력
data/generated/generated_responses_shard_*.parquet  # 병렬 출력
```

**큐레이션된 데이터:**
```
data/curated/train.parquet         # 훈련 세트
data/curated/validation.parquet    # 검증 세트
```

### 모델 경로

**체크포인트:**
```
outputs/models/grpo_trainer_lora_model_0/      # 초기 체크포인트
outputs/models/grpo_trainer_lora_model_1/      # 체크포인트 1
outputs/models/grpo_trainer_lora_model_final/  # 최종 모델
```

**캐시:**
```
/root/.cache/huggingface/          # HuggingFace 모델 캐시
/mnt/data1/models/nlp/huggingface_cache/  # 외부 캐시
```

### 로그 경로

```
outputs/logs/stage1_generate.log
outputs/logs/stage1_generate_shard_0.log
outputs/logs/stage2_curate.log
outputs/logs/stage3_train.log
outputs/logs/stage4_evaluate.log
```

### 설정 경로

**Hydra:**
```
config/config.yaml                 # 메인 오케스트레이터
config/data/raw_dataset.yaml       # Stage 1
config/data/curation.yaml          # Stage 2
config/training/lora.yaml          # Stage 3
config/evaluation/benchmarks.yaml  # Stage 4
```

**레거시:**
```
config.json                        # 평탄화된 JSON 설정
```

### 출력 경로

```
outputs/results/aime24_predictions.json
outputs/results/aime24_metrics.json
outputs/results/benchmark_summary.json
```

### 심볼릭 링크

```
output_s -> /mnt/data1/datasets/nlp/conf_agg/
```

**참고:** `output_s`는 외부 저장소에 대한 심볼릭 링크입니다. 경로를 가정하기 전에 실제 위치를 확인하세요.

---

## 추가 리소스

### 문서
- `README.md`: 메인 프로젝트 README (한글)
- `docs/QUICKSTART.md`: 5분 배포 가이드
- `docs/DEPLOYMENT_GUIDE.md`: 전체 배포 (영문)
- `docs/DEPLOYMENT_KR.md`: 전체 배포 (한글)
- `docs/DEPLOYMENT_NO_DOCKER.md`: Docker 없는 설정
- `RESTART_GUIDE.md`: 재시작 절차

### 스크립트 참조
- `quick_restart.sh`: 빠른 재시작 스크립트
- `restart_setup.sh`: 전체 환경 재구축
- `download_data.sh`: 벤치마크 데이터셋 다운로드
- `install_flashinfer.sh`: FlashInfer 최적화 설치
- `setup_grpo_training.sh`: GRPO 환경 설정

### 외부 링크
- **vLLM 문서**: https://docs.vllm.ai/
- **Qwen 모델**: https://huggingface.co/Qwen
- **Hydra 문서**: https://hydra.cc/
- **WandB 문서**: https://docs.wandb.ai/

---

## AI 어시스턴트를 위한 워크플로우 요약

### 코드 수정 시

1. **쓰기 전에 읽기:** 수정하기 전에 항상 기존 파일을 읽습니다
2. **설정 확인:** Hydra 설정이 코드 기대와 일치하는지 확인
3. **로컬 테스트:** 빠른 테스트를 위해 단일 GPU 모드 사용
4. **리소스 모니터링:** 개발 중 GPU 메모리 확인
5. **광범위한 로깅:** 디버깅을 위한 로깅 추가
6. **문서 업데이트:** 주요 변경 사항으로 이 파일을 업데이트 유지

### 디버깅 시

1. **로그 먼저 확인:** `outputs/logs/`에 상세한 추적 포함
2. **GPU 접근 확인:** `nvidia-smi`가 GPU를 표시해야 함
3. **컨테이너 상태 확인:** `docker-compose ps`
4. **설정 검증:** YAML 구문 및 경로가 올바른지 확인
5. **환경 확인:** `.env` 파일이 WANDB_API_KEY와 함께 존재해야 함

### 기능 추가 시

1. **구조 따르기:** 적절한 `src/` 모듈에 코드 배치
2. **타입 힌트 사용:** 모든 새 함수에 타입 주석 필요
3. **docstring 추가:** 목적, 인수, 반환값 설명
4. **설정 업데이트:** Hydra 설정에 새 파라미터 추가
5. **이 파일 업데이트:** CLAUDE.md에 새 기능 문서화

### 모범 사례

- **`uv run` 선호** 일관성을 위해 직접 `python` 대신
- **Hydra 사용** 모든 설정에 (하드코딩 방지)
- **모든 것을 로깅** 재현성을 위해 중요한 것
- **신중하게 직렬화** Parquet에 저장할 때
- **작은 데이터로 테스트** 전체 실행 전에
- **GPU 모니터링** 집약적인 작업 중
- **버전 관리** 코드와 함께 설정

---

**CLAUDE.md 끝**
