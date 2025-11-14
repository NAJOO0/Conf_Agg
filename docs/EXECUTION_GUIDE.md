# 벤치마크 평가 실행 가이드

## 실행 방법

### 1. 기본 실행

```bash
cd /mnt/data1/projects/Conf_Agg
python scripts/stage4_comprehensive_evaluate.py
```

### 2. 설정 파일 확인

스크립트는 Hydra를 사용하여 설정을 관리합니다. 주요 설정 파일:

- `config/config.yaml`: 기본 설정
- `config/evaluation/benchmarks.yaml`: 벤치마크 평가 설정
- `config/data/raw_dataset.yaml`: 신뢰도 계산 설정

### 3. 필요한 사전 준비

#### 3.1 AggLLM 모델 (선택사항)
- 경로: `{model_dir}/checkpoint-final`
- 기본 경로: `/mnt/data1/models/nlp/conf_agg/checkpoint-final`
- 모델이 없으면 Baseline 모델만 사용하여 평가합니다.

#### 3.2 벤치마크 데이터셋
- HuggingFace에서 자동으로 다운로드됩니다:
  - `math-ai/aime24`
  - `math-ai/aime25`
  - `MathArena/hmmt_feb_2024`
  - `MathArena/hmmt_feb_2025`

### 4. 실행 옵션

#### 4.1 Hydra 오버라이드 사용

```bash
# 모델 경로 변경
python scripts/stage4_comprehensive_evaluate.py paths.model_dir=/custom/path/to/model

# 온도 변경
python scripts/stage4_comprehensive_evaluate.py evaluation.benchmarks.evaluation.temperature=1.0

# 최대 토큰 수 변경
python scripts/stage4_comprehensive_evaluate.py evaluation.benchmarks.evaluation.max_tokens=8192

# 여러 옵션 동시 변경
python scripts/stage4_comprehensive_evaluate.py \
    evaluation.benchmarks.evaluation.temperature=1.0 \
    evaluation.benchmarks.evaluation.max_tokens=8192 \
    paths.model_dir=/custom/path/to/model
```

#### 4.2 로그 레벨 변경

```bash
# DEBUG 레벨로 실행
python scripts/stage4_comprehensive_evaluate.py experiment.log_level=DEBUG

# WARNING 레벨로 실행
python scripts/stage4_comprehensive_evaluate.py experiment.log_level=WARNING
```

### 5. 출력 결과

#### 5.1 결과 파일 위치
- 기본 경로: `{output_dir}/comprehensive_results/`
- 기본 경로: `/mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/`

#### 5.2 생성되는 파일
- `{benchmark_name}_results.json`: 각 벤치마크별 상세 결과
- `comprehensive_summary.json`: 전체 요약 결과

#### 5.3 로그 파일
- 위치: `{log_dir}/stage4_comprehensive_evaluate.log`
- 기본 경로: `/mnt/data1/datasets/nlp/conf_agg/logs/stage4_comprehensive_evaluate.log`

### 6. 실행 예시

#### 6.1 전체 벤치마크 평가 (Baseline + AggLLM)
```bash
python scripts/stage4_comprehensive_evaluate.py
```

#### 6.2 Baseline만 평가 (AggLLM 모델 없을 때)
```bash
# AggLLM 모델이 없으면 자동으로 Baseline만 평가합니다
python scripts/stage4_comprehensive_evaluate.py
```

#### 6.3 특정 벤치마크만 평가 (코드 수정 필요)
현재는 모든 벤치마크를 순차적으로 평가합니다. 특정 벤치마크만 평가하려면 스크립트를 수정하거나, 평가 중단 후 결과 확인 후 재실행하세요.

### 7. 주의사항

#### 7.1 GPU 메모리
- vLLM은 GPU 메모리를 많이 사용합니다
- 기본 설정: `gpu_memory_utilization=0.85`
- 메모리 부족 시 `config/data/raw_dataset.yaml`에서 `vllm.gpu_memory_utilization` 값을 낮추세요

#### 7.2 실행 시간
- 각 벤치마크마다 16개 solution × 문제 수만큼 생성하므로 시간이 오래 걸릴 수 있습니다
- 예상 시간: 벤치마크당 수십 분 ~ 수 시간 (데이터셋 크기에 따라 다름)

#### 7.3 HuggingFace 데이터셋 다운로드
- 첫 실행 시 HuggingFace에서 데이터셋을 다운로드합니다
- 인터넷 연결이 필요합니다
- 캐시 위치: `~/.cache/huggingface/datasets/`

### 8. 문제 해결

#### 8.1 모델 로드 실패
```
ERROR: 모델을 찾을 수 없습니다
```
- 해결: `config/config.yaml`의 `paths.model_dir` 경로 확인
- 또는 `paths.model_dir=/correct/path`로 오버라이드

#### 8.2 GPU 메모리 부족
```
CUDA out of memory
```
- 해결: `config/data/raw_dataset.yaml`에서 `vllm.gpu_memory_utilization` 값을 낮추기 (예: 0.7)
- 또는 `num_solutions` 값을 줄이기 (스크립트 수정 필요)

#### 8.3 데이터셋 다운로드 실패
```
ERROR: 데이터셋 로드 실패
```
- 해결: 인터넷 연결 확인
- 또는 HuggingFace 토큰 설정 확인 (필요시)

### 9. 결과 확인

#### 9.1 콘솔 출력
실행 중 콘솔에 실시간으로 진행 상황과 결과가 출력됩니다.

#### 9.2 JSON 결과 파일
```bash
# 결과 파일 확인
cat /mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/comprehensive_summary.json | jq

# 특정 벤치마크 결과 확인
cat /mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/AIME24_results.json | jq
```

#### 9.3 로그 파일
```bash
# 로그 파일 확인
tail -f /mnt/data1/datasets/nlp/conf_agg/logs/stage4_comprehensive_evaluate.log
```

### 10. 빠른 시작 예시

```bash
# 1. 프로젝트 디렉토리로 이동
cd /mnt/data1/projects/Conf_Agg

# 2. 기본 실행
python scripts/stage4_comprehensive_evaluate.py

# 3. 결과 확인 (다른 터미널에서)
tail -f /mnt/data1/datasets/nlp/conf_agg/logs/stage4_comprehensive_evaluate.log
```

