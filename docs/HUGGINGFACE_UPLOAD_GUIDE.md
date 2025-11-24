# HuggingFace Hub에 LoRA 모델 업로드 가이드

이 가이드는 Stage 3에서 훈련한 LoRA 모델을 HuggingFace Hub에 업로드하는 전체 과정을 설명합니다.

## 📋 목차

1. [사전 준비](#1-사전-준비)
2. [HuggingFace 계정 및 토큰 생성](#2-huggingface-계정-및-토큰-생성)
3. [저장소 생성 (웹 UI)](#3-저장소-생성-웹-ui-선택사항)
4. [모델 업로드](#4-모델-업로드)
5. [업로드 확인](#5-업로드-확인)
6. [다른 사람들이 모델 사용하는 법](#6-다른-사람들이-모델-사용하는-법)

---

## 1. 사전 준비

### 필수 패키지 설치

```bash
pip install huggingface_hub
```

또는 uv를 사용하는 경우:

```bash
uv pip install huggingface_hub
```

### 훈련된 모델 확인

Stage 3에서 훈련한 모델이 다음 경로에 있는지 확인:

```bash
ls /mnt/data1/models/nlp/conf_agg/

# 예시 출력:
# enable_think_true_20251120_8/
```

모델 폴더 안에 다음 파일들이 있어야 합니다:

- `adapter_config.json` - LoRA 설정
- `adapter_model.safetensors` 또는 `adapter_model.bin` - LoRA 가중치
- `tokenizer_config.json`, `tokenizer.json` - Tokenizer 설정 (선택)

```bash
ls /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8/

# 예시 출력:
# adapter_config.json
# adapter_model.safetensors
# tokenizer_config.json
# ...
```

---

## 2. HuggingFace 계정 및 토큰 생성

### 2.1 계정 만들기

1. https://huggingface.co/join 접속
2. 이메일, 사용자명, 비밀번호 입력하여 회원가입
3. 이메일 인증 완료

### 2.2 API 토큰 생성

1. https://huggingface.co/settings/tokens 접속
2. **"New token"** 버튼 클릭
3. 토큰 설정:
   - **Name**: 토큰 이름 입력 (예: `conf-agg-upload`)
   - **Type**: **"Write"** 선택 (업로드하려면 Write 권한 필요)
4. **"Generate a token"** 클릭
5. 생성된 토큰 복사 (예: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

⚠️ **중요**: 토큰은 한 번만 표시되므로 안전한 곳에 저장하세요!

### 2.3 토큰 저장 (3가지 방법)

#### 방법 1: CLI로 로그인 (권장)

```bash
huggingface-cli login
```

- 토큰 입력 프롬프트가 나오면 붙여넣기
- 토큰이 `~/.huggingface/token`에 저장됨

#### 방법 2: 환경변수 설정

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

`.bashrc` 또는 `.zshrc`에 추가하면 영구 저장됩니다.

#### 방법 3: 스크립트에 직접 전달

업로드 시 `--token` 인자로 전달 (나중에 설명)

---

## 3. 저장소 생성 (웹 UI) [선택사항]

스크립트가 자동으로 저장소를 생성하므로 이 단계는 **선택사항**입니다.

하지만 미리 만들고 싶다면:

1. https://huggingface.co/new 접속
2. 저장소 설정:
   - **Owner**: 본인 계정 선택
   - **Model name**: 모델 이름 입력 (예: `qwen3-1.7b-math-lora`)
   - **License**: Apache 2.0 선택 (권장)
   - **Visibility**: Public 또는 Private 선택
3. **"Create model"** 클릭

생성된 저장소 ID: `your-username/qwen3-1.7b-math-lora`

---

## 4. 모델 업로드

### 4.1 기본 업로드 (Public 저장소)

```bash
python scripts/upload_to_huggingface.py \
    --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \
    --repo_id "your-username/qwen3-1.7b-math-lora"
```

**주의**: `your-username`을 본인의 HuggingFace 사용자명으로 변경하세요!

### 4.2 Private 저장소로 업로드

```bash
python scripts/upload_to_huggingface.py \
    --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \
    --repo_id "your-username/qwen3-1.7b-math-lora" \
    --private
```

### 4.3 훈련 정보 포함

```bash
python scripts/upload_to_huggingface.py \
    --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \
    --repo_id "your-username/qwen3-1.7b-math-lora" \
    --num_generations 8 \
    --enable_think
```

### 4.4 토큰 직접 지정

```bash
python scripts/upload_to_huggingface.py \
    --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \
    --repo_id "your-username/qwen3-1.7b-math-lora" \
    --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 4.5 커밋 메시지 지정

```bash
python scripts/upload_to_huggingface.py \
    --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \
    --repo_id "your-username/qwen3-1.7b-math-lora" \
    --commit_message "Add GRPO-trained LoRA adapter for math reasoning"
```

### 업로드 과정

스크립트 실행 시 다음과 같은 출력을 볼 수 있습니다:

```
📦 저장소 생성 중: your-username/qwen3-1.7b-math-lora (private=False)
✅ 저장소 생성 완료 (또는 이미 존재)
📝 Model card 생성 중...
✅ Model card 저장: /path/to/model/README.md
📤 파일 업로드 중...
   업로드 경로: /path/to/model
   저장소: your-username/qwen3-1.7b-math-lora
✅ 업로드 완료!
🔗 모델 URL: https://huggingface.co/your-username/qwen3-1.7b-math-lora
```

---

## 5. 업로드 확인

### 5.1 웹에서 확인

1. 출력된 URL 접속: `https://huggingface.co/your-username/qwen3-1.7b-math-lora`
2. 다음 파일들이 있는지 확인:
   - `README.md` - 자동 생성된 모델 카드
   - `adapter_config.json`
   - `adapter_model.safetensors` (또는 `.bin`)
   - (선택) `tokenizer_config.json`, `tokenizer.json`

### 5.2 CLI로 확인

```bash
huggingface-cli scan-cache | grep your-model-name
```

또는 직접 다운로드 테스트:

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('your-username/qwen3-1.7b-math-lora')"
```

---

## 6. 다른 사람들이 모델 사용하는 법

업로드된 모델을 다른 사람들이 사용할 수 있는 3가지 방법:

### 방법 1: PEFT 라이브러리 (표준)

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRA 어댑터 설정 로드
config = PeftConfig.from_pretrained("your-username/qwen3-1.7b-math-lora")

# Base 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,  # Qwen/Qwen3-1.7B
    trust_remote_code=True,
    device_map="auto"
)

# LoRA 어댑터 적용
model = PeftModel.from_pretrained(model, "your-username/qwen3-1.7b-math-lora")

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 추론
messages = [{"role": "user", "content": "Solve: 2x + 5 = 13"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 방법 2: Unsloth (빠른 추론)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="your-username/qwen3-1.7b-math-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# 추론 코드는 동일
```

### 방법 3: vLLM (프로덕션 고속 추론)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    enable_lora=True,
    max_lora_rank=16,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=1.5,
    top_p=0.95,
    max_tokens=512
)

outputs = llm.generate(
    prompts=["Solve: 2x + 5 = 13"],
    sampling_params=sampling_params,
    lora_request=("qwen3-math", "your-username/qwen3-1.7b-math-lora")
)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 🚨 문제 해결

### 문제 1: `403 Forbidden` 에러

**원인**: 토큰 권한 부족 또는 잘못된 토큰

**해결**:
1. 토큰이 **Write** 권한이 있는지 확인
2. `huggingface-cli login` 다시 실행
3. 토큰이 만료되지 않았는지 확인

### 문제 2: `Repository not found` 에러

**원인**: 저장소 이름이 잘못되었거나 권한 없음

**해결**:
1. `repo_id`가 `username/model-name` 형식인지 확인
2. `username`이 본인의 HuggingFace 사용자명인지 확인

### 문제 3: `adapter_config.json not found` 에러

**원인**: 모델 디렉토리가 잘못되었거나 LoRA 어댑터가 아님

**해결**:
1. `--model_dir` 경로가 정확한지 확인
2. 해당 디렉토리에 `adapter_config.json`이 있는지 확인

### 문제 4: 업로드가 너무 느림

**원인**: 네트워크 속도 또는 큰 파일 크기

**해결**:
1. 안정적인 네트워크 환경에서 업로드
2. `.safetensors` 형식 사용 (더 빠름)
3. 여러 번 시도 (중단된 곳부터 재개됨)

---

## 📚 추가 자료

- [HuggingFace Hub 문서](https://huggingface.co/docs/hub/index)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
- [Unsloth 문서](https://github.com/unslothai/unsloth)
- [vLLM 문서](https://docs.vllm.ai/)

---

## 💡 팁

1. **Private 저장소 추천**: 실험 중인 모델은 private으로 올리고, 완성되면 public으로 변경
2. **Model Card 수정**: 업로드 후 웹에서 README.md를 편집하여 더 자세한 설명 추가 가능
3. **버전 관리**: 같은 저장소에 여러 번 업로드하면 Git처럼 버전이 관리됨
4. **태그 활용**: 중요한 버전에 태그를 달아서 관리 (`git tag` 처럼)

```bash
# 태그 생성 (웹 UI 또는 git)
git tag v1.0
git push origin v1.0
```

---

**축하합니다!** 이제 LoRA 모델을 HuggingFace Hub에 성공적으로 업로드했습니다! 🎉
