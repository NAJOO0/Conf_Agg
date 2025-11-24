"""
HuggingFace Hub에 LoRA 어댑터 업로드 스크립트

사용법:
    python scripts/upload_to_huggingface.py \
        --model_dir /path/to/lora/model \
        --repo_id "your-username/model-name" \
        --private  # (선택) private 저장소로 생성
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model_card(
    base_model: str,
    repo_id: str,
    model_dir: str,
    training_config: Optional[dict] = None
) -> str:
    """README.md (Model Card) 생성"""

    # 훈련 설정 정보 추출
    num_generations = "N/A"
    enable_think = "N/A"
    lora_r = "N/A"
    lora_alpha = "N/A"

    if training_config:
        num_generations = training_config.get("num_generations", "N/A")
        enable_think = training_config.get("enable_think", "N/A")
        lora_r = training_config.get("lora_r", "N/A")
        lora_alpha = training_config.get("lora_alpha", "N/A")

    # adapter_config.json에서 정보 추출
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            lora_r = adapter_config.get("r", lora_r)
            lora_alpha = adapter_config.get("lora_alpha", lora_alpha)

    model_card = f"""---
library_name: peft
base_model: {base_model}
tags:
  - math-reasoning
  - confidence-aggregation
  - grpo
  - lora
  - qwen
license: apache-2.0
---

# {repo_id.split('/')[-1]}

이 모델은 **Confidence-Aware Aggregation LLM** 프레임워크로 훈련된 LoRA 어댑터입니다.

## 모델 정보

- **Base Model**: `{base_model}`
- **Training Method**: GRPO (Group-Relative Policy Optimization)
- **LoRA Rank**: {lora_r}
- **LoRA Alpha**: {lora_alpha}
- **Number of Generations**: {num_generations}
- **Enable Think**: {enable_think}

## 사용 방법

### 1. 기본 로드 (PEFT)

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRA 어댑터 설정 로드
config = PeftConfig.from_pretrained("{repo_id}")

# Base 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    device_map="auto"
)

# LoRA 어댑터 적용
model = PeftModel.from_pretrained(model, "{repo_id}")

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 추론
messages = [
    {{"role": "user", "content": "Solve: 2x + 5 = 13"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Unsloth를 사용한 빠른 추론

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{repo_id}",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 추론 모드로 전환
FastLanguageModel.for_inference(model)

messages = [
    {{"role": "user", "content": "Solve: 2x + 5 = 13"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. vLLM을 사용한 고속 추론

```python
from vllm import LLM, SamplingParams

# LoRA 어댑터와 함께 vLLM 로드
llm = LLM(
    model="{base_model}",
    enable_lora=True,
    max_lora_rank={lora_r},
    trust_remote_code=True
)

# Sampling 파라미터
sampling_params = SamplingParams(
    temperature=1.5,
    top_p=0.95,
    max_tokens=512
)

# 추론
prompts = ["Solve: 2x + 5 = 13"]
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=("{repo_id}", "{repo_id}")
)

for output in outputs:
    print(output.outputs[0].text)
```

## 훈련 세부사항

이 LoRA 어댑터는 다음과 같은 방법으로 훈련되었습니다:

1. **데이터**: 수학 문제 (MATH, GSM8K 등)
2. **알고리즘**: GRPO - 그룹 기반 relative advantage를 사용한 policy optimization
3. **목표**: 적은 샘플(1-2개)과 confidence score로 많은 샘플(50-100개)의 majority voting 성능 달성

## 라이선스

이 모델은 Apache 2.0 라이선스 하에 배포됩니다.

## 인용

```bibtex
@software{{conf_agg_llm,
  title = {{Confidence-Aware Aggregation LLM}},
  author = {{Your Name}},
  year = {{2025}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## 문의

문제가 있거나 질문이 있으시면 GitHub Issue를 생성해주세요.
"""
    return model_card


def save_training_info(model_dir: str, training_info: dict):
    """훈련 정보를 JSON 파일로 저장"""
    info_path = os.path.join(model_dir, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 훈련 정보 저장: {info_path}")


def upload_to_hub(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    subfolder: Optional[str] = None
):
    """
    LoRA 어댑터를 HuggingFace Hub에 업로드

    Args:
        model_dir: LoRA 어댑터가 저장된 디렉토리
        repo_id: HuggingFace 저장소 ID (예: "username/model-name")
        private: Private 저장소로 생성할지 여부
        token: HuggingFace API 토큰 (없으면 환경변수 또는 캐시된 토큰 사용)
        commit_message: 커밋 메시지
        subfolder: 저장소 내 하위 폴더 (예: "checkpoint-100", None이면 루트에 저장)
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        logger.error("❌ huggingface_hub 패키지가 설치되지 않았습니다.")
        logger.error("   설치: pip install huggingface_hub")
        return

    # 모델 디렉토리 확인
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"❌ 모델 디렉토리를 찾을 수 없습니다: {model_dir}")
        return

    # 필수 파일 확인
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            # safetensors가 없으면 .bin 파일 확인
            if file == "adapter_model.safetensors" and (model_path / "adapter_model.bin").exists():
                continue
            missing_files.append(file)

    if missing_files:
        logger.warning(f"⚠️ 일부 파일이 없습니다: {missing_files}")
        logger.warning("   계속 진행하지만 업로드가 실패할 수 있습니다.")

    # API 초기화
    api = HfApi(token=token)

    # 저장소 생성 (이미 있으면 무시됨)
    try:
        logger.info(f"📦 저장소 생성 중: {repo_id} (private={private})")
        create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
            repo_type="model",
            exist_ok=True
        )
        logger.info("✅ 저장소 생성 완료 (또는 이미 존재)")
    except Exception as e:
        logger.error(f"❌ 저장소 생성 실패: {e}")
        logger.error("   HuggingFace 토큰을 확인하세요.")
        logger.error("   토큰 생성: https://huggingface.co/settings/tokens")
        return

    # Base model 정보 읽기
    adapter_config_path = model_path / "adapter_config.json"
    base_model = "Qwen/Qwen3-1.7B"  # 기본값
    training_config = {}

    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path", base_model)
            training_config["lora_r"] = adapter_config.get("r")
            training_config["lora_alpha"] = adapter_config.get("lora_alpha")

    # training_info.json 확인
    training_info_path = model_path / "training_info.json"
    if training_info_path.exists():
        with open(training_info_path, 'r') as f:
            training_config.update(json.load(f))

    # Model card 생성
    logger.info("📝 Model card 생성 중...")
    model_card = create_model_card(
        base_model=base_model,
        repo_id=repo_id,
        model_dir=str(model_dir),
        training_config=training_config
    )

    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    logger.info(f"✅ Model card 저장: {readme_path}")

    # 업로드할 파일 목록
    logger.info("📤 파일 업로드 중...")
    logger.info(f"   로컬 경로: {model_dir}")
    logger.info(f"   저장소: {repo_id}")
    if subfolder:
        logger.info(f"   저장소 내 경로: {subfolder}/")
    else:
        logger.info(f"   저장소 내 경로: / (루트)")

    # 업로드
    try:
        commit_msg = commit_message or f"Upload LoRA adapter trained with GRPO"
        if subfolder:
            commit_msg = commit_message or f"Upload {subfolder}"

        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            token=token,
            commit_message=commit_msg,
            repo_type="model",
            path_in_repo=subfolder,  # 저장소 내 하위 폴더 지정
        )

        logger.info("✅ 업로드 완료!")
        if subfolder:
            logger.info(f"🔗 모델 URL: https://huggingface.co/{repo_id}/tree/main/{subfolder}")
        else:
            logger.info(f"🔗 모델 URL: https://huggingface.co/{repo_id}")

    except Exception as e:
        logger.error(f"❌ 업로드 실패: {e}")
        return


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub에 LoRA 어댑터 업로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 업로드
  python scripts/upload_to_huggingface.py \\
      --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \\
      --repo_id "your-username/qwen3-1.7b-math-lora"

  # Private 저장소로 업로드
  python scripts/upload_to_huggingface.py \\
      --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \\
      --repo_id "your-username/qwen3-1.7b-math-lora" \\
      --private

  # 토큰 직접 지정
  python scripts/upload_to_huggingface.py \\
      --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \\
      --repo_id "your-username/qwen3-1.7b-math-lora" \\
      --token "hf_xxxxxxxxxxxxx"

  # 훈련 정보 포함
  python scripts/upload_to_huggingface.py \\
      --model_dir /mnt/data1/models/nlp/conf_agg/enable_think_true_20251120_8 \\
      --repo_id "your-username/qwen3-1.7b-math-lora" \\
      --num_generations 8 \\
      --enable_think
        """
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="업로드할 LoRA 모델이 저장된 디렉토리"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace 저장소 ID (예: username/model-name)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Private 저장소로 생성 (기본: Private)"
    )

    parser.add_argument(
        "--public",
        action="store_true",
        help="Public 저장소로 생성 (--private 무시)"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API 토큰 (없으면 환경변수 HF_TOKEN 또는 캐시된 토큰 사용)"
    )

    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="커밋 메시지 (선택사항)"
    )

    # 훈련 정보 (선택사항)
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="GRPO 훈련 시 사용한 num_generations"
    )

    parser.add_argument(
        "--enable_think",
        action="store_true",
        help="Think mode 활성화 여부"
    )

    # 저장소 내 경로 설정
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="저장소 내 하위 폴더 경로 (예: checkpoint-100, final). None이면 루트에 저장"
    )

    parser.add_argument(
        "--auto_subfolder",
        action="store_true",
        help="model_dir의 마지막 폴더명을 자동으로 subfolder로 사용"
    )

    args = parser.parse_args()

    # 토큰 확인
    token = args.token or os.getenv("HF_TOKEN")

    if not token:
        logger.warning("⚠️ HuggingFace 토큰이 지정되지 않았습니다.")
        logger.warning("   환경변수 HF_TOKEN을 설정하거나 --token 인자를 사용하세요.")
        logger.warning("   또는 `huggingface-cli login`으로 로그인하세요.")
        logger.warning("   계속 진행하지만 업로드가 실패할 수 있습니다.")

    # 훈련 정보 저장 (제공된 경우)
    if args.num_generations or args.enable_think:
        training_info = {}
        if args.num_generations:
            training_info["num_generations"] = args.num_generations
        if args.enable_think:
            training_info["enable_think"] = True

        save_training_info(args.model_dir, training_info)

    # subfolder 처리
    subfolder = args.subfolder
    if args.auto_subfolder and not subfolder:
        # model_dir의 마지막 폴더명 사용
        subfolder = Path(args.model_dir).name
        logger.info(f"🔧 자동 subfolder 설정: {subfolder}")

    # private/public 처리
    private = args.private and not args.public
    if args.public:
        logger.info("🌐 Public 저장소로 생성")
    else:
        logger.info("🔒 Private 저장소로 생성 (기본값)")

    # 업로드
    upload_to_hub(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=private,
        token=token,
        commit_message=args.commit_message,
        subfolder=subfolder
    )


if __name__ == "__main__":
    main()
