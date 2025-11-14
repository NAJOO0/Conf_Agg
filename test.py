from unsloth import FastLanguageModel

# 캐시 클리어 후 재로드
import torch
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-1.7B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 다시 확인
attention_class = model.model.layers[0].self_attn.__class__.__name__
print(f"Attention 타입: {attention_class}")

# Flash Attention 강제 적용 시도
if "Flash" not in attention_class:
    print("⚠️ 여전히 일반 Attention 사용 중")
    print("수동으로 attn_implementation 지정 시도...")
    
    # 명시적 지정
    model, tokenizer = FastLanguageModel.from_pretrained(
        "Qwen/Qwen3-1.7B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
    )
    
    attention_class = model.model.layers[0].self_attn.__class__.__name__
    print(f"재시도 후 Attention 타입: {attention_class}")