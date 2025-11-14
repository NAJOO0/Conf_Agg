#!/usr/bin/env bash
# Prompt Token Count 분포 분석 스크립트 실행 예시

# 기본 사용법 (제한 없음)
# uv run python scripts/analyze_prompt_token_distribution.py \
#     --train-path /mnt/data1/datasets/nlp/conf_agg/curated/train_curated.parquet \
#     --validation-path /mnt/data1/datasets/nlp/conf_agg/curated/validation_curated.parquet \
#     --model-name Qwen/Qwen3-1.7B \
#     --output-dir outputs/analysis/prompt_tokens

# Max Input Length 제한 적용 (예: 8192 - 100 = 8092)
uv run python scripts/analyze_prompt_token_distribution.py \
    --train-path /mnt/data1/datasets/nlp/conf_agg/curated/train_curated.parquet \
    --validation-path /mnt/data1/datasets/nlp/conf_agg/curated/validation_curated.parquet \
    --model-name Qwen/Qwen3-1.7B \
    --output-dir outputs/analysis/prompt_tokens \
    --max-input-length 8092

# 또는 config.yaml의 경로 사용
# python scripts/analyze_prompt_token_distribution.py \
#     --train-path ${DATA_DIR}/curated/train_curated.parquet \
#     --validation-path ${DATA_DIR}/curated/validation_curated.parquet \
#     --model-name Qwen/Qwen3-1.7B \
#     --max-input-length 8092

