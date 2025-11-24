# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conf-AggLLM is an experimental framework implementing a **Confidence-Aware Aggregation Model** for improving LLM mathematical reasoning performance. The goal is to achieve the performance of 50-100 reasoning results through majority voting using only 1-2 reasoning results plus confidence scores.

**Base Model**: Qwen/Qwen3-1.7B
**Inference Engine**: vLLM + Ray Serve (supports 4 GPUs)
**Training Algorithm**: GRPO (Group-Relative Policy Optimization)
**Configuration**: Hydra-based configuration management
**Package Manager**: uv (fast dependency management)

## Development Commands

### Environment Setup

```bash
# Setup with uv
./setup.sh

# Start Docker container
docker compose up -d

# Access container
docker compose exec conf-agg-llm bash

# Verify GPU availability
nvidia-smi
```

### 4-Stage Pipeline Execution

```bash
# Stage 1: Raw data generation (multiple candidate solutions + confidence scores)
python scripts/stage1_generate.py

# Stage 1 (Async version - faster with continuous batching)
python scripts/stage1_generate_async.py

# Stage 2: Data curation (Hard/Easy classification, train/validation split)
python scripts/stage2_curate.py

# Stage 3: Model training (GRPO-based confidence-aware aggregation)
# Single GPU
python scripts/stage3_train.py

# Multi-GPU (2 GPUs)
./run_stage3_2gpu.sh

# Stage 4: Benchmark evaluation (split into 3 sub-stages)
# 4-1: Generate baseline and AggLLM solutions
python scripts/stage4_1_generate.py

# 4-2: Evaluate metrics (Pass@k, Majority Voting, Confidence Weighted Voting)
python scripts/stage4_2_evaluate_metrics.py

# 4-3: Aggregation evaluation (Prompt Aggregation vs AggLLM Aggregation)
python scripts/stage4_3_evaluate_aggregation.py
```

### Hydra Configuration Overrides

```bash
# Override model path
python scripts/stage4_1_generate.py paths.model_dir=/custom/path

# Override temperature
python scripts/stage4_1_generate.py evaluation.benchmarks.evaluation.temperature=1.0

# Override GPU memory utilization
python scripts/stage1_generate.py vllm.gpu_memory_utilization=0.7
```

### Testing

```bash
# Simple vLLM test
python test_vllm_simple.py

# General test script
python test.py
```

## Architecture

### 4-Stage Pipeline

1. **Stage 1 (Data Generation)**: Generate multiple candidate solutions with confidence scores
   - Uses vLLM AsyncEngine for continuous batching
   - Extracts logprobs for confidence calculation
   - Saves results to `data/generated/`

2. **Stage 2 (Data Curation)**: Classify and prepare training data
   - Hard/Easy classification based on solution accuracy
   - Creates training sets with multiple solution candidates per problem
   - Outputs to `data/curated/train_curated.parquet` and `validation_curated.parquet`

3. **Stage 3 (Training)**: GRPO-based model training
   - Uses LoRA for efficient fine-tuning
   - Supports single or multi-GPU training with DDP
   - Optional vLLM colocate mode for faster generation during training
   - Saves checkpoints to `{model_dir}/checkpoint-{step}/`

4. **Stage 4 (Evaluation)**: Benchmark evaluation on 4 datasets
   - AIME24, AIME25, HMMT24, HMMT25
   - Evaluates Pass@k, Majority Voting, Confidence Weighted Voting
   - Compares Prompt Aggregation vs AggLLM Aggregation

### Key Components

**Inference (`src/inference/`)**:
- `vllm_engine.py`: vLLM-based high-speed inference engine
  - Supports tensor parallelism for multi-GPU inference
  - Batched generation with logprobs extraction
  - FlashInfer optimization for sampling speed

**Data Processing (`src/data/`)**:
- `confidence.py`: Confidence score calculation from logprobs
  - Methods: mean_group_confidence, bottom_10_percent_confidence, tail_confidence
- `curation.py`: Data curation with Hard/Easy classification
- `dataset.py`: Dataset loading and preprocessing
- `training_dataset.py`: Training dataset format with problem/solutions/target

**Training (`src/models/`)**:
- `grpo_trainer.py`: GRPO trainer implementation
  - Group-based reward calculation
  - KL divergence with reference model
  - LoRA-based parameter-efficient training

**Evaluation (`src/evaluation/`)**:
- `math_verifier.py`: Mathematical answer verification using math-verify library
- `benchmark.py`: Standard benchmark evaluation
- `comprehensive_benchmark.py`: Extended evaluation with multiple metrics

### Configuration Files

Main configuration is in `config/config.yaml` with sub-configs:
- `config/data/raw_dataset.yaml`: Stage 1 generation settings (vLLM, sampling params)
- `config/data/curation.yaml`: Stage 2 curation settings (strategy, set_size)
- `config/training/lora.yaml`: Stage 3 training settings (LoRA, GRPO, optimizer)
- `config/evaluation/benchmarks.yaml`: Stage 4 evaluation settings (datasets, metrics)

### Important Path Conventions

Default paths are configured in `config/config.yaml`:
- `paths.data_dir`: `/mnt/data1/datasets/nlp/conf_agg` - Main data directory
- `paths.output_dir`: `/mnt/data1/datasets/nlp/conf_agg/outputs` - Outputs
- `paths.model_dir`: `/mnt/data1/models/nlp/conf_agg` - Trained models
- `paths.log_dir`: `/mnt/data1/datasets/nlp/conf_agg/logs` - Logs
- `paths.cache_dir`: `/mnt/data1/datasets/nlp/cache` - Cache
- `paths.huggingface_cache`: `/mnt/data1/models/nlp/huggingface_cache` - HF cache

### Sampling Parameters

Standard sampling params across stages:
- Temperature: 1.5 (evaluation), 0.7 (training data generation)
- TopP: 0.95 (evaluation), 0.8 (training data generation)
- TopK: 20
- MinP: 0.0
- max_tokens: 16384 (evaluation), 10240 (data generation)

### Memory Optimization

- **float16 logprobs**: 50% memory savings
- **FP8 KV cache**: Enabled in vLLM config
- **Prefix caching**: Enabled for repeated prompts
- **GPU memory utilization**: 0.95 (Stage 1), 0.85 (Stage 4)

## Common Workflows

### Adding a New Benchmark Dataset

1. Add dataset config to `config/evaluation/benchmarks.yaml`:
   ```yaml
   datasets:
     - name: "NEW_DATASET"
       path: "data/benchmarks/new_dataset.jsonl"
   ```

2. Ensure dataset has required fields: `problem`/`question` and `answer`/`solution`

3. Update `scripts/stage4_1_generate.py` if special handling needed for dataset splits

### Modifying Confidence Calculation

1. Edit `src/data/confidence.py` ConfidenceCalculator class
2. Add new method following naming pattern: `*_confidence`
3. Update `config/data/raw_dataset.yaml` to include new method:
   ```yaml
   confidence:
     methods:
       - "mean_group_confidence"
       - "your_new_confidence"
   ```

### Training with Different LoRA Configurations

Edit `config/training/lora.yaml`:
```yaml
lora:
  r: 16              # LoRA rank
  lora_alpha: 32     # LoRA alpha
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Multi-GPU Training Setup

For DDP training, use the provided script:
```bash
# 2 GPUs
./run_stage3_2gpu.sh 0,1

# Check training logs
tail -f logs/stage3_2gpu_*.log
```

The script sets required environment variables:
- `CUDA_VISIBLE_DEVICES`
- `MASTER_ADDR`, `MASTER_PORT`
- `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`
- `TORCH_DISTRIBUTED_TIMEOUT`

## Important Implementation Details

### vLLM Engine Initialization

The `VLLMInferenceEngine` in `src/inference/vllm_engine.py` requires:
- `tensor_parallel_size`: Number of GPUs for inference
- `gpu_memory_utilization`: Fraction of GPU memory to use
- `max_model_len`: Maximum sequence length
- `trust_remote_code`: Required for Qwen models

### GRPO Training Loop

GRPO training (Stage 3) follows this workflow:
1. Load curated training data (problem + multiple solution candidates)
2. Generate new solutions using current policy
3. Calculate rewards based on correctness (math_verify)
4. Compute group-relative advantages
5. Optimize policy using PPO-style updates with KL penalty

Key parameter: `group_size` - number of solutions per problem to compare for advantage calculation

### Confidence Score Calculation

Confidence is calculated from logprobs at the token level:
- **mean_group_confidence**: Average logprob over token groups
- **bottom_10_percent_confidence**: Average of lowest 10% of token logprobs
- **tail_confidence**: Average logprob of last N% of tokens

Token groups of size `group_size` (default 512) are used to reduce memory.

### Data Curation Strategy

The `DataCurator` supports three strategies:
- **naive**: Random sampling of solutions
- **curriculum**: Progressively harder problems (Easy → Hard)
- **multitask**: Mixed difficulty levels

Currently configured to use "naive" strategy with 50% easy samples.

### Solution Verification

`MathVerifier` uses the `math-verify` library to check mathematical equivalence:
- Supports symbolic math comparison
- Timeout-based execution (default 30s)
- Returns boolean correctness

## WandB Integration

WandB logging is enabled by default:
1. Set `WANDB_API_KEY` in `.env` file or environment
2. Configure in `config/config.yaml`:
   ```yaml
   experiment:
     wandb:
       enabled: true
       project: "conf-agg-llm"
   ```

## Docker Configuration

The `docker-compose.yml` mounts:
- `/home/najoo0/Conf_Agg:/workspace` - Source code
- `/data1:/data1`, `/data2:/data2` - Data volumes
- `/root/.cache/huggingface:/root/.cache/huggingface` - HF cache

Environment variables:
- `VLLM_USE_FLASHINFER=1` - Enable FlashInfer optimization
- `UV_CACHE_DIR=/tmp/uv-cache` - uv package cache
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` - CUDA memory allocator

## Troubleshooting

### GPU Memory Issues

If OOM during Stage 1/4:
- Reduce `vllm.gpu_memory_utilization` in config
- Reduce `max_model_len` or `max_tokens`
- Reduce `max_num_seqs` or `max_num_batched_tokens`

If OOM during Stage 3:
- Reduce `training.batch_size` in `config/training/lora.yaml`
- Reduce `grpo.num_generations`
- Reduce `training.max_prompt_length` or `training.max_response_length`
- If using vLLM colocate mode, reduce `vllm_gpu_memory_utilization`

### vLLM Initialization Failures

Check:
- CUDA version compatibility (requires CUDA 12.x)
- Sufficient GPU memory available
- `trust_remote_code: true` is set for Qwen models
- FlashInfer installation: `./install_flashinfer.sh`

### Multi-GPU Training Issues

If DDP fails to initialize:
- Verify `MASTER_ADDR` and `MASTER_PORT` are set correctly
- Check firewall allows localhost communication on port 29500
- Ensure GPUs are visible: `nvidia-smi`
- Check NCCL environment variables are set (see `run_stage3_2gpu.sh`)
