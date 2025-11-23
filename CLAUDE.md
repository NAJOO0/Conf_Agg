# CLAUDE.md - AI Assistant Guide for Conf-AggLLM

**Last Updated:** 2025-11-23
**Project:** Conf-AggLLM - Confidence-Aware Aggregation Model Framework
**Language:** Python 3.10+
**Primary Use:** Mathematical Reasoning Research

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Technologies](#core-technologies)
4. [Four-Stage Pipeline](#four-stage-pipeline)
5. [Development Environment](#development-environment)
6. [Configuration Management](#configuration-management)
7. [Code Conventions](#code-conventions)
8. [Common Tasks](#common-tasks)
9. [Key Source Modules](#key-source-modules)
10. [Debugging & Troubleshooting](#debugging--troubleshooting)
11. [Important Paths](#important-paths)

---

## Project Overview

### Goal
Achieve the same mathematical reasoning performance as 50-100 majority-voted inference results using only 1-2 inference results with confidence scores. This dramatically reduces computational costs while maintaining high accuracy.

### Key Innovation
**Confidence-Aware Aggregation**: A GRPO-trained model that intelligently aggregates multiple candidate solutions weighted by confidence scores, rather than naive majority voting.

### Performance Targets
- **Benchmarks**: AIME24, AIME25, HMMT24, HMMT25
- **Base Model**: Qwen2.5-1.5B / Qwen3-1.7B-FP8
- **Optimization**: 3-4x speedup via 4-GPU Ray Serve parallel processing
- **Memory**: 50% reduction using float16 logprobs

---

## Repository Structure

```
Conf_Agg/
├── config/                          # Hydra configuration files
│   ├── config.yaml                  # Main config (orchestrates all stages)
│   ├── data/
│   │   ├── raw_dataset.yaml         # Stage 1: Generation settings
│   │   └── curation.yaml            # Stage 2: Curation settings
│   ├── training/
│   │   └── lora.yaml                # Stage 3: GRPO training config
│   └── evaluation/
│       └── benchmarks.yaml          # Stage 4: Evaluation settings
│
├── src/                             # Source code modules
│   ├── data/
│   │   ├── dataset.py               # Dataset loading and management
│   │   ├── curation.py              # Data curation logic (Hard/Easy split)
│   │   ├── confidence.py            # Confidence score calculators
│   │   ├── training_dataset.py      # GRPO training dataset
│   │   └── clean_dataset.py         # Data cleaning utilities
│   ├── models/
│   │   └── grpo_trainer.py          # GRPO trainer implementation
│   ├── inference/
│   │   ├── vllm_engine.py           # vLLM inference engine wrapper
│   │   └── local_engine.py          # Local inference fallback
│   ├── evaluation/
│   │   ├── math_verifier.py         # Math answer verification
│   │   ├── benchmark.py             # Benchmark evaluation
│   │   └── comprehensive_benchmark.py # Full evaluation suite
│   └── utils/
│       ├── logging.py               # Logging utilities
│       └── metrics.py               # Metrics computation
│
├── scripts/                         # Executable scripts
│   ├── stage1_generate.py           # Stage 1: Single-GPU generation
│   ├── stage1_generate_async.py     # Stage 1: Async multi-GPU generation
│   ├── run_stage1_async.sh          # Stage 1: 4-GPU parallel launcher
│   ├── stage2_curate.py             # Stage 2: Data curation
│   ├── stage3_train.py              # Stage 3: GRPO training
│   ├── stage3_train_2.py            # Stage 3: Alternative trainer
│   ├── stage4_1_generate.py         # Stage 4: Generate benchmark responses
│   ├── stage4_2_evaluate_metrics.py # Stage 4: Compute metrics
│   ├── stage4_3_evaluate_aggregation.py # Stage 4: Aggregation eval
│   └── stage4_evaluate.py           # Stage 4: Full evaluation
│
├── data/                            # Data directories (gitignored)
│   ├── raw/                         # Original datasets (e.g., deepscaler.jsonl)
│   ├── generated/                   # Stage 1 outputs
│   ├── curated/                     # Stage 2 outputs
│   └── benchmarks/                  # AIME/HMMT benchmark datasets
│
├── outputs/                         # Output directories (gitignored)
│   ├── models/                      # Trained LoRA models
│   ├── logs/                        # Experiment logs
│   └── results/                     # Evaluation results
│
├── docs/                            # Documentation
│   ├── QUICKSTART.md                # 5-minute quick start
│   ├── DEPLOYMENT_GUIDE.md          # Full deployment (English)
│   ├── DEPLOYMENT_KR.md             # Full deployment (Korean)
│   └── DEPLOYMENT_NO_DOCKER.md      # Non-Docker deployment
│
├── Dockerfile                       # Docker container definition
├── docker-compose.yml               # Docker Compose orchestration
├── requirements.txt                 # Python dependencies
├── uv.toml                          # UV package manager config
├── uv.lock                          # UV lock file
├── config.json                      # Consolidated JSON config (legacy)
└── README.md                        # Project README (Korean)
```

### Data Flow

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

## Core Technologies

### ML/DL Frameworks
- **PyTorch** (≥2.1.0): Core deep learning framework
- **Transformers** (≥4.51.0): HuggingFace model loading
- **PEFT** (≥0.6.0): LoRA parameter-efficient fine-tuning
- **TRL** (≥0.7.0): GRPO reinforcement learning trainer
- **vLLM**: High-speed inference engine with continuous batching

### Inference Optimization
- **FlashInfer**: CUDA kernel optimization for sampling
- **Ray Serve**: Multi-GPU parallel processing (4 GPUs)
- **FP8 KV Cache**: Memory-efficient key-value caching
- **Tensor Parallelism**: Model parallelism across GPUs

### Configuration & Logging
- **Hydra** (≥1.3.0): Hierarchical configuration management
- **WandB** (≥0.15.0): Experiment tracking and visualization

### Math Verification
- **math_verify** (≥0.1.0): Mathematical answer verification

### Package Management
- **uv**: Fast Python package manager (replaces pip/poetry)

### Sampling Parameters (Default)
- TopP: 0.95
- TopK: 20
- MinP: 0.0
- Temperature: 0.6 (generation), 1.5 (aggregation)
- Max Tokens: 32768

---

## Four-Stage Pipeline

### Stage 1: Raw Data Generation
**Script:** `scripts/stage1_generate_async.py`
**Config:** `config/data/raw_dataset.yaml`

**Purpose:** Generate multiple candidate solutions with confidence scores.

**Process:**
1. Load raw math problems from `data/raw/deepscaler.jsonl`
2. Use vLLM AsyncLLMEngine for continuous batching
3. Generate `num_responses_per_problem` solutions per problem
4. Extract top-k logprobs (default k=5) for confidence calculation
5. Calculate confidence scores:
   - `mean_group_confidence`: Average logprob over groups
   - `bottom_10_percent_confidence`: Robustness to low-confidence tokens
   - `tail_confidence`: Confidence in final tokens
6. Save to `data/generated/generated_responses.parquet`

**Key Parameters:**
- `num_responses_per_problem`: 2 (default)
- `temperature`: 0.6
- `logprobs`: 5 (top-5 logprobs)
- `group_size`: 512 tokens per group
- `max_tokens`: 32768

**GPU Usage:**
- Single GPU: `SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py --gpu-id 0`
- 4 GPUs parallel: `bash scripts/run_stage1_async.sh`

**Output Format (Parquet):**
```python
{
    "problem_id": str,
    "problem_text": str,
    "ground_truth": str,
    "response_id": str,
    "generated_text": str,
    "output_token_count": int,
    "logprobs": List[List[float16]],  # Memory-optimized
    "mean_group_confidence": float,
    "bottom_10_percent_confidence": float,
    "tail_confidence": float,
    "worker_gpu": str,
    "worker_replica": str
}
```

---

### Stage 2: Data Curation
**Script:** `scripts/stage2_curate.py`
**Config:** `config/data/curation.yaml`

**Purpose:** Create training sets by categorizing problems and creating diverse solution sets.

**Process:**
1. Load generated responses from Stage 1
2. Verify each solution using `math_verify` library
3. Classify problems as Hard (low solve rate) or Easy (high solve rate)
4. Create `num_sets_per_problem` solution sets per problem
5. Each set contains `set_size` candidate solutions
6. Split into train (80%) and validation (20%)

**Key Parameters:**
- `strategy`: "curriculum" (prioritize hard problems)
- `easy_sample_percentage`: 50 (balance hard/easy)
- `num_sets_per_problem`: 16 sets per problem
- `set_size`: 8 solutions per set
- `confidence_key`: "tail_confidence" (which confidence to use)

**Curation Strategies:**
- **Curriculum**: Focus on hard problems for better learning
- **Naive**: Random sampling
- **Multitask**: Mixed difficulty levels

**Output:**
- `data/curated/train.parquet`
- `data/curated/validation.parquet`

**Data Structure:**
```python
{
    "problem_text": str,
    "ground_truth": str,
    "solutions": List[str],  # JSON-serialized for Parquet
    "confidence_scores": Dict[str, List[float]],  # JSON-serialized
    "aggregator_prompt": str,  # Full prompt for training
    "num_correct": int,
    "is_hard": bool
}
```

---

### Stage 3: GRPO Training
**Script:** `scripts/stage3_train.py`
**Config:** `config/training/lora.yaml`

**Purpose:** Train a LoRA-adapted model using Group-Relative Policy Optimization.

**Process:**
1. Load base model (Qwen3-1.7B-FP8)
2. Apply LoRA adapters to attention and MLP layers
3. Train using GRPO algorithm with groups of solutions
4. Optimize for selecting correct solutions based on confidence
5. Save checkpoints to `outputs/models/`

**GRPO Details:**
- **Group Size**: 8 solutions per optimization step
- **KL Coefficient**: 0.001 (regularization)
- **Aggregator Temperature**: 1.5 (exploration)
- **Reward**: +1 for correct selection, 0 otherwise

**LoRA Configuration:**
```python
{
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # Scaling factor
    "lora_dropout": 0.1,        # Dropout rate
    "target_modules": [         # Modules to adapt
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

**Training Configuration:**
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

**Output:**
- `outputs/models/grpo_trainer_lora_model_0/`
- `outputs/models/grpo_trainer_lora_model_1/`
- `outputs/models/grpo_trainer_lora_model_final/`

---

### Stage 4: Benchmark Evaluation
**Scripts:**
- `scripts/stage4_1_generate.py`: Generate responses
- `scripts/stage4_2_evaluate_metrics.py`: Compute metrics
- `scripts/stage4_3_evaluate_aggregation.py`: Evaluate aggregation

**Config:** `config/evaluation/benchmarks.yaml`

**Purpose:** Evaluate trained model on benchmark datasets.

**Benchmarks:**
- **AIME24** (`data/benchmarks/aime24.jsonl`)
- **AIME25** (`data/benchmarks/aime25.jsonl`)
- **HMMT24** (`data/benchmarks/hmmt24.jsonl`)
- **HMMT25** (`data/benchmarks/hmmt25.jsonl`)

**Process:**
1. Generate 8 candidate solutions per benchmark problem
2. Use trained model to select best solution based on confidence
3. Verify correctness using math_verify
4. Compute metrics: pass@1, pass@k, confidence correlation

**Metrics:**
- **pass@1**: Single-solution accuracy
- **pass@k**: At least one correct in k attempts
- **confidence_correlation**: How well confidence predicts correctness

**Output:**
- `outputs/results/{benchmark}_predictions.json`
- `outputs/results/{benchmark}_metrics.json`

---

## Development Environment

### Docker (Recommended)

**Setup:**
```bash
# Build and start container
docker-compose up -d

# Access container
docker-compose exec conf-agg-llm bash

# Inside container: sync dependencies
uv sync

# Verify GPU access
nvidia-smi
```

**Container Details:**
- **Base Image**: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
- **Python**: 3.10+
- **GPUs**: 0,1,2,3 (configurable in docker-compose.yml)
- **Shared Memory**: 16GB (`shm_size: "16g"`)
- **Working Directory**: `/workspace`

**Volume Mounts:**
```yaml
volumes:
  - /home/najoo0/Conf_Agg:/workspace           # Code
  - /data1:/data1                               # Data storage
  - /data2:/data2                               # Additional storage
  - /root/.cache/huggingface:/root/.cache/huggingface  # Model cache
  - uv-cache:/tmp/uv-cache                      # UV cache
```

### Environment Variables

**Required:**
```bash
WANDB_API_KEY=your_wandb_key_here              # WandB tracking
```

**Automatically Set:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3                   # GPU visibility
NVIDIA_VISIBLE_DEVICES=all                     # All GPUs available
PYTHONPATH=/workspace                          # Python module path
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # CUDA memory management
VLLM_USE_FLASHINFER=1                          # Enable FlashInfer optimization
UV_CACHE_DIR=/tmp/uv-cache                     # UV cache location
```

### Non-Docker Setup

See `docs/DEPLOYMENT_NO_DOCKER.md` for direct Python installation.

---

## Configuration Management

### Hydra Configuration System

**Main Config:** `config/config.yaml`

Hydra uses **composition** to build final configuration:

```yaml
# config/config.yaml
defaults:
  - data: raw_dataset
  - training: lora
  - evaluation: benchmarks
  - _self_

# Override from CLI:
# python script.py data.num_responses=4 training.epochs=2
```

### Configuration Files

**1. Data Generation** (`config/data/raw_dataset.yaml`)
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

**2. Curation** (`config/data/curation.yaml`)
```yaml
strategy: curriculum
easy_sample_percentage: 50
num_sets_per_problem: 16
set_size: 8
```

**3. Training** (`config/training/lora.yaml`)
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

**4. Evaluation** (`config/evaluation/benchmarks.yaml`)
```yaml
datasets:
  - name: AIME24
    path: data/benchmarks/aime24.jsonl

evaluation:
  num_candidates: 8
  temperature: 1.5
  max_tokens: 16384
```

### Legacy Config

`config.json` contains a flattened JSON version of all configs. Prefer Hydra YAML files for modifications.

---

## Code Conventions

### File Organization

**Scripts** (`scripts/`):
- Entry points for each stage
- Use Hydra for configuration loading
- Include logging setup and error handling
- Follow naming: `stage{N}_{action}.py`

**Source Modules** (`src/`):
- Reusable logic (no CLI arguments)
- Type hints required
- Docstrings for all public functions/classes
- Organized by functionality (data, models, inference, evaluation)

### Python Style

**Imports:**
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel

# Local
from src.data.dataset import RawDataset
from src.utils.logging import setup_logging
```

**Type Hints:**
```python
def process_data(
    input_path: str,
    config: Dict[str, Any],
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process input data according to config."""
    pass
```

**Docstrings:**
```python
def calculate_confidence(logprobs: List[List[float]], method: str = "mean") -> float:
    """
    Calculate confidence score from logprobs.

    Args:
        logprobs: List of token-level logprob distributions
        method: Calculation method ("mean", "tail", "bottom_10")

    Returns:
        Confidence score (0.0-1.0)

    Raises:
        ValueError: If method is unknown
    """
```

### Logging

**Setup:**
```python
import logging
from src.utils.logging import setup_logging

setup_logging(
    log_level="INFO",
    log_file="outputs/logs/stage1.log"
)
logger = logging.getLogger(__name__)
```

**Usage:**
```python
logger.info(f"Processing {len(problems)} problems")
logger.warning(f"GPU memory usage high: {usage:.2f}%")
logger.error(f"Failed to load model: {e}")
```

### Error Handling

**Graceful Failures:**
```python
try:
    result = math_verify.verify(prediction, ground_truth)
except Exception as e:
    logger.warning(f"Verification failed: {e}, falling back to string match")
    result = prediction.strip() == ground_truth.strip()
```

**Early Returns:**
```python
if not logprobs:
    logger.warning("Empty logprobs, returning default confidence")
    return 0.0
```

### Data Serialization

**For Parquet (no nested lists/dicts):**
```python
# Serialize nested structures
serialized_set = {
    "solutions": json.dumps(solutions_list),
    "confidence_scores": json.dumps(scores_dict)
}
```

**For JSON (nested OK):**
```python
with open("output.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### Memory Optimization

**float16 for logprobs:**
```python
# Convert to float16 to save 50% memory
logprobs_fp16 = np.array(logprobs, dtype=np.float16).tolist()
```

**Streaming saves:**
```python
# Save incrementally, don't accumulate in memory
async with aiofiles.open(output_file, "a") as f:
    await f.write(json.dumps(result) + "\n")
```

---

## Common Tasks

### 1. Running Stage 1 (Data Generation)

**Single GPU:**
```bash
# Inside container
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --config-path config \
    --config-name config \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1
```

**4 GPUs Parallel:**
```bash
# Launches 4 workers (GPU 0,1,2,3) in background
uv run bash scripts/run_stage1_async.sh
```

**Monitoring:**
```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f outputs/logs/stage1_generate_shard_0.log
tail -f outputs/logs/stage1_generate_shard_1.log
```

### 2. Running Stage 2 (Curation)

```bash
uv run python scripts/stage2_curate.py
```

**Output:**
- `data/curated/train.parquet`
- `data/curated/validation.parquet`

### 3. Running Stage 3 (Training)

```bash
uv run python scripts/stage3_train.py
```

**Monitor with WandB:**
- Project: `conf-agg-llm`
- Metrics: loss, reward, kl_divergence

**Checkpoints:**
- Saved every 500 steps to `outputs/models/`

### 4. Running Stage 4 (Evaluation)

**Full Pipeline:**
```bash
# Generate predictions
uv run python scripts/stage4_1_generate.py

# Compute metrics
uv run python scripts/stage4_2_evaluate_metrics.py

# Evaluate aggregation
uv run python scripts/stage4_3_evaluate_aggregation.py
```

**Or All-in-One:**
```bash
uv run python scripts/stage4_comprehensive_evaluate.py
```

### 5. Modifying Configuration

**Via CLI Override:**
```bash
python scripts/stage1_generate.py \
    data.generation.num_responses_per_problem=4 \
    data.vllm.gpu_memory_utilization=0.8
```

**Via File Edit:**
```bash
# Edit config
nano config/data/raw_dataset.yaml

# Run with updated config
python scripts/stage1_generate.py
```

### 6. Adding New Benchmark

**1. Add dataset file:**
```bash
# Add to data/benchmarks/new_benchmark.jsonl
# Format: {"problem": "...", "answer": "..."}
```

**2. Update config:**
```yaml
# config/evaluation/benchmarks.yaml
datasets:
  - name: NEW_BENCHMARK
    path: data/benchmarks/new_benchmark.jsonl
```

**3. Run evaluation:**
```bash
uv run python scripts/stage4_comprehensive_evaluate.py
```

### 7. Debugging GPU Issues

**Check GPU visibility:**
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

**Reduce memory usage:**
```yaml
# config/data/raw_dataset.yaml
vllm:
  gpu_memory_utilization: 0.7  # Lower from 0.95
  max_num_seqs: 20              # Lower from 40
```

**Clear GPU memory:**
```python
import torch
torch.cuda.empty_cache()
```

### 8. Resuming Interrupted Training

```python
# scripts/stage3_train.py supports checkpoint resuming
# Modify script to load from checkpoint:
model = AutoModelForCausalLM.from_pretrained(
    "outputs/models/grpo_trainer_lora_model_1"
)
```

### 9. Uploading Model to HuggingFace

```bash
uv run python scripts/upload_to_hf.py \
    --model-path outputs/models/grpo_trainer_lora_model_final \
    --repo-name your-username/conf-agg-model
```

---

## Key Source Modules

### `src/data/dataset.py`
**Purpose:** Load and manage raw datasets.

**Key Classes:**
- `RawDataset`: Loads JSONL math problems
- `GeneratedDataset`: Loads Stage 1 parquet outputs

**Usage:**
```python
from src.data.dataset import RawDataset

dataset = RawDataset(data_path="data/raw/deepscaler.jsonl")
problems = dataset.load()  # Returns List[Dict]
```

### `src/data/curation.py`
**Purpose:** Curate training data from generated responses.

**Key Classes:**
- `DataCurator`: Implements curation strategies

**Key Methods:**
- `classify_hard_easy_sets()`: Categorize by difficulty
- `create_solution_sets()`: Build diverse training sets
- `curate()`: Full curation pipeline

**Usage:**
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
**Purpose:** Calculate confidence scores from logprobs.

**Key Classes:**
- `ConfidenceCalculator`: Implements multiple confidence methods

**Confidence Methods:**
1. **mean_group_confidence**: Average logprob over token groups
2. **bottom_10_percent_confidence**: Robustness to worst tokens
3. **tail_confidence**: Confidence in final tokens

**Usage:**
```python
from src.data.confidence import ConfidenceCalculator

calc = ConfidenceCalculator(group_size=512)
scores = calc.calculate_all_confidence_scores(logprobs)
# Returns: {"mean_group_confidence": 0.85, ...}
```

### `src/models/grpo_trainer.py`
**Purpose:** GRPO training implementation.

**Key Classes:**
- `GRPOTrainer`: Manages GRPO training loop

**Key Methods:**
- `_initialize_model()`: Load model with LoRA
- `train()`: Execute training
- `compute_rewards()`: Calculate GRPO rewards

**Key Concepts:**
- Uses reference model for KL divergence
- Group-based reward calculation
- LoRA for parameter efficiency

### `src/inference/vllm_engine.py`
**Purpose:** High-speed inference via vLLM.

**Key Classes:**
- `VLLMInferenceEngine`: Wrapper for vLLM LLM class

**Key Methods:**
- `generate_multiple_responses()`: Batch generation
- `_initialize_model()`: Setup vLLM engine

**Optimization Features:**
- Tensor parallelism
- KV cache with FP8
- Continuous batching
- Prefix caching

**Usage:**
```python
from src.inference.vllm_engine import VLLMInferenceEngine

engine = VLLMInferenceEngine(
    model_name="Qwen/Qwen3-1.7B-FP8",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95
)
responses = engine.generate_multiple_responses(
    prompt="Solve: 2x + 3 = 7",
    n=8,
    temperature=0.6
)
```

### `src/evaluation/math_verifier.py`
**Purpose:** Verify mathematical answers.

**Key Classes:**
- `MathVerifier`: Wrapper for math_verify library

**Key Methods:**
- `verify_answer()`: Single answer verification
- `verify_batch()`: Batch verification
- `extract_final_answer_from_content()`: Extract `\boxed{}` answers

**Usage:**
```python
from src.evaluation.math_verifier import MathVerifier

verifier = MathVerifier(timeout=30)
is_correct = verifier.verify_answer(
    predicted="42",
    ground_truth="42"
)
```

### `src/evaluation/benchmark.py`
**Purpose:** Benchmark evaluation logic.

**Key Functions:**
- Load benchmark datasets
- Run model on benchmarks
- Compute pass@k metrics

### `src/utils/logging.py`
**Purpose:** Centralized logging setup.

**Key Functions:**
- `setup_logging()`: Configure file + console logging

---

## Debugging & Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```yaml
# Reduce in config/data/raw_dataset.yaml
vllm:
  gpu_memory_utilization: 0.7  # Down from 0.95
  max_num_seqs: 20              # Down from 40
  max_num_batched_tokens: 8192  # Down from 16384
```

#### 2. Container Keeps Restarting

**Check logs:**
```bash
docker-compose logs conf-agg-llm
```

**Common causes:**
- Missing .env file
- Invalid volume paths
- GPU driver mismatch

**Solution:**
```bash
# Rebuild without cache
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### 3. vLLM Model Loading Fails

**Symptoms:**
```
Failed to load model: trust_remote_code
```

**Solution:**
```yaml
# Ensure in config
vllm:
  trust_remote_code: true
```

#### 4. math_verify Timeout

**Symptoms:**
```
Verification timeout after 30s
```

**Solution:**
```python
# Increase timeout in curation config
verification:
  timeout: 60  # Increase from 30
```

#### 5. WandB Not Logging

**Check:**
```bash
echo $WANDB_API_KEY
```

**Solution:**
```bash
# Add to .env file
WANDB_API_KEY=your_key_here

# Restart container
docker-compose restart
```

#### 6. Hydra Config Not Found

**Symptoms:**
```
FileNotFoundError: config/config.yaml
```

**Solution:**
```bash
# Run from project root
cd /workspace  # In container
python scripts/stage1_generate.py
```

#### 7. Parquet Read/Write Errors

**Symptoms:**
```
ArrowNotImplementedError: Nested types not supported
```

**Cause:** Parquet doesn't support nested lists/dicts.

**Solution:**
```python
# Serialize nested structures
data["solutions"] = json.dumps(solutions_list)
```

### Performance Debugging

**Profile GPU usage:**
```bash
# Install nvitop (already in requirements)
nvitop

# Or use standard nvidia-smi
watch -n 1 nvidia-smi
```

**Profile Python:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(20)
```

**Monitor WandB:**
- Check `conf-agg-llm` project
- Look for metrics plateaus
- Watch GPU utilization charts

### Logging Best Practices

**Set appropriate log levels:**
```python
# During development
setup_logging(log_level="DEBUG")

# In production
setup_logging(log_level="INFO")
```

**Check logs:**
```bash
# Container logs
docker-compose logs -f conf-agg-llm

# Stage-specific logs
tail -f outputs/logs/stage1_generate.log
tail -f outputs/logs/stage2_curate.log
tail -f outputs/logs/stage3_train.log
```

---

## Important Paths

### Data Paths

**Raw Data:**
```
data/raw/deepscaler.jsonl          # Training problems
data/benchmarks/aime24.jsonl       # AIME 2024 benchmark
data/benchmarks/aime25.jsonl       # AIME 2025 benchmark
data/benchmarks/hmmt24.jsonl       # HMMT 2024 benchmark
data/benchmarks/hmmt25.jsonl       # HMMT 2025 benchmark
```

**Generated Data:**
```
data/generated/generated_responses.parquet       # Stage 1 output
data/generated/generated_responses_shard_*.parquet  # Parallel outputs
```

**Curated Data:**
```
data/curated/train.parquet         # Training set
data/curated/validation.parquet    # Validation set
```

### Model Paths

**Checkpoints:**
```
outputs/models/grpo_trainer_lora_model_0/      # Initial checkpoint
outputs/models/grpo_trainer_lora_model_1/      # Checkpoint 1
outputs/models/grpo_trainer_lora_model_final/  # Final model
```

**Cache:**
```
/root/.cache/huggingface/          # HuggingFace model cache
/mnt/data1/models/nlp/huggingface_cache/  # External cache
```

### Log Paths

```
outputs/logs/stage1_generate.log
outputs/logs/stage1_generate_shard_0.log
outputs/logs/stage2_curate.log
outputs/logs/stage3_train.log
outputs/logs/stage4_evaluate.log
```

### Config Paths

**Hydra:**
```
config/config.yaml                 # Main orchestrator
config/data/raw_dataset.yaml       # Stage 1
config/data/curation.yaml          # Stage 2
config/training/lora.yaml          # Stage 3
config/evaluation/benchmarks.yaml  # Stage 4
```

**Legacy:**
```
config.json                        # Flattened JSON config
```

### Output Paths

```
outputs/results/aime24_predictions.json
outputs/results/aime24_metrics.json
outputs/results/benchmark_summary.json
```

### Symbolic Links

```
output_s -> /mnt/data1/datasets/nlp/conf_agg/
```

**Note:** `output_s` is a symbolic link to external storage. Check actual location before assuming paths.

---

## Additional Resources

### Documentation
- `README.md`: Main project README (Korean)
- `docs/QUICKSTART.md`: 5-minute deployment guide
- `docs/DEPLOYMENT_GUIDE.md`: Full deployment (English)
- `docs/DEPLOYMENT_KR.md`: Full deployment (Korean)
- `docs/DEPLOYMENT_NO_DOCKER.md`: Non-Docker setup
- `RESTART_GUIDE.md`: Restart procedures

### Scripts Reference
- `quick_restart.sh`: Fast restart script
- `restart_setup.sh`: Full environment rebuild
- `download_data.sh`: Download benchmark datasets
- `install_flashinfer.sh`: Install FlashInfer optimization
- `setup_grpo_training.sh`: GRPO environment setup

### External Links
- **vLLM Docs**: https://docs.vllm.ai/
- **Qwen Models**: https://huggingface.co/Qwen
- **Hydra Docs**: https://hydra.cc/
- **WandB Docs**: https://docs.wandb.ai/

---

## Workflow Summary for AI Assistants

### When Modifying Code

1. **Read before writing:** Always read existing files before modifying
2. **Check configs:** Verify Hydra configs match code expectations
3. **Test locally:** Use single-GPU mode for quick testing
4. **Monitor resources:** Watch GPU memory during development
5. **Log extensively:** Add logging for debugging
6. **Update docs:** Keep this file updated with major changes

### When Debugging

1. **Check logs first:** `outputs/logs/` contains detailed traces
2. **Verify GPU access:** `nvidia-smi` should show GPUs
3. **Check container status:** `docker-compose ps`
4. **Validate configs:** Ensure YAML syntax and paths correct
5. **Check environment:** `.env` file must exist with WANDB_API_KEY

### When Adding Features

1. **Follow structure:** Put code in appropriate `src/` module
2. **Use type hints:** All new functions need type annotations
3. **Add docstrings:** Explain purpose, args, returns
4. **Update configs:** Add new parameters to Hydra configs
5. **Update this file:** Document new features in CLAUDE.md

### Best Practices

- **Prefer `uv run`** over direct `python` for consistency
- **Use Hydra** for all configuration (avoid hardcoding)
- **Log everything** important for reproducibility
- **Serialize carefully** when saving to Parquet
- **Test with small data** before full runs
- **Monitor GPU** during intensive operations
- **Version control** configs along with code

---

**End of CLAUDE.md**
