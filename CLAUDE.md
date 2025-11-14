# CLAUDE.md - AI Assistant Guide for Conf-AggLLM

> **Last Updated**: 2025-11-14
> **Version**: 1.0.0
> **Purpose**: This document provides comprehensive guidance for AI assistants (like Claude) working with the Conf-AggLLM codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Development Environment](#development-environment)
4. [Key Technologies & Frameworks](#key-technologies--frameworks)
5. [Configuration System](#configuration-system)
6. [Development Workflows](#development-workflows)
7. [Testing & Validation](#testing--validation)
8. [Code Conventions](#code-conventions)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)
11. [Important Constraints](#important-constraints)

---

## Project Overview

### What is Conf-AggLLM?

**Conf-AggLLM** (Confidence-Aware Aggregation for Large Language Models) is a research framework designed to improve LLM mathematical reasoning performance and efficiency. The key innovation is achieving ensemble-level performance (typically requiring 50-100 inference samples with majority voting) using only 1-2 inference attempts by leveraging confidence scores.

### Project Goals

1. **Performance**: Match or exceed majority-voting ensemble performance with minimal inference
2. **Efficiency**: Reduce computational cost by 50-100x compared to traditional ensemble methods
3. **Generalization**: Train small models (e.g., Qwen3-1.7B) to aggregate predictions effectively
4. **Reproducibility**: Provide complete pipeline with configuration management via Hydra

### Core Innovation

The framework trains a confidence-aware aggregation model using GRPO (Group Relative Policy Optimization) that learns to:
- Extract reliable confidence scores from token log-probabilities
- Aggregate multiple predictions weighted by confidence
- Achieve high accuracy with minimal computational overhead

---

## Codebase Structure

### Directory Layout

```
Conf_Agg/
├── config/                      # Hydra configuration files (YAML)
│   ├── config.yaml              # Main configuration
│   ├── data/
│   │   ├── raw_dataset.yaml     # Stage 1: Data generation settings
│   │   └── curation.yaml        # Stage 2: Data curation settings
│   ├── training/
│   │   └── lora.yaml            # Stage 3: Training configuration (GRPO + LoRA)
│   └── evaluation/
│       └── benchmarks.yaml      # Stage 4: Evaluation benchmarks
│
├── src/                         # Source code (~4,361 Python lines)
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── confidence.py        # Confidence score calculation from logprobs
│   │   ├── curation.py          # Dataset curation (Hard/Easy classification)
│   │   ├── dataset.py           # Dataset loading and preprocessing
│   │   ├── clean_dataset.py     # Data cleaning utilities
│   │   └── training_dataset.py  # Training data preparation
│   ├── inference/               # Inference engines
│   │   ├── vllm_engine.py       # vLLM-based high-speed inference
│   │   └── local_engine.py      # Local inference engine (fallback)
│   ├── models/                  # Model training
│   │   └── grpo_trainer.py      # GRPO algorithm implementation
│   ├── evaluation/              # Evaluation modules
│   │   ├── math_verifier.py     # Mathematical answer verification
│   │   ├── benchmark.py         # Benchmark evaluation logic
│   │   └── comprehensive_benchmark.py
│   └── utils/                   # Utility modules
│       ├── logging.py           # Logging configuration
│       └── metrics.py           # Performance metrics
│
├── scripts/                     # Execution scripts
│   ├── stage1_generate.py       # Raw data generation (multi-GPU)
│   ├── stage1_generate_async.py # Async version with Ray Serve
│   ├── stage2_curate.py         # Data curation
│   ├── stage3_train.py          # Model training (GRPO)
│   ├── stage3_train_2.py        # Alternative training script
│   ├── stage4_1_generate.py     # Generate evaluation data
│   ├── stage4_2_evaluate_metrics.py  # Compute metrics
│   ├── stage4_3_evaluate_aggregation.py  # Aggregation evaluation
│   ├── stage4_evaluate.py       # Complete benchmark evaluation
│   ├── run_stage1_async.sh      # Multi-GPU async execution
│   └── prev/                    # Archived/previous scripts
│
├── data/                        # Data directory (mostly gitignored)
│   ├── raw/                     # Original datasets (e.g., deepscaler.jsonl)
│   ├── generated/               # Stage 1 outputs (parquet files)
│   ├── curated/                 # Stage 2 outputs (train/val splits)
│   └── benchmarks/              # Evaluation datasets
│       ├── aime24.jsonl         # AIME 2024
│       ├── aime25.jsonl         # AIME 2025
│       ├── hmmt24.jsonl         # HMMT 2024
│       └── hmmt25.jsonl         # HMMT 2025
│
├── outputs/                     # Output directory (gitignored)
│   ├── models/                  # Trained model checkpoints
│   ├── logs/                    # Training and execution logs
│   └── results/                 # Evaluation results
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md            # 5-minute quick start guide
│   ├── DEPLOYMENT_KR.md         # Korean deployment guide
│   ├── DEPLOYMENT_GUIDE.md      # English deployment guide
│   ├── DEPLOYMENT_NO_DOCKER.md  # Direct Python execution
│   └── EXECUTION_GUIDE.md       # Execution instructions
│
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Multi-container orchestration
├── pyproject.toml               # Python project configuration (uv)
├── uv.toml                      # UV package manager config
├── uv.lock                      # Locked dependencies
├── requirements.txt             # Python dependencies (legacy)
├── config.json                  # JSON configuration (alternative)
├── vllm_servers.json            # vLLM server configuration
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variables template
├── setup.sh                     # Initial setup script
├── restart_setup.sh             # Server restart/deployment
└── README.md                    # Project README (Korean)
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main Hydra configuration with project-wide settings |
| `scripts/stage1_generate.py` | Primary data generation script |
| `src/data/confidence.py` | Core confidence calculation logic |
| `src/models/grpo_trainer.py` | GRPO training algorithm |
| `docker-compose.yml` | Container orchestration with GPU support |
| `.gitignore` | Excludes data/, outputs/, .env from version control |

---

## Development Environment

### Technology Stack

**Core Languages:**
- Python 3.12 (primary)
- Bash (deployment scripts)
- YAML/JSON (configuration)

**Package Management:**
- **UV** (primary): Fast, modern Python package manager
- **pip** (fallback): Traditional package manager

**Container Platform:**
- Docker with NVIDIA GPU support
- Docker Compose for multi-service orchestration

### Environment Setup

#### Method 1: Docker (Recommended)

```bash
# Build and start container
docker-compose up -d

# Enter container
docker-compose exec conf-agg-llm bash

# Inside container: sync dependencies
uv sync

# Verify GPU access
nvidia-smi
```

#### Method 2: Direct Python

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=/path/to/Conf_Agg
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Required Environment Variables

Create a `.env` file with:

```bash
# WandB (experiment tracking)
WANDB_API_KEY=your_wandb_api_key

# Sampling limit (for testing)
SAMPLE_LIMIT=400

# CUDA settings
CUDA_VISIBLE_DEVICES=0,1,2,3

# vLLM optimizations
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_FLASHINFER_SAMPLER=1
```

### GPU Requirements

- **Minimum**: 1 GPU with 40GB+ VRAM (e.g., A100)
- **Recommended**: 4 GPUs for parallel processing
- **CUDA**: 12.8+ with compatible drivers
- **Tensor Parallelism**: Supported via vLLM

---

## Key Technologies & Frameworks

### Machine Learning Stack

#### PyTorch Ecosystem
- **PyTorch 2.1.0+**: Core deep learning framework
- **Transformers 4.51.0+**: HuggingFace model library
- **PEFT**: Parameter-efficient fine-tuning (LoRA)
- **Accelerate**: Distributed training
- **TRL**: Transformer Reinforcement Learning

#### Inference Optimization
- **vLLM**: High-throughput LLM inference with PagedAttention
  - Supports tensor parallelism across multiple GPUs
  - FP8 KV cache for memory efficiency
  - Prefix caching for repeated prompts
  - FlashInfer backend for sampling optimization
- **Unsloth**: Fast fine-tuning with memory optimizations

#### Training Algorithm
- **GRPO (Group Relative Policy Optimization)**:
  - Reinforcement learning for math reasoning
  - Groups responses by problem for relative comparison
  - Uses math_verify library for reward signals
  - Implemented in `src/models/grpo_trainer.py`

### Data Processing

- **Pandas 2.0.0+**: DataFrame operations
- **Datasets 2.14.0+**: HuggingFace dataset library
- **Parquet**: Columnar storage format for efficiency
- **math_verify**: Mathematical answer verification

### Configuration Management

- **Hydra 1.3.0+**: Hierarchical configuration composition
  - YAML-based configuration files
  - Command-line overrides
  - Multi-run support for hyperparameter sweeps

### Experiment Tracking

- **Weights & Biases (WandB)**: Experiment logging and visualization
  - Tracks metrics, hyperparameters, and artifacts
  - Requires API key in `.env`

---

## Configuration System

### Hydra Configuration Hierarchy

The project uses Hydra for configuration management with a hierarchical structure:

```yaml
# config/config.yaml (main)
defaults:
  - data: raw_dataset
  - training: lora
  - evaluation: benchmarks

project:
  name: "conf_agg_llm"
  version: "1.0.0"

model:
  base_model: "Qwen/Qwen3-1.7B-FP8"
  max_length: 32768
  trust_remote_code: true

paths:
  data_dir: "/mnt/data1/datasets/nlp/conf_agg"
  output_dir: "/mnt/data1/datasets/nlp/conf_agg/outputs"
  model_dir: "/mnt/data1/models/nlp/conf_agg"

experiment:
  seed: 42
  wandb:
    enabled: true
    project: "conf-agg-llm"
```

### Key Configuration Files

#### 1. Data Generation (`config/data/raw_dataset.yaml`)

```yaml
generation:
  num_responses_per_problem: 2
  temperature: 0.6
  max_tokens: 32768
  top_p: 0.95
  top_k: 20

vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.95
  max_model_len: 32768
  kv_cache_dtype: "fp8"
  enable_prefix_caching: true
```

#### 2. Training (`config/training/lora.yaml`)

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

grpo:
  group_size: 8
  kl_coefficient: 0.001
  aggregator_temperature: 1.5

training:
  epochs: 1
  batch_size: 1024
  learning_rate: 5e-05
```

### Overriding Configuration

```bash
# Override specific values via CLI
python scripts/stage1_generate.py \
    data.generation.temperature=0.8 \
    data.vllm.gpu_memory_utilization=0.85
```

---

## Development Workflows

### 4-Stage Pipeline

The project follows a systematic 4-stage pipeline:

#### Stage 1: Raw Data Generation

**Purpose**: Generate multiple candidate answers with confidence scores

**Script**: `scripts/stage1_generate.py`

**Process**:
1. Load base dataset (e.g., deepscaler.jsonl)
2. Generate N responses per problem using vLLM
3. Extract confidence scores from token log-probabilities
4. Save to parquet files with columns: [problem, answer, confidence, correctness]

**Key Features**:
- Multi-GPU support via sharding
- Float16 logprob compression (50% memory savings)
- Confidence methods: mean_group, bottom_10_percent, tail

**Example**:
```bash
# Single GPU
SAMPLE_LIMIT=400 uv run python scripts/stage1_generate.py \
    --gpu-id "0" \
    --shard-id 0 \
    --total-shards 1

# Multi-GPU (4 GPUs)
bash scripts/run_stage1_async.sh
```

**Output**: `data/generated/sample_400/raw_generated_shard_0.parquet`

#### Stage 2: Data Curation

**Purpose**: Create high-quality training dataset with curriculum learning

**Script**: `scripts/stage2_curate.py`

**Process**:
1. Classify problems as Hard/Easy based on model performance
2. Verify answers using math_verify library
3. Create balanced train/validation splits (80/20)
4. Filter low-quality samples

**Example**:
```bash
uv run python scripts/stage2_curate.py
```

**Output**: `data/curated/train.parquet`, `data/curated/val.parquet`

#### Stage 3: Model Training

**Purpose**: Fine-tune aggregation model using GRPO

**Script**: `scripts/stage3_train.py`

**Process**:
1. Load curated dataset
2. Initialize base model with LoRA adapters
3. Train using GRPO algorithm with group-based rewards
4. Save checkpoints to outputs/models/

**Key Features**:
- LoRA for parameter efficiency
- Unsloth optimizations
- vLLM colocate mode (training + inference)
- DDP for multi-GPU training

**Example**:
```bash
uv run python scripts/stage3_train.py
```

**Output**: `outputs/models/checkpoint-{step}/`

#### Stage 4: Benchmark Evaluation

**Purpose**: Evaluate model on held-out benchmarks

**Script**: `scripts/stage4_evaluate.py`

**Process**:
1. Load trained model
2. Generate predictions on benchmarks
3. Compute metrics: pass@1, pass@k, confidence correlation
4. Save detailed results

**Benchmarks**:
- AIME 2024/2025 (American Invitational Mathematics Examination)
- HMMT 2024/2025 (Harvard-MIT Mathematics Tournament)

**Example**:
```bash
uv run python scripts/stage4_evaluate.py
```

**Output**: `outputs/results/evaluation_report.json`

### Development Cycle

```
1. Make code changes
   ↓
2. Test locally (single GPU, SAMPLE_LIMIT=10)
   ↓
3. Run full pipeline on subset (SAMPLE_LIMIT=400)
   ↓
4. Validate results in WandB
   ↓
5. Scale to full dataset (SAMPLE_LIMIT=unset)
   ↓
6. Commit and push
```

---

## Testing & Validation

### Test Files

- `test.py`: Basic unit tests
- `test_vllm_simple.py`: vLLM inference tests
- `test.ipynb`: Jupyter notebook for interactive testing

### Validation Checklist

Before committing major changes:

- [ ] Code runs without errors on SAMPLE_LIMIT=10
- [ ] GPU memory usage is reasonable (<95%)
- [ ] Output files are generated correctly
- [ ] Confidence scores are within [0, 1]
- [ ] Math verification produces correct results
- [ ] WandB logging works
- [ ] Multi-GPU execution succeeds

### Common Test Commands

```bash
# Quick smoke test (Stage 1)
SAMPLE_LIMIT=10 uv run python scripts/stage1_generate.py \
    --gpu-id "0" --shard-id 0 --total-shards 1

# Verify output
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/generated/sample_10/raw_generated_shard_0.parquet')
print(f'Rows: {len(df)}')
print(df.head())
"

# Test vLLM
uv run python test_vllm_simple.py

# Test math verification
uv run python -c "
from src.evaluation.math_verifier import verify_answer
result = verify_answer('42', '42')
print(f'Verification: {result}')
"
```

---

## Code Conventions

### Python Style

- **PEP 8** compliant
- **Type hints**: Encouraged but not strictly enforced
- **Docstrings**: Use for public functions
- **Imports**: Group by standard library, third-party, local

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private**: Prefix with `_`

### Configuration Conventions

- **YAML files**: Use lowercase with underscores
- **Hydra configs**: Follow hierarchical structure
- **Paths**: Use absolute paths or Hydra interpolation

### Git Conventions

#### Commit Messages

```
<type>: <short summary>

<optional detailed description>

Examples:
- feat: Add confidence-aware aggregation training
- fix: Correct GPU memory leak in vLLM engine
- docs: Update deployment guide with Docker instructions
- refactor: Simplify confidence calculation logic
```

#### Branch Naming

- Feature: `feature/<description>`
- Bugfix: `fix/<description>`
- Experimental: `exp/<description>`

#### Files to NEVER Commit

Per `.gitignore`:
- `.env` (contains secrets)
- `data/raw/*`, `data/generated/*`, `data/curated/*` (large files)
- `outputs/*` (experiment artifacts)
- `__pycache__/`, `*.pyc` (Python cache)
- `venv/`, `.venv/` (virtual environments)

---

## Common Tasks

### Adding a New Confidence Method

1. **Edit** `src/data/confidence.py`
2. **Add method** to `calculate_confidence()` function
3. **Update config** `config/data/raw_dataset.yaml`:
   ```yaml
   confidence:
     methods:
       - mean_group_confidence
       - bottom_10_percent_confidence
       - tail_confidence
       - your_new_method  # Add here
   ```
4. **Test** with SAMPLE_LIMIT=10
5. **Commit** with descriptive message

### Modifying Training Hyperparameters

1. **Edit** `config/training/lora.yaml`
2. **Change values**:
   ```yaml
   training:
     learning_rate: 1e-05  # Changed from 5e-05
     batch_size: 2048       # Changed from 1024
   ```
3. **Run training** with WandB enabled
4. **Compare** results in WandB dashboard
5. **Document** findings in experiment notes

### Adding a New Benchmark

1. **Add dataset** to `data/benchmarks/your_benchmark.jsonl`
2. **Update config** `config/evaluation/benchmarks.yaml`:
   ```yaml
   datasets:
     - name: "YourBenchmark"
       path: "data/benchmarks/your_benchmark.jsonl"
   ```
3. **Run evaluation**: `uv run python scripts/stage4_evaluate.py`
4. **Verify results** in `outputs/results/`

### Debugging GPU Memory Issues

1. **Check current usage**: `nvidia-smi`
2. **Reduce memory utilization** in config:
   ```yaml
   vllm:
     gpu_memory_utilization: 0.85  # Reduced from 0.95
     max_model_len: 24576          # Reduced from 32768
   ```
3. **Enable eager mode** (disables CUDA graph caching):
   ```yaml
   vllm:
     enforce_eager: true
   ```
4. **Clear cache**: `docker-compose restart`

### Scaling to Multiple GPUs

For data generation:

```bash
# Create separate shards for each GPU
for shard_id in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard_id uv run python scripts/stage1_generate.py \
        --gpu-id "$shard_id" \
        --shard-id $shard_id \
        --total-shards 4 &
done
wait  # Wait for all to complete
```

For training (automatic via DDP):

```bash
# Specify number of GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/stage3_train.py
```

---

## Troubleshooting

### Common Issues

#### 1. vLLM Model Loading Fails

**Symptom**: `OSError: Qwen/Qwen3-1.7B-FP8 not found`

**Solution**:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here

# Ensure trust_remote_code is enabled
# In config/config.yaml:
model:
  trust_remote_code: true
```

#### 2. CUDA Out of Memory

**Symptom**: `torch.cuda.OutOfMemoryError`

**Solution**:
```yaml
# Reduce GPU memory utilization
vllm:
  gpu_memory_utilization: 0.75

# Reduce max_model_len
vllm:
  max_model_len: 16384

# Reduce batch size
vllm:
  max_num_seqs: 20
```

#### 3. Math Verification Timeout

**Symptom**: `TimeoutError in math_verify`

**Solution**:
```yaml
# Increase timeout in config/data/curation.yaml
verification:
  timeout: 60  # Increased from 30
```

#### 4. Docker GPU Not Accessible

**Symptom**: `nvidia-smi` fails inside container

**Solution**:
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify on host
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

#### 5. UV Sync Fails

**Symptom**: Dependency resolution errors

**Solution**:
```bash
# Clear cache and regenerate lock
rm uv.lock
rm -rf .venv
uv sync

# Or use pip fallback
pip install -r requirements.txt
```

### Debugging Workflow

1. **Check logs**: `outputs/logs/sample_*/stage*.log`
2. **Monitor GPU**: `watch -n 1 nvidia-smi`
3. **Inspect data**: Load parquet files with pandas
4. **Use Jupyter**: `jupyter lab` for interactive debugging
5. **Enable verbose logging**: Set `logging.level: DEBUG` in config

---

## Important Constraints

### Performance Considerations

- **GPU Memory**: Each vLLM instance requires ~30-40GB VRAM for Qwen3-1.7B
- **Disk Space**: Generated data can reach 100GB+ for full pipeline
- **Computation Time**:
  - Stage 1: ~2-4 hours (4 GPUs, SAMPLE_LIMIT=400)
  - Stage 2: ~30 minutes
  - Stage 3: ~4-8 hours (depends on dataset size)
  - Stage 4: ~1-2 hours

### Resource Limits

- **Max Sequence Length**: 32,768 tokens (model limit)
- **Max Batch Size**: Constrained by GPU memory
- **Concurrent vLLM Servers**: 1 per GPU (due to memory)

### Data Constraints

- **Input Format**: JSONL with `problem` and `answer` fields
- **Answer Format**: LaTeX math expressions wrapped in `\boxed{}`
- **Confidence Range**: [0, 1] normalized scores

### Security Constraints

- **Never commit**:
  - `.env` files (contain API keys)
  - Model checkpoints (too large)
  - Generated data (privacy concerns)
- **Use secrets manager** for production deployments
- **Validate inputs** to prevent code injection in math expressions

### Compatibility Constraints

- **Python**: 3.12 only (due to vLLM requirements)
- **CUDA**: 12.8+ required
- **GPU Architecture**: Ampere (A100) or newer recommended
- **Docker**: 24.0+ with NVIDIA Container Toolkit

---

## Working with AI Assistants

### What AI Assistants Should Know

When assisting with this codebase:

1. **Always check configuration files** before modifying code
2. **Test with SAMPLE_LIMIT** for quick iterations
3. **Monitor GPU memory** usage to prevent OOM errors
4. **Use Hydra overrides** for experimentation
5. **Validate math outputs** with math_verify
6. **Check WandB logs** for experiment tracking
7. **Respect .gitignore** rules - never commit large files
8. **Document experiments** in commit messages

### Recommended Workflow for Code Changes

1. **Understand**: Read relevant source files and configs
2. **Plan**: Outline changes and potential impacts
3. **Test**: Use SAMPLE_LIMIT=10 for rapid testing
4. **Validate**: Check outputs, logs, and metrics
5. **Scale**: Test with SAMPLE_LIMIT=400
6. **Document**: Update this file and commit messages
7. **Monitor**: Track in WandB

### Questions to Ask

Before making changes:

- What stage of the pipeline does this affect?
- Will this change GPU memory requirements?
- Does this require configuration updates?
- How will this impact experiment reproducibility?
- Are there backward compatibility concerns?

---

## Additional Resources

### Documentation

- [README.md](README.md): Project overview (Korean)
- [docs/QUICKSTART.md](docs/QUICKSTART.md): 5-minute deployment guide
- [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md): Full deployment instructions
- [docs/DEPLOYMENT_NO_DOCKER.md](docs/DEPLOYMENT_NO_DOCKER.md): Non-Docker setup

### External Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Hydra Documentation](https://hydra.cc/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Key Papers

- AggLM: Majority voting for math reasoning
- GRPO: Group-based policy optimization
- Qwen Technical Report: Base model architecture

---

## Changelog

### Version 1.0.0 (2025-11-14)

- Initial CLAUDE.md creation
- Comprehensive codebase documentation
- Development workflow guidelines
- Troubleshooting guide

---

## Contact & Support

For questions or issues:

1. Check this CLAUDE.md file
2. Review logs in `outputs/logs/`
3. Search existing GitHub issues
4. Create new issue with:
   - Error messages
   - Configuration used
   - Steps to reproduce
   - System information (GPU, CUDA version, etc.)

---

**Note**: This document is intended for AI assistants and should be updated as the codebase evolves. When making significant changes to the project structure, workflows, or conventions, please update this file accordingly.
