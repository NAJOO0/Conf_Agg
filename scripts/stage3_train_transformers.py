# stage3_train_transformers.py
"""
Unsloth 완전 대체 버전
- Transformers + Flash Attention 2 (검증됨)
- TRL GRPO (동일)
- Liger Kernel (메모리 최적화)
- vLLM 통합 (동일)
- DDP 지원 (동일)
"""
import os
os.environ['WANDB_API_KEY'] = 'cef6d541e9983fb4a433b2e72a63997ed465e0ac'
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import torch
from typing import Optional

import torch.distributed as dist
from datetime import timedelta, datetime

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# PEFT imports
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# TRL imports
from trl import GRPOConfig, GRPOTrainer as TRL_GRPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# 프로젝트 루트
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.evaluation.math_verifier import MathVerifier

logger = logging.getLogger(__name__)


def create_math_reward_function(math_verifier: MathVerifier):
    """Reward function (기존과 동일)"""
    def reward_func(completions, ground_truth=None, **kwargs):
        if ground_truth is None:
            ground_truth = kwargs.get("ground_truth", None)
        
        if isinstance(ground_truth, list):
            if len(ground_truth) == len(completions):
                rewards = []
                for completion, gt in zip(completions, ground_truth):
                    predicted_answer = math_verifier.extract_final_answer_from_content(completion)
                    is_correct = math_verifier.verify_answer(predicted_answer, gt)
                    rewards.append(1.0 if is_correct else 0.0)
                return rewards
            else:
                gt = ground_truth[0] if ground_truth else ""
        else:
            gt = ground_truth or ""
        
        rewards = []
        for completion in completions:
            predicted_answer = math_verifier.extract_final_answer_from_content(completion)
            is_correct = math_verifier.verify_answer(predicted_answer, gt)
            rewards.append(1.0 if is_correct else 0.0)
        
        return rewards
    
    return reward_func


class OptimizedGRPOTrainer:
    """
    Transformers 기반 GRPO 트레이너
    - Flash Attention 2 보장
    - Liger Kernel 통합
    - vLLM 지원
    - DDP 지원
    """
    
    def __init__(self, model_name, lora_config=None, grpo_config=None, training_config=None, device="cuda"):
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.grpo_config = grpo_config or {}
        self.training_config = training_config or {}
        self.device = device
        
        # GPU 개수
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"🎮 감지된 GPU: {self.num_gpus}개")
        
        for i in range(self.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # DDP 환경 감지
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = rank >= 0
        
        # ===== 모델 로드 =====
        logger.info(f"📥 모델 로드: {model_name}")
        
        max_seq_length = (
            self.training_config.get("max_prompt_length", 512) + 
            self.training_config.get("max_response_length", 1024)
        )
        
        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 모델 로드
        # DDP 사용 시: device_map 사용 안 함 (각 프로세스가 자신의 GPU에 로드)
        # 단일 GPU 사용 시: device_map="auto" 사용 가능
        if is_distributed:
            logger.info(f"🚀 Flash Attention 2로 모델 로드 (DDP 모드, GPU {local_rank})...")
            # DDP에서는 device_map을 사용하지 않고, 각 프로세스가 자신의 GPU에 로드
            # local_rank GPU에 직접 로드
            torch.cuda.set_device(local_rank)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",  # ✅ 보장!
                device_map={"": local_rank},  # 특정 GPU에 로드
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            logger.info("🚀 Flash Attention 2로 모델 로드...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2",  # ✅ 보장!
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ===== 확인 =====
        attn_impl = self.model.config._attn_implementation
        layer_type = type(self.model.model.layers[0].self_attn).__name__
        
        logger.info(f"✅ Attention: {attn_impl}")
        logger.info(f"✅ Layer: {layer_type}")
        
        if attn_impl != "flash_attention_2":
            raise RuntimeError(f"❌ Flash Attention 2 실패: {attn_impl}")
        
        logger.info("✅ 모델 로드 완료")
        
        # Chat template
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        
        # LoRA
        if self.lora_config:
            self._setup_lora()
        
        # Reward function
        timeout = self.training_config.get("verification_timeout", 30)
        self.math_verifier = MathVerifier(timeout=timeout)
        self.reward_func = create_math_reward_function(self.math_verifier)
    
    def _setup_lora(self):
        """LoRA 설정"""
        logger.info("🔧 LoRA 설정 중...")
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.0),
            bias=self.lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ LoRA 설정 완료")
    
    def train(self, train_dataset, validation_dataset=None, save_dir="./output"):
        """GRPO 학습"""
        logger.info("🎯 GRPO 학습 시작...")
        
        # DDP 환경 감지
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = rank >= 0

        if is_distributed:
            logger.info(
                f"✅ DDP 모드\n"
                f"   RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}"
            )

        # vLLM 설정 (기존과 동일)
        use_vllm = self.grpo_config.get("use_vllm", True)

        if use_vllm and is_distributed:
            vllm_mode = "colocate"
            logger.info("🚀 DDP + vLLM Colocate")
        elif use_vllm and not is_distributed:
            requested_mode = self.grpo_config.get("vllm_mode", "colocate")
            if self.num_gpus >= 2 and requested_mode == "separate":
                vllm_mode = "separate"
                logger.info("🔧 vLLM Separate (GPU 1)")
            else:
                vllm_mode = "colocate"
                logger.info("🔧 vLLM Colocate")
        else:
            vllm_mode = None
            logger.info("⚙️ vLLM 비활성화")
        
        # 배치 크기
        num_generations = self.grpo_config.get("num_generations", 2)
        batch_size = self.training_config.get("batch_size", 1)
        
        # Warmup steps
        warmup_steps_raw = self.training_config.get("warmup_steps", None)
        warmup_steps = None
        
        if warmup_steps_raw is not None:
            if isinstance(warmup_steps_raw, str):
                if warmup_steps_raw.lower() not in ["none", "null", ""]:
                    try:
                        warmup_steps = int(warmup_steps_raw)
                    except (ValueError, TypeError):
                        warmup_steps = None
            elif isinstance(warmup_steps_raw, (int, float)) and warmup_steps_raw > 0:
                warmup_steps = int(warmup_steps_raw)
        
        warmup_ratio = None if warmup_steps is not None else self.training_config.get("warmup_ratio", 0.1)
        
        # GRPO Config
        grpo_config = GRPOConfig(
            output_dir=save_dir,
            num_train_epochs=self.training_config.get("epochs", 1),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 8),
            
            learning_rate=self.training_config.get("learning_rate", 5e-6),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            **({"warmup_steps": warmup_steps} if warmup_steps is not None else {"warmup_ratio": warmup_ratio}),

            # GRPO
            num_generations=num_generations,
            max_prompt_length=self.training_config.get("max_prompt_length", 512),
            max_completion_length=self.training_config.get("max_response_length", 1024),
            temperature=self.grpo_config.get("temperature", 1.0),
            beta=self.grpo_config.get("beta", 0.01),
            
            # 최적화
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            bf16=True,
            
            # vLLM
            use_vllm=use_vllm,
            vllm_mode=vllm_mode,
            vllm_gpu_memory_utilization=self.grpo_config.get("vllm_gpu_memory_utilization", 0.6),
            vllm_enable_sleep_mode=True,
            
            # 🔥 Liger Kernel
            # use_liger=True,  # 40% 메모리 절감
            
            # 로깅
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            
            eval_strategy="no",
            eval_steps=self.training_config.get("eval_steps", 500) if validation_dataset else None,
            per_device_eval_batch_size=num_generations,
            
            report_to="wandb" if self.training_config.get("use_wandb", False) else "none",
            
            seed=self.training_config.get("seed", 42),
            dataloader_num_workers=1,
            remove_unused_columns=False,
        )
        
        # 유효 배치
        effective_batch = (
            grpo_config.per_device_train_batch_size
            * grpo_config.gradient_accumulation_steps
            * max(world_size, 1)
        )
        
        logger.info(
            f"📦 설정:\n"
            f"   GPU당 배치: {grpo_config.per_device_train_batch_size}\n"
            f"   Gradient accumulation: {grpo_config.gradient_accumulation_steps}\n"
            f"   유효 배치: {effective_batch}\n"
            f"   Num generations: {grpo_config.num_generations}\n"
            # f"   Liger: {grpo_config.use_liger}"
        )
        
        # Trainer
        trainer = TRL_GRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_func,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            eval_dataset=validation_dataset,
        )
        
        # 학습
        logger.info("🏃 학습 시작!")
        trainer.train()
        
        # 저장
        logger.info(f"💾 모델 저장: {save_dir}")
        trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        torch.cuda.empty_cache()
        logger.info("✅ 학습 완료!")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 함수 (기존과 동일)"""
    
    # DDP 환경 변수 (기존과 동일)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '300'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank >= 0
    
    if rank >= 0:
        os.environ['VLLM_WORKER_NAME'] = f'worker_{rank}'
        os.environ['VLLM_INSTANCE_ID'] = str(rank)

    # DDP 초기화 (기존과 동일)
    if is_distributed and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=30)
            )
            torch.cuda.set_device(local_rank)
        except Exception as e:
            print(f"❌ DDP 초기화 실패: {e}", file=sys.stderr)
            raise

    # 로깅 (기존과 동일)
    log_file = os.path.join(cfg.paths.log_dir, "stage3_train.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 3: GRPO 학습 (Transformers + Flash Attention 2)")
    
    # 디렉토리 (기존과 동일)
    enable_think = cfg.training.training.enable_think
    train_date = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(cfg.paths.model_dir, f"enable_think_{enable_think}_{train_date}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 데이터셋 (기존과 동일)
    train_data_path = os.path.join(cfg.paths.data_dir, "curated", "train_filtered.parquet")
    validation_data_path = os.path.join(cfg.paths.data_dir, "curated", "valid_filtered.parquet")
    
    if not os.path.exists(train_data_path):
        logger.error(f"훈련 데이터 없음: {train_data_path}")
        return
    
    from src.data.training_dataset import CuratedTrainingDataset
    
    train_dataset = CuratedTrainingDataset(train_data_path)
    logger.info(f"훈련: {len(train_dataset)} 샘플")
    
    validation_dataset = None
    if os.path.exists(validation_data_path):
        validation_dataset = CuratedTrainingDataset(validation_data_path)
        logger.info(f"검증: {len(validation_dataset)} 샘플")
    
    # 트레이너
    logger.info("🔧 트레이너 초기화...")
    trainer = OptimizedGRPOTrainer(
        model_name=cfg.model.base_model,
        lora_config=cfg.training.lora if cfg.training.method == "lora" else None,
        grpo_config=cfg.training.grpo,
        training_config=cfg.training.training,
        device=cfg.experiment.device
    )
    
    # 학습
    logger.info("🏋️ 학습 시작...")
    trainer.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        save_dir=model_dir
    )
    
    logger.info(f"✅ 완료: {model_dir}")


if __name__ == "__main__":
    main()