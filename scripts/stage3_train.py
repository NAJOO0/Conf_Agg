"""
Stage 3: GRPO 모델 훈련 스크립트 (Unsloth + TRL)
소형 모델용 - Data Parallelism (DDP) 사용
Qwen3-1.7B 같은 작은 모델에 최적화
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import torch
from typing import Optional

import torch.distributed as dist
from datetime import timedelta, datetime

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import GenerationConfig   
# TRL imports
from trl import GRPOConfig, GRPOTrainer as TRL_GRPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# Transformers imports
from transformers import TrainingArguments
from peft import LoraConfig

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.evaluation.math_verifier import MathVerifier

logger = logging.getLogger(__name__)

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ['UNSLOTH_DISABLE_FAST_LORA'] = '1'

def create_math_reward_function(math_verifier: MathVerifier):
    """
    math_verify를 사용하는 reward function을 생성합니다.
    
    Args:
        math_verifier: MathVerifier 인스턴스
    
    Returns:
        reward function (completions, ground_truth, **kwargs) -> List[float]
    """
    def reward_func(completions, ground_truth=None, **kwargs):
        """
        생성된 응답에 대해 reward를 계산합니다.
        
        Args:
            completions: 생성된 응답 리스트
            ground_truth: 정답 (dataset의 'ground_truth' 컬럼에서 가져옴)
            **kwargs: 추가 인자 (dataset의 다른 컬럼들)
        
        Returns:
            reward 점수 리스트 (각 completion에 대해 1.0 또는 0.0)
        """
        # ground_truth가 kwargs에 있을 수도 있음
        if ground_truth is None:
            ground_truth = kwargs.get("ground_truth", None)
        
        # 여러 개의 ground_truth가 리스트로 올 수 있음
        if isinstance(ground_truth, list):
            if len(ground_truth) == len(completions):
                rewards = []
                for completion, gt in zip(completions, ground_truth):
                    # 응답에서 최종 답안 추출
                    predicted_answer = math_verifier.extract_final_answer_from_content(completion)
                    # 정답 검증
                    is_correct = math_verifier.verify_answer(predicted_answer, gt)
                    rewards.append(1.0 if is_correct else 0.0)
                return rewards
            else:
                # 길이가 맞지 않으면 첫 번째 ground_truth 사용
                gt = ground_truth[0] if ground_truth else ""
        else:
            gt = ground_truth or ""
        
        # 단일 ground_truth를 모든 completion에 대해 사용
        rewards = []
        for completion in completions:
            predicted_answer = math_verifier.extract_final_answer_from_content(completion)
            is_correct = math_verifier.verify_answer(predicted_answer, gt)
            rewards.append(1.0 if is_correct else 0.0)
        
        return rewards
    
    return reward_func


class OptimizedGRPOTrainer:
    """
    소형 모델용 GRPO 트레이너 (Unsloth + TRL)
    
    특징:
    - vLLM 지원 (선택적 - inference 최적화)
    - Data Parallelism (DDP) 사용으로 2배 속도
    - 1.7B~8B 모델에 최적화
    """
    
    def __init__(self, model_name, lora_config=None, grpo_config=None, training_config=None, device="cuda"):
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.grpo_config = grpo_config or {}
        self.training_config = training_config or {}
        self.device = device
        
        # GPU 개수 감지
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"🎮 감지된 GPU 개수: {self.num_gpus}개")
        
        for i in range(self.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # ===== 수정: vLLM 사용 여부는 train()에서 결정 =====
        # 여기서는 모델만 로드
        use_vllm = self.grpo_config.get("use_vllm", True)
        
        # ===== 모델 로드 (항상 DDP 호환 모드) =====
        logger.info(f"📥 모델 로드 중: {model_name}")
        max_seq_length = (
            self.training_config.get("max_prompt_length", 512) + 
            self.training_config.get("max_response_length", 1024)
        )

        # ✅ 항상 device_map=None (DDP 호환)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            fast_inference=use_vllm, 
            unsloth_vllm_stanby=True,
            float8_kv_cache=use_vllm,
            # gpu_memory_utilization=0.4,
        )
        
        logger.info("✅ 모델 로드 완료 (DDP 호환 모드)")
        
        # Chat template
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        
        # LoRA 설정
        if self.lora_config:
            self._setup_lora()
        
        # Reward function
        timeout = self.training_config.get("verification_timeout", 30)
        self.math_verifier = MathVerifier(timeout=timeout)
        self.reward_func = create_math_reward_function(self.math_verifier)
        
    def _setup_lora(self):
        """LoRA 어댑터 설정"""
        logger.info("🔧 LoRA 어댑터 설정 중...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.get("r", 16),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_dropout=self.lora_config.get("lora_dropout", 0.0),
            bias=self.lora_config.get("bias", "none"),
            use_gradient_checkpointing="unsloth",  # Unsloth 최적화
            random_state=self.training_config.get("seed", 42),
            use_rslora=self.lora_config.get("use_rslora", False),
            loftq_config=None,
        )
        
        # 훈련 가능한 파라미터 계산
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"✅ LoRA 설정 완료\n"
            f"   - Rank: {self.lora_config.get('r', 16)}\n"
            f"   - 훈련 파라미터: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)"
        )
    
    # stage3_train.py 수정

    def train(self, train_dataset, validation_dataset=None, save_dir="./output"):
        """GRPO 훈련 실행"""
        logger.info("🎯 GRPO 훈련 시작...")
        
        # ===== DDP 환경 감지 =====
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = rank >= 0

        if is_distributed:
            logger.info(
                f"✅ DDP 모드 감지\n"
                f"   - RANK: {rank}\n"
                f"   - LOCAL_RANK: {local_rank}\n"
                f"   - WORLD_SIZE: {world_size}"
            )

        # ===== vLLM 설정 결정 =====
        use_vllm = self.grpo_config.get("use_vllm", True)

        if use_vllm and is_distributed:
            # DDP + vLLM Colocate (최고 조합!)
            vllm_mode = "colocate"
            vllm_device = None
            logger.info(
                "🚀 DDP + vLLM Colocate 모드\n"
                "   - 각 GPU가 독립적으로 vLLM 실행\n"
                "   - Gradient all-reduce로 동기화"
            )
        elif use_vllm and not is_distributed:
            # 단일 프로세스 + vLLM
            requested_mode = self.grpo_config.get("vllm_mode", "colocate")
            
            if self.num_gpus >= 2:
                # GPU 2개 이상일 때 colocate 또는 separate 선택 가능
                if requested_mode == "colocate":
                    vllm_mode = "colocate"
                    vllm_device = None
                    logger.info(
                        f"🔧 단일 프로세스 vLLM Colocate 모드\n"
                        f"   - GPU {self.num_gpus}개 활용\n"
                        f"   - 같은 프로세스에서 training + vLLM 동시 실행"
                    )
                else:
                    # separate 모드: 별도 GPU 사용
                    vllm_mode = "separate"
                    vllm_device = "cuda:1" if self.num_gpus >= 2 else None
                    logger.info(
                        f"🔧 단일 프로세스 vLLM Separate 모드\n"
                        f"   - Training: GPU 0\n"
                        f"   - vLLM: GPU 1"
                    )
            else:
                # GPU 1개일 때는 colocate만 가능
                vllm_mode = "colocate"
                vllm_device = None
                logger.info("🔧 단일 GPU vLLM Colocate 모드")
        else:
            vllm_mode = None
            vllm_device = None
            logger.info("⚙️ vLLM 비활성화")
        
        # ===== Batch size 검증 =====
        num_generations = self.grpo_config.get("num_generations", 8)
        batch_size = self.training_config.get("batch_size", 8)
        
        # if batch_size % num_generations != 0:
        #     adjusted_batch_size = (batch_size // num_generations) * num_generations
        #     if adjusted_batch_size == 0:
        #         adjusted_batch_size = num_generations
        #     logger.warning(
        #         f"⚠️ batch_size 조정: {batch_size} -> {adjusted_batch_size}"
        #     )
        #     batch_size = adjusted_batch_size
        
        # ===== GRPO Config 설정 =====
        # warmup_steps 처리: Hydra에서 문자열로 올 수 있으므로 변환
        warmup_steps_raw = self.training_config.get("warmup_steps", None)
        warmup_steps = None
        
        if warmup_steps_raw is not None:
            # "None" 문자열이나 실제 None 체크
            if isinstance(warmup_steps_raw, str):
                if warmup_steps_raw.lower() not in ["none", "null", ""]:
                    try:
                        warmup_steps = int(warmup_steps_raw)
                    except (ValueError, TypeError):
                        warmup_steps = None
            elif isinstance(warmup_steps_raw, (int, float)) and warmup_steps_raw > 0:
                warmup_steps = int(warmup_steps_raw)
        
        # warmup_steps가 있으면 사용, 없으면 warmup_ratio 사용
        warmup_ratio = None if warmup_steps is not None else self.training_config.get("warmup_ratio", 0.1)
        
        grpo_config = GRPOConfig(
            # 기본 설정
            output_dir=save_dir,
            num_train_epochs=self.training_config.get("epochs", 1),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            
            # Learning rate
            learning_rate=self.training_config.get("learning_rate", 5e-6),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            **({"warmup_steps": warmup_steps} if warmup_steps is not None else {"warmup_ratio": warmup_ratio}),

            # GRPO 설정
            num_generations=num_generations,
            max_prompt_length=self.training_config.get("max_prompt_length", 512),
            max_completion_length=self.training_config.get("max_response_length", 1024),
            temperature=self.grpo_config.get("temperature", 1.0),
            beta=self.grpo_config.get("beta", 0.01),  # 0이 아닌 작은 값
            
            # 최적화
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            bf16=is_bfloat16_supported(),
            # DDP 설정
            
            # ===== 🎯 vLLM 설정 (수정됨) =====
            use_vllm=use_vllm,
            vllm_mode=vllm_mode,  # ✅ separate 또는 None
            vllm_gpu_memory_utilization=self.grpo_config.get("vllm_gpu_memory_utilization", 0.85),  # ✅ 높임
            vllm_enable_sleep_mode=self.grpo_config.get("vllm_enable_sleep_mode", True),
            # 로깅 및 저장
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            
            # Evaluation
            # eval_strategy="steps" if validation_dataset else "no",
            eval_strategy="no",
            eval_steps=self.training_config.get("eval_steps", 500) if validation_dataset else None,
            per_device_eval_batch_size=num_generations,
            
            # WandB
            report_to="wandb" if self.training_config.get("use_wandb", False) else "none",
            
            # 기타
            seed=self.training_config.get("seed", 42),
            dataloader_num_workers=1,
            remove_unused_columns=False,
        )
        
        # 유효 배치 크기
        effective_batch_size = (
            grpo_config.per_device_train_batch_size
            * grpo_config.gradient_accumulation_steps
            * max(world_size, 1)
        )
        
        logger.info(
            f"📦 최종 설정:\n"
            f"   - 모드: {'DDP + vLLM Colocate 🏆' if (is_distributed and use_vllm) else 'DDP only' if is_distributed else 'Single'}\n"
            f"   - World size: {max(world_size, 1)}\n"
            f"   - GPU당 배치: {grpo_config.per_device_train_batch_size}\n"
            f"   - Gradient accumulation: {grpo_config.gradient_accumulation_steps}\n"
            f"   - 유효 배치 크기: {effective_batch_size}\n"
            f"   - 예상 throughput: {'🚀 최고!' if (is_distributed and use_vllm) else '✅ 빠름'}"
        )
        
        # Trainer 초기화
        trainer = TRL_GRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_func,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            eval_dataset=validation_dataset,
        )
        
        # 훈련 실행
        logger.info("🏃 훈련 시작!")
        trainer.train()
        
        # 저장
        logger.info(f"💾 모델 저장: {save_dir}")
        trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        torch.cuda.empty_cache()
        logger.info("✅ 훈련 완료!")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 3 메인 함수"""

     # 🔥 환경 변수 강제 설정 (재초기화 X)
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


    print(f"[RANK {rank}] torch.distributed.is_initialized(): {dist.is_initialized()}", file=sys.stderr)
    print(f"[RANK {rank}] MASTER_ADDR: {os.environ['MASTER_ADDR']}", file=sys.stderr)
    print(f"[RANK {rank}] MASTER_PORT: {os.environ['MASTER_PORT']}", file=sys.stderr)
    sys.stderr.flush()
    
    # ===== 🚀 DDP 수동 초기화 (TRL 전에) =====
    if is_distributed and not dist.is_initialized():
        print(f"⏱️  [RANK {rank}] DDP 초기화 시도 (30초 타임아웃)...", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=30)
            )
            torch.cuda.set_device(local_rank)
            print(f"✅ [RANK {rank}] DDP 초기화 성공! GPU {local_rank}", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"❌ [RANK {rank}] DDP 초기화 실패: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise

    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage3_train.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 3: GRPO 모델 훈련 시작 (Unsloth + TRL)")
    logger.info(f"설정: {cfg.training}")
    
    # GPU 개수 확인
    num_gpus = torch.cuda.device_count()
    logger.info(f"🎮 사용 가능한 GPU: {num_gpus}개")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 디렉토리 생성
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    # enavle_think 및 훈련 날짜 폴더데 model 저장
    enable_think = cfg.training.training.enable_think
    train_date = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(cfg.paths.model_dir, f"enable_think_{enable_think}_{train_date}")
    os.makedirs(model_dir, exist_ok=True)
    # 입력 파일 경로 확인
    train_data_path = os.path.join(cfg.paths.data_dir, "curated", "train_curated.parquet")
    validation_data_path = os.path.join(cfg.paths.data_dir, "curated", "validation_curated.parquet")
    
    if not os.path.exists(train_data_path):
        logger.error(f"훈련 데이터 파일을 찾을 수 없습니다: {train_data_path}")
        logger.error("먼저 Stage 2를 실행해주세요: python scripts/stage2_curate.py")
        return
    
    # 데이터셋 생성 (TRL GRPOTrainer는 Dataset을 받음)
    logger.info("📦 데이터셋 생성 중...")
    from src.data.training_dataset import CuratedTrainingDataset
    
    train_dataset = CuratedTrainingDataset(train_data_path)
    logger.info(f"훈련 데이터셋 생성: {len(train_dataset)} 샘플")
    
    validation_dataset = None
    if os.path.exists(validation_data_path):
        validation_dataset = CuratedTrainingDataset(validation_data_path)
        logger.info(f"검증 데이터셋 생성: {len(validation_dataset)} 샘플")
    
    # GRPO 트레이너 초기화
    logger.info("🦥 Optimized GRPO 트레이너 초기화 중...")
    trainer = OptimizedGRPOTrainer(
        model_name=cfg.model.base_model,
        lora_config=cfg.training.lora if cfg.training.method == "lora" else None,
        grpo_config=cfg.training.grpo,
        training_config=cfg.training.training,
        device=cfg.experiment.device
    )
    
    # 훈련 실행
    logger.info("🏋️ 모델 훈련 시작...")
    trainer.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        save_dir=model_dir
    )
    
    logger.info("✅ Stage 3 완료")
    logger.info(f"📁 훈련된 모델 저장 위치: {model_dir}")


if __name__ == "__main__":
    main()