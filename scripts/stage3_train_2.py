"""
Stage 3: GRPO 모델 훈련 스크립트 (Unsloth + vLLM + TRL)
최적화 버전 - 2025년 2월 기준 베스트 프랙티스 적용

주요 개선사항:
1. vLLM Colocate 모드 활성화 (DDP + vLLM 최적 조합)
2. FP8 KV Cache 지원 (메모리 2배 절약)
3. Unsloth 최신 메모리 최적화 활용
4. 배치 크기 자동 조정
5. 향상된 에러 핸들링
"""
import os
# Flash Attention 없이도 작동하도록 환경 변수 설정
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # ✅ vLLM 메모리 최적화
# Flash Attention 2가 없어도 xFormers로 자동 폴백
# os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # 필요시 주석 해제
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import torch
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any

import torch.distributed as dist
from datetime import timedelta

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported, vLLMSamplingParams
from transformers import GenerationConfig   

# TRL imports
from trl import GRPOConfig, GRPOTrainer as TRL_GRPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# PEFT imports
from peft import LoraConfig

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging
from src.evaluation.math_verifier import MathVerifier

logger = logging.getLogger(__name__)



def create_math_reward_function(math_verifier: MathVerifier):
    """
    수학 문제 검증용 reward function 생성
    
    Args:
        math_verifier: MathVerifier 인스턴스
    
    Returns:
        reward function (completions, ground_truth, **kwargs) -> List[float]
    """
    def reward_func(completions, ground_truth=None, **kwargs):
        """
        생성된 응답에 대해 reward를 계산합니다.
        
        Args:
            completions: 생성된 응답 리스트 (각 요소는 str 또는 list of dict)
            ground_truth: 정답 (dataset의 'ground_truth' 컬럼에서 가져옴)
            **kwargs: 추가 인자 (dataset의 다른 컬럼들)
        
        Returns:
            reward 점수 리스트 (각 completion에 대해 1.0 또는 0.0)
        """
        # ground_truth가 kwargs에 있을 수도 있음
        if ground_truth is None:
            ground_truth = kwargs.get("ground_truth", None)
        
        # completions 형식 정규화 (list of dict -> str)
        normalized_completions = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                # [{"role": "assistant", "content": "..."}] 형식
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    normalized_completions.append(completion[0]["content"])
                else:
                    normalized_completions.append(str(completion))
            elif isinstance(completion, str):
                normalized_completions.append(completion)
            else:
                normalized_completions.append(str(completion))
        
        # 여러 개의 ground_truth가 리스트로 올 수 있음
        if isinstance(ground_truth, list):
            if len(ground_truth) == len(normalized_completions):
                # 각 completion마다 대응하는 ground_truth 사용
                rewards = []
                for completion, gt in zip(normalized_completions, ground_truth):
                    predicted_answer = math_verifier.extract_final_answer_from_content(completion)
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
        for completion in normalized_completions:
            predicted_answer = math_verifier.extract_final_answer_from_content(completion)
            is_correct = math_verifier.verify_answer(predicted_answer, gt)
            rewards.append(1.0 if is_correct else 0.0)
        
        return rewards
    
    return reward_func


class OptimizedGRPOTrainer:
    """
    Unsloth + vLLM + GRPO 최적화 트레이너
    
    특징:
    1. vLLM Colocate 모드: DDP와 완벽 호환
    2. FP8 KV Cache: 메모리 2배 절약 (RTX 3090/A100 이상)
    3. Unsloth 메모리 최적화: 90% VRAM 절감
    4. 자동 배치 크기 조정
    5. 멀티 GPU DDP 지원
    """
    
    def __init__(
        self, 
        model_name: str,
        lora_config: Optional[Dict[str, Any]] = None,
        grpo_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.grpo_config = grpo_config or {}
        self.training_config = training_config or {}
        self.device = device
        
        # GPU 환경 감지
        self.num_gpus = torch.cuda.device_count()
        self.is_distributed = int(os.environ.get("RANK", -1)) >= 0
        self.rank = int(os.environ.get("RANK", -1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # GPU 정보 로깅 (rank 0에서만)
        if not self.is_distributed or self.rank == 0:
            logger.info(f"🎮 GPU 환경 정보")
            logger.info(f"   - GPU 개수: {self.num_gpus}")
            logger.info(f"   - 분산 모드: {'✅ DDP' if self.is_distributed else '❌ Single'}")
            if self.is_distributed:
                logger.info(f"   - Rank: {self.rank}/{self.world_size}")
            
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_capability = torch.cuda.get_device_capability(i)
                logger.info(
                    f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, "
                    f"Compute {gpu_capability[0]}.{gpu_capability[1]})"
                )
        
        # FP8 KV Cache 지원 여부 확인 (Compute Capability >= 8.0)
        self.supports_fp8 = False
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            self.supports_fp8 = capability[0] >= 8  # Ampere(A100) 이상
            logger.info(f"FP8 KV Cache 지원 여부: {self.supports_fp8}")
        
        # 모델 로드
        self._load_model()
        
        # Reward function 설정
        self._setup_reward_function()
        
    def _load_model(self):
        """모델 로드 (DDP 호환 모드)"""
        if not self.is_distributed or self.rank == 0:
            logger.info(f"📥 모델 로드 중: {self.model_name}")
        
        # 시퀀스 길이 계산
        max_seq_length = (
            self.training_config.get("max_prompt_length", 512) + 
            self.training_config.get("max_response_length", 1024)
        )
        
        # ✅ enable_think가 켜져있으면 응답 길이 증가
        enable_think = self.grpo_config.get("enable_think", False)
        if enable_think:
            # Thinking은 보통 답변보다 2-3배 길어질 수 있음
            max_seq_length = int(max_seq_length * 1.5)
            if not self.is_distributed or self.rank == 0:
                logger.info(f"📏 enable_think=True: max_seq_length 증가 → {max_seq_length}")
        
        load_in_4bit = False if max_seq_length > 16384 else True
        load_in_8bit = True if max_seq_length > 16384 else False
        use_vllm = self.grpo_config.get("use_vllm", True)
        
        # 모델 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=None,
            fast_inference=use_vllm,
            float8_kv_cache=self.supports_fp8 and use_vllm,
        )
        
        if not self.is_distributed or self.rank == 0:
            logger.info(
                f"✅ 모델 로드 완료\n"
                f"   - Max sequence: {max_seq_length}\n"
                f"   - Quantization: {'4-bit' if load_in_4bit else '8-bit'}\n"
                f"   - vLLM: {'✅' if use_vllm else '❌'}\n"
                f"   - enable_think: {'✅' if enable_think else '❌'}"
            )
        
        # Chat template 설정
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        
        # LoRA 설정
        if self.lora_config:
            self._setup_lora()

    def _setup_lora(self):
        """LoRA 어댑터 설정"""
        if not self.is_distributed or self.rank == 0:
            logger.info("🔧 LoRA 어댑터 설정 중...")
        
        lora_r = self.lora_config.get("r", 16)
        lora_alpha = self.lora_config.get("lora_alpha", lora_r)  # 기본값: r과 동일
        
        # target_modules를 리스트로 변환 (Hydra의 ListConfig를 일반 리스트로)
        target_modules_raw = self.lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        # ListConfig나 다른 타입을 일반 리스트로 변환
        if hasattr(target_modules_raw, '__iter__') and not isinstance(target_modules_raw, str):
            target_modules = list(target_modules_raw)
        else:
            target_modules = target_modules_raw if isinstance(target_modules_raw, list) else [target_modules_raw]
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=self.lora_config.get("lora_dropout", 0.0),
            bias=self.lora_config.get("bias", "none"),
            use_gradient_checkpointing="unsloth",  # Unsloth 최적화
            random_state=self.training_config.get("seed", 42),
            use_rslora=self.lora_config.get("use_rslora", False),
            loftq_config=None,
        )
        
        # 훈련 가능한 파라미터 계산
        if not self.is_distributed or self.rank == 0:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"✅ LoRA 설정 완료\n"
                f"   - Rank (r): {lora_r}\n"
                f"   - Alpha: {lora_alpha}\n"
                f"   - Dropout: {self.lora_config.get('lora_dropout', 0.0)}\n"
                f"   - 훈련 파라미터: {trainable_params / 1e6:.2f}M "
                f"({trainable_params / total_params * 100:.2f}%)"
            )
    
    def _setup_reward_function(self):
        """Reward function 설정"""
        timeout = self.training_config.get("verification_timeout", 30)
        self.math_verifier = MathVerifier(timeout=timeout)
        self.reward_func = create_math_reward_function(self.math_verifier)
        
        if not self.is_distributed or self.rank == 0:
            logger.info(f"✅ Reward function 설정 완료 (timeout: {timeout}s)")
    
    def _validate_and_adjust_batch_size(
        self, 
        batch_size: int, 
        num_generations: int
    ) -> int:
        """
        배치 크기 검증 및 자동 조정
        
        GRPO에서 batch_size는 num_generations의 배수여야 함
        """
        if batch_size % num_generations != 0:
            adjusted_batch_size = (batch_size // num_generations) * num_generations
            if adjusted_batch_size == 0:
                adjusted_batch_size = num_generations
            
            if not self.is_distributed or self.rank == 0:
                logger.warning(
                    f"⚠️ batch_size가 num_generations의 배수가 아닙니다\n"
                    f"   - 원래: {batch_size}\n"
                    f"   - 조정: {adjusted_batch_size}\n"
                    f"   - num_generations: {num_generations}"
                )
            return adjusted_batch_size
        
        return batch_size
    
    def _create_vllm_sampling_params(self) -> Optional[vLLMSamplingParams]:
        """vLLM 샘플링 파라미터 생성"""
        use_vllm = self.grpo_config.get("use_vllm", True)
        if not use_vllm:
            return None
        
        # 사용자 정의 샘플링 파라미터 (있으면)
        sampling_config = self.grpo_config.get("vllm_sampling", {})
        
        if sampling_config:
            params = vLLMSamplingParams(
                temperature=sampling_config.get("temperature", 0.7),
                top_p=sampling_config.get("top_p", 0.8),
                top_k=sampling_config.get("top_k", 20),
                min_p=sampling_config.get("min_p", 0.0),
                seed=self.training_config.get("seed", 42),
            )
            
            if not self.is_distributed or self.rank == 0:
                logger.info(f"🎲 vLLM 샘플링 파라미터 설정: {sampling_config}")
            
            return params
        
        return None
    
    def train(
        self, 
        train_dataset, 
        validation_dataset=None, 
        save_dir="./output"
    ):
        """GRPO 훈련 실행"""
        if not self.is_distributed or self.rank == 0:
            logger.info("🎯 GRPO 훈련 준비 중...")
        
        # ===== 배치 크기 설정 및 검증 =====
        num_generations = self.grpo_config.get("num_generations", 8)
        batch_size = self.training_config.get("batch_size", 8)
        # batch_size = self._validate_and_adjust_batch_size(batch_size, num_generations)
        
        # ===== Warmup 설정 =====
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
        
        # ===== vLLM 모드 결정 =====
        use_vllm = self.grpo_config.get("use_vllm", True)
        
        if use_vllm:
            if self.is_distributed:
                # ===== 🏆 DDP + vLLM Colocate (최적 조합!) =====
                vllm_mode = "colocate"
                if not self.is_distributed or self.rank == 0:
                    logger.info(
                        "🚀 vLLM Colocate 모드 (DDP)\n"
                        "   - 각 GPU에서 독립적으로 vLLM 실행\n"
                        "   - Gradient all-reduce로 동기화\n"
                        "   - 최고의 throughput!"
                    )
            else:
                # 단일 프로세스: colocate 사용
                vllm_mode = "colocate"
                if not self.is_distributed or self.rank == 0:
                    logger.info("🔧 vLLM Colocate 모드 (Single GPU)")
        else:
            vllm_mode = None
            if not self.is_distributed or self.rank == 0:
                logger.info("⚙️ vLLM 비활성화")
        
        # ===== GRPO Config 생성 =====
        eval_batch_size = None
        eval_enabled = False

        if validation_dataset is not None:
            val_len = len(validation_dataset)
            num_gens = num_generations
            
            if val_len < num_gens:
                if not self.is_distributed or self.rank == 0:
                    logger.warning(
                        f"⚠️ 검증 샘플 수({val_len})가 num_generations({num_gens})보다 작습니다. "
                        f"평가를 비활성화합니다."
                    )
                validation_dataset = None
                eval_enabled = False
            else:
                # ✅ 평가 배치 크기: 최소 1, 최대 num_generations, 데이터셋 크기 고려
                # 빈 배치 방지를 위해 더 작은 배치 크기 사용
                eval_batch_size = max(1, min(num_gens, val_len // 10))  # 데이터셋 크기의 10% 이하
                eval_enabled = True
                if not self.is_distributed or self.rank == 0:
                    logger.info(f"✅ 검증 배치 크기: {eval_batch_size} (데이터셋 크기: {val_len})")
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
            
            # GRPO 알고리즘 설정
            num_generations=num_generations,
            max_prompt_length=self.training_config.get("max_prompt_length", 512),
            max_completion_length=self.training_config.get("max_response_length", 1024),
            temperature=self.grpo_config.get("temperature", 1.0),
            beta=self.grpo_config.get("beta", 0.00),
            # mask_truncated_completions: False로 설정하여 truncated completion도 사용
            # True로 설정하면 모든 completion이 truncated되면 빈 배치가 발생할 수 있음
            mask_truncated_completions=False,
            
            # 최적화 설정
            bf16=is_bfloat16_supported(),  # Auto-detect
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",  # 메모리 절약
            gradient_checkpointing=True,
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            
            # DDP 설정
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            
            # ===== vLLM 설정 =====
            # use_vllm: GRPO 훈련 중 generation 단계에서 vLLM 사용 (필수!)
            # vllm_mode: "colocate" = training과 같은 GPU에서 vLLM 실행
            use_vllm=use_vllm,
            vllm_mode=vllm_mode,
            vllm_gpu_memory_utilization=self.grpo_config.get(
                "vllm_gpu_memory_utilization", 
                0.85  # DDP에서 높게 설정 가능
            ),
            vllm_enable_sleep_mode=self.grpo_config.get("vllm_enable_sleep_mode", True),
            
            # 로깅 및 저장
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            load_best_model_at_end=False,
            
            # Evaluation
            # 빈 배치 문제가 지속되면 평가를 비활성화할 수 있음
            # eval_strategy="no",  # 평가 비활성화 (빈 배치 문제 해결 전까지)
            eval_strategy="steps" if eval_enabled else "no",
            # 빈 배치 문제로 평가 빈도 줄임 (config에서 eval_steps가 1이면 100으로 변경)
            eval_steps = self.training_config.get("eval_steps", 500) if eval_enabled else None,
            # 평가 배치 크기: 빈 배치 방지를 위해 최소 1, 최대 num_generations과 데이터셋 크기 중 작은 값
            per_device_eval_batch_size=max(1, min(eval_batch_size, len(validation_dataset) if validation_dataset else 1)),
            eval_accumulation_steps=1 if eval_enabled else None,
            # 빈 배치 필터링
            dataloader_drop_last=False,  # 마지막 배치도 유지하되, 빈 배치는 필터링
            
            # WandB
            report_to="wandb" if self.training_config.get("use_wandb", False) else "none",
            
            # 기타
            seed=self.training_config.get("seed", 42),
            dataloader_num_workers=1,
            remove_unused_columns=False,
            
            # 추가 안정성 설정
            logging_nan_inf_filter=True,
            skip_memory_metrics=True,
        )
        
        # ===== 유효 배치 크기 계산 =====
        effective_batch_size = (
            grpo_config.per_device_train_batch_size
            * grpo_config.gradient_accumulation_steps
            * max(self.world_size, 1)
        )
        
        if not self.is_distributed or self.rank == 0:
            logger.info(
                f"\n{'='*60}\n"
                f"📦 최종 훈련 설정\n"
                f"{'='*60}\n"
                f"🎮 Hardware:\n"
                f"   - 모드: {'DDP' if self.is_distributed else 'Single GPU'}\n"
                f"   - World size: {max(self.world_size, 1)}\n"
                f"   - GPU 메모리: {self.grpo_config.get('vllm_gpu_memory_utilization', 0.85):.0%}\n"
                f"\n"
                f"🚀 vLLM:\n"
                f"   - 사용: {'✅' if use_vllm else '❌'}\n"
                f"   - 모드: {vllm_mode or 'N/A'}\n"
                f"   - FP8 KV Cache: {'✅' if self.supports_fp8 else '❌'}\n"
                f"\n"
                f"📊 Batch Size:\n"
                f"   - GPU당 배치: {grpo_config.per_device_train_batch_size}\n"
                f"   - Gradient accumulation: {grpo_config.gradient_accumulation_steps}\n"
                f"   - Num generations: {num_generations}\n"
                f"   - 유효 배치: {effective_batch_size}\n"
                f"\n"
                f"🎓 Training:\n"
                f"   - Epochs: {grpo_config.num_train_epochs}\n"
                f"   - Learning rate: {grpo_config.learning_rate}\n"
                f"   - Warmup: {warmup_steps or f'{warmup_ratio:.1%} ratio'}\n"
                f"   - Beta: {grpo_config.beta}\n"
                f"\n"
                f"💾 Checkpoints:\n"
                f"   - Save every: {grpo_config.save_steps} steps\n"
                f"   - Max keep: {grpo_config.save_total_limit}\n"
                f"{'='*60}\n"
            )
        
        # ===== vLLM 샘플링 파라미터 (선택적) =====
        vllm_sampling_params = self._create_vllm_sampling_params()
        
        # ===== Trainer 초기화 =====
        trainer = TRL_GRPOTrainer(
            model=self.model,
            reward_funcs=self.reward_func,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            # vllm_sampling_params=vllm_sampling_params,  # 필요 시 사용
        )
        
        # ===== _generate_and_score_completions 함수 래핑 (디버깅 로그 추가) =====
        try:
            original_generate_and_score = trainer._generate_and_score_completions
            
            def wrapped_generate_and_score_completions(inputs):
                """디버깅 로그가 추가된 _generate_and_score_completions"""
                # 입력 배치 크기 확인
                input_batch_size = len(inputs) if inputs else 0
                print(f"[DEBUG _generate_and_score_completions] 입력 배치 크기: {input_batch_size}", 
                      file=sys.stderr, flush=True)
                
                if input_batch_size == 0:
                    print(f"[WARNING _generate_and_score_completions] 빈 입력 배치 감지!", 
                          file=sys.stderr, flush=True)
                    # 빈 배치에 대한 기본 반환값 생성
                    device = trainer.accelerator.device
                    return {
                        'prompt_completion_ids': torch.empty((0, 0), dtype=torch.long, device=device),
                        'attention_mask': torch.empty((0, 0), dtype=torch.long, device=device),
                        'completion_mask': torch.empty((0, 0), dtype=torch.long, device=device),
                        'completion_ids': torch.empty((0, 0), dtype=torch.long, device=device),
                        'completion_ids_list': [],
                        'old_per_token_logps': None,
                        'ref_per_token_logps': None,
                        'sampling_per_token_logps': None,
                        'advantages': torch.empty((0,), dtype=torch.float32, device=device),
                    }
                
                # 원본 함수 호출
                try:
                    result = original_generate_and_score(inputs)
                    
                    # 결과 전체 구조 확인
                    print(f"[DEBUG _generate_and_score_completions] 결과 키: {list(result.keys()) if isinstance(result, dict) else 'N/A'}", 
                          file=sys.stderr, flush=True)
                    
                    # prompt_ids와 completion_ids 확인 (실제 반환값 구조)
                    if 'prompt_ids' in result:
                        prompt_ids_bsz = result['prompt_ids'].shape[0] if hasattr(result['prompt_ids'], 'shape') else 0
                        prompt_ids_len = result['prompt_ids'].shape[1] if hasattr(result['prompt_ids'], 'shape') and len(result['prompt_ids'].shape) > 1 else 0
                        print(f"[DEBUG _generate_and_score_completions] prompt_ids.shape={result['prompt_ids'].shape if hasattr(result['prompt_ids'], 'shape') else 'N/A'}, bsz={prompt_ids_bsz}", 
                              file=sys.stderr, flush=True)
                    
                    if 'completion_ids' in result:
                        completion_ids_bsz = result['completion_ids'].shape[0] if hasattr(result['completion_ids'], 'shape') else 0
                        completion_ids_len = result['completion_ids'].shape[1] if hasattr(result['completion_ids'], 'shape') and len(result['completion_ids'].shape) > 1 else 0
                        print(f"[DEBUG _generate_and_score_completions] completion_ids.shape={result['completion_ids'].shape if hasattr(result['completion_ids'], 'shape') else 'N/A'}, bsz={completion_ids_bsz}", 
                              file=sys.stderr, flush=True)
                        
                        if completion_ids_bsz == 0:
                            print(f"[WARNING _generate_and_score_completions] completion_ids가 빈 배치입니다! 입력: {input_batch_size}개", 
                                  file=sys.stderr, flush=True)
                    
                    if 'completion_mask' in result:
                        comp_mask_bsz = result['completion_mask'].shape[0] if hasattr(result['completion_mask'], 'shape') else 0
                        print(f"[DEBUG _generate_and_score_completions] completion_mask.shape={result['completion_mask'].shape if hasattr(result['completion_mask'], 'shape') else 'N/A'}, bsz={comp_mask_bsz}", 
                              file=sys.stderr, flush=True)
                    
                    # 결과 배치 크기 확인 (prompt_completion_ids는 반환값에 없을 수 있음)
                    if 'prompt_completion_ids' in result:
                        result_bsz = result['prompt_completion_ids'].shape[0] if hasattr(result['prompt_completion_ids'], 'shape') else 0
                        result_qlen = result['prompt_completion_ids'].shape[1] if hasattr(result['prompt_completion_ids'], 'shape') and len(result['prompt_completion_ids'].shape) > 1 else 0
                        print(f"[DEBUG _generate_and_score_completions] 결과 배치 크기: {result_bsz}, qlen: {result_qlen}", 
                              file=sys.stderr, flush=True)
                        
                        if result_bsz == 0:
                            print(f"[WARNING _generate_and_score_completions] ========== 빈 결과 배치 생성! ==========", 
                                  file=sys.stderr, flush=True)
                            print(f"[WARNING _generate_and_score_completions] 입력 배치 크기: {input_batch_size}개", 
                                  file=sys.stderr, flush=True)
                            print(f"[WARNING _generate_and_score_completions] prompt_completion_ids.shape={result['prompt_completion_ids'].shape if hasattr(result['prompt_completion_ids'], 'shape') else 'N/A'}", 
                                  file=sys.stderr, flush=True)
                            
                            # completion_mask 확인
                            if 'completion_mask' in result and result['completion_mask'] is not None:
                                comp_mask = result['completion_mask']
                                if hasattr(comp_mask, 'shape'):
                                    print(f"[WARNING _generate_and_score_completions] completion_mask.shape={comp_mask.shape}", 
                                          file=sys.stderr, flush=True)
                                if hasattr(comp_mask, 'sum'):
                                    mask_sum = comp_mask.sum().item()
                                    print(f"[WARNING _generate_and_score_completions] completion_mask.sum()={mask_sum}", 
                                          file=sys.stderr, flush=True)
                            
                            # completion_ids 확인
                            if 'completion_ids' in result and result['completion_ids'] is not None:
                                comp_ids = result['completion_ids']
                                if hasattr(comp_ids, 'shape'):
                                    print(f"[WARNING _generate_and_score_completions] completion_ids.shape={comp_ids.shape}", 
                                          file=sys.stderr, flush=True)
                            
                            # attention_mask 확인
                            if 'attention_mask' in result and result['attention_mask'] is not None:
                                attn_mask = result['attention_mask']
                                if hasattr(attn_mask, 'shape'):
                                    print(f"[WARNING _generate_and_score_completions] attention_mask.shape={attn_mask.shape}", 
                                          file=sys.stderr, flush=True)
                            
                            print(f"[WARNING _generate_and_score_completions] =========================================", 
                                  file=sys.stderr, flush=True)
                        else:
                            # prompt와 completion 길이 확인
                            prompt_ids = result['prompt_completion_ids']
                            if 'completion_mask' in result and result['completion_mask'] is not None:
                                comp_mask = result['completion_mask']
                                if hasattr(comp_mask, 'shape') and len(comp_mask.shape) >= 2:
                                    comp_len = comp_mask.shape[1]
                                    prompt_len = prompt_ids.shape[1] - comp_len if len(prompt_ids.shape) > 1 else 0
                                    print(f"[DEBUG _generate_and_score_completions] prompt_len={prompt_len}, completion_len={comp_len}", 
                                          file=sys.stderr, flush=True)
                    else:
                        print(f"[ERROR _generate_and_score_completions] 'prompt_completion_ids' 키가 결과에 없습니다!", 
                              file=sys.stderr, flush=True)
                        print(f"[ERROR _generate_and_score_completions] 사용 가능한 키: {list(result.keys()) if isinstance(result, dict) else 'N/A'}", 
                              file=sys.stderr, flush=True)
                    
                    return result
                except Exception as e:
                    print(f"[ERROR _generate_and_score_completions] 에러 발생: {type(e).__name__}: {e}", 
                          file=sys.stderr, flush=True)
                    import traceback
                    print(f"[ERROR _generate_and_score_completions] 트레이스백:\n{traceback.format_exc()}", 
                          file=sys.stderr, flush=True)
                    raise
            
            # 함수 교체
            trainer._generate_and_score_completions = wrapped_generate_and_score_completions
            if not self.is_distributed or self.rank == 0:
                logger.info("✅ _generate_and_score_completions 함수 래핑 완료 (디버깅 로그 추가)")
        except Exception as e:
            if not self.is_distributed or self.rank == 0:
                logger.warning(f"⚠️ _generate_and_score_completions 래핑 실패: {e}")
        
        # ===== compute_loss 함수 래핑 (디버깅 로그 추가) =====
        # 주의: compute_loss는 Trainer의 메서드이므로 래핑 시 재귀 문제가 발생할 수 있음
        # 대신 grpo_accumulated_loss에서만 로그를 추가하여 문제 추적
        # compute_loss 래핑은 제거 (재귀 에러 방지)
        
        # ===== grpo_accumulated_loss 함수 런타임 패치 (빈 배치 방어) =====
        try:
            import sys
            import numpy as np
            import importlib
            
            # 캐시 파일에서 직접 grpo_accumulated_loss 함수 찾기 (가장 확실한 방법)
            cache_file = Path(__file__).parent.parent / "unsloth_compiled_cache" / "UnslothGRPOTrainer.py"
            import time
            max_wait = 10
            waited = 0
            while not cache_file.exists() and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            
            grpo_func = None
            if cache_file.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("unsloth_compiled_cache_UnslothGRPOTrainer", cache_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'grpo_accumulated_loss'):
                        candidate = module.grpo_accumulated_loss
                        # callable인지 확인
                        if callable(candidate):
                            grpo_func = candidate
                            if not self.is_distributed or self.rank == 0:
                                logger.info(f"✅ grpo_accumulated_loss 함수 발견: 캐시 파일 (callable 확인 완료)")
                        else:
                            if not self.is_distributed or self.rank == 0:
                                logger.warning(f"⚠️ 캐시 파일의 grpo_accumulated_loss는 callable이 아닙니다: {type(candidate)}")
            
            # sys.modules에서도 찾기 (백업 방법)
            if grpo_func is None:
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'grpo_accumulated_loss'):
                        candidate = getattr(module, 'grpo_accumulated_loss')
                        # callable이고 unsloth 관련 모듈인지 확인
                        if callable(candidate) and ('unsloth' in module_name.lower() or 'grpo' in module_name.lower()):
                            grpo_func = candidate
                            if not self.is_distributed or self.rank == 0:
                                logger.info(f"✅ grpo_accumulated_loss 함수 발견: {module_name}")
                            break
            
            if grpo_func is not None:
                # 원본 함수 저장
                original_grpo_accumulated_loss = grpo_func
                
                # 패치된 함수 정의
                def patched_grpo_accumulated_loss(trainer, input_ids, attention_mask, logits_to_keep, 
                                                  completion_mask, advantages, old_hidden_states, 
                                                  ref_hidden_states, n_chunks=-1, **kwargs):
                    """빈 배치 방어가 추가된 grpo_accumulated_loss"""
                    # 입력 확인 (가장 먼저)
                    print(f"[DEBUG grpo_accumulated_loss] ========== 함수 호출 시작 ==========", 
                          file=sys.stderr, flush=True)
                    if input_ids is not None:
                        print(f"[DEBUG grpo_accumulated_loss] 입력 input_ids.shape={input_ids.shape if hasattr(input_ids, 'shape') else 'N/A'}", 
                              file=sys.stderr, flush=True)
                    if completion_mask is not None:
                        print(f"[DEBUG grpo_accumulated_loss] 입력 completion_mask.shape={completion_mask.shape if hasattr(completion_mask, 'shape') else 'N/A'}", 
                              file=sys.stderr, flush=True)
                    if advantages is not None:
                        print(f"[DEBUG grpo_accumulated_loss] 입력 advantages.shape={advantages.shape if hasattr(advantages, 'shape') else 'N/A'}", 
                              file=sys.stderr, flush=True)
                    
                    # input_ids shape 안전하게 가져오기
                    try:
                        if input_ids is None or not hasattr(input_ids, 'shape') or len(input_ids.shape) < 2:
                            raise ValueError(f"Invalid input_ids: {input_ids}")
                        bsz, qlen = input_ids.shape
                    except (AttributeError, ValueError, TypeError) as e:
                        print(f"[DEBUG grpo_accumulated_loss] input_ids shape 오류: {e}, input_ids={input_ids}", 
                              file=sys.stderr, flush=True)
                        # 기본값으로 처리
                        device = getattr(trainer.model, 'device', None) if hasattr(trainer, 'model') else None
                        if device is None:
                            try:
                                device = input_ids.device if hasattr(input_ids, 'device') else None
                            except:
                                device = None
                        zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                        completion_length = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                        empty_tensor = torch.tensor([], device=device, dtype=torch.float32).detach() if device is not None else torch.tensor([], dtype=torch.float32).detach()
                        return zero, completion_length, zero, zero, empty_tensor
                    
                    # 디버깅: stderr로 강제 출력
                    # max_prompt_length와 max_completion_length 확인
                    max_prompt_len = getattr(trainer.args, 'max_prompt_length', None) if hasattr(trainer, 'args') else None
                    max_completion_len = getattr(trainer.args, 'max_completion_length', None) if hasattr(trainer, 'args') else None
                    max_total = (max_prompt_len + max_completion_len) if (max_prompt_len and max_completion_len) else None
                    
                    # prompt 길이 계산 (completion_mask의 길이로 completion 길이를 알 수 있음)
                    completion_len = completion_mask.size(1) if completion_mask is not None and hasattr(completion_mask, 'size') else None
                    prompt_len = qlen - completion_len if completion_len is not None else None
                    
                    # completion_mask 상세 정보
                    comp_mask_info = ""
                    if completion_mask is not None and hasattr(completion_mask, 'sum'):
                        comp_mask_sum = completion_mask.sum().item()
                        comp_mask_shape = completion_mask.shape if hasattr(completion_mask, 'shape') else "unknown"
                        comp_mask_info = f", completion_mask.shape={comp_mask_shape}, completion_mask.sum()={comp_mask_sum}"
                    
                    # 입력 상세 정보 (빈 배치 추적용)
                    print(f"[DEBUG grpo_accumulated_loss] ========== 함수 진입 ==========", 
                          file=sys.stderr, flush=True)
                    print(f"[DEBUG grpo_accumulated_loss] bsz={bsz}, qlen={qlen} (prompt+completion), n_chunks={n_chunks}{comp_mask_info}", 
                          file=sys.stderr, flush=True)
                    if prompt_len is not None:
                        print(f"[DEBUG grpo_accumulated_loss] prompt_len={prompt_len}, completion_len={completion_len}", 
                              file=sys.stderr, flush=True)
                    
                    # 모든 입력 텐서 shape 확인
                    if input_ids is not None and hasattr(input_ids, 'shape'):
                        input_shape_before = input_ids.shape
                        print(f"[DEBUG grpo_accumulated_loss] input_ids.shape (before left_pack)={input_shape_before}", 
                              file=sys.stderr, flush=True)
                    
                    if attention_mask is not None and hasattr(attention_mask, 'shape'):
                        print(f"[DEBUG grpo_accumulated_loss] attention_mask.shape={attention_mask.shape}", 
                              file=sys.stderr, flush=True)
                    
                    if advantages is not None and hasattr(advantages, 'shape'):
                        print(f"[DEBUG grpo_accumulated_loss] advantages.shape={advantages.shape}", 
                              file=sys.stderr, flush=True)
                    
                    print(f"[DEBUG grpo_accumulated_loss] =============================", 
                          file=sys.stderr, flush=True)
                    if max_prompt_len and prompt_len is not None and prompt_len > max_prompt_len:
                        print(f"[WARNING grpo_accumulated_loss] prompt_len({prompt_len}) > max_prompt_length({max_prompt_len})!", 
                              file=sys.stderr, flush=True)
                    if max_total and qlen > max_total:
                        print(f"[WARNING grpo_accumulated_loss] qlen({qlen}=prompt+completion) > max_total({max_total}=max_prompt+max_completion)!", 
                              file=sys.stderr, flush=True)
                    
                    # 방어: 빈 배치가 들어오면 0 loss 반환 (모든 반환값을 텐서로 맞춤)
                    if bsz == 0:
                        device = getattr(trainer.model, 'device', None) if hasattr(trainer, 'model') else None
                        if device is None:
                            try:
                                device = input_ids.device if input_ids is not None and hasattr(input_ids, 'device') else None
                            except:
                                device = None
                        zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                        completion_length = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                        # flat_is_ratio는 빈 텐서로 반환 (None이면 compute_loss에서 .numel() 호출 시 에러)
                        empty_tensor = torch.tensor([], device=device, dtype=torch.float32).detach() if device is not None else torch.tensor([], dtype=torch.float32).detach()
                        
                        # 빈 배치 원인 상세 분석
                        print(f"[WARNING grpo_accumulated_loss] ========== 빈 배치 감지 ==========", 
                              file=sys.stderr, flush=True)
                        print(f"[WARNING grpo_accumulated_loss] bsz=0, qlen={qlen}", 
                              file=sys.stderr, flush=True)
                        print(f"[WARNING grpo_accumulated_loss] prompt_len={prompt_len if prompt_len else 'unknown'}, completion_len={completion_len if completion_len else 'unknown'}", 
                              file=sys.stderr, flush=True)
                        print(f"[WARNING grpo_accumulated_loss] max_prompt_length={max_prompt_len}, max_completion_length={max_completion_len}", 
                              file=sys.stderr, flush=True)
                        
                        # input_ids 상세 정보
                        if input_ids is not None:
                            print(f"[WARNING grpo_accumulated_loss] input_ids.shape={input_ids.shape if hasattr(input_ids, 'shape') else 'N/A'}", 
                                  file=sys.stderr, flush=True)
                        
                        # completion_mask 상세 분석
                        if completion_mask is not None and hasattr(completion_mask, 'sum'):
                            completion_mask_sum = completion_mask.sum().item()
                            completion_mask_shape = completion_mask.shape if hasattr(completion_mask, 'shape') else 'unknown'
                            print(f"[WARNING grpo_accumulated_loss] completion_mask.shape={completion_mask_shape}, completion_mask.sum()={completion_mask_sum}", 
                                  file=sys.stderr, flush=True)
                            
                            if completion_mask_sum == 0:
                                print(f"[WARNING grpo_accumulated_loss] 원인: 모든 completion이 필터링되었습니다!", 
                                      file=sys.stderr, flush=True)
                                print(f"[WARNING grpo_accumulated_loss] 해결책: max_completion_length를 늘리거나 prompt를 줄이는 것을 고려하세요.", 
                                      file=sys.stderr, flush=True)
                            else:
                                print(f"[WARNING grpo_accumulated_loss] 원인: bsz=0인데 completion_mask.sum()={completion_mask_sum} (이상함!)", 
                                      file=sys.stderr, flush=True)
                        else:
                            print(f"[WARNING grpo_accumulated_loss] 원인: completion_mask가 None이거나 유효하지 않음", 
                                  file=sys.stderr, flush=True)
                        
                        # attention_mask 확인
                        if attention_mask is not None:
                            attn_mask_shape = attention_mask.shape if hasattr(attention_mask, 'shape') else 'unknown'
                            attn_mask_sum = attention_mask.sum().item() if hasattr(attention_mask, 'sum') else 'N/A'
                            print(f"[WARNING grpo_accumulated_loss] attention_mask.shape={attn_mask_shape}, attention_mask.sum()={attn_mask_sum}", 
                                  file=sys.stderr, flush=True)
                        
                        print(f"[WARNING grpo_accumulated_loss] =========================================", 
                              file=sys.stderr, flush=True)
                        
                        return zero, completion_length, zero, zero, empty_tensor
                    
                    # left_pack_padding 후 상태 확인 (원본 함수 내부에서 호출됨)
                    # 원본 함수 호출 전에 try-except로 감싸서 에러 처리
                    try:
                        result = original_grpo_accumulated_loss(
                            trainer, input_ids, attention_mask, logits_to_keep, completion_mask,
                            advantages, old_hidden_states, ref_hidden_states, n_chunks, **kwargs
                        )
                        
                        # 결과 검증
                        if result is not None and len(result) > 0:
                            # 첫 번째 반환값이 loss인지 확인
                            if isinstance(result[0], torch.Tensor):
                                loss_val = result[0].item() if hasattr(result[0], 'item') else None
                                if loss_val is not None and (torch.isnan(result[0]) or torch.isinf(result[0])):
                                    print(f"[WARNING grpo_accumulated_loss] 반환된 loss가 NaN 또는 Inf: {loss_val}", 
                                          file=sys.stderr, flush=True)
                        
                        return result
                    except (IndexError, AttributeError, RuntimeError, ValueError) as e:
                        # 다양한 에러 타입 처리 (IndexError, AttributeError, RuntimeError, ValueError 등)
                        error_type = type(e).__name__
                        
                        # IndexError의 경우 factors 계산 시도
                        factors = []
                        if isinstance(e, IndexError):
                            try:
                                factors = [i for i in range(1, bsz + 1) if bsz % i == 0] if bsz > 0 else []
                            except:
                                factors = []
                        
                        print(f"[DEBUG grpo_accumulated_loss] {error_type} 발생! bsz={bsz}, factors={factors}, len(factors)={len(factors)}, n_chunks={n_chunks}", 
                              file=sys.stderr, flush=True)
                        print(f"[DEBUG grpo_accumulated_loss] 에러 메시지: {e}", file=sys.stderr, flush=True)
                        
                        # factors가 비어있거나 문제가 있는 경우 0 loss 반환 (모든 반환값을 텐서로 맞춤)
                        if len(factors) == 0 or bsz <= 0:
                            device = getattr(trainer.model, 'device', None) if hasattr(trainer, 'model') else None
                            if device is None:
                                try:
                                    device = input_ids.device if hasattr(input_ids, 'device') else None
                                except:
                                    device = None
                            zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                            completion_length = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                            # flat_is_ratio는 빈 텐서로 반환 (None이면 compute_loss에서 .numel() 호출 시 에러)
                            empty_tensor = torch.tensor([], device=device, dtype=torch.float32).detach() if device is not None else torch.tensor([], dtype=torch.float32).detach()
                            print(f"[DEBUG grpo_accumulated_loss] factors 문제로 0 loss 반환", file=sys.stderr, flush=True)
                            return zero, completion_length, zero, zero, empty_tensor
                        
                        # 다시 시도 (이론상 발생하지 않아야 하지만)
                        print(f"[DEBUG grpo_accumulated_loss] {error_type} 재발생", file=sys.stderr, flush=True)
                        raise
                
                # 모든 모듈에서 함수 교체 (callable인 경우만)
                patched_count = 0
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'grpo_accumulated_loss'):
                        existing = getattr(module, 'grpo_accumulated_loss')
                        # callable이고 unsloth 관련 모듈인 경우만 패치
                        if callable(existing) and ('unsloth' in module_name.lower() or 'grpo' in module_name.lower()):
                            setattr(module, 'grpo_accumulated_loss', patched_grpo_accumulated_loss)
                            patched_count += 1
                            if not self.is_distributed or self.rank == 0:
                                logger.info(f"✅ {module_name}의 grpo_accumulated_loss 패치 완료")
                
                # 캐시 파일 모듈에도 직접 패치
                if cache_file.exists():
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("unsloth_compiled_cache_UnslothGRPOTrainer", cache_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            if hasattr(module, 'grpo_accumulated_loss'):
                                module.grpo_accumulated_loss = patched_grpo_accumulated_loss
                                patched_count += 1
                                if not self.is_distributed or self.rank == 0:
                                    logger.info(f"✅ 캐시 파일 모듈의 grpo_accumulated_loss 패치 완료")
                    except Exception as e:
                        if not self.is_distributed or self.rank == 0:
                            logger.debug(f"캐시 파일 모듈 패치 시도 중 에러 (무시): {e}")
                
                if not self.is_distributed or self.rank == 0:
                    logger.info(f"✅ 총 {patched_count}개 모듈에서 grpo_accumulated_loss 패치 완료")
                
                # trainer 내부에서도 직접 참조할 수 있으므로 compute_loss 오버라이드
                original_compute_loss = trainer.compute_loss
                def patched_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
                    """compute_loss 래퍼 - 빈 배치 체크 (다양한 에러 방어)"""
                    try:
                        # 원본 함수 호출
                        return original_compute_loss(model, inputs, return_outputs, num_items_in_batch)
                    except (IndexError, AttributeError, RuntimeError) as e:
                        # 다양한 에러 타입 처리 (IndexError, AttributeError, RuntimeError 등)
                        error_type = type(e).__name__
                        prompt_ids = inputs.get("prompt_ids", None)
                        completion_ids = inputs.get("completion_ids", None)
                        
                        print(f"[DEBUG compute_loss] {error_type} 발생: {e}", file=sys.stderr, flush=True)
                        if prompt_ids is not None:
                            try:
                                print(f"[DEBUG compute_loss] prompt_ids.shape={prompt_ids.shape}", file=sys.stderr, flush=True)
                            except:
                                print(f"[DEBUG compute_loss] prompt_ids={prompt_ids}", file=sys.stderr, flush=True)
                        if completion_ids is not None:
                            try:
                                print(f"[DEBUG compute_loss] completion_ids.shape={completion_ids.shape}", file=sys.stderr, flush=True)
                            except:
                                print(f"[DEBUG compute_loss] completion_ids={completion_ids}", file=sys.stderr, flush=True)
                        
                        # 실제로 빈 배치인 경우에만 0 loss 반환
                        if prompt_ids is not None and completion_ids is not None:
                            try:
                                prompt_bsz = prompt_ids.shape[0] if hasattr(prompt_ids, 'shape') and len(prompt_ids.shape) > 0 else -1
                                completion_bsz = completion_ids.shape[0] if hasattr(completion_ids, 'shape') and len(completion_ids.shape) > 0 else -1
                                
                                if prompt_bsz == 0 or completion_bsz == 0:
                                    device = getattr(model, 'device', None) if hasattr(model, 'device') else None
                                    if device is None:
                                        try:
                                            if prompt_bsz > 0 and hasattr(prompt_ids, 'device'):
                                                device = prompt_ids.device
                                            elif completion_bsz > 0 and hasattr(completion_ids, 'device'):
                                                device = completion_ids.device
                                        except:
                                            device = None
                                    zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                                    print(f"[DEBUG compute_loss] 빈 배치 확인! 0 loss 반환 (prompt_bsz={prompt_bsz}, completion_bsz={completion_bsz})", 
                                          file=sys.stderr, flush=True)
                                    return zero
                            except Exception as shape_err:
                                print(f"[DEBUG compute_loss] shape 체크 중 에러: {shape_err}", file=sys.stderr, flush=True)
                        
                        # 빈 배치가 아닌데 에러가 발생한 경우는 재발생
                        print(f"[DEBUG compute_loss] 빈 배치가 아닌데 {error_type} 발생, 재발생", file=sys.stderr, flush=True)
                        raise
                
                trainer.compute_loss = patched_compute_loss
                
                if not self.is_distributed or self.rank == 0:
                    logger.info("✅ grpo_accumulated_loss 함수 패치 완료 (빈 배치 방어 + 디버깅)")
            else:
                if not self.is_distributed or self.rank == 0:
                    logger.warning("⚠️ grpo_accumulated_loss 함수를 찾을 수 없습니다")
        except Exception as e:
            if not self.is_distributed or self.rank == 0:
                logger.warning(f"⚠️ grpo_accumulated_loss 패치 실패 (계속 진행): {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # ===== 훈련 실행 =====
        if not self.is_distributed or self.rank == 0:
            logger.info("🏃 훈련 시작!\n")
        
        try:
            trainer.train()
            
            # ===== 모델 저장 (rank 0에서만) =====
            if not self.is_distributed or self.rank == 0:
                logger.info(f"\n💾 모델 저장 중: {save_dir}")
                trainer.save_model(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                logger.info("✅ 모델 저장 완료!")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 훈련 중 에러 발생: {error_msg}", exc_info=True)
            
            # CUDA illegal memory access 에러에 대한 추가 정보
            if "illegal memory access" in error_msg.lower() or "CUDA error" in error_msg:
                logger.error(
                    "\n" + "="*60 + "\n"
                    "🔍 CUDA Illegal Memory Access 에러 해결 방법:\n"
                    "1. vLLM GPU 메모리 사용률을 더 낮추세요 (현재: 0.2)\n"
                    "2. 배치 크기나 시퀀스 길이를 줄이세요\n"
                    "3. CUDA_LAUNCH_BLOCKING=1 환경 변수를 설정하여 디버깅하세요:\n"
                    "   export CUDA_LAUNCH_BLOCKING=1\n"
                    "4. vLLM을 server 모드로 변경하여 별도 프로세스로 분리하세요\n"
                    "="*60
                )
            
            raise
        finally:
            # CUDA 메모리 정리 (에러가 발생해도 시도)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass  # CUDA 에러가 발생한 경우 empty_cache도 실패할 수 있음
        
        if not self.is_distributed or self.rank == 0:
            logger.info("\n🎉 훈련 완료!")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 3 메인 함수"""
    
    # ===== DDP 환경 변수 설정 =====
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank >= 0
    
    if is_distributed:
        # DDP 필수 환경 변수
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '300'
        os.environ['NCCL_IB_DISABLE'] = '1'
        
        # vLLM 환경 변수 (각 rank별로 고유하게)
        os.environ['VLLM_WORKER_NAME'] = f'worker_{rank}'
        os.environ['VLLM_INSTANCE_ID'] = str(rank)
        
        # 디버그 출력
        print(
            f"[RANK {rank}] 환경 변수 설정 완료\n"
            f"  - MASTER_ADDR: {os.environ['MASTER_ADDR']}\n"
            f"  - MASTER_PORT: {os.environ['MASTER_PORT']}\n"
            f"  - LOCAL_RANK: {local_rank}\n"
            f"  - WORLD_SIZE: {world_size}",
            file=sys.stderr
        )
        sys.stderr.flush()
    
    # ===== DDP 초기화 (TRL 전에 수동으로) =====
    if is_distributed and not dist.is_initialized():
        print(f"[RANK {rank}] DDP 초기화 시도...", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=60)  # 타임아웃 증가
            )
            torch.cuda.set_device(local_rank)
            print(f"✅ [RANK {rank}] DDP 초기화 성공! (GPU {local_rank})", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"❌ [RANK {rank}] DDP 초기화 실패: {e}", file=sys.stderr)
            sys.stderr.flush()
            raise
    
    # ===== 로깅 설정 (rank 0에서만) =====
    if not is_distributed or rank == 0:
        log_file = os.path.join(cfg.paths.log_dir, "stage3_train.log")
        setup_logging(
            log_level=cfg.experiment.get("log_level", "INFO"),
            log_file=log_file,
            wandb_enabled=cfg.experiment.wandb.enabled,
            wandb_project=cfg.experiment.wandb.project,
            wandb_tags=cfg.experiment.wandb.tags
        )
        
        logger.info("🚀 Stage 3: GRPO 모델 훈련 시작 (Unsloth + vLLM + TRL)")
        logger.info(f"설정: {cfg.training}")
        
        # 디렉토리 생성
        os.makedirs(cfg.paths.model_dir, exist_ok=True)
    
    # ===== 데이터 경로 확인 =====
    # train_data_path = os.path.join(cfg.paths.data_dir, "curated", "train_filtered.parquet")
    # validation_data_path = os.path.join(cfg.paths.data_dir, "curated", "validation_filtered.parquet")
    train_data_path = os.path.join(cfg.paths.data_dir, "curated", "train_curated.parquet")
    validation_data_path = os.path.join(cfg.paths.data_dir, "curated", "validation_curated.parquet")

    if not os.path.exists(train_data_path):
        if not is_distributed or rank == 0:
            logger.error(f"❌ 훈련 데이터 파일을 찾을 수 없습니다: {train_data_path}")
            logger.error("먼저 Stage 2를 실행해주세요: python scripts/stage2_curate.py")
        return
    
    # ===== 데이터셋 생성 =====
    if not is_distributed or rank == 0:
        logger.info("📦 데이터셋 생성 중...")
    
    from src.data.training_dataset import CuratedTrainingDataset
    
    train_dataset = CuratedTrainingDataset(train_data_path)
    
    if not is_distributed or rank == 0:
        logger.info(f"✅ 훈련 데이터셋: {len(train_dataset)} 샘플")
    
    validation_dataset = None
    # logger.info("검증 데이터셋 비활성화")
    if os.path.exists(validation_data_path):
        validation_dataset = CuratedTrainingDataset(validation_data_path)
        # 빈 검증 세트는 평가 루프에서 빈 배치(bsz==0)를 유발하여 크래시할 수 있으므로 비활성화
        if len(validation_dataset) == 0:
            if not is_distributed or rank == 0:
                logger.warning("⚠️ 검증 데이터셋이 비어 있습니다. 평가를 비활성화합니다.")
            validation_dataset = None
        else:
            if not is_distributed or rank == 0:
                logger.info(f"✅ 검증 데이터셋: {len(validation_dataset)} 샘플")
    
    # ===== GRPO 트레이너 초기화 =====
    if not is_distributed or rank == 0:
        logger.info("🦥 Unsloth GRPO 트레이너 초기화 중...")
    
    trainer = OptimizedGRPOTrainer(
        model_name=cfg.model.base_model,
        lora_config=cfg.training.lora if cfg.training.method == "lora" else None,
        grpo_config=cfg.training.grpo,
        training_config=cfg.training.training,
        device=cfg.experiment.device
    )
    
    # ===== 훈련 실행 =====
    if not is_distributed or rank == 0:
        logger.info("🏋️ 모델 훈련 시작...")
    
    trainer.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        save_dir=cfg.paths.model_dir
    )
    
    if not is_distributed or rank == 0:
        logger.info("✅ Stage 3 완료")
        logger.info(f"📁 훈련된 모델 저장 위치: {cfg.paths.model_dir}")
    
    # ===== DDP 정리 =====
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()