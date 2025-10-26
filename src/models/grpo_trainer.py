"""
GRPO 트레이너 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Group-Relative Policy Optimization 트레이너"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        grpo_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Args:
            model_name: 모델 이름
            tokenizer_name: 토크나이저 이름 (None이면 model_name 사용)
            lora_config: LoRA 설정
            grpo_config: GRPO 설정
            training_config: 훈련 설정
            device: 디바이스
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 설정 초기화
        self.lora_config = lora_config or {}
        self.grpo_config = grpo_config or {}
        self.training_config = training_config or {}
        
        # 기본값 설정
        self.group_size = self.grpo_config.get("group_size", 8)
        self.kl_coefficient = self.grpo_config.get("kl_coefficient", 0.001)
        self.aggregator_temperature = self.grpo_config.get("aggregator_temperature", 1.5)
        
        # 모델과 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self.reference_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """모델과 토크나이저를 초기화합니다."""
        logger.info(f"모델 초기화: {self.model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA 적용
        if self.lora_config:
            lora_config = LoraConfig(
                r=self.lora_config.get("r", 16),
                lora_alpha=self.lora_config.get("lora_alpha", 32),
                lora_dropout=self.lora_config.get("lora_dropout", 0.1),
                target_modules=self.lora_config.get("target_modules", [
                    "q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ]),
                bias=self.lora_config.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA 적용 완료")
        
        # 참조 모델 (KL divergence 계산용)
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 참조 모델은 훈련하지 않음
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        logger.info("모델 초기화 완료")
    
    def compute_grpo_advantages(
        self, 
        rewards: List[float], 
        group_size: Optional[int] = None
    ) -> List[float]:
        """
        GRPO의 핵심: 그룹 기반 이점 계산
        
        Args:
            rewards: 보상 점수 리스트
            group_size: 그룹 크기
        
        Returns:
            이점 점수 리스트
        """
        if group_size is None:
            group_size = self.group_size
        
        advantages = []
        rewards = np.array(rewards)
        
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i + group_size]
            
            if len(group_rewards) == 0:
                continue
            
            # 그룹 내 평균과 표준편차 계산
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8  # 안정성을 위해 작은 값 추가
            
            # 그룹 기반 이점 계산
            group_advantages = (group_rewards - mean_reward) / std_reward
            advantages.extend(group_advantages.tolist())
        
        return advantages
    
    def compute_kl_divergence(
        self, 
        logits: torch.Tensor, 
        reference_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence를 계산합니다.
        
        Args:
            logits: 현재 모델의 로짓
            reference_logits: 참조 모델의 로짓
        
        Returns:
            KL divergence
        """
        # 확률 분포로 변환
        probs = F.softmax(logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        # KL divergence 계산
        kl_div = F.kl_div(
            F.log_softmax(logits, dim=-1),
            reference_probs,
            reduction="batchmean"
        )
        
        return kl_div
    
    def aggregate_responses(
        self, 
        responses: List[str], 
        confidence_scores: List[Dict[str, float]]
    ) -> str:
        """
        신뢰도 기반으로 응답을 집계합니다.
        
        Args:
            responses: 응답 리스트
            confidence_scores: 신뢰도 점수 리스트
        
        Returns:
            집계된 최종 응답
        """
        if not responses:
            return ""
        
        # 신뢰도 점수 계산 (여러 메트릭의 평균)
        scores = []
        for conf_dict in confidence_scores:
            score = np.mean(list(conf_dict.values()))
            scores.append(score)
        
        # 온도 기반 샘플링
        scores = np.array(scores)
        if self.aggregator_temperature > 0:
            # 온도 적용
            scaled_scores = scores / self.aggregator_temperature
            probs = F.softmax(torch.tensor(scaled_scores), dim=0).numpy()
        else:
            # 최고 점수 선택
            probs = np.zeros_like(scores)
            probs[np.argmax(scores)] = 1.0
        
        # 확률에 따라 응답 선택
        selected_idx = np.random.choice(len(responses), p=probs)
        return responses[selected_idx]
    
    def train_step(
        self, 
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        단일 훈련 스텝을 수행합니다.
        
        Args:
            batch: 배치 데이터
        
        Returns:
            훈련 메트릭
        """
        self.model.train()
        
        # 배치 데이터 추출
        prompts = batch["prompts"]
        responses = batch["responses"]
        confidence_scores = batch["confidence_scores"]
        ground_truths = batch["ground_truths"]
        
        # 집계된 응답 생성
        aggregated_responses = []
        for resp_group, conf_group in zip(responses, confidence_scores):
            agg_resp = self.aggregate_responses(resp_group, conf_group)
            aggregated_responses.append(agg_resp)
        
        # 보상 계산 (간단한 정확도 기반)
        rewards = []
        for agg_resp, gt in zip(aggregated_responses, ground_truths):
            # 간단한 문자열 매칭으로 보상 계산
            reward = 1.0 if agg_resp.strip().lower() == gt.strip().lower() else 0.0
            rewards.append(reward)
        
        # GRPO 이점 계산
        advantages = self.compute_grpo_advantages(rewards)
        
        # 텍스트를 토큰으로 변환
        all_texts = []
        for prompt, resp in zip(prompts, aggregated_responses):
            text = prompt + resp
            all_texts.append(text)
        
        # 토크나이징
        inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.get("max_prompt_length", 16384) + 
                      self.training_config.get("max_response_length", 16384)
        )
        
        # 모델 출력
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # 참조 모델 출력
        with torch.no_grad():
            ref_outputs = self.reference_model(**inputs)
            ref_logits = ref_outputs.logits
        
        # KL divergence 계산
        kl_div = self.compute_kl_divergence(logits, ref_logits)
        
        # 손실 계산 (간단한 버전)
        loss = -torch.mean(torch.tensor(advantages)) + self.kl_coefficient * kl_div
        
        # 역전파
        loss.backward()
        
        return {
            "loss": loss.item(),
            "kl_divergence": kl_div.item(),
            "mean_reward": np.mean(rewards),
            "mean_advantage": np.mean(advantages)
        }
    
    def train(
        self, 
        train_dataset: DataLoader,
        validation_dataset: Optional[DataLoader] = None,
        save_dir: str = "outputs/models"
    ):
        """
        모델을 훈련합니다.
        
        Args:
            train_dataset: 훈련 데이터셋
            validation_dataset: 검증 데이터셋 (선택사항)
            save_dir: 모델 저장 디렉토리
        """
        logger.info("모델 훈련 시작")
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 5e-5),
            weight_decay=self.training_config.get("weight_decay", 0.01)
        )
        
        # 훈련 루프
        epochs = self.training_config.get("epochs", 1)
        logging_steps = self.training_config.get("logging_steps", 50)
        save_steps = self.training_config.get("save_steps", 500)
        
        global_step = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} 시작")
            
            epoch_metrics = {
                "loss": [],
                "kl_divergence": [],
                "mean_reward": [],
                "mean_advantage": []
            }
            
            progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # 훈련 스텝
                metrics = self.train_step(batch)
                
                # 메트릭 수집
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                
                # 옵티마이저 스텝
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 로깅
                if global_step % logging_steps == 0:
                    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
                    logger.info(f"Step {global_step}: {avg_metrics}")
                    
                    if wandb.run:
                        wandb.log(avg_metrics, step=global_step)
                
                # 체크포인트 저장
                if global_step % save_steps == 0:
                    self.save_checkpoint(save_dir, global_step)
                
                # 진행률 업데이트
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.4f}"
                })
            
            # 에포크 완료 로깅
            epoch_avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch + 1} 완료: {epoch_avg_metrics}")
        
        # 최종 모델 저장
        self.save_checkpoint(save_dir, "final")
        logger.info("모델 훈련 완료")
    
    def save_checkpoint(self, save_dir: str, step: str):
        """체크포인트를 저장합니다."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint-{step}")
        
        # LoRA 가중치만 저장
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_path)
        
        logger.info(f"체크포인트 저장: {checkpoint_path}")

