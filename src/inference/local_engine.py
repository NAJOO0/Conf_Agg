"""
로컬 모델 기반 추론 엔진 (transformers 사용)
"""
import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logger = logging.getLogger(__name__)


class LocalInferenceEngine:
    """로컬 모델 기반 추론 엔진 클래스"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        cache_dir: str = None
    ):
        """
        Args:
            model_name: 모델 이름 (예: "Qwen/Qwen3-1.7B")
            device: 디바이스
            torch_dtype: 텐서 데이터 타입
            trust_remote_code: 원격 코드 신뢰 여부
            cache_dir: Hugging Face 캐시 디렉토리
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """모델과 토크나이저를 초기화합니다."""
        try:
            logger.info(f"로컬 모델 초기화 중: {self.model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 - 최적화된 병렬화 방식
            if self.device == "cuda" and torch.cuda.device_count() > 1:
                # Tensor Parallelism: 각 GPU가 모델의 다른 부분을 담당하되 통신 최소화
                logger.info(f"Tensor Parallelism 모드: {torch.cuda.device_count()}개 GPU 사용")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",  # Transformers가 최적의 분산 전략 선택
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    low_cpu_mem_usage=True,  # 메모리 효율성
                    max_memory={i: "40GB" for i in range(torch.cuda.device_count())}  # 각 GPU 메모리 제한
                )
            else:
                # 단일 GPU 또는 CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir
                )
            
            # 디바이스 매핑 정보 출력
            if hasattr(self.model, 'hf_device_map'):
                logger.info("모델 디바이스 매핑:")
                for name, device in self.model.hf_device_map.items():
                    logger.info(f"  {name}: {device}")
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("로컬 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"로컬 모델 초기화 실패: {e}")
            raise
    
    def generate_with_logprobs(
        self,
        prompt: str,
        temperature: float = 1.5,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 16384,
        num_logprobs: int = 5
    ) -> Dict[str, Any]:
        """
        로그 확률과 함께 텍스트를 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            temperature: 샘플링 온도
            top_p: Top-p 값
            top_k: Top-k 값
            min_p: Min-p 값
            max_tokens: 최대 토큰 수
            num_logprobs: 반환할 로그 확률 개수
        
        Returns:
            생성 결과와 로그 확률
        """
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32768 - max_tokens  # 전체 컨텍스트 길이 고려
        )
        
        # 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 생성 파라미터 설정
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,  # 로그 확률을 얻기 위해 필요
            "use_cache": True
        }
        
        # Min-p 필터링을 위한 커스텀 샘플링 함수
        if min_p > 0:
            generation_kwargs["min_p"] = min_p
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # 생성된 텍스트 추출
        generated_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 로그 확률 추출
        logprobs = self._extract_logprobs_from_scores(
            outputs.scores, 
            generated_tokens, 
            num_logprobs
        )
        
        return {
            "generated_text": generated_text,
            "logprobs": logprobs,
            "finish_reason": "stop"  # 간단히 설정
        }
    
    def generate_batch_with_logprobs(
        self,
        prompt: str,
        batch_size: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,
        max_tokens: int = 16384,
        num_logprobs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        배치로 여러 응답을 동시에 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            batch_size: 배치 크기
            temperature: 샘플링 온도
            top_p: Top-p 값
            top_k: Top-k 값
            min_p: Min-p 값
            max_tokens: 최대 토큰 수
            num_logprobs: 반환할 로그 확률 개수
        
        Returns:
            생성된 응답 리스트
        """
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32768 - max_tokens
        )
        
        # 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 배치로 확장 (같은 프롬프트를 batch_size만큼 반복)
        batch_inputs = {
            "input_ids": inputs["input_ids"].repeat(batch_size, 1),
            "attention_mask": inputs["attention_mask"].repeat(batch_size, 1)
        }
        
        # 생성 파라미터 설정
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True
        }
        
        # Min-p 필터링을 위한 커스텀 샘플링 함수
        if min_p > 0:
            generation_kwargs["min_p"] = min_p
        
        with torch.no_grad():
            outputs = self.model.generate(**batch_inputs, **generation_kwargs)
        
        # 배치 결과 처리
        results = []
        input_length = inputs["input_ids"].shape[1]
        
        for i in range(batch_size):
            # 생성된 텍스트 추출
            generated_tokens = outputs.sequences[i][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 로그 확률 추출
            logprobs = self._extract_logprobs_from_scores_batch(
                outputs.scores, 
                generated_tokens, 
                num_logprobs,
                i  # 배치 인덱스
            )
            
            results.append({
                "generated_text": generated_text,
                "logprobs": logprobs,
                "finish_reason": "stop"
            })
        
        return results
    
    def _extract_logprobs_from_scores(
        self, 
        scores: List[torch.Tensor], 
        generated_tokens: torch.Tensor,
        num_logprobs: int
    ) -> List[List[float]]:
        """
        생성 점수에서 로그 확률을 추출합니다.
        
        Args:
            scores: 생성된 각 스텝의 점수
            generated_tokens: 생성된 토큰들
            num_logprobs: 반환할 로그 확률 개수
        
        Returns:
            토큰별 로그 확률 리스트
        """
        logprobs = []
        
        for i, (score, token_id) in enumerate(zip(scores, generated_tokens)):
            # 로그 소프트맥스 적용
            log_probs = F.log_softmax(score[0], dim=-1)
            
            # 상위 num_logprobs개 로그 확률 추출
            top_logprobs, top_indices = torch.topk(log_probs, num_logprobs, dim=-1)
            
            # 실제 생성된 토큰의 로그 확률도 포함
            actual_logprob = log_probs[token_id].item()
            
            # 상위 로그 확률들을 리스트로 변환
            token_logprobs = [actual_logprob]  # 실제 토큰의 로그 확률을 첫 번째로
            for j in range(num_logprobs - 1):
                token_logprobs.append(top_logprobs[j].item())
            
            logprobs.append(token_logprobs)
        
        return logprobs
    
    def generate_multiple_responses(
        self,
        prompt: str,
        num_responses: int = 128,
        temperature: float = 1.5,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 16384,
        num_logprobs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        단일 프롬프트에 대해 여러 응답을 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            num_responses: 생성할 응답 개수
            temperature: 샘플링 온도
            top_p: Top-p 값
            top_k: Top-k 값
            min_p: Min-p 값
            max_tokens: 최대 토큰 수
            num_logprobs: 반환할 로그 확률 개수
        
        Returns:
            생성된 응답 리스트
        """
        logger.info(f"여러 응답 생성 시작: {num_responses}개")
        
        results = []
        batch_size = min(32, num_responses)  # 배치 크기 증가 (GPU 활용도 향상)
        
        for i in range(0, num_responses, batch_size):
            current_batch_size = min(batch_size, num_responses - i)
            
            try:
                # 진짜 배치 처리: 한 번에 여러 응답 생성
                batch_results = self.generate_batch_with_logprobs(
                    prompt=prompt,
                    batch_size=current_batch_size,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    num_logprobs=num_logprobs
                )
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"배치 생성 실패 (배치 {i//batch_size + 1}): {e}")
                # 실패한 경우 개별 처리로 폴백
                for j in range(current_batch_size):
                    try:
                        result = self.generate_with_logprobs(
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            max_tokens=max_tokens,
                            num_logprobs=num_logprobs
                        )
                        results.append(result)
                    except Exception as e2:
                        logger.warning(f"개별 응답 생성 실패 (응답 {i+j}): {e2}")
                        results.append({
                            "generated_text": "",
                            "logprobs": [],
                            "finish_reason": "error"
                        })
            
            logger.info(f"진행률: {len(results)}/{num_responses}")
        
        logger.info(f"여러 응답 생성 완료: {len(results)}개")
        return results
    
    def _extract_logprobs_from_scores_batch(
        self, 
        scores: List[torch.Tensor], 
        generated_tokens: torch.Tensor,
        num_logprobs: int,
        batch_idx: int
    ) -> List[List[float]]:
        """
        배치 생성 점수에서 로그 확률을 추출합니다.
        
        Args:
            scores: 생성된 각 스텝의 점수
            generated_tokens: 생성된 토큰들
            num_logprobs: 반환할 로그 확률 개수
            batch_idx: 배치 내 인덱스
        
        Returns:
            토큰별 로그 확률 리스트
        """
        logprobs = []
        
        for i, (score, token_id) in enumerate(zip(scores, generated_tokens)):
            # 로그 소프트맥스 적용 (배치 인덱스 사용)
            log_probs = F.log_softmax(score[batch_idx], dim=-1)
            
            # 상위 num_logprobs개 로그 확률 추출
            top_logprobs, top_indices = torch.topk(log_probs, num_logprobs, dim=-1)
            
            # 실제 생성된 토큰의 로그 확률도 포함
            actual_logprob = log_probs[token_id].item()
            
            # 상위 로그 확률들을 리스트로 변환
            token_logprobs = [actual_logprob]  # 실제 토큰의 로그 확률을 첫 번째로
            for j in range(num_logprobs - 1):
                token_logprobs.append(top_logprobs[j].item())
            
            logprobs.append(token_logprobs)
        
        return logprobs
    
    def cleanup(self):
        """리소스를 정리합니다."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("로컬 모델 리소스 정리 완료")

