"""
vLLM 기반 고속 추론 엔진
"""
import os
import torch
from typing import List, Dict, Any, Optional
import logging
from vllm import LLM, SamplingParams
import numpy as np

logger = logging.getLogger(__name__)


class VLLMInferenceEngine:
    """vLLM 기반 고속 추론 엔진 클래스"""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 32768,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        cache_dir: str = None,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 128,
    ):
        """
        Args:
            model_name: 모델 이름 (예: "Qwen/Qwen3-1.7B")
            tensor_parallel_size: 텐서 병렬 크기 (GPU 개수)
            gpu_memory_utilization: GPU 메모리 사용률
            max_model_len: 최대 모델 길이
            dtype: 데이터 타입
            trust_remote_code: 원격 코드 신뢰 여부
            cache_dir: 캐시 디렉토리
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """vLLM 모델을 초기화합니다."""
        try:
            logger.info(f"vLLM 모델 초기화 중: {self.model_name}")
            logger.info(f"텐서 병렬 크기: {self.tensor_parallel_size}")
            logger.info(f"GPU 메모리 사용률: {self.gpu_memory_utilization}")
            
            # vLLM 엔진 초기화
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                download_dir=self.cache_dir,
                max_num_batched_tokens=self._max_num_batched_tokens,
                max_num_seqs=self._max_num_seqs,
                enforce_eager=False,
            )
            
            logger.info("vLLM 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"vLLM 모델 초기화 실패: {e}")
            raise
    
    def generate_multiple_responses(
        self,
        prompt: str,
        num_responses: int = 128,
        temperature: float = 1.5,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 16384,
        num_logprobs: int = 5,
        # Advanced sampling options
        prompt_logprobs: int = 0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        단일 프롬프트에 대해 여러 응답을 고속으로 생성합니다.
        
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
        logger.info(f"vLLM 여러 응답 생성 시작: {num_responses}개")
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=prompt_logprobs,
            stop=stop,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        
        # 프롬프트를 num_responses만큼 복제
        prompts = [prompt] * num_responses
        
        try:
            # vLLM으로 배치 생성
            outputs = self.llm.generate(prompts, sampling_params)
            
            # 결과 처리
            results = []
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                
                # 로그 확률 추출
                logprobs = []
                token_topk_seq = getattr(output.outputs[0], "logprobs", None)
                if token_topk_seq:
                    for token_topk in token_topk_seq:
                        if not token_topk:
                            logprobs.append([])
                            continue
                        token_logprobs: list[float] = []
                        for lp in token_topk:
                            # vLLM may return objects with attribute, dicts, or tuples
                            if hasattr(lp, "logprob"):
                                token_logprobs.append(float(lp.logprob))
                            elif isinstance(lp, dict) and "logprob" in lp:
                                token_logprobs.append(float(lp["logprob"]))
                            elif isinstance(lp, (tuple, list)) and len(lp) >= 2 and isinstance(lp[1], (int, float)):
                                token_logprobs.append(float(lp[1]))
                        logprobs.append(token_logprobs)
                
                results.append({
                    "generated_text": generated_text,
                    "logprobs": logprobs,
                    "finish_reason": output.outputs[0].finish_reason
                })
            
            logger.info(f"vLLM 여러 응답 생성 완료: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"vLLM 생성 실패: {e}")
            # 실패한 경우 빈 결과 반환
            return [{
                "generated_text": "",
                "logprobs": [],
                "finish_reason": "error"
            }] * num_responses
    
    def cleanup(self):
        """리소스를 정리합니다."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            logger.info("vLLM 모델 리소스 정리 완료")