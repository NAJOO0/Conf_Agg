"""
종합 벤치마크 평가 모듈
Baseline과 AggLLM 모델의 다양한 aggregation 방법 비교
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.data.confidence import ConfidenceCalculator
from src.evaluation.math_verifier import MathVerifier
from src.utils.metrics import calculate_pass_at_k

logger = logging.getLogger(__name__)


def extract_reasoning_content(text: str) -> str:
    """
    <think>부터 </think> 전까지 추출 (enable_thinking=True일 때)
    
    Args:
        text: generated_text
        
    Returns:
        reasoning 내용 (없으면 빈 문자열)
    """
    if not text:
        return ""
    
    text_str = str(text)
    
    # <think> 마커 찾기
    start_marker = "<think>"
    end_marker = "</think>"
    
    start_pos = text_str.find(start_marker)
    if start_pos == -1:
        return ""
    
    end_pos = text_str.find(end_marker, start_pos)
    if end_pos == -1:
        return ""
    
    # reasoning 부분 추출
    reasoning_start = start_pos + len(start_marker)
    reasoning_content = text_str[reasoning_start:end_pos].strip()
    
    return reasoning_content


def extract_content(text: str) -> str:
    """
    </think> 토큰 이후 값들 추출 (enable_thinking=True일 때)
    마커가 없으면 전체 텍스트 반환
    
    Args:
        text: generated_text
        
    Returns:
        </think> 이후 내용 (마커가 없으면 전체 텍스트)
    """
    if not text:
        return ""
    
    text_str = str(text)
    marker = "</think>"
    
    marker_pos = text_str.find(marker)
    if marker_pos == -1:
        # 마커가 없으면 전체 텍스트 반환 (enable_thinking=False인 경우)
        return text_str.strip()
    
    # 마커 이후 텍스트 추출
    content = text_str[marker_pos + len(marker):].strip()
    return content


def simple_extract_topk(gen_logprobs: List[Dict[int, Any]], k: int) -> List[List[float]]:
    """최적화된 logprob 추출 함수 (float16으로 메모리 절약)"""
    if not gen_logprobs:
        return []
    
    results = []
    for token_step_dict in gen_logprobs:
        if not token_step_dict:
            results.append([])
            continue
        
        lps = []
        for i, entry in enumerate(token_step_dict.values()):
            if i >= k:
                break
            if hasattr(entry, "logprob"):
                lps.append(float(entry.logprob))
            elif isinstance(entry, dict) and "logprob" in entry:
                lps.append(float(entry["logprob"]))
        
        if lps:
            # float16 변환 (메모리 절약)
            results.append(np.array(lps, dtype=np.float16).tolist())
        else:
            results.append([])
    
    return results


class ComprehensiveBenchmarkEvaluator:
    """종합 벤치마크 평가 클래스"""
    
    def __init__(
        self,
        model_name: str,
        aggllm_model_path: Optional[str] = None,
        confidence_calculator: Optional[ConfidenceCalculator] = None,
        math_verifier: Optional[MathVerifier] = None,
        num_solutions: int = 16,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        logprobs: int = 5,
        enable_thinking: bool = False,
        device: str = "cuda",
        use_vllm_for_aggllm: bool = True,
        merged_model_cache_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        aggllm_gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None
    ):
        """
        Args:
            model_name: 모델 이름 (Qwen3-1.7B)
            aggllm_model_path: AggLLM LoRA 모델 경로
            confidence_calculator: 신뢰도 계산기
            math_verifier: 수학 검증기
            num_solutions: 생성할 solution 수 (기본 16개)
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p 샘플링
            top_k: Top-k 샘플링
            min_p: Min-p 샘플링
            logprobs: 로그 확률 개수
            enable_thinking: enable_thinking 플래그
            device: 디바이스
            use_vllm_for_aggllm: AggLLM도 vLLM 사용 여부 (기본 True)
            merged_model_cache_dir: 병합된 모델 캐시 디렉토리 (None이면 임시 디렉토리 사용)
            gpu_memory_utilization: Baseline 모델의 GPU 메모리 사용률 (기본 0.85)
            aggllm_gpu_memory_utilization: AggLLM 모델의 GPU 메모리 사용률 (None이면 0.4로 자동 설정)
            max_model_len: 최대 모델 길이 (None이면 max_tokens + 2048로 자동 설정, 메모리 절약)
        """
        self.model_name = model_name
        self.aggllm_model_path = aggllm_model_path
        self.num_solutions = num_solutions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.logprobs = logprobs
        self.enable_thinking = enable_thinking
        self.device = device
        self.use_vllm_for_aggllm = use_vllm_for_aggllm
        self.merged_model_cache_dir = merged_model_cache_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        # AggLLM 메모리 사용률이 지정되지 않으면 자동으로 낮은 값 설정 (두 모델 동시 사용 고려)
        self.aggllm_gpu_memory_utilization = aggllm_gpu_memory_utilization if aggllm_gpu_memory_utilization is not None else 0.4
        # max_model_len이 지정되지 않으면 max_tokens + 여유분으로 설정 (KV cache 메모리 절약)
        self.max_model_len = max_model_len if max_model_len is not None else max_tokens + 8192
        
        # 신뢰도 계산기 및 검증기 초기화
        self.confidence_calculator = confidence_calculator or ConfidenceCalculator()
        self.math_verifier = math_verifier or MathVerifier()
        
        # 모델 로드
        self.baseline_llm = None  # vLLM LLM
        self.baseline_tokenizer = None
        self.aggllm_model = None  # transformers (LoRA 지원, use_vllm_for_aggllm=False일 때만 사용)
        self.aggllm_tokenizer = None
        self.aggllm_llm = None  # vLLM LLM (use_vllm_for_aggllm=True일 때 사용)
        self.use_vllm = True  # Baseline에 vLLM 사용 여부
        
        # Aggregation 프롬프트 템플릿
        self.aggregation_prompt_template = (
            "Given the following problem:\n{problem}\n"
            "and these solution attempts:\n{solutions}\n"
            "It is possible that any, all, or none of these solutions are correct or complete. Carefully review the\n"
            "provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining\n"
            "useful ideas—to produce a final, comprehensive, and correct solution to the problem."
        )
        
        # 기본 프롬프트 instruction
        self.base_instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    
    def load_baseline_model(self):
        """Baseline 모델 로드 (vLLM 사용)"""
        if self.baseline_llm is not None:
            return
        
        logger.info(f"Baseline 모델 로드 중 (vLLM): {self.model_name}")
        
        # 토크나이저 로드
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
        
        # vLLM LLM 초기화
        self.baseline_llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype="float16",
            trust_remote_code=True,
        )
        
        logger.info("Baseline 모델 로드 완료 (vLLM)")
    
    def load_aggllm_model(self):
        """AggLLM 모델 로드 (LoRA 적용)
        
        use_vllm_for_aggllm=True이면 LoRA를 병합한 후 vLLM으로 로드
        use_vllm_for_aggllm=False이면 Transformers로 로드 (기존 방식)
        """
        if self.use_vllm_for_aggllm:
            if self.aggllm_llm is not None:
                return
        else:
            if self.aggllm_model is not None:
                return
        
        if self.aggllm_model_path is None or not os.path.exists(self.aggllm_model_path):
            logger.warning(f"AggLLM 모델 경로가 없습니다: {self.aggllm_model_path}")
            return
        
        logger.info(f"AggLLM 모델 로드 중: {self.model_name} + {self.aggllm_model_path}")
        
        self.aggllm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.aggllm_tokenizer.pad_token is None:
            self.aggllm_tokenizer.pad_token = self.aggllm_tokenizer.eos_token
        
        if self.use_vllm_for_aggllm:
            # LoRA를 base 모델에 병합한 후 vLLM으로 로드
            logger.info("LoRA 가중치를 base 모델에 병합 중...")
            
            # 기반 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 가중치 로드
            peft_model = PeftModel.from_pretrained(base_model, self.aggllm_model_path)
            
            # LoRA 가중치 병합
            merged_model = peft_model.merge_and_unload()
            
            # 병합된 모델 저장 경로 결정
            if self.merged_model_cache_dir:
                merged_model_path = self.merged_model_cache_dir
            else:
                # 임시 디렉토리 사용
                import tempfile
                merged_model_path = tempfile.mkdtemp(prefix="aggllm_merged_")
            
            os.makedirs(merged_model_path, exist_ok=True)
            
            # 병합된 모델이 이미 저장되어 있는지 확인
            config_path = os.path.join(merged_model_path, "config.json")
            if not os.path.exists(config_path):
                logger.info(f"병합된 모델 저장 중: {merged_model_path}")
                merged_model.save_pretrained(merged_model_path, safe_serialization=True)
                self.aggllm_tokenizer.save_pretrained(merged_model_path)
                logger.info("병합된 모델 저장 완료")
            else:
                logger.info(f"기존 병합된 모델 사용: {merged_model_path}")
            
            # 메모리 정리
            del base_model, peft_model, merged_model
            torch.cuda.empty_cache()
            
            # vLLM으로 로드
            logger.info(f"vLLM으로 AggLLM 모델 로드 중... (GPU memory utilization: {self.aggllm_gpu_memory_utilization}, max_model_len: {self.max_model_len})")
            self.aggllm_llm = LLM(
                model=merged_model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.aggllm_gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype="float16",
                trust_remote_code=True,
            )
            logger.info("AggLLM 모델 로드 완료 (vLLM)")
        else:
            # 기존 방식: Transformers로 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 가중치 로드
            self.aggllm_model = PeftModel.from_pretrained(base_model, self.aggllm_model_path)
            self.aggllm_model.eval()
            
            logger.info("AggLLM 모델 로드 완료 (Transformers)")
    
    def generate_solutions_with_vllm(
        self,
        problem_text: str,
        num_solutions: int,
        use_aggllm: bool = False
    ) -> List[Dict[str, Any]]:
        """
        vLLM을 사용하여 주어진 문제에 대해 여러 solution을 생성합니다.
        
        Args:
            problem_text: 문제 텍스트
            num_solutions: 생성할 solution 수
            use_aggllm: AggLLM 모델 사용 여부 (False면 baseline 사용)
        
        Returns:
            solution 리스트 (각 solution은 generated_text, final_answer, confidence_scores 포함)
        """
        # 사용할 모델과 토크나이저 선택
        if use_aggllm:
            llm = self.aggllm_llm
            tokenizer = self.aggllm_tokenizer
        else:
            llm = self.baseline_llm
            tokenizer = self.baseline_tokenizer
        
        # 프롬프트 구성
        prompt = f"{problem_text}\n\n{self.base_instruction}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        # SamplingParams 설정 (config 값 사용)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            logprobs=self.logprobs,
        )
        
        # 배치로 생성 (vLLM은 배치 처리 효율적)
        prompts = [formatted_prompt] * num_solutions
        
        # vLLM으로 생성
        outputs = llm.generate(prompts, sampling_params)
        
        solutions = []
        for output in outputs:
            generated_text = output.outputs[0].text
            
            # logprobs 추출
            logprobs = []
            if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
                logprobs = simple_extract_topk(output.outputs[0].logprobs, self.logprobs)
            
            # 신뢰도 점수 계산
            if logprobs:
                confidence_scores = self.confidence_calculator.calculate_all_confidence_scores(logprobs)
            else:
                # logprobs가 없으면 기본값
                confidence_scores = {
                    "mean_group_confidence": 0.0,
                    "bottom_10_percent_confidence": 0.0,
                    "tail_confidence": 0.0,
                    "lowest_group_confidence": 0.0
                }
            
            # enable_thinking에 따라 파싱
            if self.enable_thinking:
                reasoning_content = extract_reasoning_content(generated_text)
                content = extract_content(generated_text)
                # content가 비어있으면 generated_text 사용
                if not content:
                    content = generated_text
            else:
                reasoning_content = ""
                content = generated_text
            
            # final_answer는 content에서 추출
            final_answer = self.math_verifier.extract_final_answer_from_content(content)
            
            solutions.append({
                "generated_text": generated_text,
                "reasoning_content": reasoning_content,
                "content": content,  # aggregation용 (enable_thinking=True면 </think> 이후)
                "final_answer": final_answer,
                "confidence_scores": confidence_scores
            })
        
        return solutions
    
    def generate_solutions_with_transformers(
        self,
        problem_text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        num_solutions: int
    ) -> List[Dict[str, Any]]:
        """
        Transformers를 사용하여 주어진 문제에 대해 여러 solution을 생성합니다.
        (AggLLM용 - LoRA 지원)
        
        Args:
            problem_text: 문제 텍스트
            model: 사용할 모델
            tokenizer: 토크나이저
            num_solutions: 생성할 solution 수
        
        Returns:
            solution 리스트 (각 solution은 generated_text, final_answer, confidence_scores 포함)
        """
        solutions = []
        
        # 프롬프트 구성
        prompt = f"{problem_text}\n\n{self.base_instruction}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        for i in range(num_solutions):
            # 텍스트 생성
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 생성된 텍스트 추출
            generated_text = tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # logprobs 추출 (outputs.scores에서)
            logprobs = []
            if hasattr(outputs, 'scores') and outputs.scores:
                # scores는 logits이므로 log_softmax를 적용하여 logprobs로 변환
                # scores는 (num_tokens, batch_size, vocab_size) 형태
                for token_scores in outputs.scores:
                    # log_softmax 적용
                    log_probs_tensor = torch.nn.functional.log_softmax(token_scores[0], dim=-1)
                    # 각 토큰의 top-k logprobs 추출
                    top_k_values, top_k_indices = torch.topk(log_probs_tensor, k=self.logprobs, dim=-1)
                    token_logprobs = top_k_values.cpu().numpy().tolist()
                    logprobs.append(token_logprobs)
            
            # 신뢰도 점수 계산
            if logprobs:
                confidence_scores = self.confidence_calculator.calculate_all_confidence_scores(logprobs)
            else:
                # logprobs가 없으면 기본값
                confidence_scores = {
                    "mean_group_confidence": 0.0,
                    "bottom_10_percent_confidence": 0.0,
                    "tail_confidence": 0.0,
                    "lowest_group_confidence": 0.0
                }
            
            # enable_thinking에 따라 파싱
            if self.enable_thinking:
                reasoning_content = extract_reasoning_content(generated_text)
                content = extract_content(generated_text)
                # content가 비어있으면 generated_text 사용
                if not content:
                    content = generated_text
            else:
                reasoning_content = ""
                content = generated_text
            
            # final_answer는 content에서 추출
            final_answer = self.math_verifier.extract_final_answer_from_content(content)
            
            solutions.append({
                "generated_text": generated_text,
                "reasoning_content": reasoning_content,
                "content": content,  # aggregation용 (enable_thinking=True면 </think> 이후)
                "final_answer": final_answer,
                "confidence_scores": confidence_scores
            })
        
        return solutions
    
    def calculate_pass_at_k(
        self,
        solutions: List[Dict[str, Any]],
        ground_truth: str,
        k_values: List[int] = [1, 4, 16]
    ) -> Dict[int, float]:
        """
        Pass@k 메트릭을 계산합니다.
        
        Args:
            solutions: solution 리스트
            ground_truth: 정답
            k_values: 계산할 k 값들
        
        Returns:
            각 k에 대한 Pass@k 점수
        """
        results = {}
        
        for k in k_values:
            if k == 1:
                # Pass@1: 첫 번째 solution
                if solutions:
                    final_answer = solutions[0]["final_answer"]
                    results[k] = 1.0 if self.math_verifier.verify_answer(final_answer, ground_truth) else 0.0
                else:
                    results[k] = 0.0
            elif k == 16:
                # Pass@16: 16개 중 하나라도 정답
                is_correct = False
                for sol in solutions[:k]:
                    if self.math_verifier.verify_answer(sol["final_answer"], ground_truth):
                        is_correct = True
                        break
                results[k] = 1.0 if is_correct else 0.0
            elif k == 4:
                # Pass@4: 16개를 4개 set으로 나눠서 각 set에서 하나라도 정답
                num_sets = 16 // 4
                set_correct = False
                for i in range(num_sets):
                    start_idx = i * 4
                    end_idx = start_idx + 4
                    set_solutions = solutions[start_idx:end_idx]
                    
                    # 이 set에서 하나라도 정답인지 확인
                    for sol in set_solutions:
                        if self.math_verifier.verify_answer(sol["final_answer"], ground_truth):
                            set_correct = True
                            break
                    if set_correct:
                        break
                results[k] = 1.0 if set_correct else 0.0
        
        return results
    
    def majority_voting(
        self,
        solutions: List[Dict[str, Any]],
        samples_per_set: int
    ) -> List[str]:
        """
        Majority Voting을 수행합니다.
        
        Args:
            solutions: solution 리스트 (16개)
            samples_per_set: 각 set의 sample 수 (2, 4, 8, 16)
        
        Returns:
            각 set의 majority voting 결과 리스트
        """
        num_sets = 16 // samples_per_set
        results = []
        
        for i in range(num_sets):
            start_idx = i * samples_per_set
            end_idx = start_idx + samples_per_set
            set_solutions = solutions[start_idx:end_idx]
            
            # 각 sample의 final_answer 추출
            answers = [sol["final_answer"] for sol in set_solutions]
            
            # 가장 많이 나온 답안 선택
            answer_counts = defaultdict(int)
            for answer in answers:
                answer_counts[answer] += 1
            
            if answer_counts:
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                results.append(majority_answer)
            else:
                results.append("")
        
        return results
    
    def confidence_weighted_voting(
        self,
        solutions: List[Dict[str, Any]],
        samples_per_set: int,
        confidence_metric: str
    ) -> List[str]:
        """
        Confidence Weighted Voting을 수행합니다.
        
        Args:
            solutions: solution 리스트 (16개)
            samples_per_set: 각 set의 sample 수 (2, 4, 8, 16)
            confidence_metric: 사용할 신뢰도 메트릭
        
        Returns:
            각 set의 confidence weighted voting 결과 리스트
        """
        num_sets = 16 // samples_per_set
        results = []
        
        for i in range(num_sets):
            start_idx = i * samples_per_set
            end_idx = start_idx + samples_per_set
            set_solutions = solutions[start_idx:end_idx]
            
            # 각 sample의 final_answer와 confidence 추출
            weighted_votes = defaultdict(float)
            for sol in set_solutions:
                answer = sol["final_answer"]
                conf = sol["confidence_scores"].get(confidence_metric, 0.0)
                weighted_votes[answer] += conf
            
            if weighted_votes:
                best_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
                results.append(best_answer)
            else:
                results.append("")
        
        return results
    
    def format_solutions_for_aggregation(
        self,
        solutions: List[Dict[str, Any]]
    ) -> str:
        """
        Aggregation 프롬프트를 위한 solution 텍스트를 생성합니다.
        tail_confidence를 사용하고, content만 사용합니다 (reasoning_content 제외).
        
        Args:
            solutions: solution 리스트
        
        Returns:
            포맷된 solution 텍스트
        """
        lines = []
        for idx, sol in enumerate(solutions, start=1):
            conf_value = sol["confidence_scores"].get("tail_confidence", None)
            conf_str = f"{conf_value:.4f}" if conf_value is not None else "N/A"
            # content만 사용 (reasoning_content는 제외)
            solution_content = sol.get("content", "")
            lines.append(
                f"solution{idx}:\n"
                f"{solution_content}\n"
                f"final_answer: {sol['final_answer']}\n"
                f"confidence: {conf_str}\n"
            )
        return "\n".join(lines)
    
    def prompt_aggregation_with_vllm(
        self,
        problem_text: str,
        solutions: List[Dict[str, Any]],
        use_aggllm: bool = False
    ) -> str:
        """
        vLLM을 사용하여 Prompt Aggregation을 수행합니다.
        
        Args:
            problem_text: 문제 텍스트
            solutions: solution 리스트 (4개)
            use_aggllm: AggLLM 모델 사용 여부 (False면 baseline 사용)
        
        Returns:
            Aggregation 결과 텍스트
        """
        # 사용할 모델과 토크나이저 선택
        if use_aggllm:
            llm = self.aggllm_llm
            tokenizer = self.aggllm_tokenizer
        else:
            llm = self.baseline_llm
            tokenizer = self.baseline_tokenizer
        
        # Solution 텍스트 포맷팅
        solutions_text = self.format_solutions_for_aggregation(solutions)
        
        # 프롬프트 구성
        prompt = self.aggregation_prompt_template.format(
            problem=problem_text,
            solutions=solutions_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        # SamplingParams 설정 (config 값 사용)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )
        
        # vLLM으로 생성
        outputs = llm.generate([formatted_prompt], sampling_params)
        
        aggregated_text = outputs[0].outputs[0].text
        
        return aggregated_text
    
    def prompt_aggregation(
        self,
        problem_text: str,
        solutions: List[Dict[str, Any]],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer
    ) -> str:
        """
        Transformers를 사용하여 Prompt Aggregation을 수행합니다.
        (AggLLM용)
        
        Args:
            problem_text: 문제 텍스트
            solutions: solution 리스트 (4개)
            model: 사용할 모델 (AggLLM)
            tokenizer: 토크나이저
        
        Returns:
            Aggregation 결과 텍스트
        """
        # Solution 텍스트 포맷팅
        solutions_text = self.format_solutions_for_aggregation(solutions)
        
        # 프롬프트 구성
        prompt = self.aggregation_prompt_template.format(
            problem=problem_text,
            solutions=solutions_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        # 생성
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=16384
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        aggregated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return aggregated_text
    
    def evaluate_single_problem(
        self,
        problem_text: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        단일 문제에 대해 모든 평가를 수행합니다.
        
        Args:
            problem_text: 문제 텍스트
            ground_truth: 정답
        
        Returns:
            평가 결과 딕셔너리
        """
        results = {}
        
        # Baseline 모델로 solution 생성 (vLLM 사용)
        logger.info("Baseline 모델로 solution 생성 중 (vLLM)...")
        baseline_solutions = self.generate_solutions_with_vllm(
            problem_text,
            self.num_solutions
        )
        
        # AggLLM 모델로 solution 생성
        if self.use_vllm_for_aggllm and self.aggllm_llm is not None:
            logger.info("AggLLM 모델로 solution 생성 중 (vLLM)...")
            aggllm_solutions = self.generate_solutions_with_vllm(
                problem_text,
                self.num_solutions,
                use_aggllm=True
            )
        elif not self.use_vllm_for_aggllm and self.aggllm_model is not None:
            logger.info("AggLLM 모델로 solution 생성 중 (Transformers)...")
            aggllm_solutions = self.generate_solutions_with_transformers(
                problem_text,
                self.aggllm_model,
                self.aggllm_tokenizer,
                self.num_solutions
            )
        else:
            aggllm_solutions = []
        
        # Baseline 평가
        baseline_results = self._evaluate_solutions(
            baseline_solutions,
            ground_truth,
            problem_text,
            "baseline"
        )
        results["baseline"] = baseline_results
        # Baseline 생성 결과 저장
        results["baseline_generated_solutions"] = baseline_solutions
        
        # AggLLM 평가
        if aggllm_solutions:
            aggllm_results = self._evaluate_solutions(
                aggllm_solutions,
                ground_truth,
                problem_text,
                "aggllm"
            )
            results["aggllm"] = aggllm_results
            # AggLLM 생성 결과 저장
            results["aggllm_generated_solutions"] = aggllm_solutions
        
        # Aggregation 평가
        # Baseline → AggLLM Aggregation
        if ((self.use_vllm_for_aggllm and self.aggllm_llm is not None) or 
            (not self.use_vllm_for_aggllm and self.aggllm_model is not None)) and len(baseline_solutions) >= 4:
            baseline_4_solutions = baseline_solutions[:4]
            if self.use_vllm_for_aggllm:
                agg_text = self.prompt_aggregation_with_vllm(
                    problem_text,
                    baseline_4_solutions,
                    use_aggllm=True
                )
            else:
                agg_text = self.prompt_aggregation(
                    problem_text,
                    baseline_4_solutions,
                    self.aggllm_model,
                    self.aggllm_tokenizer
                )
            agg_answer = self.math_verifier.extract_final_answer_from_content(agg_text)
            is_correct = self.math_verifier.verify_answer(agg_answer, ground_truth)
            results["baseline_to_aggllm_aggregation"] = {
                "is_correct": is_correct,
                "answer": agg_answer,
                "text": agg_text
            }
        
        # AggLLM → AggLLM Aggregation
        if ((self.use_vllm_for_aggllm and self.aggllm_llm is not None) or 
            (not self.use_vllm_for_aggllm and self.aggllm_model is not None)) and len(aggllm_solutions) >= 4:
            aggllm_4_solutions = aggllm_solutions[:4]
            if self.use_vllm_for_aggllm:
                agg_text = self.prompt_aggregation_with_vllm(
                    problem_text,
                    aggllm_4_solutions,
                    use_aggllm=True
                )
            else:
                agg_text = self.prompt_aggregation(
                    problem_text,
                    aggllm_4_solutions,
                    self.aggllm_model,
                    self.aggllm_tokenizer
                )
            agg_answer = self.math_verifier.extract_final_answer_from_content(agg_text)
            is_correct = self.math_verifier.verify_answer(agg_answer, ground_truth)
            results["aggllm_to_aggllm_aggregation"] = {
                "is_correct": is_correct,
                "answer": agg_answer,
                "text": agg_text
            }
        
        return results
    
    def _evaluate_solutions(
        self,
        solutions: List[Dict[str, Any]],
        ground_truth: str,
        problem_text: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Solution 리스트에 대해 모든 메트릭을 계산합니다.
        
        Args:
            solutions: solution 리스트
            ground_truth: 정답
            problem_text: 문제 텍스트
            model_name: 모델 이름
        
        Returns:
            평가 결과 딕셔너리
        """
        results = {}
        
        # Pass@k 계산
        pass_at_k = self.calculate_pass_at_k(solutions, ground_truth)
        results["pass_at_k"] = pass_at_k
        
        # Majority Voting
        majority_results = {}
        for samples_per_set in [2, 4, 8, 16]:
            voting_results = self.majority_voting(solutions, samples_per_set)
            # 각 set의 결과를 평가
            correct_count = 0
            for answer in voting_results:
                if self.math_verifier.verify_answer(answer, ground_truth):
                    correct_count += 1
            majority_results[f"{samples_per_set}_samples"] = {
                "correct_count": correct_count,
                "total_sets": len(voting_results),
                "accuracy": correct_count / len(voting_results) if voting_results else 0.0
            }
        results["majority_voting"] = majority_results
        
        # Confidence Weighted Voting
        confidence_metrics = [
            "bottom_10_percent_confidence",
            "tail_confidence",
            "mean_group_confidence",
            "lowest_group_confidence"
        ]
        
        confidence_results = {}
        for metric in confidence_metrics:
            metric_results = {}
            for samples_per_set in [2, 4, 8, 16]:
                voting_results = self.confidence_weighted_voting(
                    solutions,
                    samples_per_set,
                    metric
                )
                # 각 set의 결과를 평가
                correct_count = 0
                for answer in voting_results:
                    if self.math_verifier.verify_answer(answer, ground_truth):
                        correct_count += 1
                metric_results[f"{samples_per_set}_samples"] = {
                    "correct_count": correct_count,
                    "total_sets": len(voting_results),
                    "accuracy": correct_count / len(voting_results) if voting_results else 0.0
                }
            confidence_results[metric] = metric_results
        results["confidence_weighted_voting"] = confidence_results
        
        # Prompt Aggregation (non-trained baseline)
        if len(solutions) >= 16:
            # 16개를 4개 set으로 나누기
            aggregation_results = []
            for i in range(4):
                set_solutions = solutions[i*4:(i+1)*4]
                # Prompt Aggregation은 vLLM 사용 (더 빠름)
                agg_text = self.prompt_aggregation_with_vllm(
                    problem_text,
                    set_solutions
                )
                agg_answer = self.math_verifier.extract_final_answer_from_content(agg_text)
                is_correct = self.math_verifier.verify_answer(agg_answer, ground_truth)
                aggregation_results.append({
                    "is_correct": is_correct,
                    "answer": agg_answer
                })
            
            correct_count = sum(1 for r in aggregation_results if r["is_correct"])
            results["prompt_aggregation"] = {
                "correct_count": correct_count,
                "total_sets": len(aggregation_results),
                "accuracy": correct_count / len(aggregation_results) if aggregation_results else 0.0
            }
        
        return results
    
    def load_benchmark_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        HuggingFace에서 벤치마크 데이터셋을 로드합니다.
        
        Args:
            dataset_name: 데이터셋 이름 (예: "math-ai/aime24")
        
        Returns:
            데이터 리스트
        """
        logger.info(f"벤치마크 데이터셋 로드 중: {dataset_name}")
        
        try:
            # aime는 test, hmmt는 train으로 split을 다르게 로드
            if "aime" in dataset_name.lower():
                dataset = load_dataset(dataset_name, split="test")
            elif "hmmt" in dataset_name.lower():
                dataset = load_dataset(dataset_name, split="train")
            else:
                dataset = load_dataset(dataset_name, split="test")
            data = []
            
            for item in dataset:
                # 데이터셋 구조에 따라 필드명 조정
                problem_text = item.get("problem", item.get("question", ""))
                ground_truth = item.get("answer", item.get("solution", ""))
                # INSERT_YOUR_CODE
                # aime24의 answer가 \boxed{xxx} 형태라면 중괄호 안 값만 저장
                if "aime24" in dataset_name.lower() and isinstance(ground_truth, str):
                    import re
                    match = re.search(r"\\boxed\{([^{}]+)\}", ground_truth)
                    if match:
                        ground_truth = match.group(1).strip()
                data.append({
                    "problem": problem_text,
                    "answer": ground_truth
                })
            
            logger.info(f"데이터셋 로드 완료: {len(data)}개 문제")
            return data
            
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            return []
    
    def evaluate_benchmark(
        self,
        dataset_name: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        벤치마크 데이터셋에 대해 평가를 수행합니다.
        
        Args:
            dataset_name: 데이터셋 이름
            output_dir: 결과 저장 디렉토리
        
        Returns:
            평가 결과 딕셔너리
        """
        # 모델 로드
        self.load_baseline_model()
        if self.aggllm_model_path:
            self.load_aggllm_model()
        
        # 데이터셋 로드
        data = self.load_benchmark_dataset(dataset_name)
        
        if not data:
            logger.error(f"데이터셋을 로드할 수 없습니다: {dataset_name}")
            return {}
        
        # 평가 결과 저장
        all_results = []
        baseline_metrics = defaultdict(list)
        aggllm_metrics = defaultdict(list)
        
        for i, item in enumerate(data):
            problem_text = item["problem"]
            ground_truth = item["answer"]
            
            logger.info(f"문제 {i+1}/{len(data)} 평가 중...")
            
            problem_results = self.evaluate_single_problem(problem_text, ground_truth)
            all_results.append(problem_results)
            
            # 메트릭 누적
            if "baseline" in problem_results:
                baseline_result = problem_results["baseline"]
                # Pass@k 누적
                for k, v in baseline_result.get("pass_at_k", {}).items():
                    baseline_metrics[f"pass_at_{k}"].append(v)
                # Majority Voting 누적
                for key, value in baseline_result.get("majority_voting", {}).items():
                    baseline_metrics[f"majority_voting_{key}"].append(value["accuracy"])
                # Confidence Weighted Voting 누적
                for metric, metric_results in baseline_result.get("confidence_weighted_voting", {}).items():
                    for key, value in metric_results.items():
                        baseline_metrics[f"confidence_weighted_{metric}_{key}"].append(value["accuracy"])
                # Prompt Aggregation 누적
                if "prompt_aggregation" in baseline_result:
                    baseline_metrics["prompt_aggregation"].append(
                        baseline_result["prompt_aggregation"]["accuracy"]
                    )
            
            if "aggllm" in problem_results:
                aggllm_result = problem_results["aggllm"]
                # Pass@k 누적
                for k, v in aggllm_result.get("pass_at_k", {}).items():
                    aggllm_metrics[f"pass_at_{k}"].append(v)
                # Majority Voting 누적
                for key, value in aggllm_result.get("majority_voting", {}).items():
                    aggllm_metrics[f"majority_voting_{key}"].append(value["accuracy"])
                # Confidence Weighted Voting 누적
                for metric, metric_results in aggllm_result.get("confidence_weighted_voting", {}).items():
                    for key, value in metric_results.items():
                        aggllm_metrics[f"confidence_weighted_{metric}_{key}"].append(value["accuracy"])
                # Prompt Aggregation 누적
                if "prompt_aggregation" in aggllm_result:
                    aggllm_metrics["prompt_aggregation"].append(
                        aggllm_result["prompt_aggregation"]["accuracy"]
                    )
        
        # 최종 메트릭 계산 (평균)
        final_results = {
            "dataset_name": dataset_name,
            "total_problems": len(data),
            "baseline": {},
            "aggllm": {}
        }
        
        for key, values in baseline_metrics.items():
            final_results["baseline"][key] = np.mean(values) if values else 0.0
        
        for key, values in aggllm_metrics.items():
            final_results["aggllm"][key] = np.mean(values) if values else 0.0
        
        # Aggregation 결과 계산
        baseline_to_aggllm_correct = sum(
            1 for r in all_results
            if r.get("baseline_to_aggllm_aggregation", {}).get("is_correct", False)
        )
        aggllm_to_aggllm_correct = sum(
            1 for r in all_results
            if r.get("aggllm_to_aggllm_aggregation", {}).get("is_correct", False)
        )
        
        final_results["baseline_to_aggllm_aggregation"] = {
            "correct": baseline_to_aggllm_correct,
            "total": len(data),
            "accuracy": baseline_to_aggllm_correct / len(data) if data else 0.0
        }
        
        final_results["aggllm_to_aggllm_aggregation"] = {
            "correct": aggllm_to_aggllm_correct,
            "total": len(data),
            "accuracy": aggllm_to_aggllm_correct / len(data) if data else 0.0
        }
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        dataset_safe_name = dataset_name.replace('/', '_')
        result_path = os.path.join(output_dir, f"{dataset_safe_name}_results.json")
        
        # 최종 결과에 상세 생성 결과 추가 (선택적)
        final_results["detailed_results"] = all_results
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과 저장: {result_path}")
        
        # 각 모델의 생성 결과를 별도로 저장
        # Baseline 생성 결과 추출 및 저장
        baseline_generated = []
        aggllm_generated = []
        for i, result in enumerate(all_results):
            problem_data = {
                "problem_id": i,
                "problem_text": data[i]["problem"],
                "ground_truth": data[i]["answer"]
            }
            
            if "baseline_generated_solutions" in result:
                baseline_generated.append({
                    **problem_data,
                    "solutions": result["baseline_generated_solutions"]
                })
            
            if "aggllm_generated_solutions" in result:
                aggllm_generated.append({
                    **problem_data,
                    "solutions": result["aggllm_generated_solutions"]
                })
        
        # Baseline 생성 결과 저장
        if baseline_generated:
            baseline_output_path = os.path.join(output_dir, f"{dataset_safe_name}_baseline_generated.json")
            with open(baseline_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "total_problems": len(baseline_generated),
                    "generated_solutions": baseline_generated
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Baseline 생성 결과 저장: {baseline_output_path}")
        
        # AggLLM 생성 결과 저장
        if aggllm_generated:
            aggllm_output_path = os.path.join(output_dir, f"{dataset_safe_name}_aggllm_generated.json")
            with open(aggllm_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "total_problems": len(aggllm_generated),
                    "generated_solutions": aggllm_generated
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"AggLLM 생성 결과 저장: {aggllm_output_path}")
        
        # 상세 결과 저장 (모든 평가 결과 포함)
        save_detailed = True  # 필요시 config로 제어 가능
        if save_detailed:
            detailed_path = os.path.join(output_dir, f"{dataset_safe_name}_detailed_results.json")
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"상세 결과 저장: {detailed_path}")
        
        return final_results

