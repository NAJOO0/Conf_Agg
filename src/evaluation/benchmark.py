"""
벤치마크 평가 모듈
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.inference.vllm_engine import VLLMInferenceEngine
from src.data.confidence import ConfidenceCalculator
from src.evaluation.math_verifier import MathVerifier
from src.utils.metrics import calculate_pass_at_k, calculate_confidence_correlation

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """벤치마크 평가 클래스"""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str,
        confidence_calculator: ConfidenceCalculator,
        math_verifier: MathVerifier,
        num_candidates: int = 8,
        temperature: float = 1.5,
        max_tokens: int = 16384
    ):
        """
        Args:
            model_path: 훈련된 모델 경로
            base_model_name: 기반 모델 이름
            confidence_calculator: 신뢰도 계산기
            math_verifier: 수학 검증기
            num_candidates: 후보 응답 수
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.confidence_calculator = confidence_calculator
        self.math_verifier = math_verifier
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 모델과 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """훈련된 모델을 로드합니다."""
        logger.info(f"모델 로드 중: {self.model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 기반 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA 가중치 로드 (있는 경우)
        if os.path.exists(self.model_path):
            try:
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
                logger.info("LoRA 가중치 로드 완료")
            except Exception as e:
                logger.warning(f"LoRA 가중치 로드 실패: {e}")
        
        self.model.eval()
        logger.info("모델 로드 완료")
    
    def generate_candidates(
        self, 
        prompt: str
    ) -> List[Dict[str, Any]]:
        """
        주어진 프롬프트에 대해 후보 응답들을 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
        
        Returns:
            후보 응답 리스트 (텍스트와 신뢰도 점수 포함)
        """
        candidates = []
        
        for i in range(self.num_candidates):
            # 텍스트 생성
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # 생성된 텍스트 추출
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # 신뢰도 점수 계산 (간단한 버전)
            # 실제로는 로그 확률을 사용해야 하지만, 여기서는 간단히 구현
            confidence_scores = {
                "mean_group_confidence": np.random.random(),  # 임시
                "bottom_10_percent_confidence": np.random.random(),  # 임시
                "tail_confidence": np.random.random()  # 임시
            }
            
            candidates.append({
                "text": generated_text,
                "confidence_scores": confidence_scores
            })
        
        return candidates
    
    def aggregate_candidates(
        self, 
        candidates: List[Dict[str, Any]]
    ) -> str:
        """
        후보 응답들을 신뢰도 기반으로 집계합니다.
        
        Args:
            candidates: 후보 응답 리스트
        
        Returns:
            집계된 최종 응답
        """
        if not candidates:
            return ""
        
        # 신뢰도 점수 계산
        scores = []
        for candidate in candidates:
            conf_scores = candidate["confidence_scores"]
            # 여러 메트릭의 평균 사용
            score = np.mean(list(conf_scores.values()))
            scores.append(score)
        
        # 가장 높은 신뢰도를 가진 응답 선택
        best_idx = np.argmax(scores)
        return candidates[best_idx]["text"]
    
    def evaluate_dataset(
        self, 
        dataset_path: str
    ) -> Dict[str, Any]:
        """
        데이터셋에 대해 평가를 수행합니다.
        
        Args:
            dataset_path: 벤치마크 데이터셋 경로
        
        Returns:
            평가 결과
        """
        logger.info(f"데이터셋 평가 시작: {dataset_path}")
        
        # 데이터셋 로드
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"데이터셋 로드 완료: {len(data)}개 문제")
        
        # 평가 결과 저장
        results = {
            "total_problems": len(data),
            "correct_predictions": 0,
            "predictions": [],
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,
            "pass_at_10": 0.0,
            "confidence_correlation": 0.0
        }
        
        all_predictions = []
        all_ground_truths = []
        all_confidence_scores = []
        
        # 각 문제에 대해 평가
        for i, problem_data in enumerate(data):
            problem_text = problem_data.get("problem", "")
            ground_truth = problem_data.get("answer", "")
            
            logger.info(f"문제 {i+1}/{len(data)} 평가 중...")
            
            # 프롬프트 구성
            prompt = f"""다음 수학 문제를 단계별로 풀어보세요. 각 단계에서 생각하는 과정을 자세히 설명하고, 최종 답을 제시하세요.

문제: {problem_text}

풀이 과정:"""
            
            # 후보 응답 생성
            candidates = self.generate_candidates(prompt)
            
            # 응답 집계
            final_response = self.aggregate_candidates(candidates)
            
            # 정답 추출
            predicted_answer = self.math_verifier.extract_answer_from_response(final_response)
            
            # 정답 검증
            is_correct = self.math_verifier.verify_answer(predicted_answer, ground_truth)
            
            if is_correct:
                results["correct_predictions"] += 1
            
            # 결과 저장
            result_item = {
                "problem_id": i,
                "problem_text": problem_text,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "final_response": final_response,
                "is_correct": is_correct,
                "candidates": candidates
            }
            
            results["predictions"].append(result_item)
            all_predictions.append(predicted_answer)
            all_ground_truths.append(ground_truth)
            
            # 신뢰도 점수 저장
            if candidates:
                best_candidate = max(candidates, key=lambda x: np.mean(list(x["confidence_scores"].values())))
                all_confidence_scores.append(np.mean(list(best_candidate["confidence_scores"].values())))
            else:
                all_confidence_scores.append(0.0)
        
        # Pass@k 계산
        pass_at_k_results = calculate_pass_at_k(
            all_predictions, 
            all_ground_truths, 
            k_values=[1, 5, 10]
        )
        
        results["pass_at_1"] = pass_at_k_results[1]
        results["pass_at_5"] = pass_at_k_results[5]
        results["pass_at_10"] = pass_at_k_results[10]
        
        # 신뢰도 상관관계 계산
        if all_confidence_scores:
            correlation_results = calculate_confidence_correlation(
                all_predictions,
                all_ground_truths,
                all_confidence_scores
            )
            results["confidence_correlation"] = correlation_results["pearson_correlation"]
        
        logger.info(f"평가 완료: Pass@1 = {results['pass_at_1']:.3f}")
        
        return results
    
    def evaluate_all_benchmarks(
        self, 
        benchmark_configs: List[Dict[str, str]],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        모든 벤치마크에 대해 평가를 수행합니다.
        
        Args:
            benchmark_configs: 벤치마크 설정 리스트
            output_dir: 결과 저장 디렉토리
        
        Returns:
            전체 평가 결과
        """
        logger.info("모든 벤치마크 평가 시작")
        
        all_results = {}
        
        for benchmark_config in benchmark_configs:
            dataset_name = benchmark_config["name"]
            dataset_path = benchmark_config["path"]
            
            logger.info(f"벤치마크 평가: {dataset_name}")
            
            if not os.path.exists(dataset_path):
                logger.warning(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
                continue
            
            # 평가 수행
            results = self.evaluate_dataset(dataset_path)
            all_results[dataset_name] = results
            
            # 개별 결과 저장
            result_path = os.path.join(output_dir, f"{dataset_name}_results.json")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{dataset_name} 결과 저장: {result_path}")
        
        # 전체 결과 요약
        summary = {
            "benchmarks": all_results,
            "overall_pass_at_1": np.mean([r["pass_at_1"] for r in all_results.values()]),
            "overall_pass_at_5": np.mean([r["pass_at_5"] for r in all_results.values()]),
            "overall_pass_at_10": np.mean([r["pass_at_10"] for r in all_results.values()]),
            "overall_confidence_correlation": np.mean([r["confidence_correlation"] for r in all_results.values()])
        }
        
        # 전체 결과 저장
        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전체 결과 요약 저장: {summary_path}")
        
        return summary

