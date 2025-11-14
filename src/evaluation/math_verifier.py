"""
math_verify 라이브러리 래퍼
"""
import math_verify
import logging
from typing import List, Tuple, Optional
import time
import re

logger = logging.getLogger(__name__)


class MathVerifier:
    """math_verify 라이브러리 래퍼 클래스"""
    
    def __init__(self, timeout: int = 30):
        """
        Args:
            timeout: 검증 타임아웃 (초)
        """
        self.timeout = timeout
    
    def verify_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        math_verify를 사용하여 답안을 검증합니다.
        
        Args:
            predicted: 예측된 답안
            ground_truth: 정답
        
        Returns:
            검증 결과 (True/False)
        """
        try:
            # math_verify를 사용한 검증
            result = math_verify.verify(predicted, ground_truth)
            return result
        except Exception as e:
            logger.warning(f"math_verify 검증 실패: {e}")
            # 폴백: 문자열 비교
            return predicted.strip().lower() == ground_truth.strip().lower()
    
    def verify_batch(
        self, 
        predictions: List[str], 
        ground_truths: List[str]
    ) -> List[bool]:
        """
        배치 단위로 답안을 검증합니다.
        
        Args:
            predictions: 예측된 답안 리스트
            ground_truths: 정답 리스트
        
        Returns:
            검증 결과 리스트
        """
        results = []
        for pred, gt in zip(predictions, ground_truths):
            results.append(self.verify_answer(pred, gt))
        return results
    
    def extract_final_answer_from_content(self, content: str) -> str:
        """content에서 최종 답안 추출"""
        if not content:
            return ""
        
        content_str = str(content).strip()
        
        # \boxed{} 찾기
        boxed_matches = list(re.finditer(r'\\boxed\{', content_str))
        if boxed_matches:
            last_start = boxed_matches[-1].end()
            brace_count = 1
            end_pos = last_start
            
            while end_pos < len(content_str) and brace_count > 0:
                if content_str[end_pos] == '{' and (end_pos == 0 or content_str[end_pos-1] != '\\'):
                    brace_count += 1
                elif content_str[end_pos] == '}' and (end_pos == 0 or content_str[end_pos-1] != '\\'):
                    brace_count -= 1
                end_pos += 1
            
            if brace_count == 0:
                return content_str[last_start:end_pos-1].strip()
        
        return ""

