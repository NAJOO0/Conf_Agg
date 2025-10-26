"""
math_verify 라이브러리 래퍼
"""
import math_verify
import logging
from typing import List, Tuple, Optional
import time

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
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        응답에서 최종 답안을 추출합니다.
        
        Args:
            response: 모델 응답
        
        Returns:
            추출된 답안
        """
        # 간단한 패턴 매칭으로 답안 추출
        lines = response.strip().split('\n')
        
        # "답:", "정답:", "최종 답:" 등의 패턴 찾기
        for line in reversed(lines):  # 마지막부터 검색
            line = line.strip().lower()
            if any(keyword in line for keyword in ['답:', '정답:', '최종 답:', 'answer:', 'final answer:']):
                # 콜론 뒤의 내용 추출
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # 패턴을 찾지 못한 경우 마지막 줄 반환
        if lines:
            return lines[-1].strip()
        
        return response.strip()

