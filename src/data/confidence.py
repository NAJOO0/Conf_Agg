"""
신뢰도 점수 계산 모듈
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """신뢰도 점수 계산 클래스"""
    
    def __init__(self, group_size: int = 10):
        """
        Args:
            group_size: 토큰 그룹 크기
        """
        self.group_size = group_size
    
    def calculate_mean_group_confidence(
        self, 
        logprobs: List[List[float]]
    ) -> float:
        """
        전체 추론 경로의 평균 그룹 신뢰도를 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트
        
        Returns:
            평균 그룹 신뢰도 점수
        """
        if logprobs is None or len(logprobs) == 0:
            return 0.0
        
        # 각 토큰의 최대 로그 확률 (신뢰도)
        token_confidences = [max(token_logprobs) for token_logprobs in logprobs]
        
        # 그룹별 평균 신뢰도 계산
        group_confidences = []
        for i in range(0, len(token_confidences), self.group_size):
            group = token_confidences[i:i + self.group_size]
            group_confidences.append(np.mean(group))
        
        return np.mean(group_confidences) if group_confidences else 0.0
    
    def calculate_bottom_10_percent_confidence(
        self, 
        logprobs: List[List[float]]
    ) -> float:
        """
        신뢰도가 가장 낮았던 하위 10% 그룹들의 평균을 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트
        
        Returns:
            하위 10% 그룹들의 평균 신뢰도
        """
        if logprobs is None or len(logprobs) == 0:
            return 0.0
        
        # 각 토큰의 최대 로그 확률
        token_confidences = [max(token_logprobs) for token_logprobs in logprobs]
        
        # 그룹별 평균 신뢰도 계산
        group_confidences = []
        for i in range(0, len(token_confidences), self.group_size):
            group = token_confidences[i:i + self.group_size]
            group_confidences.append(np.mean(group))
        
        if len(group_confidences) == 0:
            return 0.0
        
        # 하위 10% 그룹 선택
        num_bottom_groups = max(1, int(len(group_confidences) * 0.1))
        bottom_groups = sorted(group_confidences)[:num_bottom_groups]
        
        return np.mean(bottom_groups)
    
    def calculate_tail_confidence(
        self, 
        logprobs: List[List[float]]
    ) -> float:
        """
        마지막 1개 그룹의 신뢰도 점수를 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트
        
        Returns:
            마지막 그룹의 신뢰도 점수
        """
        if logprobs is None or len(logprobs) == 0:
            return 0.0
        
        # 각 토큰의 최대 로그 확률
        token_confidences = [max(token_logprobs) for token_logprobs in logprobs]
        
        # 마지막 그룹의 신뢰도 계산
        if len(token_confidences) < self.group_size:
            return np.mean(token_confidences)
        
        last_group = token_confidences[-self.group_size:]
        return np.mean(last_group)
    
    def calculate_all_confidence_scores(
        self, 
        logprobs: List[List[float]]
    ) -> Dict[str, float]:
        """
        모든 신뢰도 점수를 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트
        
        Returns:
            신뢰도 점수 딕셔너리
        """
        return {
            "mean_group_confidence": self.calculate_mean_group_confidence(logprobs),
            "bottom_10_percent_confidence": self.calculate_bottom_10_percent_confidence(logprobs),
            "tail_confidence": self.calculate_tail_confidence(logprobs)
        }
    
    def batch_calculate_confidence(
        self, 
        batch_logprobs: List[List[List[float]]]
    ) -> List[Dict[str, float]]:
        """
        배치 단위로 신뢰도 점수를 계산합니다.
        
        Args:
            batch_logprobs: 배치별 토큰 로그 확률 리스트
        
        Returns:
            배치별 신뢰도 점수 리스트
        """
        results = []
        for logprobs in batch_logprobs:
            results.append(self.calculate_all_confidence_scores(logprobs))
        
        return results

