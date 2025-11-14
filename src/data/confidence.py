"""
신뢰도 점수 계산 모듈
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
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
    
    def _calculate_group_confidences(self, token_confidences: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        슬라이딩 윈도우 방식으로 모든 연속 그룹의 신뢰도를 계산합니다.
        
        예: n=20, size=10일 때
        - [0:10], [1:11], [2:12], ..., [10:20] 총 11개 그룹 생성
        
        Args:
            token_confidences: 토큰별 신뢰도 리스트
            
        Returns:
            그룹별 평균 신뢰도 numpy 배열
        """
        if len(token_confidences) == 0:
            return np.array([])
        
        token_confidences = np.asarray(token_confidences, dtype=np.float32)
        
        # numpy의 convolve를 사용한 슬라이딩 윈도우 평균 계산 (훨씬 빠름)
        # kernel은 모두 1로 이루어진 group_size 길이의 배열
        kernel = np.ones(self.group_size, dtype=np.float32) / self.group_size
        group_confidences = np.convolve(token_confidences, kernel, mode='valid')
        
        return group_confidences
    
    def calculate_mean_group_confidence(
        self, 
        logprobs: Optional[List[List[float]]] = None,
        group_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        전체 추론 경로의 평균 그룹 신뢰도를 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (group_confidences가 None일 때만 사용)
            group_confidences: 미리 계산된 그룹 신뢰도 리스트 (최적화용)
            
        Returns:
            평균 그룹 신뢰도 점수
        """
        if group_confidences is not None:
            if len(group_confidences) == 0:
                return 0.0
            group_confidences = np.asarray(group_confidences)
            return float(np.mean(group_confidences))
        
        if logprobs is None or len(logprobs) == 0:
            return 0.0
        
        # numpy 배열로 변환 후 벡터화 연산으로 토큰 신뢰도 계산
        logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
        token_confidences = -logprobs_array
        
        # 슬라이딩 윈도우로 모든 그룹 생성
        group_confidences = self._calculate_group_confidences(token_confidences)
        
        return float(np.mean(group_confidences)) if len(group_confidences) > 0 else 0.0
    
    def calculate_bottom_10_percent_confidence(
        self, 
        logprobs: Optional[List[List[float]]] = None,
        group_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        신뢰도가 가장 낮았던 하위 10% 그룹들의 평균을 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (group_confidences가 None일 때만 사용)
            group_confidences: 미리 계산된 그룹 신뢰도 리스트 (최적화용)
            
        Returns:
            하위 10% 그룹들의 평균 신뢰도
        """
        if group_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            
            # numpy 배열로 변환 후 벡터화 연산으로 토큰 신뢰도 계산
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
            
            # 슬라이딩 윈도우로 모든 그룹 생성
            group_confidences = self._calculate_group_confidences(token_confidences)
        
        if len(group_confidences) == 0:
            return 0.0
        
        # numpy 배열로 변환
        group_confidences = np.asarray(group_confidences)
        
        # 하위 10% 그룹 선택: 전체 정렬 대신 partition 사용 (훨씬 빠름)
        num_bottom_groups = max(1, int(np.ceil(len(group_confidences) * 0.1)))
        # partition을 사용하여 하위 num_bottom_groups개만 정렬
        partitioned = np.partition(group_confidences, num_bottom_groups - 1)
        bottom_groups = partitioned[:num_bottom_groups]
        
        return float(np.mean(bottom_groups))
    
    def calculate_lowest_group_confidence(
        self,
        logprobs: Optional[List[List[float]]] = None,
        group_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        모든 그룹 중 최저 그룹 신뢰도를 반환합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (group_confidences가 None일 때만 사용)
            group_confidences: 미리 계산된 그룹 신뢰도 리스트 (최적화용)
        """
        if group_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
            if len(token_confidences) == 0:
                return 0.0
            
            # 슬라이딩 윈도우로 모든 그룹 생성
            group_confidences = self._calculate_group_confidences(token_confidences)
        
        if len(group_confidences) == 0:
            return 0.0
        
        group_confidences = np.asarray(group_confidences)
        return float(np.min(group_confidences))

    def calculate_top_10_percent_confidence(
        self,
        logprobs: Optional[List[List[float]]] = None,
        group_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        신뢰도가 가장 높았던 상위 10% 그룹들의 평균을 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (group_confidences가 None일 때만 사용)
            group_confidences: 미리 계산된 그룹 신뢰도 리스트 (최적화용)
        """
        if group_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
            
            # 슬라이딩 윈도우로 모든 그룹 생성
            group_confidences = self._calculate_group_confidences(token_confidences)
        
        if len(group_confidences) == 0:
            return 0.0
        
        # numpy 배열로 변환
        group_confidences = np.asarray(group_confidences)
        
        # 상위 10% 그룹 선택: 전체 정렬 대신 partition 사용 (훨씬 빠름)
        num_top_groups = max(1, int(np.ceil(len(group_confidences) * 0.1)))
        # partition을 사용하여 상위 num_top_groups개만 정렬
        partitioned = np.partition(group_confidences, -num_top_groups)
        top_groups = partitioned[-num_top_groups:]
        
        return float(np.mean(top_groups))

    def calculate_highest_group_confidence(
        self,
        logprobs: Optional[List[List[float]]] = None,
        group_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        모든 그룹 중 최고 그룹 신뢰도를 반환합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (group_confidences가 None일 때만 사용)
            group_confidences: 미리 계산된 그룹 신뢰도 리스트 (최적화용)
        """
        if group_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
            if len(token_confidences) == 0:
                return 0.0
            
            # 슬라이딩 윈도우로 모든 그룹 생성
            group_confidences = self._calculate_group_confidences(token_confidences)
        
        if len(group_confidences) == 0:
            return 0.0
        
        group_confidences = np.asarray(group_confidences)
        return float(np.max(group_confidences))

    def calculate_tail_confidence(
        self, 
        logprobs: Optional[List[List[float]]] = None,
        token_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        마지막 1개 그룹의 신뢰도 점수를 계산합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트 (token_confidences가 None일 때만 사용)
            token_confidences: 미리 계산된 토큰 신뢰도 리스트 (최적화용)
        
        Returns:
            마지막 그룹의 신뢰도 점수
        """
        if token_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            
            # numpy 배열로 변환 후 벡터화 연산으로 토큰 신뢰도 계산
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
        
        # numpy 배열로 변환
        token_confidences = np.asarray(token_confidences)
        
        # 마지막 그룹의 신뢰도 계산
        if len(token_confidences) < self.group_size:
            return float(np.mean(token_confidences))
        
        last_group = token_confidences[-self.group_size:]
        return float(np.mean(last_group))
    
    def calculate_head_confidence(
        self,
        logprobs: Optional[List[List[float]]] = None,
        token_confidences: Optional[Union[List[float], np.ndarray]] = None
    ) -> float:
        """
        첫 번째 그룹의 신뢰도 점수를 계산합니다.
    
        Args:
            logprobs: 토큰별 로그 확률 리스트 (token_confidences가 None일 때만 사용)
            token_confidences: 미리 계산된 토큰 신뢰도 리스트 (최적화용)
        
        Returns:
            첫 번째 그룹의 신뢰도 점수
        """
        if token_confidences is None:
            if logprobs is None or len(logprobs) == 0:
                return 0.0
            
            # numpy 배열로 변환 후 벡터화 연산으로 토큰 신뢰도 계산
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
        
        # numpy 배열로 변환
        token_confidences = np.asarray(token_confidences)
        
        # 첫 번째 그룹의 신뢰도 계산
        if len(token_confidences) < self.group_size:
            return float(np.mean(token_confidences))
        
        first_group = token_confidences[:self.group_size]
        return float(np.mean(first_group))
    
    def calculate_all_confidence_scores(
        self, 
        logprobs: List[List[float]]
    ) -> Dict[str, float]:
        """
        모든 신뢰도 점수를 계산합니다.
        최적화: group_confidences를 한 번만 계산하여 모든 함수에 재사용합니다.
        
        Args:
            logprobs: 토큰별 로그 확률 리스트
        
        Returns:
            신뢰도 점수 딕셔너리
        """
        # 한 번만 token_confidences와 group_confidences 계산
        if logprobs is None or len(logprobs) == 0:
            token_confidences = np.array([], dtype=np.float32)
            group_confidences = np.array([], dtype=np.float32)
        else:
            # numpy 배열로 변환 후 벡터화 연산으로 토큰 신뢰도 계산
            logprobs_array = np.array([np.mean(token_logprobs) for token_logprobs in logprobs], dtype=np.float32)
            token_confidences = -logprobs_array
            # 슬라이딩 윈도우로 모든 그룹 생성 (한 번만 계산)
            group_confidences = self._calculate_group_confidences(token_confidences)
        
        # 계산된 token_confidences와 group_confidences를 모든 함수에 재사용
        return {
            "mean_group_confidence": self.calculate_mean_group_confidence(
                group_confidences=group_confidences
            ),
            "bottom_10_percent_confidence": self.calculate_bottom_10_percent_confidence(
                group_confidences=group_confidences
            ),
            "tail_confidence": self.calculate_tail_confidence(
                token_confidences=token_confidences
            ),
            "head_confidence": self.calculate_head_confidence(
                token_confidences=token_confidences
            ),
            "lowest_group_confidence": self.calculate_lowest_group_confidence(
                group_confidences=group_confidences
            ),
            "top_10_percent_confidence": self.calculate_top_10_percent_confidence(
                group_confidences=group_confidences
            ),
            "highest_group_confidence": self.calculate_highest_group_confidence(
                group_confidences=group_confidences
            )
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

