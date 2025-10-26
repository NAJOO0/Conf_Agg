"""
메트릭 계산 유틸리티 모듈
"""
import numpy as np
from typing import List, Dict, Any, Optional
import math_verify


def calculate_pass_at_k(
    predictions: List[str],
    ground_truths: List[str],
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Pass@k 메트릭을 계산합니다.
    
    Args:
        predictions: 예측 결과 리스트
        ground_truths: 정답 리스트
        k_values: 계산할 k 값들
    
    Returns:
        각 k에 대한 Pass@k 점수
    """
    results = {}
    
    for k in k_values:
        correct_count = 0
        total_count = len(predictions)
        
        for i in range(total_count):
            # 각 문제에 대해 상위 k개 예측 중 하나라도 맞으면 성공
            problem_predictions = predictions[i][:k] if isinstance(predictions[i], list) else [predictions[i]]
            
            for pred in problem_predictions:
                if verify_answer(pred, ground_truths[i]):
                    correct_count += 1
                    break
        
        results[k] = correct_count / total_count if total_count > 0 else 0.0
    
    return results


def verify_answer(predicted: str, ground_truth: str) -> bool:
    """
    math_verify를 사용하여 답안을 검증합니다.
    
    Args:
        predicted: 예측된 답안
        ground_truth: 정답
    
    Returns:
        검증 결과 (True/False)
    """
    try:
        return math_verify.verify(predicted, ground_truth)
    except Exception:
        # math_verify 실패 시 문자열 비교로 폴백
        return predicted.strip().lower() == ground_truth.strip().lower()


def calculate_confidence_correlation(
    predictions: List[str],
    ground_truths: List[str],
    confidence_scores: List[float]
) -> Dict[str, float]:
    """
    신뢰도 점수와 정확도 간의 상관관계를 계산합니다.
    
    Args:
        predictions: 예측 결과 리스트
        ground_truths: 정답 리스트
        confidence_scores: 신뢰도 점수 리스트
    
    Returns:
        상관관계 메트릭들
    """
    # 정확도 계산
    accuracies = []
    for pred, gt in zip(predictions, ground_truths):
        accuracies.append(1.0 if verify_answer(pred, gt) else 0.0)
    
    accuracies = np.array(accuracies)
    confidence_scores = np.array(confidence_scores)
    
    # 피어슨 상관계수
    correlation = np.corrcoef(confidence_scores, accuracies)[0, 1]
    
    # 신뢰도 구간별 정확도
    confidence_bins = np.percentile(confidence_scores, [0, 25, 50, 75, 100])
    bin_accuracies = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(accuracies[mask]))
        else:
            bin_accuracies.append(0.0)
    
    return {
        "pearson_correlation": correlation,
        "bin_accuracies": bin_accuracies,
        "confidence_bins": confidence_bins.tolist()
    }


def calculate_group_statistics(
    rewards: List[float],
    group_size: int = 8
) -> Dict[str, Any]:
    """
    그룹 기반 통계를 계산합니다.
    
    Args:
        rewards: 보상 점수 리스트
        group_size: 그룹 크기
    
    Returns:
        그룹 통계 정보
    """
    rewards = np.array(rewards)
    num_groups = len(rewards) // group_size
    
    group_means = []
    group_stds = []
    
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]
        
        group_means.append(np.mean(group_rewards))
        group_stds.append(np.std(group_rewards))
    
    return {
        "num_groups": num_groups,
        "group_size": group_size,
        "group_means": group_means,
        "group_stds": group_stds,
        "overall_mean": np.mean(rewards),
        "overall_std": np.std(rewards)
    }

