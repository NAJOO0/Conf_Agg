"""
데이터 큐레이션 모듈
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict
import random

from src.evaluation.math_verifier import MathVerifier
from src.data.dataset import GeneratedDataset

logger = logging.getLogger(__name__)


class DataCurator:
    """데이터 큐레이션 클래스"""
    
    def __init__(
        self,
        strategy: str = "curriculum",
        easy_sample_percentage: int = 50,
        num_sets_per_problem: int = 16,
        set_size: int = 8,
        timeout: int = 30
    ):
        """
        Args:
            strategy: 큐레이션 전략 (naive, curriculum, multitask)
            easy_sample_percentage: Easy 샘플 비율
            num_sets_per_problem: 문제당 세트 수
            set_size: 각 세트의 크기
            timeout: 검증 타임아웃
        """
        self.strategy = strategy
        self.easy_sample_percentage = easy_sample_percentage
        self.num_sets_per_problem = num_sets_per_problem
        self.set_size = set_size
        self.verifier = MathVerifier(timeout=timeout)
        
        # 시드 설정
        random.seed(42)
        np.random.seed(42)
    
    def classify_hard_easy(
        self, 
        generated_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Hard/Easy 분류를 수행합니다.
        
        Args:
            generated_data: Stage 1에서 생성된 데이터
        
        Returns:
            (hard_data, easy_data) 튜플
        """
        logger.info("Hard/Easy 분류 시작")
        
        # 문제별로 그룹화
        problem_groups = generated_data.groupby('problem_id')
        
        hard_problems = []
        easy_problems = []
        
        for problem_id, group in problem_groups:
            # 다수결 투표로 정답 결정
            predictions = group['generated_text'].tolist()
            ground_truth = group['ground_truth'].iloc[0]
            
            # 각 응답에서 답안 추출
            extracted_answers = []
            for pred in predictions:
                answer = self.verifier.extract_answer_from_response(pred)
                extracted_answers.append(answer)
            
            # 다수결 투표
            majority_answer = self._get_majority_answer(extracted_answers)
            
            # 다수결 결과와 실제 정답 비교
            is_correct = self.verifier.verify_answer(majority_answer, ground_truth)
            
            if is_correct:
                easy_problems.append(problem_id)
            else:
                hard_problems.append(problem_id)
        
        logger.info(f"Hard 문제: {len(hard_problems)}개, Easy 문제: {len(easy_problems)}개")
        
        # 데이터 분할
        hard_data = generated_data[generated_data['problem_id'].isin(hard_problems)]
        easy_data = generated_data[generated_data['problem_id'].isin(easy_problems)]
        
        return hard_data, easy_data
    
    def _get_majority_answer(self, answers: List[str]) -> str:
        """다수결 투표로 답안을 결정합니다."""
        # 답안별 빈도 계산
        answer_counts = defaultdict(int)
        for answer in answers:
            answer_counts[answer] += 1
        
        # 가장 많이 나온 답안 반환
        return max(answer_counts.items(), key=lambda x: x[1])[0]
    
    def create_response_sets(
        self, 
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        응답을 세트로 나눕니다.
        
        Args:
            data: 문제별 응답 데이터
        
        Returns:
            세트별 데이터 리스트
        """
        logger.info("응답 세트 생성 시작")
        
        sets = []
        problem_groups = data.groupby('problem_id')
        
        for problem_id, group in problem_groups:
            responses = group.to_dict('records')
            
            # 응답을 세트 크기로 나누기
            for i in range(0, len(responses), self.set_size):
                if i + self.set_size <= len(responses):
                    response_set = responses[i:i + self.set_size]
                    
                    # 세트 정보 생성
                    set_info = {
                        'problem_id': problem_id,
                        'problem_text': response_set[0]['problem_text'],
                        'ground_truth': response_set[0]['ground_truth'],
                        'set_id': f"{problem_id}_set_{i // self.set_size}",
                        'responses': [r['generated_text'] for r in response_set],
                        'confidence_scores': {
                            'mean_group_confidence': [r['mean_group_confidence'] for r in response_set],
                            'bottom_10_percent_confidence': [r['bottom_10_percent_confidence'] for r in response_set],
                            'tail_confidence': [r['tail_confidence'] for r in response_set]
                        }
                    }
                    
                    sets.append(set_info)
        
        logger.info(f"총 {len(sets)}개 세트 생성")
        return sets
    
    def apply_curation_strategy(
        self, 
        hard_sets: List[Dict[str, Any]], 
        easy_sets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        큐레이션 전략을 적용합니다.
        
        Args:
            hard_sets: Hard 문제 세트들
            easy_sets: Easy 문제 세트들
        
        Returns:
            큐레이션된 세트들
        """
        logger.info(f"큐레이션 전략 적용: {self.strategy}")
        
        if self.strategy == "naive":
            # 모든 세트 사용
            return hard_sets + easy_sets
        
        elif self.strategy == "curriculum":
            # Easy 샘플 비율만큼 Easy 세트 추가
            num_easy_to_add = int(len(hard_sets) * self.easy_sample_percentage / 100)
            selected_easy = random.sample(easy_sets, min(num_easy_to_add, len(easy_sets)))
            return hard_sets + selected_easy
        
        elif self.strategy == "multitask":
            # Hard와 Easy를 균등하게 혼합
            min_sets = min(len(hard_sets), len(easy_sets))
            selected_hard = random.sample(hard_sets, min_sets)
            selected_easy = random.sample(easy_sets, min_sets)
            return selected_hard + selected_easy
        
        else:
            raise ValueError(f"지원하지 않는 큐레이션 전략: {self.strategy}")
    
    def split_train_validation(
        self, 
        curated_sets: List[Dict[str, Any]], 
        train_split: float = 0.8
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        훈련/검증 데이터로 분할합니다.
        
        Args:
            curated_sets: 큐레이션된 세트들
            train_split: 훈련 데이터 비율
        
        Returns:
            (train_sets, validation_sets) 튜플
        """
        logger.info(f"훈련/검증 데이터 분할 (훈련 비율: {train_split})")
        
        # 문제별로 그룹화하여 분할
        problem_sets = defaultdict(list)
        for set_data in curated_sets:
            problem_sets[set_data['problem_id']].append(set_data)
        
        problems = list(problem_sets.keys())
        random.shuffle(problems)
        
        split_idx = int(len(problems) * train_split)
        train_problems = problems[:split_idx]
        validation_problems = problems[split_idx:]
        
        train_sets = []
        validation_sets = []
        
        for problem in train_problems:
            train_sets.extend(problem_sets[problem])
        
        for problem in validation_problems:
            validation_sets.extend(problem_sets[problem])
        
        logger.info(f"훈련 세트: {len(train_sets)}개, 검증 세트: {len(validation_sets)}개")
        
        return train_sets, validation_sets
    
    def curate_data(
        self, 
        generated_data_path: str,
        output_dir: str
    ) -> Tuple[str, str]:
        """
        전체 데이터 큐레이션 과정을 수행합니다.
        
        Args:
            generated_data_path: Stage 1 결과 파일 경로
            output_dir: 출력 디렉토리
        
        Returns:
            (train_path, validation_path) 튜플
        """
        logger.info("데이터 큐레이션 시작")
        
        # 생성된 데이터 로드
        generated_data = pd.read_parquet(generated_data_path)
        logger.info(f"생성된 데이터 로드: {len(generated_data)}개 응답")
        
        # Hard/Easy 분류
        hard_data, easy_data = self.classify_hard_easy(generated_data)
        
        # 응답 세트 생성
        hard_sets = self.create_response_sets(hard_data)
        easy_sets = self.create_response_sets(easy_data)
        
        # 큐레이션 전략 적용
        curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)
        
        # 훈련/검증 분할
        train_sets, validation_sets = self.split_train_validation(curated_sets)
        
        # 결과 저장
        train_df = pd.DataFrame(train_sets)
        validation_df = pd.DataFrame(validation_sets)
        
        train_path = f"{output_dir}/train_curated.parquet"
        validation_path = f"{output_dir}/validation_curated.parquet"
        
        train_df.to_parquet(train_path, index=False)
        validation_df.to_parquet(validation_path, index=False)
        
        logger.info(f"큐레이션 완료: {train_path}, {validation_path}")
        
        return train_path, validation_path

