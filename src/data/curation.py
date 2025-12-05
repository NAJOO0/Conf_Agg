"""
데이터 큐레이션 모듈
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict
import random
import os
import json
from transformers import AutoTokenizer
# PyArrow 가용성 확인
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

from src.evaluation.math_verifier import MathVerifier
from src.data.dataset import GeneratedDataset

logger = logging.getLogger(__name__)


def _serialize_nested_data(sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    중첩된 데이터 구조를 JSON 문자열로 변환하여 Parquet 저장 호환성 확보
    
    Args:
        sets: 세트 리스트
    
    Returns:
        JSON 문자열로 변환된 세트 리스트
    """
    serialized_sets = []
    for set_info in sets:
        serialized_set = set_info.copy()
        # solutions 리스트를 JSON 문자열로 변환
        if 'solutions' in serialized_set and isinstance(serialized_set['solutions'], list):
            serialized_set['solutions'] = json.dumps(serialized_set['solutions'], ensure_ascii=False)
        # confidence_scores 딕셔너리를 JSON 문자열로 변환
        if 'confidence_scores' in serialized_set and isinstance(serialized_set['confidence_scores'], dict):
            serialized_set['confidence_scores'] = json.dumps(serialized_set['confidence_scores'], ensure_ascii=False)
        serialized_sets.append(serialized_set)
    return serialized_sets


class DataCurator:
    """데이터 큐레이션 클래스"""
    
    def __init__(
        self,
        strategy: str = "curriculum",
        enable_thinking: bool = False,
        easy_sample_percentage: int = 50,
        num_sets_per_problem: int = 4,
        set_size: int = 4,
        timeout: int = 30,
        confidence_key: str = "bottom_10_percent_confidence",
        fill_insufficient_with_sampling: bool = True,
        tokenizer_model: str = "Qwen/Qwen3-1.7B",
        easy_threshold: float = 0.5,
        order_strategy: str = "hard_first",
        diverse_confidence_sampling: bool = False,
        prompt_template: str =
            (
                "You are an expert mathematician and critical analyst.\n"
                "Your task is to synthesize multiple, potentially flawed, solution attempts "
                "into a single, correct, and comprehensive final answer.\n\n"
                "You will be given a problem, followed by several solution attempts.\n"
                "Solution attempts include confidence scores to help estimate their quality.\n\n"
                "Carefully review all the provided information. It is possible that any, all, or none "
                "of the solutions are correct or complete.\n"
                "Use them as starting points—correcting mistakes, filling in gaps, and/or combining "
                "useful ideas—to produce your final solution.\n\n"
                "---\n"
                "GIVEN THE FOLLOWING PROBLEM:\n{problem}\n\n"
                "AND THESE SOLUTION ATTEMPTS:\n{solutions}\n\n"
                "---\n"
                "Now, provide the final, comprehensive, and correct solution to the problem."
            ),
    ):
        """
        Args:
            strategy: 큐레이션 전략 (naive, curriculum, multitask, baseline)
                - naive: 기본 전략 (Hard/Easy 비율 조정)
                - curriculum: set_size를 점진적으로 감소시켜 여러 데이터셋 생성
                - multitask: 여러 set_size를 하나의 데이터셋에 포함
                - baseline: confidence 없이 solution만 포함하는 프롬프트 생성
            easy_sample_percentage: Easy 샘플 비율
            num_sets_per_problem: 문제당 세트 수
            set_size: 각 세트의 크기
            timeout: 검증 타임아웃
            confidence_key: 사용할 컨피던스 키 (default: "tail_confidence")
            fill_insufficient_with_sampling: 응답이 부족할 때 샘플링으로 채울지 여부 (default: True)
                - True: 중복을 허용하여 필요한 개수만큼 채움 (random.choices)
                - False: 있는 응답만 사용하여 세트 수가 줄어듦
            tokenizer_model: 토크나이저 모델 이름 (default: "Qwen/Qwen3-1.7B")
            easy_threshold: Easy/Hard 분류 임계값 (default: 0.5, 정답률 >= 이 값이면 Easy)
            order_strategy: 데이터 순서 전략 (default: "hard_first")
                - "hard_first": Hard 세트 먼저, Easy 세트 나중
                - "easy_first": Easy 세트 먼저, Hard 세트 나중
                - "shuffle": Hard와 Easy를 무작위로 섞기
            diverse_confidence_sampling: confidence가 diverse하게 분포하도록 샘플링 (default: False)
                - True: confidence를 정렬하고 구간별로 균등하게 샘플링 (stratified sampling)
                - False: 기존 랜덤 샘플링 방식 사용
            prompt_template: 프롬프트 템플릿 문자열
        """
        self.strategy = strategy
        self.easy_sample_percentage = easy_sample_percentage
        self.num_sets_per_problem = num_sets_per_problem
        self.set_size = set_size
        self.enable_thinking = enable_thinking
        self.verifier = MathVerifier(timeout=timeout)
        self.confidence_key = confidence_key
        self.fill_insufficient_with_sampling = fill_insufficient_with_sampling
        self.easy_threshold = easy_threshold
        self.order_strategy = order_strategy
        self.diverse_confidence_sampling = diverse_confidence_sampling
        self.prompt_template = prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        # 시드 설정
        random.seed(42)
        np.random.seed(42)
        logger.info(f"Enable Thinking: {self.enable_thinking}")
        logger.info(f"Order Strategy: {self.order_strategy}")
        logger.info(f"Diverse Confidence Sampling: {self.diverse_confidence_sampling}")
    def classify_hard_easy_sets(
        self,
        sets: List[Dict[str, Any]],
        save_distribution: bool = False,
        output_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        세트 기반 Hard/Easy 분류를 수행합니다.
        각 세트의 정답률을 계산하여 Hard/Easy로 분류합니다.

        Args:
            sets: 생성된 세트 리스트
            save_distribution: 정답 개수 분포를 파일로 저장할지 여부
            output_dir: 분포 파일을 저장할 디렉토리 (save_distribution=True일 때 필요)
            suffix: 파일 이름에 추가할 접미사 (예: "_size_4")

        Returns:
            (hard_sets, easy_sets) 튜플
        """
        logger.info("세트 기반 Hard/Easy 분류 시작")

        hard_sets = []
        easy_sets = []

        # 정답 개수 분포를 추적하기 위한 리스트
        hard_correct_counts = []
        easy_correct_counts = []

        for set_info in sets:
            ground_truth = set_info['ground_truth']
            solutions = set_info['solutions']

            # 세트 내 정답 개수 계산
            correct_count = 0
            for solution in solutions:
                final_answer = solution.get('final_answer', '')
                if self.verifier.verify_answer(final_answer, ground_truth):
                    correct_count += 1

            # 정답률 계산
            total_count = len(solutions)
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            # 정답률이 임계값 이상이면 Easy, 아니면 Hard
            if accuracy >= self.easy_threshold:
                easy_sets.append(set_info)
                easy_correct_counts.append(correct_count)
            else:
                hard_sets.append(set_info)
                hard_correct_counts.append(correct_count)

        logger.info(f"Hard 세트: {len(hard_sets)}개, Easy 세트: {len(easy_sets)}개")
        logger.info(f"Easy 비율: {len(easy_sets) / len(sets) * 100:.2f}%")

        # 분포 정보 저장
        if save_distribution and output_dir:
            self._save_correct_answer_distribution(
                hard_correct_counts,
                easy_correct_counts,
                output_dir,
                suffix
            )

        return hard_sets, easy_sets

    def _save_correct_answer_distribution(
        self,
        hard_correct_counts: List[int],
        easy_correct_counts: List[int],
        output_dir: str,
        suffix: str = ""
    ):
        """
        Hard/Easy 세트의 정답 개수 분포를 분석하고 파일로 저장합니다.

        Args:
            hard_correct_counts: Hard 세트들의 정답 개수 리스트
            easy_correct_counts: Easy 세트들의 정답 개수 리스트
            output_dir: 결과 파일을 저장할 디렉토리
            suffix: 파일 이름에 추가할 접미사 (예: "_size_4")
        """
        logger.info("정답 개수 분포 분석 및 저장 시작")

        # 분포 계산 함수
        def calculate_distribution(counts: List[int], category: str) -> Dict[str, Any]:
            if not counts:
                return {
                    'category': category,
                    'total_sets': 0,
                    'distribution': {},
                    'statistics': {}
                }

            # 정답 개수별 빈도 계산
            count_freq = defaultdict(int)
            for count in counts:
                count_freq[count] += 1

            # 정렬된 분포
            sorted_distribution = dict(sorted(count_freq.items()))

            # 통계 계산
            counts_array = np.array(counts)
            statistics = {
                'mean': float(np.mean(counts_array)),
                'median': float(np.median(counts_array)),
                'std': float(np.std(counts_array)),
                'min': int(np.min(counts_array)),
                'max': int(np.max(counts_array)),
                'q25': float(np.percentile(counts_array, 25)),
                'q75': float(np.percentile(counts_array, 75))
            }

            return {
                'category': category,
                'total_sets': len(counts),
                'distribution': sorted_distribution,
                'statistics': statistics
            }

        # Hard와 Easy 각각의 분포 계산
        hard_dist = calculate_distribution(hard_correct_counts, 'Hard')
        easy_dist = calculate_distribution(easy_correct_counts, 'Easy')

        # 결과 딕셔너리 생성
        result = {
            'metadata': {
                'set_size': self.set_size,
                'easy_threshold': self.easy_threshold,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'hard': hard_dist,
            'easy': easy_dist
        }

        # JSON 파일로 저장 (suffix가 있으면 파일명에 추가)
        filename = f"correct_answer_distribution{suffix}.json"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"정답 개수 분포 저장 완료: {output_path}")

        # 로그에도 요약 정보 출력
        logger.info(f"Hard 세트 정답 개수 분포 (총 {hard_dist['total_sets']}개):")
        if hard_dist['distribution']:
            for count, freq in hard_dist['distribution'].items():
                logger.info(f"  정답 {count}개: {freq}개 세트 ({freq/hard_dist['total_sets']*100:.2f}%)")
            logger.info(f"  평균: {hard_dist['statistics']['mean']:.2f}, "
                       f"중앙값: {hard_dist['statistics']['median']:.2f}, "
                       f"표준편차: {hard_dist['statistics']['std']:.2f}")

        logger.info(f"Easy 세트 정답 개수 분포 (총 {easy_dist['total_sets']}개):")
        if easy_dist['distribution']:
            for count, freq in easy_dist['distribution'].items():
                logger.info(f"  정답 {count}개: {freq}개 세트 ({freq/easy_dist['total_sets']*100:.2f}%)")
            logger.info(f"  평균: {easy_dist['statistics']['mean']:.2f}, "
                       f"중앙값: {easy_dist['statistics']['median']:.2f}, "
                       f"표준편차: {easy_dist['statistics']['std']:.2f}")

    def _get_majority_answer(self, answers: List[str]) -> str:
        """다수결 투표로 답안을 결정합니다."""
        # 답안별 빈도 계산
        answer_counts = defaultdict(int)
        for answer in answers:
            answer_counts[answer] += 1

        # 가장 많이 나온 답안 반환
        return max(answer_counts.items(), key=lambda x: x[1])[0]

    def _select_diverse_by_confidence(
        self,
        responses: List[Dict[str, Any]],
        num_samples: int,
        problem_id: Any
    ) -> List[Dict[str, Any]]:
        """
        Confidence 기반 diverse sampling을 수행합니다.

        응답들을 confidence로 정렬하고, num_samples개 구간으로 나누어
        각 구간에서 균등하게 샘플링하여 diverse한 confidence 분포를 보장합니다.

        Args:
            responses: 응답 리스트
            num_samples: 선택할 샘플 수 (보통 set_size)
            problem_id: 문제 ID (로깅용)

        Returns:
            Diverse하게 선택된 응답 리스트
        """
        # 응답에서 confidence 값 추출 및 정렬
        responses_with_conf = []
        for r in responses:
            conf_val = r.get(self.confidence_key)
            if conf_val is None:
                conf_val = 0.0
            responses_with_conf.append((conf_val, r))

        # Confidence 기준으로 정렬 (오름차순)
        responses_with_conf.sort(key=lambda x: x[0])

        # Stratified sampling: num_samples개 구간으로 나누고 각 구간에서 1개씩 선택
        selected_responses = []
        n_responses = len(responses_with_conf)

        if n_responses < num_samples:
            # 응답이 필요한 수보다 적으면 중복 샘플링
            if self.fill_insufficient_with_sampling:
                # 있는 것을 모두 선택하고 부족한 만큼 랜덤으로 추가
                selected_responses = [r for _, r in responses_with_conf]
                while len(selected_responses) < num_samples:
                    selected_responses.append(random.choice([r for _, r in responses_with_conf]))
            else:
                selected_responses = [r for _, r in responses_with_conf]
        else:
            # 구간 크기 계산
            bucket_size = n_responses / num_samples

            for i in range(num_samples):
                # 각 구간의 시작과 끝 인덱스 계산
                start_idx = int(i * bucket_size)
                end_idx = int((i + 1) * bucket_size)

                # 마지막 구간은 끝까지 포함
                if i == num_samples - 1:
                    end_idx = n_responses

                # 구간이 비어있지 않은지 확인
                if start_idx < end_idx:
                    # 구간 내에서 무작위로 1개 선택
                    selected_idx = random.randint(start_idx, end_idx - 1)
                    selected_responses.append(responses_with_conf[selected_idx][1])
                else:
                    # 구간이 비어있으면 (이론상 발생하지 않아야 함) 마지막 응답 선택
                    selected_responses.append(responses_with_conf[-1][1])

        return selected_responses
    
    def create_response_sets(
        self, 
        data: pd.DataFrame,
        num_sets: Optional[int] = None,
        shuffle_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        응답을 세트로 나눕니다.
        
        Args:
            data: 문제별 응답 데이터
            num_sets: 각 문제당 생성할 세트 수 (None이면 num_sets_per_problem 사용)
            shuffle_seed: 각 문제별 응답을 shuffle할 때 사용할 seed (None이면 기본 seed 사용)
        
        Returns:
            세트별 데이터 리스트
        """
        logger.info("응답 세트 생성 시작")
        
        if num_sets is None:
            num_sets = self.num_sets_per_problem
        
        sets = []
        problem_groups = data.groupby('problem_id')
        
        for problem_id, group in problem_groups:
            responses = group.to_dict('records')

            # 필요한 총 응답 수 계산
            required_responses = num_sets * self.set_size

            # Diverse confidence sampling을 사용하는 경우
            if self.diverse_confidence_sampling:
                # shuffle_seed가 제공되면 해당 seed 사용
                if shuffle_seed is not None:
                    problem_seed = hash(str(problem_id) + str(shuffle_seed)) % (2**31)
                    random.seed(problem_seed)

                # 각 세트마다 독립적으로 diverse sampling 수행
                selected_responses = []
                for set_idx in range(num_sets):
                    # 각 세트마다 set_size개를 diverse하게 선택
                    set_responses = self._select_diverse_by_confidence(
                        responses,
                        self.set_size,
                        problem_id
                    )
                    selected_responses.extend(set_responses)

                # 전역 seed 복원
                if shuffle_seed is not None:
                    random.seed(42)

            else:
                # 기존 랜덤 샘플링 방식
                # shuffle_seed가 제공되면 해당 seed로 shuffle 및 샘플링
                if shuffle_seed is not None:
                    # 문제별로 고유한 seed 생성 (problem_id와 shuffle_seed 조합)
                    problem_seed = hash(str(problem_id) + str(shuffle_seed)) % (2**31)
                    random.seed(problem_seed)

                    if len(responses) >= required_responses:
                        # 충분한 응답이 있으면 중복 없이 샘플링
                        selected_responses = random.sample(responses, required_responses)
                    else:
                        # 응답이 부족한 경우
                        if self.fill_insufficient_with_sampling:
                            # 중복을 허용하여 필요한 개수만큼 채움
                            selected_responses = random.choices(responses, k=required_responses)
                            logger.info(
                                f"문제 {problem_id}: 응답 {len(responses)}개에서 중복 샘플링으로 {required_responses}개 생성"
                            )
                        else:
                            # 있는 만큼만 사용 (세트 수가 줄어듦)
                            selected_responses = responses
                            logger.warning(
                                f"문제 {problem_id}: 응답 {len(responses)}개로는 세트 생성에 부족합니다. "
                                f"필요: {required_responses}개, 세트 수가 {len(responses) // self.set_size}개로 제한됩니다."
                            )

                    # 전역 seed 복원
                    random.seed(42)
                else:
                    # shuffle_seed가 없으면 기존 방식 사용
                    if len(responses) >= required_responses:
                        # 충분한 응답이 있으면 중복 없이 샘플링
                        selected_responses = random.sample(responses, required_responses)
                    else:
                        # 응답이 부족한 경우
                        if self.fill_insufficient_with_sampling:
                            # 중복을 허용하여 필요한 개수만큼 채움
                            selected_responses = random.choices(responses, k=required_responses)
                            logger.info(
                                f"문제 {problem_id}: 응답 {len(responses)}개에서 중복 샘플링으로 {required_responses}개 생성"
                            )
                        else:
                            # 있는 만큼만 사용 (세트 수가 줄어듦)
                            selected_responses = responses
                            logger.warning(
                                f"문제 {problem_id}: 응답 {len(responses)}개로는 세트 생성에 부족합니다. "
                                f"필요: {required_responses}개, 세트 수가 {len(responses) // self.set_size}개로 제한됩니다."
                            )
            
            # 응답을 세트 크기로 나누기
            actual_num_sets = len(selected_responses) // self.set_size
            for i in range(actual_num_sets):
                start_idx = i * self.set_size
                end_idx = start_idx + self.set_size
                response_set = selected_responses[start_idx:end_idx]
                
                problem_text = response_set[0].get('problem_text', '')
                ground_truth = response_set[0].get('ground_truth', '')
                
                # solutions 리스트(컨텐츠/최종답/선택 컨피던스) 구성
                solutions = []
                selected_conf_values = []
                for r in response_set:
                    # content/final_answer 우선 사용, 없으면 generated_text 백업
                    content = r.get('content') if r.get('content') is not None else r.get('generated_text', '')
                    final_answer = r.get('final_answer') if r.get('final_answer') is not None else ''
                    # confidence_key 값 가져오기 (없으면 기본값 0.0)
                    conf_val = r.get(self.confidence_key)
                    if conf_val is None:
                        logger.warning(
                            f"confidence_key '{self.confidence_key}' not found in response "
                            f"for problem {problem_id}, using default value 0.0"
                        )
                        conf_val = 0.0
                    solutions.append({
                        'content': content,
                        'final_answer': final_answer,
                        'confidence': {
                            'key': self.confidence_key,
                            'value': conf_val
                        }
                    })
                    selected_conf_values.append(conf_val)
                
                # 프롬프트용 solutions 텍스트 생성 (baseline 전략일 경우 confidence 제외)
                lines = []
                for idx, s in enumerate(solutions, start=1):
                    if self.strategy == "baseline":
                        # baseline 전략: confidence 없이 solution만 포함
                        lines.append(
                            f"solution{idx}:\n"
                            f"{s['content']}\n"
                            f"final_answer: {s['final_answer']}\n"
                        )
                    else:
                        # 기존 전략: confidence 포함
                        conf_value = s['confidence']['value']
                        conf_str = f"{conf_value:.4f}" if conf_value is not None else "N/A"
                        lines.append(
                            f"solution{idx}:\n"
                            f"{s['content']}\n"
                            f"final_answer: {s['final_answer']}\n"
                            f"confidence: {conf_str}\n"
                        )
                solutions_text = "\n".join(lines)
                prompt = self.prompt_template.format(problem=problem_text, solutions=solutions_text)
                
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                # 세트 정보 생성(호환 필드 유지)
                set_info = {
                    'problem_id': problem_id,
                    'problem_text': problem_text,
                    'ground_truth': ground_truth,
                    'set_id': f"{problem_id}_set_{i}",
                    'prompt': prompt,
                    'solutions': solutions,
                    'selected_confidence_key': self.confidence_key,
                    'selected_confidence': selected_conf_values,
                    # enable_thinking: Stage 3에서 chat template 적용 시 사용
                    'enable_thinking': self.enable_thinking,
                    # 기존 파이프라인 호환을 위해 유지
                    'responses': [r.get('generated_text', '') for r in response_set],
                    'confidence_scores': {
                        'mean_group_confidence': [r.get('mean_group_confidence') for r in response_set],
                        'bottom_10_percent_confidence': [r.get('bottom_10_percent_confidence') for r in response_set],
                        'tail_confidence': [r.get('tail_confidence') for r in response_set]
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
        
        기본 전략: Easy가 Hard의 50%가 되도록 조정
        - Easy가 부족하면 Hard를 줄여서 비율 맞춤
        - Easy가 많으면 Easy를 줄여서 비율 맞춤
        
        Args:
            hard_sets: Hard 문제 세트들
            easy_sets: Easy 문제 세트들
        
        Returns:
            큐레이션된 세트들
        """
        logger.info(f"큐레이션 전략 적용: {self.strategy}")
        
        # 목표: Easy가 Hard의 50%가 되도록 조정
        num_easy_target = int(len(hard_sets) * self.easy_sample_percentage / 100)
        
        if len(easy_sets) == 0:
            # Easy가 없으면 Hard만 반환
            selected_hard = hard_sets.copy()
            selected_easy = []
        elif len(easy_sets) < num_easy_target:
            # Easy가 부족하면 Hard를 줄여서 비율 맞춤
            # Easy = Hard * 0.5 이므로, Hard = Easy * 2
            num_hard_target = int(len(easy_sets) * 100 / self.easy_sample_percentage)
            
            # Hard 세트 줄이기 전략: Zero-correct (정답 0개) 우선 제거
            # 1. Zero-correct 세트와 나머지 세트 분리
            zero_correct_sets = []
            other_hard_sets = []
            
            for s in hard_sets:
                # solutions 내의 정답 개수 계산
                correct_count = 0
                for sol in s['solutions']:
                    if self.verifier.verify_answer(sol.get('final_answer', ''), s['ground_truth']):
                        correct_count += 1
                
                if correct_count == 0:
                    zero_correct_sets.append(s)
                else:
                    other_hard_sets.append(s)
            
            logger.info(f"Hard 세트 분석: Zero-correct {len(zero_correct_sets)}개, Others {len(other_hard_sets)}개")
            
            if len(other_hard_sets) >= num_hard_target:
                # Others만으로 충분하면 Others에서 샘플링 (Zero-correct는 모두 버림)
                selected_hard = random.sample(other_hard_sets, num_hard_target)
                logger.info(f"Hard 조정: Others에서 {num_hard_target}개 선택, Zero-correct {len(zero_correct_sets)}개 제거")
            else:
                # Others를 모두 포함하고 부족분은 Zero-correct에서 채움
                needed_from_zero = num_hard_target - len(other_hard_sets)
                selected_hard = other_hard_sets + random.sample(zero_correct_sets, min(needed_from_zero, len(zero_correct_sets)))
                logger.info(f"Hard 조정: Others {len(other_hard_sets)}개 모두 선택, Zero-correct에서 {min(needed_from_zero, len(zero_correct_sets))}개 추가")

            selected_easy = easy_sets.copy()
            logger.info(
                f"Easy가 부족하여 Hard를 조정: "
                f"Hard {len(hard_sets)}개 -> {len(selected_hard)}개, "
                f"Easy {len(easy_sets)}개 유지"
            )
        else:
            # Easy가 충분하면 Easy를 줄여서 비율 맞춤
            
            # Easy 세트 줄이기 전략: All-correct (정답 set_size개) 우선 제거
            # 1. All-correct 세트와 나머지 세트 분리
            all_correct_sets = []
            other_easy_sets = []
            
            for s in easy_sets:
                # solutions 내의 정답 개수 계산
                correct_count = 0
                for sol in s['solutions']:
                    if self.verifier.verify_answer(sol.get('final_answer', ''), s['ground_truth']):
                        correct_count += 1
                
                if correct_count == self.set_size:
                    all_correct_sets.append(s)
                else:
                    other_easy_sets.append(s)
            
            logger.info(f"Easy 세트 분석: All-correct {len(all_correct_sets)}개, Others {len(other_easy_sets)}개")
            
            if len(other_easy_sets) >= num_easy_target:
                # Others만으로 충분하면 Others에서 샘플링 (All-correct는 모두 버림)
                selected_easy = random.sample(other_easy_sets, num_easy_target)
                logger.info(f"Easy 조정: Others에서 {num_easy_target}개 선택, All-correct {len(all_correct_sets)}개 제거")
            else:
                # Others를 모두 포함하고 부족분은 All-correct에서 채움
                needed_from_all = num_easy_target - len(other_easy_sets)
                selected_easy = other_easy_sets + random.sample(all_correct_sets, needed_from_all)
                logger.info(f"Easy 조정: Others {len(other_easy_sets)}개 모두 선택, All-correct에서 {needed_from_all}개 추가")

            selected_hard = hard_sets.copy()
            logger.info(
                f"Easy가 충분하여 Easy를 조정: "
                f"Hard {len(hard_sets)}개 유지, "
                f"Easy {len(easy_sets)}개 -> {len(selected_easy)}개"
            )
        
        logger.info(f"최종 선택: Hard 세트 {len(selected_hard)}개, Easy 세트 {len(selected_easy)}개")
        if len(selected_hard) > 0:
            logger.info(f"Easy 비율: {len(selected_easy) / len(selected_hard) * 100:.2f}% (목표: {self.easy_sample_percentage}%)")
        else:
            logger.info("Hard 세트가 없어 비율 계산 불가")
        
        # 순서 전략에 따라 데이터 정렬
        if self.order_strategy == "hard_first":
            # Hard 먼저, Easy 나중 (기본 동작)
            curated_sets = selected_hard + selected_easy
            logger.info(f"데이터 순서: Hard 먼저 ({len(selected_hard)}개), Easy 나중 ({len(selected_easy)}개)")
        elif self.order_strategy == "easy_first":
            # Easy 먼저, Hard 나중
            curated_sets = selected_easy + selected_hard
            logger.info(f"데이터 순서: Easy 먼저 ({len(selected_easy)}개), Hard 나중 ({len(selected_hard)}개)")
        elif self.order_strategy == "shuffle":
            # Hard와 Easy를 무작위로 섞기
            curated_sets = selected_hard + selected_easy
            random.shuffle(curated_sets)
            logger.info(f"데이터 순서: 무작위로 섞임 (총 {len(curated_sets)}개)")
        else:
            # 기본값: hard_first
            logger.warning(f"알 수 없는 order_strategy: {self.order_strategy}, 기본값(hard_first) 사용")
            curated_sets = selected_hard + selected_easy
        
        return curated_sets
    
    def split_train_validation(
        self, 
        curated_sets: List[Dict[str, Any]], 
        validation_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Train/Validation 분할 (Problem ID 기준)
        
        주의: Data Leakage를 방지하기 위해 반드시 Problem ID 기준으로 분할해야 함.
        같은 문제의 변형들이 Train과 Validation에 섞이면 안 됨.
        
        수정: 분할 후에도 curated_sets의 원래 순서(Hard/Easy 정렬)를 유지함.
        """
        logger.info(f"Train/Validation 분할 시작 (비율: {validation_ratio})")
        
        # 1. Problem ID 수집
        problem_ids = list(set(item['problem_id'] for item in curated_sets))
        logger.info(f"총 문제 개수: {len(problem_ids)}")
        
        # 2. Problem ID 섞기 (어떤 문제가 Train/Val로 갈지 결정)
        random.shuffle(problem_ids)
        
        # 3. 분할 지점 계산
        split_idx = int(len(problem_ids) * (1 - validation_ratio))
        train_problem_ids = set(problem_ids[:split_idx])
        valid_problem_ids = set(problem_ids[split_idx:])
        
        logger.info(f"Train 문제: {len(train_problem_ids)}개, Validation 문제: {len(valid_problem_ids)}개")
        
        # 4. 원래 순서를 유지하며 필터링
        train_sets = []
        valid_sets = []
        
        for item in curated_sets:
            if item['problem_id'] in train_problem_ids:
                train_sets.append(item)
            else:
                valid_sets.append(item)
        
        logger.info(f"분할 완료: Train {len(train_sets)}개, Validation {len(valid_sets)}개")
        return train_sets, valid_sets
    
    def curate_data(
        self, 
        generated_data_path: str,
        output_dir: str,
        train_split: float = 0.8
    ) -> Dict[str, Any]:
        """
        전체 데이터 큐레이션 과정을 수행합니다.
        
        Args:
            generated_data_path: Stage 1 결과 파일 경로
            output_dir: 출력 디렉토리
            train_split: 훈련 데이터 비율
        
        Returns:
            생성된 파일 경로 딕셔너리
        """
        logger.info("데이터 큐레이션 시작")
        os.makedirs(output_dir, exist_ok=True)
        
        # 생성된 데이터 로드 (PyArrow를 사용하여 중첩 리스트 타입 처리)
        if HAS_PYARROW:
            try:
                # memory_map=False로 시도 (큰 파일의 경우 더 안정적)
                table = pq.read_table(generated_data_path, memory_map=False)
                generated_data = table.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception as e:
                logger.warning(f"PyArrow memory_map=False 실패: {e}, memory_map=True로 재시도...")
                try:
                    # memory_map=True로 재시도
                    table = pq.read_table(generated_data_path, memory_map=True)
                    generated_data = table.to_pandas(types_mapper=pd.ArrowDtype)
                except Exception as e2:
                    logger.warning(f"PyArrow types_mapper 사용 실패: {e2}, 기본 변환으로 재시도...")
                    # types_mapper 없이 시도
                    table = pq.read_table(generated_data_path, memory_map=False)
                    generated_data = table.to_pandas()
        else:
            generated_data = pd.read_parquet(generated_data_path)
        logger.info(f"생성된 데이터 로드: {len(generated_data)}개 응답, {generated_data['problem_id'].nunique()}개 문제")
        # 2GB+ 문자열 오류 (offset overflow) 방지를 위해
        # string[pyarrow] 타입을 large_string[pyarrow]로 즉시 변환
        logger.info("string 타입을 large_string으로 변환 중 (offset overflow 방지)...")
        string_cols = generated_data.select_dtypes(include=['string[pyarrow]']).columns
        
        if not string_cols.empty:
            logger.info(f"변환 대상 컬럼: {string_cols.to_list()}")
            for col in string_cols:
                try:
                    generated_data[col] = generated_data[col].astype('large_string[pyarrow]')
                except Exception as e:
                    logger.warning(f"'{col}' 컬럼 large_string 변환 실패: {e}")
        else:
            # types_mapper가 실패했거나 object 타입을 로드된 경우
            logger.info("pyarrow string 타입 컬럼이 없거나, object 타입으로 로드됨. object 타입 변환 시도...")
            object_cols = generated_data.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    # 'object' 컬럼이 실제 문자열 데이터인지 확인 (선택적)
                    if not generated_data[col].empty and isinstance(generated_data[col].dropna().iloc[0], str):
                        generated_data[col] = generated_data[col].astype('large_string[pyarrow]')
                        logger.info(f"'{col}' (object) 컬럼을 large_string으로 변환.")
                except Exception as e:
                    # 문자열이 아닌 object일 수 있으므로 경고만 하고 넘어감
                    logger.warning(f"'{col}' (object) 컬럼 변환 중 오류 발생 (무시): {e}")
        
        logger.info("large_string 변환 완료.")
        
        if self.strategy == "curriculum":
            # Curriculum: set_size는 나누기 2씩 감소, set_num은 통일
            set_sizes = []
            current_size = self.set_size
            while current_size >= 1:
                set_sizes.append(current_size)
                if current_size == 1:
                    break
                current_size = current_size // 2
            
            # set_num은 통일 (사용자 설정값 그대로 사용)
            set_num = self.num_sets_per_problem
            
            result_paths = {}
            
            logger.info(f"Curriculum 전략: set_size={set_sizes}, set_num={set_num} (통일)")
            
            for idx, set_size in enumerate(set_sizes):
                logger.info(f"Curriculum 데이터셋 생성: set_size={set_size}, set_num={set_num}")

                # 임시로 set_size 변경 (set_num은 변경하지 않음)
                original_set_size = self.set_size
                self.set_size = set_size

                # 각 set_size마다 다른 seed로 shuffle하여 서로 다른 solution 선택
                # set_size를 seed로 사용하여 각 set_size마다 다른 순서로 solution 선택
                shuffle_seed = set_size * 1000 + idx  # set_size와 인덱스를 조합하여 고유한 seed 생성

                # 모든 데이터로 응답 세트 생성 (shuffle_seed로 서로 다른 solution 선택)
                all_sets = self.create_response_sets(generated_data, num_sets=set_num, shuffle_seed=shuffle_seed)

                # 세트 기반 Hard/Easy 분류 (분포 저장 활성화)
                hard_sets, easy_sets = self.classify_hard_easy_sets(
                    all_sets,
                    save_distribution=True,
                    output_dir=output_dir,
                    suffix=f"_size_{set_size}"
                )

                # 큐레이션 전략 적용 (기본 전략)
                curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)

                # 훈련/검증 분할
                train_sets, validation_sets = self.split_train_validation(curated_sets, 1.0 - train_split)

                # 결과 저장 (중첩 데이터를 JSON으로 직렬화)
                train_sets_serialized = _serialize_nested_data(train_sets)
                validation_sets_serialized = _serialize_nested_data(validation_sets)
                train_df = pd.DataFrame(train_sets_serialized)
                validation_df = pd.DataFrame(validation_sets_serialized)

                train_path = os.path.join(output_dir, f"train_curated_size_{set_size}.parquet")
                validation_path = os.path.join(output_dir, f"validation_curated_size_{set_size}.parquet")
                
                train_df.to_parquet(train_path, index=False)
                validation_df.to_parquet(validation_path, index=False)
                
                result_paths[f"train_size_{set_size}"] = train_path
                result_paths[f"validation_size_{set_size}"] = validation_path
                
                logger.info(f"Curriculum 데이터셋 저장 완료: set_size={set_size}, set_num={set_num} (통일)")
                logger.info(f"  Train: {train_path} ({len(train_sets)}개 세트)")
                logger.info(f"  Validation: {validation_path} ({len(validation_sets)}개 세트)")
                
                # set_size 복원
                self.set_size = original_set_size
            
            return result_paths
        
        elif self.strategy == "multitask":
            # Multitask: 하나의 데이터셋에 여러 set_size 포함, set_num은 통일
            set_sizes = []
            current_size = self.set_size
            while current_size >= 1:
                set_sizes.append(current_size)
                if current_size == 1:
                    break
                current_size = current_size // 2
            
            # set_num은 통일 (사용자 설정값 그대로 사용)
            set_num = self.num_sets_per_problem
            
            all_train_sets = []
            all_validation_sets = []
            
            logger.info(f"Multitask 전략: set_size={set_sizes}, set_num={set_num} (통일)")
            
            original_set_size = self.set_size
            
            for idx, set_size in enumerate(set_sizes):
                logger.info(f"Multitask 데이터셋 생성 중: set_size={set_size}, set_num={set_num}")

                # 임시로 set_size 변경 (set_num은 변경하지 않음)
                self.set_size = set_size

                # 각 set_size마다 다른 seed로 shuffle하여 서로 다른 solution 선택
                # set_size를 seed로 사용하여 각 set_size마다 다른 순서로 solution 선택
                shuffle_seed = set_size * 1000 + idx  # set_size와 인덱스를 조합하여 고유한 seed 생성

                # 모든 데이터로 응답 세트 생성 (shuffle_seed로 서로 다른 solution 선택)
                all_sets = self.create_response_sets(generated_data, num_sets=set_num, shuffle_seed=shuffle_seed)

                # 세트 기반 Hard/Easy 분류 (분포 저장 활성화)
                hard_sets, easy_sets = self.classify_hard_easy_sets(
                    all_sets,
                    save_distribution=True,
                    output_dir=output_dir,
                    suffix=f"_size_{set_size}"
                )

                # 큐레이션 전략 적용 (기본 전략)
                curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)

                # 훈련/검증 분할
                train_sets, validation_sets = self.split_train_validation(curated_sets, 1.0 - train_split)

                # set_size 정보를 각 세트에 추가
                for train_set in train_sets:
                    train_set['set_size'] = set_size
                    all_train_sets.append(train_set)

                for val_set in validation_sets:
                    val_set['set_size'] = set_size
                    all_validation_sets.append(val_set)
            
            # set_size 복원
            self.set_size = original_set_size
            
            # 모든 set_size를 포함한 최종 데이터셋 저장 (중첩 데이터를 JSON으로 직렬화)
            all_train_sets_serialized = _serialize_nested_data(all_train_sets)
            all_validation_sets_serialized = _serialize_nested_data(all_validation_sets)
            train_df = pd.DataFrame(all_train_sets_serialized)
            validation_df = pd.DataFrame(all_validation_sets_serialized)
            
            train_path = os.path.join(output_dir, "train_curated_multitask.parquet")
            validation_path = os.path.join(output_dir, "validation_curated_multitask.parquet")
            
            train_df.to_parquet(train_path, index=False)
            validation_df.to_parquet(validation_path, index=False)
            
            logger.info(f"Multitask 데이터셋 저장 완료:")
            logger.info(f"  Train: {train_path} ({len(all_train_sets)}개 세트)")
            logger.info(f"  Validation: {validation_path} ({len(all_validation_sets)}개 세트)")
            
            return {
                "train": train_path,
                "validation": validation_path
            }
        
        else:
            # 기본 전략
            # 모든 데이터로 응답 세트 생성
            all_sets = self.create_response_sets(generated_data)

            # 세트 기반 Hard/Easy 분류 (분포 저장 활성화)
            hard_sets, easy_sets = self.classify_hard_easy_sets(
                all_sets,
                save_distribution=True,
                output_dir=output_dir
            )
            
            # 큐레이션 전략 적용
            curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)
            
            # 훈련/검증 분할
            train_sets, validation_sets = self.split_train_validation(curated_sets, 1.0 - train_split)
            
            # 결과 저장 (중첩 데이터를 JSON으로 직렬화)
            train_sets_serialized = _serialize_nested_data(train_sets)
            validation_sets_serialized = _serialize_nested_data(validation_sets)
            train_df = pd.DataFrame(train_sets_serialized)
            validation_df = pd.DataFrame(validation_sets_serialized)
            
            train_path = os.path.join(output_dir, "train_curated.parquet")
            validation_path = os.path.join(output_dir, "validation_curated.parquet")
            
            train_df.to_parquet(train_path, index=False)
            validation_df.to_parquet(validation_path, index=False)
            
            logger.info(f"큐레이션 완료: {train_path}, {validation_path}")
            
            return {
                "train": train_path,
                "validation": validation_path
            }

