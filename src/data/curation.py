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


class DataCurator:
    """데이터 큐레이션 클래스"""
    
    def __init__(
        self,
        strategy: str = "curriculum",
        easy_sample_percentage: int = 50,
        num_sets_per_problem: int = 4,
        set_size: int = 4,
        timeout: int = 30,
        confidence_key: str = "tail_confidence",
        fill_insufficient_with_sampling: bool = True,
        prompt_template: str = (
            "Given the following problem:\n{problem}\n"
            "and these solution attempts:\n{solutions}\n"
            "It is possible that any, all, or none of these solutions are correct or complete. Carefully review the\n"
            "provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining\n"
            "useful ideas—to produce a final, comprehensive, and correct solution to the problem."
        )
    ):
        """
        Args:
            strategy: 큐레이션 전략 (naive, curriculum, multitask)
            easy_sample_percentage: Easy 샘플 비율
            num_sets_per_problem: 문제당 세트 수
            set_size: 각 세트의 크기
            timeout: 검증 타임아웃
            confidence_key: 사용할 컨피던스 키 (default: "bottom_10_percent_confidence")
            fill_insufficient_with_sampling: 응답이 부족할 때 샘플링으로 채울지 여부 (default: False)
            prompt_template: 프롬프트 템플릿 문자열
        """
        self.strategy = strategy
        self.easy_sample_percentage = easy_sample_percentage
        self.num_sets_per_problem = num_sets_per_problem
        self.set_size = set_size
        self.verifier = MathVerifier(timeout=timeout)
        self.confidence_key = confidence_key
        self.fill_insufficient_with_sampling = fill_insufficient_with_sampling
        self.prompt_template = prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        # 시드 설정
        random.seed(42)
        np.random.seed(42)
    
    def classify_hard_easy_sets(
        self, 
        sets: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        세트 기반 Hard/Easy 분류를 수행합니다.
        각 세트의 정답률을 계산하여 Hard/Easy로 분류합니다.
        
        Args:
            sets: 생성된 세트 리스트
        
        Returns:
            (hard_sets, easy_sets) 튜플
        """
        logger.info("세트 기반 Hard/Easy 분류 시작")
        
        hard_sets = []
        easy_sets = []
        
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
            
            # 다수결이 맞으면 Easy, 아니면 Hard
            # (과반수 이상이 맞으면 Easy로 분류)
            if accuracy >= 0.5:
                easy_sets.append(set_info)
            else:
                hard_sets.append(set_info)
        
        logger.info(f"Hard 세트: {len(hard_sets)}개, Easy 세트: {len(easy_sets)}개")
        logger.info(f"Easy 비율: {len(easy_sets) / len(sets) * 100:.2f}%")
        
        return hard_sets, easy_sets
    
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
        data: pd.DataFrame,
        num_sets: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        응답을 세트로 나눕니다.
        
        Args:
            data: 문제별 응답 데이터
            num_sets: 각 문제당 생성할 세트 수 (None이면 num_sets_per_problem 사용)
        
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
            
            if len(responses) >= required_responses:
                # 충분한 응답이 있으면 정확히 필요한 만큼만 사용 (랜덤 샘플링)
                selected_responses = random.sample(responses, required_responses)
            else:
                # 응답이 부족한 경우
                if self.fill_insufficient_with_sampling:
                    # 샘플링으로 채우기 (중복 허용)
                    selected_responses = responses.copy()
                    while len(selected_responses) < required_responses:
                        selected_responses.append(random.choice(responses))
                    logger.info(
                        f"문제 {problem_id}: 응답 {len(responses)}개에서 샘플링으로 {required_responses}개 채움"
                    )
                else:
                    # 모두 사용 (이 경우 세트 수가 줄어듦)
                    selected_responses = responses
                    logger.warning(
                        f"문제 {problem_id}: 응답 {len(responses)}개가 부족합니다. "
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
                    conf_val = r.get(self.confidence_key)
                    solutions.append({
                        'content': content,
                        'final_answer': final_answer,
                        'confidence': {
                            'key': self.confidence_key,
                            'value': conf_val
                        }
                    })
                    selected_conf_values.append(conf_val)
                
                # 프롬프트용 solutions 텍스트 생성 (confidence 포함)
                lines = []
                for idx, s in enumerate(solutions, start=1):
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
                    enable_thinking=False,
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
                    'enable_thinking': False,
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
        
        기본 전략: Hard 전부 + Hard의 50%만큼 Easy
        
        Args:
            hard_sets: Hard 문제 세트들
            easy_sets: Easy 문제 세트들
        
        Returns:
            큐레이션된 세트들
        """
        logger.info(f"큐레이션 전략 적용: {self.strategy}")
        
        # 기본 전략: Hard 전부 + Hard의 50%만큼 Easy
        num_easy_to_add = int(len(hard_sets) * self.easy_sample_percentage / 100)
        if len(easy_sets) > 0:
            selected_easy = random.sample(easy_sets, min(num_easy_to_add, len(easy_sets)))
        else:
            selected_easy = []
        
        logger.info(f"Hard 세트: {len(hard_sets)}개, 선택된 Easy 세트: {len(selected_easy)}개")
        
        return hard_sets + selected_easy
    
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
            
            for set_size in set_sizes:
                logger.info(f"Curriculum 데이터셋 생성: set_size={set_size}, set_num={set_num}")
                
                # 임시로 set_size 변경 (set_num은 변경하지 않음)
                original_set_size = self.set_size
                self.set_size = set_size
                
                # 모든 데이터로 응답 세트 생성
                all_sets = self.create_response_sets(generated_data, num_sets=set_num)
                
                # 세트 기반 Hard/Easy 분류
                hard_sets, easy_sets = self.classify_hard_easy_sets(all_sets)
                
                # 큐레이션 전략 적용 (기본 전략)
                curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)
                
                # 훈련/검증 분할
                train_sets, validation_sets = self.split_train_validation(curated_sets, train_split)
                
                # 결과 저장
                train_df = pd.DataFrame(train_sets)
                validation_df = pd.DataFrame(validation_sets)
                
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
            
            for set_size in set_sizes:
                logger.info(f"Multitask 데이터셋 생성 중: set_size={set_size}, set_num={set_num}")
                
                # 임시로 set_size 변경 (set_num은 변경하지 않음)
                self.set_size = set_size
                
                # 모든 데이터로 응답 세트 생성
                all_sets = self.create_response_sets(generated_data, num_sets=set_num)
                
                # 세트 기반 Hard/Easy 분류
                hard_sets, easy_sets = self.classify_hard_easy_sets(all_sets)
                
                # 큐레이션 전략 적용 (기본 전략)
                curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)
                
                # 훈련/검증 분할
                train_sets, validation_sets = self.split_train_validation(curated_sets, train_split)
                
                # set_size 정보를 각 세트에 추가
                for train_set in train_sets:
                    train_set['set_size'] = set_size
                    all_train_sets.append(train_set)
                
                for val_set in validation_sets:
                    val_set['set_size'] = set_size
                    all_validation_sets.append(val_set)
            
            # set_size 복원
            self.set_size = original_set_size
            
            # 모든 set_size를 포함한 최종 데이터셋 저장
            train_df = pd.DataFrame(all_train_sets)
            validation_df = pd.DataFrame(all_validation_sets)
            
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
            
            # 세트 기반 Hard/Easy 분류
            hard_sets, easy_sets = self.classify_hard_easy_sets(all_sets)
            
            # 큐레이션 전략 적용
            curated_sets = self.apply_curation_strategy(hard_sets, easy_sets)
            
            # 훈련/검증 분할
            train_sets, validation_sets = self.split_train_validation(curated_sets, train_split)
            
            # 결과 저장
            train_df = pd.DataFrame(train_sets)
            validation_df = pd.DataFrame(validation_sets)
            
            train_path = os.path.join(output_dir, "train_curated.parquet")
            validation_path = os.path.join(output_dir, "validation_curated.parquet")
            
            train_df.to_parquet(train_path, index=False)
            validation_df.to_parquet(validation_path, index=False)
            
            logger.info(f"큐레이션 완료: {train_path}, {validation_path}")
            
            return {
                "train": train_path,
                "validation": validation_path
            }

