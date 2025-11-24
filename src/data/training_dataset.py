"""
훈련용 데이터셋 클래스
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# PyArrow import (nested data 처리용)
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("PyArrow를 사용할 수 없습니다. nested data 로드에 문제가 있을 수 있습니다.")


class CuratedTrainingDataset(Dataset):
    """큐레이션된 훈련 데이터셋"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 큐레이션된 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """데이터를 로드합니다."""
        # 필요한 컬럼만 로드하여 메모리 최적화
        required_columns = ['prompt', 'problem_text', 'responses', 'confidence_scores', 'ground_truth']

        try:
            # PyArrow를 사용하여 nested data 처리
            if HAS_PYARROW:
                try:
                    # 먼저 전체 schema를 확인하여 존재하는 컬럼만 선택
                    schema = pq.read_schema(self.data_path)
                    available_columns = [col for col in required_columns if col in schema.names]

                    # 사용 가능한 컬럼만 읽기 (메모리 최적화)
                    table = pq.read_table(self.data_path, columns=available_columns, memory_map=False)
                    # types_mapper를 사용하여 Arrow 타입 유지
                    df = table.to_pandas(types_mapper=pd.ArrowDtype)
                except Exception as e1:
                    logger.warning(f"PyArrow 컬럼 필터링 실패: {e1}, 기본 변환으로 재시도...")
                    try:
                        # types_mapper 없이 시도
                        table = pq.read_table(self.data_path, memory_map=False)
                        df = table.to_pandas()
                    except Exception as e2:
                        logger.warning(f"PyArrow 기본 변환 실패: {e2}, pandas로 직접 로드 시도...")
                        df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_parquet(self.data_path)

            # 데이터가 비어있는지 확인
            if len(df) == 0:
                logger.warning(f"데이터셋이 비어있습니다: {self.data_path}")
                return []

            logger.info(f"데이터 로드 성공: {len(df)}개 샘플")
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}", exc_info=True)
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # 리스트 형태의 데이터 처리
        if isinstance(item.get('responses'), str):
            # 문자열로 저장된 경우 JSON 파싱 시도
            try:
                item['responses'] = json.loads(item['responses'])
            except (json.JSONDecodeError, TypeError) as e:
                # JSON 파싱 실패 시 안전하게 빈 리스트로 설정
                logger.warning(f"Failed to parse responses at index {idx} as JSON: {e}. Using empty list.")
                item['responses'] = []

        if isinstance(item.get('confidence_scores'), str):
            # 문자열로 저장된 경우 JSON 파싱 시도
            try:
                item['confidence_scores'] = json.loads(item['confidence_scores'])
            except (json.JSONDecodeError, TypeError) as e:
                # JSON 파싱 실패 시 안전하게 빈 딕셔너리로 설정
                logger.warning(f"Failed to parse confidence_scores at index {idx} as JSON: {e}. Using empty dict.")
                item['confidence_scores'] = {}

        return item


def create_training_dataloader(
    train_data_path: str,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    훈련용 데이터로더를 생성합니다.
    
    Args:
        train_data_path: 훈련 데이터 파일 경로
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수
    
    Returns:
        데이터로더
    """
    dataset = CuratedTrainingDataset(train_data_path)
    
    def collate_fn(batch):
        """배치 데이터를 정리하는 함수"""
        prompts = []
        responses = []
        confidence_scores = []
        ground_truths = []
        
        for item in batch:
            # 'prompt'가 있으면 우선 사용, 없으면 problem_text
            prompts.append(item.get('prompt', item.get('problem_text', '')))
            # 호환 필드: 없으면 빈 리스트/딕셔너리로 대체
            responses.append(item.get('responses', []))
            confidence_scores.append(item.get('confidence_scores', {}))
            ground_truths.append(item['ground_truth'])
        
        return {
            'prompts': prompts,
            'responses': responses,
            'confidence_scores': confidence_scores,
            'ground_truths': ground_truths
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def create_validation_dataloader(
    validation_data_path: str,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """
    검증용 데이터로더를 생성합니다.
    
    Args:
        validation_data_path: 검증 데이터 파일 경로
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수
    
    Returns:
        데이터로더
    """
    return create_training_dataloader(
        validation_data_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

