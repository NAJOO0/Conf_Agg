"""
훈련용 데이터셋 클래스
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
        try:
            df = pd.read_parquet(self.data_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # 리스트 형태의 데이터 처리
        if isinstance(item.get('responses'), str):
            # 문자열로 저장된 경우 파싱
            import ast
            try:
                item['responses'] = ast.literal_eval(item['responses'])
            except:
                item['responses'] = [item['responses']]
        
        if isinstance(item.get('confidence_scores'), str):
            # 문자열로 저장된 경우 파싱
            import ast
            try:
                item['confidence_scores'] = ast.literal_eval(item['confidence_scores'])
            except:
                item['confidence_scores'] = [{}]
        
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
            prompts.append(item['problem_text'])
            responses.append(item['responses'])
            confidence_scores.append(item['confidence_scores'])
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

