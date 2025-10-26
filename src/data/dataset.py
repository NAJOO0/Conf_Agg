"""
데이터셋 클래스들
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class RawDataset(Dataset):
    """원본 데이터셋 클래스"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 원본 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """데이터를 로드합니다."""
        try:
            if self.data_path.endswith('.jsonl'):
                data = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                return data
            elif self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
                return df.to_dict('records')
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {self.data_path}")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class GeneratedDataset(Dataset):
    """Stage 1 생성 결과 데이터셋 클래스"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 생성된 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """생성된 데이터를 로드합니다."""
        try:
            df = pd.read_parquet(self.data_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"생성된 데이터 로드 실패: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def get_problem_responses(self, problem_id: str) -> List[Dict[str, Any]]:
        """특정 문제의 모든 응답을 가져옵니다."""
        return [item for item in self.data if item.get('problem_id') == problem_id]


class CuratedDataset(Dataset):
    """Stage 2 큐레이션된 데이터셋 클래스"""
    
    def __init__(self, data_path: str, split: str = "train"):
        """
        Args:
            data_path: 큐레이션된 데이터 파일 경로
            split: 데이터 분할 ("train" 또는 "validation")
        """
        self.data_path = data_path
        self.split = split
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """큐레이션된 데이터를 로드합니다."""
        try:
            df = pd.read_parquet(self.data_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"큐레이션된 데이터 로드 실패: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # PyTorch 텐서로 변환
        if 'responses' in item and isinstance(item['responses'], list):
            item['responses'] = torch.tensor(item['responses'], dtype=torch.long)
        
        if 'confidence_scores' in item and isinstance(item['confidence_scores'], list):
            item['confidence_scores'] = torch.tensor(item['confidence_scores'], dtype=torch.float32)
        
        return item


class BenchmarkDataset(Dataset):
    """벤치마크 데이터셋 클래스"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 벤치마크 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """벤치마크 데이터를 로드합니다."""
        try:
            data = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            logger.error(f"벤치마크 데이터 로드 실패: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def get_problem_text(self, idx: int) -> str:
        """문제 텍스트를 가져옵니다."""
        item = self.data[idx]
        return item.get('problem', '')
    
    def get_answer(self, idx: int) -> str:
        """정답을 가져옵니다."""
        item = self.data[idx]
        return item.get('answer', '')

