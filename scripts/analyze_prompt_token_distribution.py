#!/usr/bin/env python3
"""
Train/Validation 데이터의 Prompt Token Count 분포 분석

주요 기능:
1. Train 데이터의 prompt token count 분포 분석
2. Validation 데이터의 prompt token count 분포 분석
3. 전체 데이터의 prompt token count 분포 분석
4. 시각화 및 통계 출력
"""
import os
import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings
from typing import Optional, List, Tuple
import time

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer


# TRL imports (chat template 적용용)
try:
    from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, maybe_apply_chat_template
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logger.warning("TRL을 사용할 수 없습니다. chat template을 적용하지 않습니다.")

# PyArrow 가용성 확인
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False


def load_tokenizer(model_name: str):
    """
    Tokenizer 로드 및 chat template 설정
    
    Args:
        model_name: 모델 이름 또는 경로
        
    Returns:
        tokenizer
    """
    logger.info(f"Tokenizer 로드 중: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer 로드 완료")
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        raise
    
    # Chat template 설정 (stage3_train_2.py와 동일하게)
    if tokenizer.chat_template is None:
        if HAS_TRL:
            tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
            logger.info("📝 기본 chat template 적용 (SIMPLE_CHAT_TEMPLATE)")
        else:
            logger.warning("⚠️ TRL이 없어 chat template을 설정하지 않습니다.")
    else:
        logger.info("📝 모델에 이미 chat template이 설정되어 있습니다.")
    
    return tokenizer


def count_tokens_with_tokenizer(tokenizer, text: str, apply_chat_template: bool = False) -> int:
    """
    Tokenizer를 사용하여 정확한 토큰 수 계산
    
    Args:
        tokenizer: tokenizer 인스턴스
        text: 텍스트 문자열 (이미 chat template이 적용된 형태일 수 있음)
        apply_chat_template: chat template 적용 여부 (기본값: False)
                           curation.py에서 이미 적용했으므로 기본값은 False
        
    Returns:
        토큰 수
    """
    if pd.isna(text) or not text:
        return 0
    
    try:
        text_str = str(text)
        
        # chat template 적용 (필요한 경우에만)
        if apply_chat_template:
            text_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": text_str}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        
        # 토큰화
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text_str, add_special_tokens=False)
            return len(tokens)
        elif hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text_str)
            return len(tokens)
        else:
            return len(text_str.split())
    except Exception as e:
        logger.warning(f"토큰 수 계산 실패: {e}")
        return len(str(text).split())


def load_parquet_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parquet 파일을 안정적으로 로드
    
    Args:
        file_path: Parquet 파일 경로
    
    Returns:
        로드된 데이터프레임 또는 None
    """
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    
    if not file_path.endswith('.parquet'):
        logger.error(f"Parquet 파일이 아닙니다: {file_path}")
        return None
    
    logger.info(f"파일 로드 중: {file_path}")
    try:
        if HAS_PYARROW:
            try:
                table = pq.read_table(file_path, memory_map=False)
                df = table.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception as e:
                logger.warning(f"PyArrow memory_map=False 실패: {e}, memory_map=True로 재시도...")
                try:
                    table = pq.read_table(file_path, memory_map=True)
                    df = table.to_pandas(types_mapper=pd.ArrowDtype)
                except Exception as e2:
                    logger.warning(f"PyArrow types_mapper 사용 실패: {e2}, 기본 변환으로 재시도...")
                    table = pq.read_table(file_path, memory_map=False)
                    df = table.to_pandas()
        else:
            df = pd.read_parquet(file_path)

        logger.info(f"로드 완료: {len(df)}개 행")
        logger.info(f"컬럼: {df.columns.to_list()}")
        
        # string 타입을 large_string으로 변환 (offset overflow 방지)
        if HAS_PYARROW:
            try:
                logger.info("string 타입을 large_string으로 변환 중...")
                string_cols = df.select_dtypes(include=['string[pyarrow]']).columns
                if not string_cols.empty:
                    for col in string_cols:
                        try:
                            df[col] = df[col].astype('large_string[pyarrow]')
                        except Exception as ce:
                            logger.warning(f"'{col}' 컬럼 large_string 변환 실패: {ce}")
                else:
                    object_cols = df.select_dtypes(include=['object']).columns
                    for col in object_cols:
                        try:
                            if not df[col].empty and isinstance(df[col].dropna().iloc[0], str):
                                df[col] = df[col].astype('large_string[pyarrow]')
                        except Exception as oe:
                            logger.warning(f"'{col}' (object) 컬럼 변환 중 오류 (무시): {oe}")
            except Exception as e:
                logger.warning(f"large_string 변환 단계에서 경고: {e}")

        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
        return None


def calculate_prompt_token_counts(df: pd.DataFrame, tokenizer, apply_chat_template: bool = False) -> pd.DataFrame:
    """
    Prompt의 token count 계산
    
    Args:
        df: 데이터프레임
        tokenizer: tokenizer 인스턴스
        apply_chat_template: chat template 적용 여부 (기본값: False)
                           curation.py에서 이미 chat template을 적용했으므로 기본값은 False
        
    Returns:
        prompt_token_count 컬럼이 추가된 데이터프레임
    """
    if 'prompt' not in df.columns:
        logger.error("'prompt' 컬럼이 없습니다.")
        return df
    
    # 로깅
    if apply_chat_template:
        logger.info("Prompt token count 계산 중... (chat template 재적용)")
    else:
        logger.info("Prompt token count 계산 중... (이미 적용된 chat template 사용)")
    
    prompt_token_counts = []
    
    total = len(df)
    for idx, prompt in enumerate(df['prompt']):
        if idx % 100 == 0 and idx > 0:
            logger.info(f"진행 중: {idx}/{total} ({idx/total*100:.1f}%)")
        
        token_count = count_tokens_with_tokenizer(tokenizer, prompt, apply_chat_template=apply_chat_template)
        prompt_token_counts.append(token_count)
    
    df['prompt_token_count'] = prompt_token_counts
    logger.info("Prompt token count 계산 완료")
    
    return df


def print_statistics(token_counts: np.ndarray, dataset_name: str):
    """
    Token count 통계 출력
    
    Args:
        token_counts: 토큰 수 배열
        dataset_name: 데이터셋 이름
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {dataset_name} Prompt Token Count 통계 ===")
    logger.info(f"{'='*60}")
    
    non_zero_mask = token_counts > 0
    non_zero_counts = token_counts[non_zero_mask]
    
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"토큰이 있는 데이터 수: {non_zero_mask.sum()} ({non_zero_mask.sum()/len(token_counts)*100:.1f}%)")
    
    if len(non_zero_counts) > 0:
        logger.info(f"\n기본 통계:")
        logger.info(f"  평균 토큰 수 (전체): {token_counts.mean():.2f}")
        logger.info(f"  평균 토큰 수 (토큰 있음): {non_zero_counts.mean():.2f}")
        logger.info(f"  중앙값 (토큰 있음): {np.median(non_zero_counts):.2f}")
        logger.info(f"  표준편차 (토큰 있음): {non_zero_counts.std():.2f}")
        logger.info(f"  최소값 (토큰 있음): {non_zero_counts.min():.2f}")
        logger.info(f"  최대값 (토큰 있음): {non_zero_counts.max():.2f}")
        
        # 분위수
        percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
        logger.info(f"\n분위수 (토큰 있음):")
        for p in percentiles:
            logger.info(f"  {p}%: {np.percentile(non_zero_counts, p):.2f}")
        
        # 구간별 분포
        logger.info(f"\n토큰 수 구간별 분포:")
        max_val = non_zero_counts.max()
        if max_val < 1000:
            bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, float('inf')]
            labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                     '500-600', '600-700', '700-800', '800-900', '900+']
        elif max_val < 5000:
            bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, float('inf')]
            labels = ['0-500', '500-1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', 
                     '2.5K-3K', '3K-3.5K', '3.5K-4K', '4K+']
        else:
            bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
            labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', 
                     '5K-6K', '6K-7K', '7K-8K', '8K+']
        
        for i in range(len(bins)-1):
            if i < len(bins) - 1:
                count = ((non_zero_counts >= bins[i]) & (non_zero_counts < bins[i+1])).sum()
            else:
                count = (non_zero_counts >= bins[i]).sum()
            percentage = count / len(non_zero_counts) * 100 if len(non_zero_counts) > 0 else 0
            logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")


def visualize_distribution(
    train_counts: np.ndarray,
    valid_counts: np.ndarray,
    all_counts: np.ndarray,
    output_dir: str
):
    """
    Token count 분포 시각화
    
    Args:
        train_counts: Train 데이터 토큰 수 배열
        valid_counts: Validation 데이터 토큰 수 배열
        all_counts: 전체 데이터 토큰 수 배열
        output_dir: 출력 디렉토리
    """
    logger.info("시각화 생성 중...")
    
    try:
        # 1. 히스토그램 비교 (Train vs Validation)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Train 히스토그램
        ax = axes[0, 0]
        train_non_zero = train_counts[train_counts > 0]
        if len(train_non_zero) > 0:
            ax.hist(train_non_zero, bins=50, alpha=0.7, color='#4C78A8', edgecolor='black')
            ax.set_xlabel('Prompt Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Train Prompt Token Count Distribution\n(n={len(train_counts)}, mean={train_non_zero.mean():.1f})')
            ax.grid(True, alpha=0.3)
        
        # Validation 히스토그램
        ax = axes[0, 1]
        valid_non_zero = valid_counts[valid_counts > 0]
        if len(valid_non_zero) > 0:
            ax.hist(valid_non_zero, bins=50, alpha=0.7, color='#E45756', edgecolor='black')
            ax.set_xlabel('Prompt Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Validation Prompt Token Count Distribution\n(n={len(valid_counts)}, mean={valid_non_zero.mean():.1f})')
            ax.grid(True, alpha=0.3)
        
        # 전체 히스토그램
        ax = axes[1, 0]
        all_non_zero = all_counts[all_counts > 0]
        if len(all_non_zero) > 0:
            ax.hist(all_non_zero, bins=50, alpha=0.7, color='#54A24B', edgecolor='black')
            ax.set_xlabel('Prompt Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'All Prompt Token Count Distribution\n(n={len(all_counts)}, mean={all_non_zero.mean():.1f})')
            ax.grid(True, alpha=0.3)
        
        # 비교 Box Plot
        ax = axes[1, 1]
        data_to_plot = []
        labels = []
        if len(train_non_zero) > 0:
            data_to_plot.append(train_non_zero)
            labels.append('Train')
        if len(valid_non_zero) > 0:
            data_to_plot.append(valid_non_zero)
            labels.append('Validation')
        if len(all_non_zero) > 0:
            data_to_plot.append(all_non_zero)
            labels.append('All')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_ylabel('Prompt Token Count')
            ax.set_title('Prompt Token Count Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'prompt_token_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {plot_path}")
        
        # 2. CDF (Cumulative Distribution Function) 비교
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(train_non_zero) > 0:
            sorted_train = np.sort(train_non_zero)
            y_train = np.arange(1, len(sorted_train) + 1) / len(sorted_train)
            ax.plot(sorted_train, y_train, label='Train', linewidth=2, color='#4C78A8')
        
        if len(valid_non_zero) > 0:
            sorted_valid = np.sort(valid_non_zero)
            y_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
            ax.plot(sorted_valid, y_valid, label='Validation', linewidth=2, color='#E45756')
        
        if len(all_non_zero) > 0:
            sorted_all = np.sort(all_non_zero)
            y_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
            ax.plot(sorted_all, y_all, label='All', linewidth=2, color='#54A24B')
        
        ax.set_xlabel('Prompt Token Count')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function (CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'prompt_token_cdf.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"CDF 시각화 저장: {plot_path}")
        
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")


def main(
    train_path: str,
    validation_path: str,
    model_name: str,
    output_dir: Optional[str] = None,
    max_input_length: Optional[int] = None
):
    """
    메인 함수: Prompt Token Count 분포 분석
    
    Args:
        train_path: Train 데이터 파일 경로
        validation_path: Validation 데이터 파일 경로
        model_name: 모델 이름 (tokenizer 로드용)
        output_dir: 출력 디렉토리
        max_input_length: 최대 입력 길이 제한 (이 값을 넘는 인스턴스 제거)
    """
    logger.info("\n" + "="*60)
    logger.info("=== Prompt Token Count 분포 분석 ===")
    logger.info("="*60)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(train_path))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # 1. Tokenizer 로드
    try:
        tokenizer = load_tokenizer(model_name)
        # curation.py에서 이미 chat template을 적용했으므로 재적용하지 않음
        logger.info("✅ curation.py에서 이미 chat template이 적용된 prompt를 사용합니다.")
        logger.info("   토큰 수 계산 시 chat template을 재적용하지 않습니다.")
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        return
    
    # 2. 데이터 로드
    logger.info("\n데이터 로드 중...")
    train_df = load_parquet_file(train_path)
    if train_df is None:
        logger.error("Train 데이터 로드 실패")
        return
    
    valid_df = load_parquet_file(validation_path)
    if valid_df is None:
        logger.error("Validation 데이터 로드 실패")
        return
    
    # 3. Prompt token count 계산
    # curation.py에서 이미 chat template이 적용되었으므로 재적용하지 않음
    logger.info("\n" + "="*60)
    logger.info("Train 데이터 Prompt Token Count 계산 중...")
    train_df = calculate_prompt_token_counts(train_df, tokenizer, apply_chat_template=False)
    
    logger.info("\n" + "="*60)
    logger.info("Validation 데이터 Prompt Token Count 계산 중...")
    valid_df = calculate_prompt_token_counts(valid_df, tokenizer, apply_chat_template=False)
    
    # 3-1. max_input_length 제한 적용 (필터링)
    if max_input_length is not None:
        logger.info("\n" + "="*60)
        logger.info(f"Max Input Length 제한 적용: {max_input_length} 토큰")
        logger.info("="*60)
        
        train_before = len(train_df)
        valid_before = len(valid_df)
        
        train_df = train_df[train_df['prompt_token_count'] <= max_input_length].copy()
        valid_df = valid_df[valid_df['prompt_token_count'] <= max_input_length].copy()
        
        train_removed = train_before - len(train_df)
        valid_removed = valid_before - len(valid_df)
        total_removed = train_removed + valid_removed
        
        logger.info(f"Train: {train_removed}개 제거 ({train_removed/train_before*100:.2f}%)")
        logger.info(f"Validation: {valid_removed}개 제거 ({valid_removed/valid_before*100:.2f}%)")
        logger.info(f"전체: {total_removed}개 제거 ({(total_removed/(train_before+valid_before)*100):.2f}%)")
        logger.info(f"남은 인스턴스: Train {len(train_df)}개, Validation {len(valid_df)}개")
        
        #filtered train_df and valid_df 저장
        dir_name = '/mnt/data1/datasets/nlp/conf_agg/curated/'
        train_df.to_parquet(os.path.join(dir_name, 'train_filtered.parquet'))
        valid_df.to_parquet(os.path.join(dir_name, 'valid_filtered.parquet'))
        
    # 4. 통계 출력
    train_counts = train_df['prompt_token_count'].values
    valid_counts = valid_df['prompt_token_count'].values
    all_counts = np.concatenate([train_counts, valid_counts])
    
    print_statistics(train_counts, "Train")
    print_statistics(valid_counts, "Validation")
    print_statistics(all_counts, "All (Train + Validation)")
    
    # 5. 시각화
    visualize_distribution(train_counts, valid_counts, all_counts, output_dir)
    
    # 6. 결과를 CSV로 저장
    try:
        summary_data = {
            'dataset': ['Train', 'Validation', 'All'],
            'total_samples': [len(train_counts), len(valid_counts), len(all_counts)],
            'mean': [
                train_counts[train_counts > 0].mean() if len(train_counts[train_counts > 0]) > 0 else 0,
                valid_counts[valid_counts > 0].mean() if len(valid_counts[valid_counts > 0]) > 0 else 0,
                all_counts[all_counts > 0].mean() if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'median': [
                np.median(train_counts[train_counts > 0]) if len(train_counts[train_counts > 0]) > 0 else 0,
                np.median(valid_counts[valid_counts > 0]) if len(valid_counts[valid_counts > 0]) > 0 else 0,
                np.median(all_counts[all_counts > 0]) if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'std': [
                train_counts[train_counts > 0].std() if len(train_counts[train_counts > 0]) > 0 else 0,
                valid_counts[valid_counts > 0].std() if len(valid_counts[valid_counts > 0]) > 0 else 0,
                all_counts[all_counts > 0].std() if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'min': [
                train_counts[train_counts > 0].min() if len(train_counts[train_counts > 0]) > 0 else 0,
                valid_counts[valid_counts > 0].min() if len(valid_counts[valid_counts > 0]) > 0 else 0,
                all_counts[all_counts > 0].min() if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'max': [
                train_counts[train_counts > 0].max() if len(train_counts[train_counts > 0]) > 0 else 0,
                valid_counts[valid_counts > 0].max() if len(valid_counts[valid_counts > 0]) > 0 else 0,
                all_counts[all_counts > 0].max() if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'p95': [
                np.percentile(train_counts[train_counts > 0], 95) if len(train_counts[train_counts > 0]) > 0 else 0,
                np.percentile(valid_counts[valid_counts > 0], 95) if len(valid_counts[valid_counts > 0]) > 0 else 0,
                np.percentile(all_counts[all_counts > 0], 95) if len(all_counts[all_counts > 0]) > 0 else 0
            ],
            'p99': [
                np.percentile(train_counts[train_counts > 0], 99) if len(train_counts[train_counts > 0]) > 0 else 0,
                np.percentile(valid_counts[valid_counts > 0], 99) if len(valid_counts[valid_counts > 0]) > 0 else 0,
                np.percentile(all_counts[all_counts > 0], 99) if len(all_counts[all_counts > 0]) > 0 else 0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'prompt_token_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n통계 요약 저장: {summary_path}")
    except Exception as e:
        logger.warning(f"통계 요약 저장 실패: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 분석 완료!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Validation 데이터의 Prompt Token Count 분포 분석")
    parser.add_argument("--train-path", type=str, default="/mnt/data1/datasets/nlp/conf_agg/curated/train_curated.parquet",
                       help="Train 데이터 Parquet 파일 경로")
    parser.add_argument("--validation-path", type=str, default="/mnt/data1/datasets/nlp/conf_agg/curated/validation_curated.parquet",
                       help="Validation 데이터 Parquet 파일 경로")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B",
                       help="모델 이름 또는 경로 (tokenizer 로드용)")
    parser.add_argument("--output-dir", type=str, default="/mnt/data1/datasets/nlp/conf_agg/curated",
                       help="출력 디렉토리 (기본값: train 파일이 있는 디렉토리)")
    parser.add_argument("--max-input-length", type=int, default=8092,
                       help="최대 입력 길이 제한 (이 값을 넘는 인스턴스 제거, 예: 8092)")
    
    args = parser.parse_args()
    
    main(
        train_path=args.train_path,
        validation_path=args.validation_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_input_length=args.max_input_length
    )

