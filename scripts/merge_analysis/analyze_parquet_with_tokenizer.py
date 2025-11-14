#!/usr/bin/env python3
"""
단일 Parquet 파일을 분석하는 스크립트 (Tokenizer 사용)

주요 기능:
1. 전체에 대해서 token 수 분포에 따른 정확도, confidence 분석
2. Generated content의 token 수 분포 분석 (tokenizer 사용)
3. Problem 별로 token count 별 정확도, confidence 분석
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
from typing import Optional, Any
import time
from scipy import stats
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.math_verifier import MathVerifier
from src.data.confidence import ConfidenceCalculator

# 경고 무시 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Unsloth imports (tokenizer 사용)
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    logger.warning("Unsloth를 사용할 수 없습니다. tokenizer를 직접 로드합니다.")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers도 사용할 수 없습니다.")
        HAS_UNSLOTH = False


def load_tokenizer(model_name: str):
    """
    Tokenizer 로드 (stage3_train_2.py와 동일한 방식)
    
    Args:
        model_name: 모델 이름 또는 경로
        
    Returns:
        tokenizer
    """
    logger.info(f"Tokenizer 로드 중: {model_name}")
    
    if HAS_UNSLOTH:
        try:
            # Unsloth 방식으로 tokenizer 로드 (모델 없이 tokenizer만)
            # FastLanguageModel.from_pretrained는 모델과 tokenizer를 함께 반환하지만
            # 실제로는 tokenizer만 필요하므로 더 가벼운 방법 사용
            _, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,  # 최소값
                dtype=None,
                load_in_4bit=False,  # tokenizer만 필요하므로 모델 로드 안 함
                device_map=None,
                fast_inference=False,  # tokenizer만 필요
            )
            logger.info("✅ Tokenizer 로드 완료 (Unsloth)")
            return tokenizer
        except Exception as e:
            logger.warning(f"Unsloth로 tokenizer 로드 실패: {e}. transformers로 시도합니다.")
    
    # Fallback: transformers 직접 사용
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer 로드 완료 (transformers)")
        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        raise


def count_tokens_with_tokenizer(tokenizer, text: str) -> int:
    """
    Tokenizer를 사용하여 정확한 토큰 수 계산
    
    Args:
        tokenizer: tokenizer 인스턴스
        text: 텍스트 문자열
        
    Returns:
        토큰 수
    """
    if pd.isna(text) or not text:
        return 0
    
    try:
        text_str = str(text)
        # tokenizer.encode 또는 tokenizer.tokenize 사용
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text_str, add_special_tokens=False)
            return len(tokens)
        elif hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text_str)
            return len(tokens)
        else:
            # Fallback: 단어 수 추정
            return len(text_str.split())
    except Exception as e:
        logger.warning(f"토큰 수 계산 실패: {e}")
        # Fallback: 단어 수 추정
        return len(str(text).split())


def extract_content(text: str) -> str:
    """
    </think> 토큰 이후 값들 추출
    
    Args:
        text: generated_text
        
    Returns:
        </think> 이후 내용
    """
    if pd.isna(text):
        return ""
    
    text_str = str(text)
    marker = "</think>"
    
    marker_pos = text_str.find(marker)
    if marker_pos == -1:
        return ""
    
    # 마커 이후 텍스트 추출
    content = text_str[marker_pos + len(marker):].strip()
    return content


def analyze_generated_content_tokens(df: pd.DataFrame, tokenizer, output_dir: str = None) -> pd.DataFrame:
    """
    Generated content의 token 수 분포 분석 (tokenizer 사용)
    
    Args:
        df: 데이터프레임
        tokenizer: tokenizer 인스턴스
        output_dir: 출력 디렉토리
        
    Returns:
        content_token_count 컬럼이 추가된 데이터프레임
    """
    logger.info("\n" + "="*50)
    logger.info("=== Generated Content Token 수 분포 분석 (Tokenizer 사용) ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return df
    
    # content 추출 (없으면 생성)
    if 'content' not in df.columns:
        logger.info("content 컬럼이 없어 추출합니다...")
        df['content'] = df['generated_text'].apply(extract_content)
    
    # tokenizer를 사용하여 정확한 토큰 수 계산
    logger.info("Tokenizer를 사용하여 content 토큰 수 계산 중...")
    content_token_counts = []
    
    for idx, content in enumerate(df['content']):
        if idx % 100 == 0 and idx > 0:
            logger.info(f"진행 중: {idx}/{len(df)}")
        
        token_count = count_tokens_with_tokenizer(tokenizer, content)
        content_token_counts.append(token_count)
    
    df['content_token_count'] = content_token_counts
    
    # 통계 출력
    token_counts = np.array(content_token_counts)
    non_zero_mask = token_counts > 0
    non_zero_counts = token_counts[non_zero_mask]
    
    logger.info(f"\n=== Generated Content Token 수 통계 ===")
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"토큰이 있는 데이터 수: {non_zero_mask.sum()} ({non_zero_mask.sum()/len(token_counts)*100:.1f}%)")
    
    if len(non_zero_counts) > 0:
        logger.info(f"평균 토큰 수 (전체): {token_counts.mean():.2f}")
        logger.info(f"평균 토큰 수 (토큰 있음): {non_zero_counts.mean():.2f}")
        logger.info(f"중앙값 (토큰 있음): {np.median(non_zero_counts):.2f}")
        logger.info(f"표준편차 (토큰 있음): {non_zero_counts.std():.2f}")
        logger.info(f"최소값 (토큰 있음): {non_zero_counts.min():.2f}")
        logger.info(f"최대값 (토큰 있음): {non_zero_counts.max():.2f}")
        
        # 분위수
        percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
        logger.info("\n분위수 (토큰 있음):")
        for p in percentiles:
            logger.info(f"  {p}%: {np.percentile(non_zero_counts, p):.2f}")
        
        # 구간별 분포
        logger.info("\n토큰 수 구간별 분포:")
        max_val = non_zero_counts.max()
        if max_val < 500:
            bins = [0, 50, 100, 150, 200, 250, 300, 350, float('inf')]
            labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
        elif max_val < 2000:
            bins = [0, 200, 400, 600, 800, 1000, 1200, 1500, float('inf')]
            labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000-1200', '1200-1500', '1500+']
        else:
            bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, float('inf')]
            labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K-6K', '6K-7K', '7K+']
        
        for i in range(len(bins)-1):
            if i < len(bins) - 1:
                count = ((non_zero_counts >= bins[i]) & (non_zero_counts < bins[i+1])).sum()
            else:
                count = (non_zero_counts >= bins[i]).sum()
            percentage = count / len(non_zero_counts) * 100 if len(non_zero_counts) > 0 else 0
            logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")
    
    # 시각화: Content Token 수 분포 히스토그램
    try:
        if len(non_zero_counts) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 히스토그램
            ax = axes[0]
            ax.hist(non_zero_counts, bins=50, alpha=0.7, color='#4C78A8', edgecolor='black')
            ax.set_xlabel('Content Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Generated Content Token Count')
            ax.grid(True, alpha=0.3)
            
            # Box plot
            ax = axes[1]
            ax.boxplot(non_zero_counts, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='#4C78A8', alpha=0.7))
            ax.set_ylabel('Content Token Count')
            ax.set_title('Box Plot of Generated Content Token Count')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'generated_content_token_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {plot_path}")
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")
    
    return df


def analyze_generated_text_tokens(df: pd.DataFrame, tokenizer, output_dir: str = None) -> pd.DataFrame:
    """
    Generated text 전체의 token 수 분포 분석 (tokenizer 사용)
    
    Args:
        df: 데이터프레임
        tokenizer: tokenizer 인스턴스
        output_dir: 출력 디렉토리
        
    Returns:
        generated_text_token_count 컬럼이 추가된 데이터프레임
    """
    logger.info("\n" + "="*50)
    logger.info("=== Generated Text 전체 Token 수 분포 분석 (Tokenizer 사용) ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return df
    
    # tokenizer를 사용하여 정확한 토큰 수 계산
    logger.info("Tokenizer를 사용하여 generated_text 전체 토큰 수 계산 중...")
    generated_text_token_counts = []
    
    for idx, generated_text in enumerate(df['generated_text']):
        if idx % 100 == 0 and idx > 0:
            logger.info(f"진행 중: {idx}/{len(df)}")
        
        token_count = count_tokens_with_tokenizer(tokenizer, generated_text)
        generated_text_token_counts.append(token_count)
    
    df['generated_text_token_count'] = generated_text_token_counts
    
    # 통계 출력
    token_counts = np.array(generated_text_token_counts)
    non_zero_mask = token_counts > 0
    non_zero_counts = token_counts[non_zero_mask]
    
    logger.info(f"\n=== Generated Text 전체 Token 수 통계 ===")
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"토큰이 있는 데이터 수: {non_zero_mask.sum()} ({non_zero_mask.sum()/len(token_counts)*100:.1f}%)")
    
    if len(non_zero_counts) > 0:
        logger.info(f"평균 토큰 수 (전체): {token_counts.mean():.2f}")
        logger.info(f"평균 토큰 수 (토큰 있음): {non_zero_counts.mean():.2f}")
        logger.info(f"중앙값 (토큰 있음): {np.median(non_zero_counts):.2f}")
        logger.info(f"표준편차 (토큰 있음): {non_zero_counts.std():.2f}")
        logger.info(f"최소값 (토큰 있음): {non_zero_counts.min():.2f}")
        logger.info(f"최대값 (토큰 있음): {non_zero_counts.max():.2f}")
        
        # 분위수
        percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
        logger.info("\n분위수 (토큰 있음):")
        for p in percentiles:
            logger.info(f"  {p}%: {np.percentile(non_zero_counts, p):.2f}")
        
        # 구간별 분포
        logger.info("\n토큰 수 구간별 분포:")
        max_val = non_zero_counts.max()
        if max_val < 1000:
            bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, float('inf')]
            labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800-900', '900+']
        elif max_val < 5000:
            bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, float('inf')]
            labels = ['0-500', '500-1K', '1K-1.5K', '1.5K-2K', '2K-2.5K', '2.5K-3K', '3K-3.5K', '3.5K-4K', '4K+']
        else:
            bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
            labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K-6K', '6K-7K', '7K-8K', '8K+']
        
        for i in range(len(bins)-1):
            if i < len(bins) - 1:
                count = ((non_zero_counts >= bins[i]) & (non_zero_counts < bins[i+1])).sum()
            else:
                count = (non_zero_counts >= bins[i]).sum()
            percentage = count / len(non_zero_counts) * 100 if len(non_zero_counts) > 0 else 0
            logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")
    
    # 시각화: Generated Text Token 수 분포 히스토그램
    try:
        if len(non_zero_counts) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 히스토그램
            ax = axes[0]
            ax.hist(non_zero_counts, bins=50, alpha=0.7, color='#E45756', edgecolor='black')
            ax.set_xlabel('Generated Text Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Generated Text Token Count (Full)')
            ax.grid(True, alpha=0.3)
            
            # Box plot
            ax = axes[1]
            ax.boxplot(non_zero_counts, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='#E45756', alpha=0.7))
            ax.set_ylabel('Generated Text Token Count')
            ax.set_title('Box Plot of Generated Text Token Count (Full)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'generated_text_token_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"시각화 저장: {plot_path}")
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")
    
    return df


def analyze_accuracy_by_token_count(df: pd.DataFrame, token_col: str, output_dir: str = None, num_bins: int = 10) -> None:
    """
    Token 수 분포에 따른 정확도, confidence 분석
    
    Args:
        df: 데이터프레임
        token_col: 토큰 수 컬럼명
        output_dir: 출력 디렉토리
        num_bins: 구간 개수
    """
    logger.info("\n" + "="*50)
    logger.info(f"=== {token_col} 분포별 정답률 및 Confidence 분석 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 정답 컬럼 확인 (MathVerifier 우선)
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.error("정답 컬럼(is_correct 또는 is_correct_math_verifier)이 없습니다.")
        return
    
    # confidence score 컬럼들 확인
    confidence_columns = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence',
        # 'lowest_group_confidence',
        # 'top_10_percent_confidence',
        # 'highest_group_confidence'
    ]
    available_confidence_cols = [col for col in confidence_columns if col in df.columns]
    
    # 유효 행 필터
    valid_mask = df[token_col].notna() & df[correct_col].notna()
    for col in available_confidence_cols:
        valid_mask = valid_mask & df[col].notna()
    
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        logger.error("분석 가능한 데이터가 없습니다.")
        return
    
    # ArrowExtensionArray 방지를 위해 numpy로 명시 변환
    token_values = valid_df[token_col].to_numpy(dtype=float)
    correct_values = valid_df[correct_col].astype(int).values
    
    # confidence 값들 준비
    confidence_dict = {}
    for col in available_confidence_cols:
        confidence_dict[col] = valid_df[col].to_numpy(dtype=float)
    
    # 분위수 기반 구간 경계 생성
    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.unique(np.percentile(token_values, percentiles))
    if len(edges) < 2:
        logger.error("구간을 생성할 수 없습니다 (토큰 값 다양성 부족).")
        return
    
    # 각 구간별 정답률 및 confidence 계산
    bin_accs = []
    bin_counts = []
    bin_labels = []
    bin_confidences = {col: [] for col in available_confidence_cols}
    
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        # 마지막 구간은 right 포함
        if i < len(edges) - 2:
            mask = (token_values >= left) & (token_values < right)
        else:
            mask = (token_values >= left) & (token_values <= right)
        count = mask.sum()
        if count > 0:
            acc = correct_values[mask].mean()
            bin_accs.append(float(acc))
            bin_counts.append(int(count))
            bin_labels.append(f"{int(left)}-{int(right)}")
            
            # 각 confidence 점수 평균 계산
            for col in available_confidence_cols:
                conf_mean = confidence_dict[col][mask].mean()
                bin_confidences[col].append(float(conf_mean))
    
    if not bin_accs:
        logger.error("구간별 결과가 비어 있습니다.")
        return
    
    # 로그 출력
    logger.info(f"총 유효 샘플: {len(valid_df)}")
    if available_confidence_cols:
        logger.info(f"Confidence 컬럼: {', '.join(available_confidence_cols)}")
        logger.info("\n구간별 정답률 및 Confidence 점수:")
        for idx, (lbl, acc, cnt) in enumerate(zip(bin_labels, bin_accs, bin_counts)):
            conf_strs = []
            for col in available_confidence_cols:
                if col in bin_confidences and idx < len(bin_confidences[col]):
                    conf_val = bin_confidences[col][idx]
                    conf_strs.append(f"{col}={conf_val:.4f}")
            conf_str = ", ".join(conf_strs) if conf_strs else ""
            logger.info(f"  [{lbl}] 정답률: {acc:.3f} ({cnt}개), {conf_str}")
    else:
        logger.info("\n구간별 정답률:")
        for lbl, acc, cnt in zip(bin_labels, bin_accs, bin_counts):
            logger.info(f"  [{lbl}] 정답률: {acc:.3f} ({cnt}개)")
    
    # 시각화
    try:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        x = np.arange(len(bin_labels))
        ax1.bar(x, bin_accs, color='#4C78A8', alpha=0.85, label='Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel(f'{token_col} Bins')
        ax1.set_title(f'Accuracy by {token_col} (quantile bins)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, rotation=30, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 보조축: 샘플 수
        ax2 = ax1.twinx()
        ax2.plot(x, bin_counts, color='#F58518', marker='o', linewidth=2, label='Count')
        ax2.set_ylabel('Count')
        
        # 범례
        lines, labels = [], []
        for ax in [ax1, ax2]:
            line, label = ax.get_legend_handles_labels()
            lines += line
            labels += label
        ax1.legend(lines, labels, loc='upper right')
        
        plot_filename = f'accuracy_by_{token_col}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {plot_path}")
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")


def analyze_problem_level_token_accuracy(df: pd.DataFrame, token_col: str, output_dir: str = None, num_bins: int = 10) -> None:
    """
    Problem별로 각 inference의 token count에 따른 정확도, confidence 분석
    
    각 문제에 대해 n번의 inference가 있고, 각 inference마다 token count가 다름.
    이 token count와 정답률/confidence의 관계를 분석합니다.
    
    Args:
        df: 데이터프레임
        token_col: 토큰 수 컬럼명 (content_token_count 또는 output_token_count)
        output_dir: 출력 디렉토리
        num_bins: 구간 개수 (문제 내에서 토큰 수 구간화용)
    """
    logger.info("\n" + "="*50)
    logger.info(f"=== Problem별 {token_col}에 따른 Inference 정답률 및 Confidence 분석 ===")
    logger.info("="*50)
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if 'problem_id' not in df.columns:
        logger.error("problem_id 컬럼이 없습니다.")
        return
    
    # 정답 컬럼 확인
    if 'is_correct_math_verifier' in df.columns:
        correct_col = 'is_correct_math_verifier'
    elif 'is_correct' in df.columns:
        correct_col = 'is_correct'
    else:
        logger.error("정답 컬럼(is_correct 또는 is_correct_math_verifier)이 없습니다.")
        return
    
    # confidence score 컬럼들 확인
    confidence_columns = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence',
        # 'lowest_group_confidence',
        # 'top_10_percent_confidence',
        # 'highest_group_confidence'
    ]
    available_confidence_cols = [col for col in confidence_columns if col in df.columns]
    
    # 유효 행 필터
    valid_mask = df[token_col].notna() & df[correct_col].notna()
    for col in available_confidence_cols:
        valid_mask = valid_mask & df[col].notna()
    
    valid_df = df[valid_mask].copy()
    if len(valid_df) == 0:
        logger.error("분석 가능한 데이터가 없습니다.")
        return
    
    logger.info(f"총 유효 샘플: {len(valid_df)}")
    logger.info(f"총 문제 수: {valid_df['problem_id'].nunique()}")
    
    # 문제별 분석: 토큰 수와 정답률/confidence의 상관관계
    problem_correlations = []
    problem_stats = []
    
    for problem_id, group in valid_df.groupby('problem_id'):
        if len(group) < 2:  # 최소 2개 이상의 inference가 있어야 분석 가능
            continue
        
        problem_token_values = group[token_col].to_numpy(dtype=float)
        problem_correct_values = group[correct_col].astype(int).values
        
        # 토큰 수와 정답률의 상관관계
        try:
            # 분산이 0이면 상관관계 계산 불가
            if len(problem_token_values) < 2 or np.std(problem_token_values) == 0:
                # 토큰 수가 모두 같은 경우, 전체 평균 정답률만 기록
                corr_coef = 0.0
                p_value = 1.0
            else:
                corr_coef, p_value = stats.pearsonr(problem_token_values, problem_correct_values)
            
            # 문제 내에서 토큰 수 구간별 정답률 계산
            token_min, token_max = problem_token_values.min(), problem_token_values.max()
            # 토큰 수가 모두 같은 경우에도 분석 진행
            if token_max == token_min:
                # 토큰 수가 모두 같은 경우, 단일 구간으로 처리
                bin_accs = [float(problem_correct_values.mean())]
                bin_tokens = [float(token_min)]
                bin_counts = [len(problem_token_values)]
                bin_confidences = {col: [float(group[col].mean())] for col in available_confidence_cols}
            else:
                # 문제 내 토큰 수 범위를 기준으로 구간 생성
                token_edges = np.linspace(token_min, token_max, num_bins + 1)
                bin_accs = []
                bin_tokens = []
                bin_counts = []
                bin_confidences = {col: [] for col in available_confidence_cols}
                
                for i in range(len(token_edges) - 1):
                    left = token_edges[i]
                    right = token_edges[i + 1]
                    if i < len(token_edges) - 2:
                        mask = (problem_token_values >= left) & (problem_token_values < right)
                    else:
                        mask = (problem_token_values >= left) & (problem_token_values <= right)
                    
                    count = mask.sum()
                    if count > 0:
                        acc = problem_correct_values[mask].mean()
                        bin_accs.append(float(acc))
                        bin_tokens.append((left + right) / 2)
                        bin_counts.append(int(count))
                        
                        for col in available_confidence_cols:
                            conf_mean = group[col].to_numpy(dtype=float)[mask].mean()
                            bin_confidences[col].append(float(conf_mean))
                
            problem_correlations.append({
                'problem_id': problem_id,
                'correlation': corr_coef,
                'p_value': p_value,
                'num_inferences': len(group),
                'avg_accuracy': problem_correct_values.mean(),
                'token_mean': problem_token_values.mean(),
                'token_std': problem_token_values.std() if len(problem_token_values) > 1 else 0.0,
            })
            
            problem_stats.append({
                'problem_id': problem_id,
                'bin_accs': bin_accs,
                'bin_tokens': bin_tokens,
                'bin_counts': bin_counts,
                'bin_confidences': bin_confidences,
                'correlation': corr_coef,
                'avg_accuracy': problem_correct_values.mean(),
            })
        except Exception as e:
            logger.warning(f"문제 {problem_id} 분석 중 오류: {e}")
            continue
    
    if not problem_correlations:
        logger.error("분석 가능한 문제가 없습니다.")
        return
    
    # 통계 요약
    correlations = np.array([p['correlation'] for p in problem_correlations])
    p_values = np.array([p['p_value'] for p in problem_correlations])
    
    # 유의미한 상관관계 (p < 0.05)
    significant_mask = p_values < 0.05
    positive_corr = correlations > 0
    negative_corr = correlations < 0
    
    logger.info(f"\n=== Problem별 {token_col}와 정답률 상관관계 통계 ===")
    logger.info(f"총 분석 문제 수: {len(problem_correlations)}")
    logger.info(f"\n상관계수 통계:")
    logger.info(f"  평균: {correlations.mean():.4f}")
    logger.info(f"  중앙값: {np.median(correlations):.4f}")
    logger.info(f"  표준편차: {correlations.std():.4f}")
    logger.info(f"  최소값: {correlations.min():.4f}")
    logger.info(f"  최대값: {correlations.max():.4f}")
    
    logger.info(f"\n유의미한 상관관계 (p < 0.05):")
    logger.info(f"  전체: {significant_mask.sum()}개 ({significant_mask.sum()/len(problem_correlations)*100:.1f}%)")
    logger.info(f"  양의 상관관계: {(significant_mask & positive_corr).sum()}개")
    logger.info(f"  음의 상관관계: {(significant_mask & negative_corr).sum()}개")
    
    logger.info(f"\n전체 문제:")
    logger.info(f"  양의 상관관계: {positive_corr.sum()}개 ({positive_corr.sum()/len(problem_correlations)*100:.1f}%)")
    logger.info(f"  음의 상관관계: {negative_corr.sum()}개 ({negative_corr.sum()/len(problem_correlations)*100:.1f}%)")
    logger.info(f"  상관관계 없음 (|r| < 0.1): {(np.abs(correlations) < 0.1).sum()}개")
    
    # 대표 문제 샘플링 (상관관계가 강한 문제들)
    sorted_by_corr = sorted(problem_stats, key=lambda x: abs(x['correlation']), reverse=True)
    
    # 양의 상관관계가 강한 문제 (상위 5개)
    positive_problems = [p for p in sorted_by_corr if p['correlation'] > 0.3][:5]
    # 음의 상관관계가 강한 문제 (상위 5개)
    negative_problems = [p for p in sorted_by_corr if p['correlation'] < -0.3][:5]
    # 상관관계가 약한 문제 (랜덤 3개)
    weak_problems = [p for p in sorted_by_corr if abs(p['correlation']) < 0.1]
    if len(weak_problems) > 3:
        import random
        weak_problems = random.sample(weak_problems, 3)
    else:
        weak_problems = weak_problems[:3]
    
    logger.info(f"\n=== 대표 문제 상세 분석 ===")
    
    # 양의 상관관계 문제
    if positive_problems:
        logger.info(f"\n양의 상관관계가 강한 문제 (토큰 수↑ → 정답률↑):")
        for p in positive_problems:
            logger.info(f"  Problem {p['problem_id']}: r={p['correlation']:.3f}, 평균 정답률={p['avg_accuracy']:.3f}")
            logger.info(f"    토큰 수 구간별 정답률:")
            for tok, acc, cnt in zip(p['bin_tokens'], p['bin_accs'], p['bin_counts']):
                logger.info(f"      토큰 {tok:.0f}: {acc:.3f} ({cnt}개)")
    
    # 음의 상관관계 문제
    if negative_problems:
        logger.info(f"\n음의 상관관계가 강한 문제 (토큰 수↑ → 정답률↓):")
        for p in negative_problems:
            logger.info(f"  Problem {p['problem_id']}: r={p['correlation']:.3f}, 평균 정답률={p['avg_accuracy']:.3f}")
            logger.info(f"    토큰 수 구간별 정답률:")
            for tok, acc, cnt in zip(p['bin_tokens'], p['bin_accs'], p['bin_counts']):
                logger.info(f"      토큰 {tok:.0f}: {acc:.3f} ({cnt}개)")
    
    # 약한 상관관계 문제
    if weak_problems:
        logger.info(f"\n상관관계가 약한 문제 (토큰 수와 정답률 무관):")
        for p in weak_problems:
            logger.info(f"  Problem {p['problem_id']}: r={p['correlation']:.3f}, 평균 정답률={p['avg_accuracy']:.3f}")
    
    # 시각화
    try:
        # 1. 상관계수 분포 히스토그램
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 상관계수 분포
        ax = axes[0, 0]
        ax.hist(correlations, bins=30, alpha=0.7, color='#4C78A8', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='r=0')
        ax.set_xlabel(f'Correlation Coefficient ({token_col} vs Accuracy)')
        ax.set_ylabel('Number of Problems')
        ax.set_title('Distribution of Correlation Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 상관계수 vs 평균 정답률
        ax = axes[0, 1]
        avg_accs = np.array([p['avg_accuracy'] for p in problem_correlations])
        ax.scatter(correlations, avg_accs, alpha=0.5, s=20)
        ax.set_xlabel(f'Correlation Coefficient')
        ax.set_ylabel('Problem Average Accuracy')
        ax.set_title('Correlation vs Average Accuracy')
        ax.grid(True, alpha=0.3)
        
        # 유의미한 상관관계 비율
        ax = axes[1, 0]
        labels = ['Significant\n(p<0.05)', 'Not Significant']
        sizes = [significant_mask.sum(), (~significant_mask).sum()]
        colors = ['#54A24B', '#E45756']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Significant Correlation Ratio')
        
        # 양/음의 상관관계 비율
        ax = axes[1, 1]
        labels = ['Positive\n(r>0)', 'Negative\n(r<0)', 'No\n(|r|<0.1)']
        sizes = [
            (correlations > 0.1).sum(),
            (correlations < -0.1).sum(),
            (np.abs(correlations) < 0.1).sum()
        ]
        colors = ['#4C78A8', '#F58518', '#E45756']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Positive/Negative Correlation Ratio')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'problem_token_correlation_{token_col}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"시각화 저장: {plot_path}")
        
        # 2. 대표 문제들의 scatter plot (토큰 수 vs 정답률)
        if positive_problems or negative_problems or weak_problems:
            n_examples = len(positive_problems) + len(negative_problems) + len(weak_problems)
            if n_examples > 0:
                fig, axes = plt.subplots(1, min(n_examples, 6), figsize=(4*min(n_examples, 6), 4))
                if n_examples == 1:
                    axes = [axes]
                
                idx = 0
                for p in (positive_problems[:2] + negative_problems[:2] + weak_problems[:2]):
                    if idx >= len(axes):
                        break
                    
                    ax = axes[idx]
                    problem_group = valid_df[valid_df['problem_id'] == p['problem_id']]
                    tokens = problem_group[token_col].values
                    accs = problem_group[correct_col].astype(int).values
                    
                    ax.scatter(tokens, accs, alpha=0.6, s=30)
                    ax.set_xlabel(f'{token_col}')
                    ax.set_ylabel('Is Correct')
                    ax.set_title(f'Problem {p["problem_id"]}\nr={p["correlation"]:.3f}')
                    ax.grid(True, alpha=0.3)
                    idx += 1
                
                # 남은 subplot 제거
                for i in range(idx, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'problem_token_examples_{token_col}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"대표 문제 시각화 저장: {plot_path}")
        
    except Exception as e:
        logger.warning(f"시각화 생성/저장 실패: {e}")


def load_single_file(file_path: str) -> pd.DataFrame:
    """
    단일 Parquet 파일을 안정적으로 로드 (pyarrow 활용 + large_string 변환)
    
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
                # memory_map=False 우선 (큰 파일에서 안정적)
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
        try:
            size_mb = os.path.getsize(file_path) / (1024 ** 2)
            logger.info(f"파일 크기: {size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"파일 크기 확인 실패: {e}")

        # 문자열 offset overflow 방지를 위한 large_string 변환
        try:
            if HAS_PYARROW:
                logger.info("string 타입을 large_string으로 변환 중 (offset overflow 방지)...")
                string_cols = df.select_dtypes(include=['string[pyarrow]']).columns
                if not string_cols.empty:
                    logger.info(f"변환 대상 컬럼: {string_cols.to_list()}")
                    for col in string_cols:
                        try:
                            df[col] = df[col].astype('large_string[pyarrow]')
                        except Exception as ce:
                            logger.warning(f"'{col}' 컬럼 large_string 변환 실패: {ce}")
                else:
                    logger.info("pyarrow string 타입 컬럼 없음 또는 object로 로드됨. object 타입 변환 시도...")
                    object_cols = df.select_dtypes(include=['object']).columns
                    for col in object_cols:
                        try:
                            if not df[col].empty and isinstance(df[col].dropna().iloc[0], str):
                                df[col] = df[col].astype('large_string[pyarrow]')
                                logger.info(f"'{col}' (object) → large_string 변환")
                        except Exception as oe:
                            logger.warning(f"'{col}' (object) 컬럼 변환 중 오류 (무시): {oe}")
                logger.info("large_string 변환 완료.")
        except Exception as e:
            logger.warning(f"large_string 변환 단계에서 경고: {e}")

        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
        return None


def _normalize_logprobs(lp):
    """logprobs를 정규화하여 numpy array로 변환"""
    if lp is None:
        return None
    
    if isinstance(lp, np.ndarray):
        # 이미 numpy array인 경우
        if lp.ndim == 2:
            return lp
        elif lp.ndim == 1:
            return lp.reshape(-1, 1)
        else:
            return lp.tolist()
    
    if isinstance(lp, list):
        # 리스트인 경우 numpy array로 변환
        try:
            # 각 토큰의 logprobs를 numpy array로 변환
            normalized = []
            for tok in lp:
                if isinstance(tok, (list, np.ndarray)):
                    tok_arr = np.array(tok, dtype=np.float32)
                    normalized.append(tok_arr)
                else:
                    normalized.append(np.array([tok], dtype=np.float32))
            
            # 길이가 같은 경우에만 2D array로 변환 가능
            if len(normalized) > 0:
                lengths = [len(x) for x in normalized]
                if len(set(lengths)) == 1:
                    return np.array(normalized, dtype=np.float32)
                else:
                    # 길이가 다른 경우는 리스트로 반환
                    return normalized
            return None
        except Exception:
            return None
    
    return None


def _calculate_confidence_scores_vectorized(logprobs, group_size: int = 10):
    """
    NumPy를 사용하여 벡터화된 confidence 계산
    
    Args:
        logprobs: 토큰별 로그 확률 (numpy array 또는 list of arrays)
        group_size: 토큰 그룹 크기
    
    Returns:
        confidence 점수 딕셔너리
    """
    if logprobs is None:
        return {
            'mean_group_confidence': 0.0,
            'bottom_10_percent_confidence': 0.0,
            'tail_confidence': 0.0,
            'lowest_group_confidence': 0.0,
            'top_10_percent_confidence': 0.0,
            'highest_group_confidence': 0.0
        }
    
    try:
        # logprobs를 numpy array로 변환
        if isinstance(logprobs, list):
            if len(logprobs) == 0:
                return {
                    'mean_group_confidence': 0.0,
                    'bottom_10_percent_confidence': 0.0,
                    'tail_confidence': 0.0,
                    'lowest_group_confidence': 0.0,
                    'top_10_percent_confidence': 0.0,
                    'highest_group_confidence': 0.0
                }
            
            # 각 토큰의 평균 로그 확률 계산
            token_confidences = np.array([-np.mean(np.array(tok, dtype=np.float32)) 
                                         for tok in logprobs], dtype=np.float32)
        elif isinstance(logprobs, np.ndarray):
            if logprobs.ndim == 2:
                # (num_tokens, num_candidates) 형태
                token_confidences = -np.mean(logprobs, axis=1, dtype=np.float32)
            elif logprobs.ndim == 1:
                # (num_tokens,) 형태
                token_confidences = -logprobs.astype(np.float32)
            else:
                return {
                    'mean_group_confidence': 0.0,
                    'bottom_10_percent_confidence': 0.0,
                    'tail_confidence': 0.0,
                    'lowest_group_confidence': 0.0,
                    'top_10_percent_confidence': 0.0,
                    'highest_group_confidence': 0.0
                }
        else:
            return {
                'mean_group_confidence': 0.0,
                'bottom_10_percent_confidence': 0.0,
                'tail_confidence': 0.0,
                'lowest_group_confidence': 0.0,
                'top_10_percent_confidence': 0.0,
                'highest_group_confidence': 0.0
            }
        
        if len(token_confidences) == 0:
            return {
                'mean_group_confidence': 0.0,
                'bottom_10_percent_confidence': 0.0,
                'tail_confidence': 0.0,
                'lowest_group_confidence': 0.0,
                'top_10_percent_confidence': 0.0,
                'highest_group_confidence': 0.0
            }
        
        # 그룹별 평균 신뢰도 계산 (numpy 벡터화)
        num_groups = (len(token_confidences) + group_size - 1) // group_size
        
        # padding을 추가하여 group_size로 나누어떨어지게 만들기
        padded_length = num_groups * group_size
        if len(token_confidences) < padded_length:
            padded = np.pad(token_confidences, (0, padded_length - len(token_confidences)), 
                          mode='constant', constant_values=0.0)
        else:
            padded = token_confidences
        
        # reshape하여 그룹별로 나누고 평균 계산 (벡터화)
        grouped = padded.reshape(num_groups, group_size)
        group_confidences = np.mean(grouped, axis=1, dtype=np.float32)
        
        # 마지막 그룹의 실제 길이 고려 (padding된 부분 제외)
        if len(token_confidences) % group_size != 0:
            last_group_start = (num_groups - 1) * group_size
            actual_last_group = token_confidences[last_group_start:]
            group_confidences[-1] = np.mean(actual_last_group, dtype=np.float32)
        
        if len(group_confidences) == 0:
            return {
                'mean_group_confidence': 0.0,
                'bottom_10_percent_confidence': 0.0,
                'tail_confidence': 0.0,
                'lowest_group_confidence': 0.0,
                'top_10_percent_confidence': 0.0,
                'highest_group_confidence': 0.0
            }
        
        # 각 confidence 점수 계산
        mean_group_confidence = float(np.mean(group_confidences))
        
        # 하위 10% 그룹
        num_bottom_groups = max(1, int(len(group_confidences) * 0.1))
        bottom_10_percent_confidence = float(np.mean(np.partition(group_confidences, num_bottom_groups - 1)[:num_bottom_groups]))
        
        # 상위 10% 그룹
        num_top_groups = max(1, int(len(group_confidences) * 0.1))
        top_10_percent_confidence = float(np.mean(np.partition(group_confidences, -num_top_groups)[-num_top_groups:]))
        
        # tail confidence (마지막 그룹)
        if len(token_confidences) < group_size:
            tail_confidence = float(np.mean(token_confidences))
        else:
            tail_confidence = float(np.mean(token_confidences[-group_size:]))
        
        # 최저/최고 그룹
        lowest_group_confidence = float(np.min(group_confidences))
        highest_group_confidence = float(np.max(group_confidences))
        
        return {
            'mean_group_confidence': mean_group_confidence,
            'bottom_10_percent_confidence': bottom_10_percent_confidence,
            'tail_confidence': tail_confidence,
            'lowest_group_confidence': lowest_group_confidence,
            'top_10_percent_confidence': top_10_percent_confidence,
            'highest_group_confidence': highest_group_confidence
        }
    except Exception as e:
        # 오류 발생 시 0.0 반환
        return {
            'mean_group_confidence': 0.0,
            'bottom_10_percent_confidence': 0.0,
            'tail_confidence': 0.0,
            'lowest_group_confidence': 0.0,
            'top_10_percent_confidence': 0.0,
            'highest_group_confidence': 0.0
        }


def ensure_confidence_columns(df: pd.DataFrame, group_size: int = 10) -> pd.DataFrame:
    """
    누락된 confidence 컬럼만 logprobs로부터 계산하여 추가합니다.
    NumPy 벡터화를 사용하여 빠르게 계산합니다.
    """
    required_cols = [
        'mean_group_confidence',
        'bottom_10_percent_confidence',
        'tail_confidence',
        # 'lowest_group_confidence',
        # 'top_10_percent_confidence',
        # 'highest_group_confidence'
    ]

    # 누락된 컬럼만 찾기
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if not missing_cols:
        logger.info("모든 confidence 컬럼이 이미 존재합니다. 계산을 건너뜁니다.")
        return df

    logger.info(f"누락된 confidence 컬럼 발견: {', '.join(missing_cols)}")
    logger.info("confidence 계산 시작 (NumPy 벡터화 버전)...")
    start_time = time.time()

    if 'logprobs' not in df.columns:
        logger.warning('logprobs 컬럼이 없어 confidence 계산을 건너뜁니다.')
        return df

    # 필요한 값들만 계산
    need_mean = 'mean_group_confidence' not in df.columns
    need_bottom10 = 'bottom_10_percent_confidence' not in df.columns
    need_tail = 'tail_confidence' not in df.columns
    need_lowest = 'lowest_group_confidence' not in df.columns
    need_top10 = 'top_10_percent_confidence' not in df.columns
    need_highest = 'highest_group_confidence' not in df.columns

    # NumPy 벡터화를 사용한 빠른 계산
    mean_vals = [] if need_mean else None
    bottom10_vals = [] if need_bottom10 else None
    tail_vals = [] if need_tail else None
    lowest_vals = [] if need_lowest else None
    top10_vals = [] if need_top10 else None
    highest_vals = [] if need_highest else None

    # 배치 처리로 진행 상황 로깅
    batch_size = 1000
    total_rows = len(df)
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end]
        
        for idx, lp in batch_df['logprobs'].items():
            try:
                scores = _calculate_confidence_scores_vectorized(lp, group_size)
                
                if need_mean:
                    mean_vals.append(scores.get('mean_group_confidence', 0.0))
                if need_bottom10:
                    bottom10_vals.append(scores.get('bottom_10_percent_confidence', 0.0))
                if need_tail:
                    tail_vals.append(scores.get('tail_confidence', 0.0))
                if need_lowest:
                    lowest_vals.append(scores.get('lowest_group_confidence', 0.0))
                if need_top10:
                    top10_vals.append(scores.get('top_10_percent_confidence', 0.0))
                if need_highest:
                    highest_vals.append(scores.get('highest_group_confidence', 0.0))
            except Exception as e:
                logger.warning(f"confidence 계산 실패 (idx={idx}): {e}")
                if need_mean:
                    mean_vals.append(0.0)
                if need_bottom10:
                    bottom10_vals.append(0.0)
                if need_tail:
                    tail_vals.append(0.0)
                if need_lowest:
                    lowest_vals.append(0.0)
                if need_top10:
                    top10_vals.append(0.0)
                if need_highest:
                    highest_vals.append(0.0)
        
        # 진행 상황 로깅
        elapsed = time.time() - start_time
        logger.info(f"진행 중: {batch_end}/{total_rows} ({elapsed:.1f}초 경과, {batch_end/elapsed:.1f} rows/sec)")

    # 누락된 컬럼만 추가
    if need_mean:
        df['mean_group_confidence'] = mean_vals
    if need_bottom10:
        df['bottom_10_percent_confidence'] = bottom10_vals
    if need_tail:
        df['tail_confidence'] = tail_vals
    if need_lowest:
        df['lowest_group_confidence'] = lowest_vals
    if need_top10:
        df['top_10_percent_confidence'] = top10_vals
    if need_highest:
        df['highest_group_confidence'] = highest_vals

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Confidence 계산 완료: {elapsed_time:.2f}초 소요 ({total_rows/elapsed_time:.1f} rows/sec)")

    return df


def main(file_path: str, model_name: str, output_dir: str = None) -> None:
    """
    메인 함수: 단일 Parquet 파일 분석
    
    Args:
        file_path: 분석할 Parquet 파일 경로
        model_name: 모델 이름 (tokenizer 로드용)
        output_dir: 출력 디렉토리 (기본값: 파일이 있는 디렉토리)
    """
    logger.info("\n" + "="*50)
    logger.info("=== 단일 Parquet 파일 분석 (Tokenizer 사용) ===")
    logger.info("="*50)
    
    # 1. 파일 로드
    df = load_single_file(file_path)
    if df is None:
        logger.error("파일 로드 실패")
        return
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))
    
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # 2. Tokenizer 로드
    try:
        tokenizer = load_tokenizer(model_name)
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        return
    
    # 3. confidence 컬럼 보강 (누락 시 logprobs로부터 계산)
    # df = ensure_confidence_columns(df)
    
    # 4. Generated content의 token 수 분포 분석 (tokenizer 사용)
    df = analyze_generated_content_tokens(df, tokenizer, output_dir)
    
    # 4-1. Generated text 전체의 token 수 분포 분석 (tokenizer 사용)
    df = analyze_generated_text_tokens(df, tokenizer, output_dir)
    
    # 5. 전체에 대해서 token 수 분포에 따른 정확도, confidence 분석
    # 5.1 content_token_count 기준
    if 'content_token_count' in df.columns:
        analyze_accuracy_by_token_count(df, 'content_token_count', output_dir, num_bins=10)
    
    # 5.2 total_token_count 기준 (있으면)
    if 'total_token_count' in df.columns:
        analyze_accuracy_by_token_count(df, 'total_token_count', output_dir, num_bins=10)
    elif 'output_token_count' in df.columns and 'prompt_token_count' in df.columns:
        df['total_token_count'] = df['output_token_count'] + df['prompt_token_count']
        analyze_accuracy_by_token_count(df, 'total_token_count', output_dir, num_bins=10)
    
    # 6. Problem 별로 token count 별 정확도, confidence 분석
    # 6.1 content_token_count 기준
    if 'content_token_count' in df.columns:
        analyze_problem_level_token_accuracy(df, 'content_token_count', output_dir, num_bins=10)
    
    # 6.2 total_token_count 기준 (있으면)
    if 'total_token_count' in df.columns:
        analyze_problem_level_token_accuracy(df, 'total_token_count', output_dir, num_bins=10)
    
    logger.info("\n" + "="*50)
    logger.info("✅ 분석 완료!")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단일 Parquet 파일 분석 (Tokenizer 사용)")
    parser.add_argument("--file", type=str, required=True,
                       help="분석할 Parquet 파일 경로")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B",
                       help="모델 이름 또는 경로 (tokenizer 로드용)")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/",
                       help="출력 디렉토리 (기본값: 파일이 있는 디렉토리)")
    
    args = parser.parse_args()
    
    main(file_path=args.file, model_name=args.model_name, output_dir=args.output_dir)

