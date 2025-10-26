#!/usr/bin/env python3
"""
통합된 데이터를 분석하는 스크립트
1. output_token_count 분포 분석
2. generated_text 마지막에 confidence score를 출력한 것들의 수와 그 값 분석
"""
import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re
from collections import Counter

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_token_distribution(df: pd.DataFrame) -> None:
    """
    output_token_count 분포 분석
    """
    logger.info("=== output_token_count 분포 분석 ===")
    
    if 'output_token_count' not in df.columns:
        logger.error("output_token_count 컬럼이 없습니다.")
        return
    
    token_counts = df['output_token_count']
    
    # 기본 통계
    logger.info(f"총 데이터 수: {len(token_counts)}")
    logger.info(f"평균 토큰 수: {token_counts.mean():.2f}")
    logger.info(f"중앙값 토큰 수: {token_counts.median():.2f}")
    logger.info(f"표준편차: {token_counts.std():.2f}")
    logger.info(f"최소값: {token_counts.min()}")
    logger.info(f"최대값: {token_counts.max()}")
    
    # 분위수
    percentiles = [25, 50, 75, 90, 95, 99]
    logger.info("분위수:")
    for p in percentiles:
        logger.info(f"  {p}%: {token_counts.quantile(p/100):.2f}")
    
    # 히스토그램 생성
    plt.figure(figsize=(12, 8))
    
    # 전체 분포
    plt.subplot(2, 2, 1)
    plt.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
    plt.title('output_token_count 전체 분포')
    plt.xlabel('토큰 수')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    # 로그 스케일 분포
    plt.subplot(2, 2, 2)
    plt.hist(np.log10(token_counts + 1), bins=50, alpha=0.7, edgecolor='black')
    plt.title('output_token_count 로그 스케일 분포')
    plt.xlabel('log10(토큰 수 + 1)')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    # 박스플롯
    plt.subplot(2, 2, 3)
    plt.boxplot(token_counts)
    plt.title('output_token_count 박스플롯')
    plt.ylabel('토큰 수')
    plt.grid(True, alpha=0.3)
    
    # 누적 분포
    plt.subplot(2, 2, 4)
    sorted_tokens = np.sort(token_counts)
    cumulative = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens)
    plt.plot(sorted_tokens, cumulative)
    plt.title('output_token_count 누적 분포')
    plt.xlabel('토큰 수')
    plt.ylabel('누적 확률')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'token_distribution.png'), dpi=300, bbox_inches='tight')
    logger.info(f"토큰 분포 그래프 저장: {os.path.join(output_dir, 'token_distribution.png')}")
    
    # 토큰 수 구간별 분포
    logger.info("\n토큰 수 구간별 분포:")
    bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-10', '11-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    
    for i in range(len(bins)-1):
        count = ((token_counts >= bins[i]) & (token_counts < bins[i+1])).sum()
        percentage = count / len(token_counts) * 100
        logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")

def extract_confidence_scores(df: pd.DataFrame) -> tuple:
    """
    generated_text에서 confidence score 추출
    """
    logger.info("=== generated_text에서 confidence score 분석 ===")
    
    if 'generated_text' not in df.columns:
        logger.error("generated_text 컬럼이 없습니다.")
        return [], []
    
    confidence_scores = []
    confidence_patterns = []
    
    # 다양한 confidence score 패턴들
    patterns = [
        r'confidence\s*score\s*:?\s*(\d+\.?\d*)',
        r'confidence\s*:?\s*(\d+\.?\d*)',
        r'신뢰도\s*:?\s*(\d+\.?\d*)',
        r'확신도\s*:?\s*(\d+\.?\d*)',
        r'confidence\s*=\s*(\d+\.?\d*)',
        r'신뢰도\s*=\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*%?\s*confidence',
        r'(\d+\.?\d*)\s*%?\s*신뢰도',
    ]
    
    total_texts = len(df)
    texts_with_confidence = 0
    
    for idx, text in enumerate(df['generated_text']):
        if pd.isna(text):
            continue
            
        text_str = str(text).lower()
        
        # 각 패턴에 대해 검색
        found_score = None
        found_pattern = None
        
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])  # 마지막 매치 사용
                    found_score = score
                    found_pattern = pattern
                    break
                except ValueError:
                    continue
        
        if found_score is not None:
            confidence_scores.append(found_score)
            confidence_patterns.append(found_pattern)
            texts_with_confidence += 1
    
    logger.info(f"총 텍스트 수: {total_texts}")
    logger.info(f"confidence score가 있는 텍스트 수: {texts_with_confidence}")
    logger.info(f"비율: {texts_with_confidence/total_texts*100:.2f}%")
    
    return confidence_scores, confidence_patterns

def analyze_confidence_scores(confidence_scores: list, confidence_patterns: list) -> None:
    """
    confidence score 분석
    """
    if not confidence_scores:
        logger.info("confidence score가 발견되지 않았습니다.")
        return
    
    logger.info(f"\n=== confidence score 분석 ===")
    logger.info(f"발견된 confidence score 수: {len(confidence_scores)}")
    
    scores = np.array(confidence_scores)
    
    # 기본 통계
    logger.info(f"평균 confidence score: {scores.mean():.3f}")
    logger.info(f"중앙값 confidence score: {np.median(scores):.3f}")
    logger.info(f"표준편차: {scores.std():.3f}")
    logger.info(f"최소값: {scores.min():.3f}")
    logger.info(f"최대값: {scores.max():.3f}")
    
    # 분위수
    percentiles = [25, 50, 75, 90, 95, 99]
    logger.info("분위수:")
    for p in percentiles:
        logger.info(f"  {p}%: {np.percentile(scores, p):.3f}")
    
    # 구간별 분포
    logger.info("\nconfidence score 구간별 분포:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
              '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    
    for i in range(len(bins)-1):
        count = ((scores >= bins[i]) & (scores < bins[i+1])).sum()
        percentage = count / len(scores) * 100
        logger.info(f"  {labels[i]}: {count}개 ({percentage:.1f}%)")
    
    # 패턴별 분포
    pattern_counts = Counter(confidence_patterns)
    logger.info("\n패턴별 분포:")
    for pattern, count in pattern_counts.most_common():
        logger.info(f"  {pattern}: {count}개")
    
    # 히스토그램 생성
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Confidence Score 분포')
    plt.xlabel('Confidence Score')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(scores)
    plt.title('Confidence Score 박스플롯')
    plt.ylabel('Confidence Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Confidence score 분포 그래프 저장: {os.path.join(output_dir, 'confidence_distribution.png')}")

def analyze_merged_data(data_path: str) -> None:
    """
    통합된 데이터 분석 메인 함수
    """
    logger.info(f"데이터 로딩: {data_path}")
    
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"데이터 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
        logger.info(f"컬럼명: {list(df.columns)}")
        
        # 샘플 데이터 확인
        logger.info("\n=== 샘플 데이터 (첫 3행) ===")
        for col in df.columns:
            logger.info(f"{col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    # 1. output_token_count 분포 분석
    analyze_token_distribution(df)
    
    # 2. confidence score 분석
    confidence_scores, confidence_patterns = extract_confidence_scores(df)
    analyze_confidence_scores(confidence_scores, confidence_patterns)
    
    logger.info("\n✅ 분석 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="통합된 데이터 분석")
    parser.add_argument("--data-path", type=str, 
                       default="/home/najoo0/Conf_Agg/output_s/generated/sample_300/raw_generated.parquet",
                       help="분석할 parquet 파일 경로")
    
    args = parser.parse_args()
    
    analyze_merged_data(args.data_path)
