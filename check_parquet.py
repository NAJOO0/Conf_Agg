#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# confidence.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.confidence import ConfidenceCalculator

# Parquet 파일 로드
df = pd.read_parquet('/workspace/output_s/generated/sample_400/raw_generated.parquet')

print('📊 output_token_count 통계 분석')
print('=' * 50)

# 기본 통계 정보
print('\n🔢 기본 통계:')
print(f'  - 전체 응답 수: {len(df):,}')
print(f'  - 평균 토큰 수: {df["output_token_count"].mean():.2f}')
print(f'  - 중앙값 토큰 수: {df["output_token_count"].median():.2f}')
print(f'  - 최소 토큰 수: {df["output_token_count"].min()}')
print(f'  - 최대 토큰 수: {df["output_token_count"].max()}')
print(f'  - 표준편차: {df["output_token_count"].std():.2f}')

# 분위수 분석
print('\n📈 분위수 분석:')
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(df["output_token_count"], p)
    print(f'  - {p}% 분위수: {value:.0f}')

# 10%씩 자르되 stride 5%로 겹치는 구간 분석
print('\n📊 토큰 수 구간별 분석 (10% 구간, stride 5%):')
token_counts = df["output_token_count"]

# 분위수 계산 (5% 간격으로)
percentiles = list(range(0, 101, 5))  # 0%, 5%, 10%, ..., 100%
percentile_values = [np.percentile(token_counts, p) for p in percentiles]

print(f'  분위수 값들: {[int(v) for v in percentile_values]}')

# ConfidenceCalculator 초기화
confidence_calc = ConfidenceCalculator(group_size=10)

# 각 구간별 분석 (10% 구간, 5% stride)
print('\n  구간별 상세 분석:')
for i in range(0, len(percentiles)-2, 1):  # stride 5% (인덱스 1씩 증가)
    start_pct = percentiles[i]
    end_pct = percentiles[i+2]  # 10% 구간
    
    start_val = int(percentile_values[i])
    end_val = int(percentile_values[i+2])
    
    # 해당 구간의 데이터 필터링
    mask = (token_counts >= start_val) & (token_counts <= end_val)
    range_df = df[mask]
    
    if len(range_df) == 0:
        continue
        
    print(f'\n  구간 {start_pct}%-{end_pct}% ({start_val}-{end_val} 토큰):')
    print(f'    - 응답 수: {len(range_df):,}개 ({len(range_df)/len(df)*100:.2f}%)')
    
    # Confidence score 계산 (logprobs 컬럼이 있는 경우)
    if 'logprobs' in df.columns:
        # 해당 구간의 logprobs 추출
        range_logprobs = range_df['logprobs'].tolist()
        
        # 각 응답의 confidence score 계산
        confidence_scores = []
        for logprobs in range_logprobs:
            # logprobs가 None이 아니고 비어있지 않은지 확인
            if logprobs is not None and hasattr(logprobs, '__len__') and len(logprobs) > 0:
                try:
                    scores = confidence_calc.calculate_all_confidence_scores(logprobs)
                    confidence_scores.append(scores)
                except Exception as e:
                    print(f"      신뢰도 계산 오류: {e}")
                    continue
        
        if confidence_scores:
            # 평균 confidence score 계산
            mean_group_conf = np.mean([s['mean_group_confidence'] for s in confidence_scores])
            bottom_10_conf = np.mean([s['bottom_10_percent_confidence'] for s in confidence_scores])
            tail_conf = np.mean([s['tail_confidence'] for s in confidence_scores])
            
            print(f'    - 평균 그룹 신뢰도: {mean_group_conf:.4f}')
            print(f'    - 하위 10% 신뢰도: {bottom_10_conf:.4f}')
            print(f'    - 꼬리 신뢰도: {tail_conf:.4f}')
        else:
            print(f'    - 신뢰도 계산 불가 (logprobs 데이터 없음)')
    else:
        print(f'    - 신뢰도 계산 불가 (logprobs 컬럼 없음)')
    
    # 토큰 수 통계
    print(f'    - 평균 토큰 수: {range_df["output_token_count"].mean():.1f}')
    print(f'    - 중앙값 토큰 수: {range_df["output_token_count"].median():.1f}')
    print(f'    - 최소/최대 토큰 수: {range_df["output_token_count"].min()}/{range_df["output_token_count"].max()}')

# 4096 토큰 응답 상세 분석
print('\n🔍 4096 토큰 응답 분석:')
max_token_responses = df[df['output_token_count'] == 4096]
print(f'  - 4096 토큰 응답 수: {len(max_token_responses):,}')
print(f'  - 전체 응답 대비 비율: {len(max_token_responses) / len(df) * 100:.2f}%')

if len(max_token_responses) > 0:
    print(f'  - 4096 토큰 응답이 있는 문제 수: {max_token_responses["problem_id"].nunique()}')
    print(f'  - 전체 문제 대비 비율: {max_token_responses["problem_id"].nunique() / df["problem_id"].nunique() * 100:.2f}%')

# 문제별 토큰 수 통계
print('\n📋 문제별 토큰 수 통계:')
problem_stats = df.groupby('problem_id')['output_token_count'].agg([
    'count', 'mean', 'min', 'max', 'std'
]).round(2)

print(f'  - 문제당 평균 응답 수: {problem_stats["count"].mean():.1f}')
print(f'  - 문제당 평균 토큰 수: {problem_stats["mean"].mean():.1f}')
print(f'  - 문제당 최대 토큰 수 평균: {problem_stats["max"].mean():.1f}')
print(f'  - 문제당 토큰 수 표준편차 평균: {problem_stats["std"].mean():.1f}')

# 토큰 수가 많은 상위 문제들
print('\n🏆 토큰 수가 많은 상위 5개 문제:')
top_problems = problem_stats.nlargest(5, 'max')
for i, (problem_id, stats) in enumerate(top_problems.iterrows(), 1):
    print(f'  {i}. 문제 ID: {problem_id}')
    print(f'     - 응답 수: {stats["count"]}')
    print(f'     - 평균 토큰 수: {stats["mean"]:.1f}')
    print(f'     - 최대 토큰 수: {stats["max"]}')
    print(f'     - 표준편차: {stats["std"]:.1f}')

print('\n✅ output_token_count 통계 분석 완료!')
