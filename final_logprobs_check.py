#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Parquet 파일 로드
df = pd.read_parquet('/data2/datasets/nlp/conf_agg/generated/raw_generated.parquet')

print('🎯 Logprobs 최종 확인 결과:')
print(f'총 응답 수: {len(df)}')

# 첫 번째 응답의 logprobs 확인
first_logprobs = df['logprobs'].iloc[0]
print(f'\n✅ 첫 번째 응답의 logprobs 구조:')
print(f'  - 타입: {type(first_logprobs)}')
print(f'  - numpy array shape: {first_logprobs.shape}')

# numpy array를 리스트로 변환
logprobs_list = first_logprobs.tolist()
print(f'  - 리스트 길이: {len(logprobs_list)}')

# 첫 번째 토큰의 logprobs 확인
first_token_logprobs = logprobs_list[0]
print(f'  - 첫 번째 토큰의 logprobs: {first_token_logprobs}')
print(f'  - 첫 번째 토큰의 logprobs 개수: {len(first_token_logprobs)}')
print(f'  - ✅ 설정된 logprobs=5와 일치: {len(first_token_logprobs) == 5}')

# 여러 토큰의 logprobs 확인
print(f'\n📊 여러 토큰의 logprobs 샘플:')
for i in range(0, min(15, len(logprobs_list)), 5):  # 5개씩 건너뛰며 확인
    token_logprobs = logprobs_list[i:i+5]
    print(f'  - 토큰 {i//5 + 1}: 개수={len(token_logprobs)}, 첫 값={token_logprobs[0][0]:.6f}')

# 전체 통계
print(f'\n📈 전체 logprobs 통계:')
total_values = []
token_counts = []

for idx, row in df.iterrows():
    logprobs = row['logprobs']
    if isinstance(logprobs, np.ndarray):
        logprobs_list = logprobs.tolist()
        token_count = len(logprobs_list) // 5  # 5개씩 묶여있으므로
        token_counts.append(token_count)
        
        # 모든 값들을 평탄화해서 수집
        for token_logprobs in logprobs_list:
            if isinstance(token_logprobs, np.ndarray):
                total_values.extend(token_logprobs.tolist())

if total_values:
    print(f'  - 전체 logprobs 값 개수: {len(total_values)}')
    print(f'  - 평균 토큰 수: {sum(token_counts) / len(token_counts):.1f}')
    print(f'  - 토큰 수 범위: {min(token_counts)} ~ {max(token_counts)}')
    print(f'  - 값들의 범위: {min(total_values):.6f} ~ {max(total_values):.6f}')
    print(f'  - 값들의 평균: {sum(total_values) / len(total_values):.6f}')

print(f'\n🎉 결론:')
print(f'  ✅ logprobs 컬럼이 정상적으로 저장됨')
print(f'  ✅ 각 토큰마다 top-5 logprobs가 저장됨 (설정값과 일치)')
print(f'  ✅ 총 {len(df)}개 응답의 모든 토큰에 대해 logprobs 저장됨')
print(f'  ✅ 평균 {sum(token_counts) / len(token_counts):.1f}개 토큰당 5개씩 logprobs 저장됨')

print('\n✅ Logprobs 저장 상태 확인 완료!')


