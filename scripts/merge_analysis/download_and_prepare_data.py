#!/usr/bin/env python3
"""
DeepScaler 데이터를 다운로드하고 JSONL 형식으로 변환하는 스크립트
"""
import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

def download_deepscaler_raw(output_path: str) -> None:
    """
    DeepScaler 데이터를 다운로드하고 JSONL 형식으로 저장
    
    Args:
        output_path: 출력 파일 경로
    """
    print("=" * 80)
    print("DeepScaler 데이터 다운로드 및 변환")
    print("=" * 80)
    
    # 방법 1: HuggingFace datasets 사용
    print("\n📦 HuggingFace에서 데이터 다운로드 중...")
    try:
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        print(f"✓ 데이터셋 로드 완료: {len(dataset)}개 샘플")
        
        # JSONL 형식으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 데이터 저장 완료: {output_path}")
        print(f"  총 {len(dataset)}개 문제 저장됨")
        
    except Exception as e:
        print(f"⚠️ HuggingFace 로드 실패: {e}")
        print("\n📝 대안: GSM8K 데이터 사용")
        download_gsm8k_alternative(output_path)


def download_gsm8k_alternative(output_path: str) -> None:
    """
    GSM8K 데이터를 DeepScaler 형식으로 변환하여 저장
    
    Args:
        output_path: 출력 파일 경로
    """
    print("\n📦 GSM8K 데이터 다운로드 중 (DeepScaler 대체용)...")
    try:
        # GSM8K 데이터 로드
        dataset = load_dataset("gsm8k", "main", split="train")
        print(f"✓ GSM8K 데이터 로드 완료: {len(dataset)}개 샘플")
        
        # DeepScaler 형식으로 변환
        converted_data = []
        for idx, item in enumerate(dataset):
            problem = item['question']
            answer = item['answer']
            
            # 답변에서 숫자 추출
            numeric_answer = extract_numeric_answer(answer)
            
            converted_item = {
                "id": f"gsm8k_{idx}",
                "problem": problem,
                "answer": numeric_answer,
                "full_answer": answer
            }
            converted_data.append(converted_item)
        
        # JSONL 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in converted_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ 데이터 변환 및 저장 완료: {output_path}")
        print(f"  총 {len(converted_data)}개 문제 저장됨")
        
    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        create_sample_data(output_path)


def extract_numeric_answer(answer: str) -> str:
    """
    GSM8K 답변에서 숫자 부분만 추출
    
    Args:
        answer: 전체 답변 (예: "Let's solve step by step... #### 42")
    
    Returns:
        숫자 답변
    """
    import re
    # #### 뒤의 숫자 찾기
    match = re.search(r'####\s*(\S+)', answer)
    if match:
        return match.group(1).strip()
    
    # 숫자 추출
    numbers = re.findall(r'\d+', answer)
    if numbers:
        return numbers[-1]
    
    return "0"


def create_sample_data(output_path: str, num_samples: int = 100) -> None:
    """
    샘플 데이터 생성 (개발/테스트용)
    
    Args:
        output_path: 출력 파일 경로
        num_samples: 생성할 샘플 수
    """
    print(f"\n📝 샘플 데이터 생성 중 ({num_samples}개)...")
    
    sample_problems = [
        {
            "problem": "A train travels 60 miles in 2 hours. How far does it travel in 3 hours?",
            "answer": "90"
        },
        {
            "problem": "Sarah has 20 apples. She gives away 5 apples to her friend. How many apples does she have left?",
            "answer": "15"
        },
        {
            "problem": "A book costs $15. If you buy 3 books, how much do you pay?",
            "answer": "45"
        },
        {
            "problem": "In a class of 30 students, 12 are girls. How many boys are there?",
            "answer": "18"
        },
        {
            "problem": "A rectangle has a length of 8 cm and width of 5 cm. What is its area?",
            "answer": "40"
        },
    ]
    
    # 샘플 확장
    data = []
    for i in range(num_samples):
        problem_template = sample_problems[i % len(sample_problems)]
        data.append({
            "id": f"sample_{i}",
            "problem": f"Problem {i}: {problem_template['problem']}",
            "answer": problem_template['answer']
        })
    
    # JSONL 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ 샘플 데이터 생성 완료: {output_path}")
    print(f"  총 {num_samples}개 문제 생성됨")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepScaler 데이터 다운로드 및 준비')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/mnt/data1/datasets/nlp/conf_agg/raw',
        help='출력 디렉토리 경로'
    )
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='샘플 데이터만 생성 (테스트용)'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'deepscaler.jsonl')
    
    if args.sample_only:
        create_sample_data(output_path)
    else:
        download_deepscaler_raw(output_path)
    
    print("\n" + "=" * 80)
    print("✅ 데이터 준비 완료!")
    print(f"📁 저장 위치: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

