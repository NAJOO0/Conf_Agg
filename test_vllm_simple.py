"""
vLLM 간단 테스트 - 샘플 4개, GPU 전체 사용, logprob 확인
"""
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

def main():
    # multiprocessing 이슈 방지
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # GPU 확인
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 샘플 문제 4개 (간단한 수학 문제)
    sample_problems = [
        "What is 2 + 2?",
        "What is 5 * 3?",
        "What is 10 - 4?",
        "What is 12 / 3?",
    ]
    
    # vLLM 모델 로드 (단일 GPU로 실행하여 multiprocessing 문제 회피)
    model_name = "Qwen/Qwen3-1.7B"
    print(f"\n🔧 모델 로드 중: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # 단일 GPU (multiprocessing 이슈 회피)
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="float16",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ 모델 로드 완료\n")
    
    # 토큰화 및 프롬프트 생성
    prompts = []
    for problem in sample_problems:
        messages = [{"role": "user", "content": problem}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    
    # 샘플링 파라미터 설정 (logprob=5는 top-5 logprob 저장)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=32768,
        logprobs=5,  # top-5 logprob 저장
    )
    
    # 추론 실행
    print("🚀 추론 시작...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"✅ 추론 완료: {len(outputs)}개 결과\n")
    
    # 결과 확인
    print("=" * 80)
    print("결과 상세 분석")
    print("=" * 80)
    
    for idx, output in enumerate(outputs):
        print(f"\n[문제 {idx + 1}] {sample_problems[idx]}")
        print(f"생성된 텍스트: {output.outputs[0].text}")
        
        # logprob 구조 분석
        if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            logprobs = output.outputs[0].logprobs
            print(f"\n📊 Logprobs 구조:")
            print(f"   - Logprobs 타입: {type(logprobs)}")
            print(f"   - 총 토큰 수: {len(logprobs)}개")
            
            # 첫 번째 토큰의 logprob 상세 확인
            if len(logprobs) > 0:
                first_token_logprobs = logprobs[0]
                print(f"\n   [첫 번째 토큰의 logprob]")
                print(f"   - 타입: {type(first_token_logprobs)}")
                
                if isinstance(first_token_logprobs, dict):
                    print(f"   - 항목 수: {len(first_token_logprobs)}개")
                    
                    # top-k 확인
                    count = 0
                    for token_id, logprob_entry in list(first_token_logprobs.items())[:10]:
                        count += 1
                        if isinstance(logprob_entry, dict):
                            logprob_value = logprob_entry.get('logprob', 'N/A')
                            token = logprob_entry.get('decoded_token', 'N/A')
                            print(f"   - Token ID {token_id}: logprob={logprob_value:.4f}, token={token}")
                        else:
                            print(f"   - Token ID {token_id}: {logprob_entry}")
                    
                    print(f"\n   ✅ 첫 번째 토큰에서 {len(first_token_logprobs)}개의 logprob 저장됨")
                    
            # 마지막 토큰의 logprob 확인
            if len(logprobs) > 1:
                last_token_logprobs = logprobs[-1]
                if isinstance(last_token_logprobs, dict):
                    print(f"\n   [마지막 토큰의 logprob]")
                    print(f"   - 항목 수: {len(last_token_logprobs)}개")
            
            # 토큰별 logprob 개수 요약
            token_logprob_counts = []
            for i, token_logprobs in enumerate(logprobs):
                if isinstance(token_logprobs, dict):
                    token_logprob_counts.append(len(token_logprobs))
            
            if token_logprob_counts:
                print(f"\n   📈 토큰별 logprob 개수:")
                print(f"   - 평균: {sum(token_logprob_counts) / len(token_logprob_counts):.1f}개")
                print(f"   - 최소: {min(token_logprob_counts)}개")
                print(f"   - 최대: {max(token_logprob_counts)}개")
            
            # 샘플 출력: 처음 3개 토큰의 logprob 개수
            print(f"\n   📋 처음 3개 토큰의 logprob 개수:")
            for i in range(min(3, len(logprobs))):
                if isinstance(logprobs[i], dict):
                    print(f"   - 토큰 {i+1}: {len(logprobs[i])}개 logprob")
        else:
            print("   ⚠️  logprob이 없습니다")
    
    print("\n" + "=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)

if __name__ == '__main__':
    main()