"""
stage4_1_regenerate.py 테스트용 불완전한 JSON 생성
"""
import json
import os

# 원본 파일 로드
source_path = "/mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/math-ai_aime24_baseline_generated.json"
with open(source_path, 'r') as f:
    data = json.load(f)

# 첫 3개 문제만 사용, 각 10개 solutions만 남기고 일부 불완전하게 만들기
test_data = {
    "dataset_name": data["dataset_name"],
    "total_problems": 3,
    "generated_solutions": []
}

for i in range(3):
    problem = data["generated_solutions"][i].copy()
    solutions = problem["solutions"][:10]

    # 일부 solution을 불완전하게 만들기
    for j in range(len(solutions)):
        if j % 3 == 0:
            # final_answer 제거
            solutions[j]["final_answer"] = ""
        elif j % 3 == 1:
            # content를 generated_text로 대체 (fallback 시뮬레이션)
            solutions[j]["content"] = solutions[j]["generated_text"]

    problem["solutions"] = solutions
    test_data["generated_solutions"].append(problem)

# 테스트 파일 저장
test_dir = "/mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/test_regenerate"
os.makedirs(test_dir, exist_ok=True)
test_path = os.path.join(test_dir, "test_baseline_generated.json")

with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"테스트 파일 생성 완료: {test_path}")
print(f"- 총 3개 문제")
print(f"- 각 문제당 10개 solutions")
print(f"- 약 1/3은 final_answer 없음")
print(f"- 약 1/3은 fallback 사용 (content == generated_text)")
print(f"- 약 1/3은 정상")
