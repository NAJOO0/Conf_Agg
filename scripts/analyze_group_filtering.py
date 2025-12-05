"""
Group filtering 분석 스크립트
set_size=8로 평가할 때 필터링되는 그룹과 각 solution 토큰 길이 확인
"""
import os
import sys
from pathlib import Path
import json
from transformers import AutoTokenizer
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))


def format_solutions_for_aggregation(solutions: list, include_confidence: bool = True) -> str:
    """Aggregation 프롬프트를 위한 solution 텍스트 생성"""
    lines = []
    for idx, sol in enumerate(solutions, start=1):
        solution_content = sol.get("content", "")
        lines.append(
            f"solution{idx}:\n"
            f"{solution_content}\n"
            f"final_answer: {sol['final_answer']}\n"
        )
        if include_confidence:
            conf_value = sol["confidence_scores"].get("bottom_10_percent_confidence", None)
            conf_str = f"{conf_value:.4f}" if conf_value is not None else "N/A"
            lines[-1] += f"confidence: {conf_str}\n"
    return "\n".join(lines)


def group_solutions_for_aggregation(
    filtered_solutions: list,
    target_group_size: int = 4,
    max_groups: int = None
) -> List[list]:
    """필터링된 solutions를 그룹으로 묶기"""
    groups = []
    for i in range(0, len(filtered_solutions), target_group_size):
        if max_groups and len(groups) >= max_groups:
            break
        group = filtered_solutions[i:i+target_group_size]
        # target_group_size보다 적으면 제외
        if len(group) == target_group_size:
            groups.append(group)
    return groups


def analyze_group_filtering(
    base_model: str,
    dataset_path: str,
    group_size: int = 8,
    max_tokens: int = 32768,
    enable_thinking: bool = True,
    include_confidence: bool = True,
    max_groups: int = 32
):
    """Group 필터링 분석"""

    # 토크나이저 로드
    print(f"토크나이저 로드 중: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Aggregation 프롬프트 템플릿
    aggregation_prompt_template = (
        "Given the following problem:\n"
        "{problem}\n"
        "and these solution attempts with their confidence scores:\n"
        "{solutions}\n"
        "It is possible that any, all, or none of these solutions are correct or complete. "
        "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
        "filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, "
        "and correct solution to the problem."
    )

    # 데이터 로드
    print(f"\n데이터 로드 중: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"총 문제 수: {len(data['generated_solutions'])}")

    # 필터링 통계
    total_groups = 0
    filtered_groups = 0
    filtering_details = []

    # 각 문제에 대해 분석
    for problem_data in data['generated_solutions']:
        problem_text = problem_data["problem_text"]
        solutions = problem_data["solutions"]
        problem_id = problem_data["problem_id"]

        # 그룹화
        solution_groups = group_solutions_for_aggregation(
            solutions,
            target_group_size=group_size,
            max_groups=max_groups
        )

        total_groups += len(solution_groups)

        # 각 그룹에 대해 토큰 수 체크
        for group_idx, group in enumerate(solution_groups):
            # aggregation prompt 생성
            solutions_text = format_solutions_for_aggregation(group, include_confidence=include_confidence)
            prompt = aggregation_prompt_template.format(
                problem=problem_text,
                solutions=solutions_text
            )
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            # 토큰 수 체크
            tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)
            token_count = len(tokens)

            # 각 solution의 토큰 수 계산
            solution_token_counts = []
            for sol in group:
                content = sol.get("content", "")
                sol_tokens = tokenizer.encode(content, add_special_tokens=False)
                solution_token_counts.append(len(sol_tokens))

            # 필터링 여부 확인
            if token_count > max_tokens:
                filtered_groups += 1
                filtering_details.append({
                    "problem_id": problem_id,
                    "group_idx": group_idx,
                    "total_prompt_tokens": token_count,
                    "solution_token_counts": solution_token_counts,
                    "num_solutions": len(group),
                    "avg_solution_tokens": sum(solution_token_counts) / len(solution_token_counts),
                    "max_solution_tokens": max(solution_token_counts),
                    "min_solution_tokens": min(solution_token_counts),
                })

    # 결과 출력
    print("\n" + "=" * 80)
    print(f"필터링 분석 결과 (group_size={group_size}, max_tokens={max_tokens})")
    print("=" * 80)
    print(f"총 그룹 수: {total_groups}")
    print(f"필터링된 그룹 수: {filtered_groups} ({filtered_groups/total_groups*100:.2f}%)")
    print(f"통과한 그룹 수: {total_groups - filtered_groups} ({(total_groups-filtered_groups)/total_groups*100:.2f}%)")

    if filtering_details:
        print("\n" + "=" * 80)
        print("필터링된 그룹 상세 정보")
        print("=" * 80)

        for detail in filtering_details:
            print(f"\nProblem ID: {detail['problem_id']}, Group Index: {detail['group_idx']}")
            print(f"  전체 프롬프트 토큰 수: {detail['total_prompt_tokens']} (한도: {max_tokens}, 초과: {detail['total_prompt_tokens'] - max_tokens})")
            print(f"  Solutions 개수: {detail['num_solutions']}")
            print(f"  각 Solution 토큰 길이: {detail['solution_token_counts']}")
            print(f"  평균 Solution 토큰: {detail['avg_solution_tokens']:.1f}")
            print(f"  최대 Solution 토큰: {detail['max_solution_tokens']}")
            print(f"  최소 Solution 토큰: {detail['min_solution_tokens']}")
    else:
        print("\n필터링된 그룹이 없습니다!")

    # 통계 요약
    if filtering_details:
        all_solution_tokens = []
        for detail in filtering_details:
            all_solution_tokens.extend(detail['solution_token_counts'])

        print("\n" + "=" * 80)
        print("필터링된 그룹의 Solution 토큰 통계")
        print("=" * 80)
        print(f"전체 Solutions 개수: {len(all_solution_tokens)}")
        print(f"평균 토큰: {sum(all_solution_tokens)/len(all_solution_tokens):.1f}")
        print(f"최대 토큰: {max(all_solution_tokens)}")
        print(f"최소 토큰: {min(all_solution_tokens)}")
        print(f"중앙값 토큰: {sorted(all_solution_tokens)[len(all_solution_tokens)//2]}")

    # 결과를 JSON으로 저장
    output_file = f"group_filtering_analysis_size{group_size}.json"
    output_data = {
        "config": {
            "base_model": base_model,
            "dataset_path": dataset_path,
            "group_size": group_size,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
            "include_confidence": include_confidence,
            "max_groups": max_groups
        },
        "summary": {
            "total_groups": total_groups,
            "filtered_groups": filtered_groups,
            "passed_groups": total_groups - filtered_groups,
            "filter_rate": filtered_groups / total_groups if total_groups > 0 else 0
        },
        "filtered_group_details": filtering_details
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    # 설정
    BASE_MODEL = "Qwen/Qwen3-1.7B"
    BASE_RESULTS_DIR = "/mnt/data1/datasets/nlp/conf_agg/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768"

    # 데이터셋별로 분석
    datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]

    for dataset in datasets:
        dataset_name = dataset["name"]
        dataset_path = dataset["path"]
        dataset_safe_name = dataset_path.replace('/', '_')

        # Baseline 결과 파일 경로
        baseline_path = os.path.join(
            BASE_RESULTS_DIR,
            f"{dataset_safe_name}_baseline_generated.json"
        )

        if not os.path.exists(baseline_path):
            print(f"\n경고: {dataset_name} baseline 파일 없음: {baseline_path}")
            continue

        print("\n" + "=" * 80)
        print(f"데이터셋: {dataset_name}")
        print("=" * 80)

        analyze_group_filtering(
            base_model=BASE_MODEL,
            dataset_path=baseline_path,
            group_size=8,
            max_tokens=32768,
            enable_thinking=True,
            include_confidence=True,
            max_groups=32
        )
