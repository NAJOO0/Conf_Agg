#!/usr/bin/env python3
"""
각 group_size 폴더의 aggregation 결과를 집계하여 jsonl 파일에 저장하는 스크립트
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

def calculate_accuracy_from_aggregation_results(
    aggregation_results: Dict[str, Any],
    aggregation_type: str
) -> Dict[str, Any]:
    """
    aggregation_results에서 특정 aggregation_type의 정확도를 계산
    
    Args:
        aggregation_results: aggregation_results.json 파일의 전체 내용
        aggregation_type: 'baseline_to_baseline_aggregation', 
                          'baseline_to_baseline_aggregation_without_confidence',
                          'baseline_to_aggllm_aggregation' 중 하나
    
    Returns:
        {'correct': int, 'total': int, 'accuracy': float}
    """
    results_dict = aggregation_results.get(aggregation_type, {})
    
    if not isinstance(results_dict, dict):
        return {'correct': 0, 'total': 0, 'accuracy': 0.0}
    
    total = 0
    correct = 0
    
    for problem_id, problem_data in results_dict.items():
        prompts = problem_data.get("prompts", [])
        for prompt in prompts:
            total += 1
            if prompt.get("is_correct", False):
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }


def aggregate_group_results(
    base_dir: str,
    output_file: str,
    dataset_name: str = "AIME24",
    dataset_path: str = "math-ai/aime24"
):
    """
    각 group_size 폴더의 결과를 집계하여 jsonl 파일에 저장
    
    Args:
        base_dir: group_size 폴더들이 있는 기본 디렉토리
        output_file: 결과를 저장할 jsonl 파일 경로
        dataset_name: 데이터셋 이름
        dataset_path: 데이터셋 경로
    """
    base_path = Path(base_dir)
    results = []
    dataset_safe_name = dataset_path.replace('/', '_')
    
    group_dirs = sorted(
        [
            p for p in base_path.iterdir()
            if p.is_dir() and p.name.startswith("group_size_")
        ],
        key=lambda p: int(p.name.split("_")[-1])
    )
    
    if not group_dirs:
        print(f"⚠️  group_size_* 폴더를 찾을 수 없습니다: {base_dir}")
        return
    
    for group_dir in group_dirs:
        try:
            group_size = int(group_dir.name.split("_")[-1])
        except ValueError:
            print(f"⚠️  알 수 없는 폴더 형식 무시: {group_dir}")
            continue
        
        checkpoint_file = group_dir / f"{dataset_safe_name}_aggregation_results_checkpoint_2400.json"
        all_checkpoints_file = group_dir / f"{dataset_safe_name}_aggregation_results.json"
        
        print(f"📂 처리 중: group_size_{group_size}")
        
        checkpoint_summary = {}
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                checkpoint_summary = checkpoint_data.get("summary", {})
                print(f"  ✓ {checkpoint_file.name} 로드 완료")
        else:
            print(f"  ⚠️  파일이 없습니다: {checkpoint_file}")
        
        all_checkpoints_summary = {}
        if all_checkpoints_file.exists():
            with open(all_checkpoints_file, 'r', encoding='utf-8') as f:
                all_checkpoints_data = json.load(f)
                all_checkpoints_summary = all_checkpoints_data.get("summary", {})
                print(f"  ✓ {all_checkpoints_file.name} 로드 완료")
        else:
            print(f"  ⚠️  파일이 없습니다: {all_checkpoints_file}")
        
        if "baseline_to_baseline_aggregation" in all_checkpoints_summary:
            summary_data = all_checkpoints_summary["baseline_to_baseline_aggregation"]
            result_entry = {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "group_size": group_size,
                "aggregation_type": "baseline_to_baseline_aggregation",
                "accuracy": summary_data.get("accuracy", 0.0),
                "correct": summary_data.get("correct", 0),
                "total": summary_data.get("total", 0)
            }
            results.append(result_entry)
            print(
                f"  ✓ baseline_to_baseline_aggregation: accuracy={result_entry['accuracy']:.4f}, "
                f"correct={result_entry['correct']}/{result_entry['total']}"
            )
        
        if "baseline_to_baseline_aggregation_without_confidence" in all_checkpoints_summary:
            summary_data = all_checkpoints_summary["baseline_to_baseline_aggregation_without_confidence"]
            result_entry = {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "group_size": group_size,
                "aggregation_type": "baseline_to_baseline_aggregation_without_confidence",
                "accuracy": summary_data.get("accuracy", 0.0),
                "correct": summary_data.get("correct", 0),
                "total": summary_data.get("total", 0)
            }
            results.append(result_entry)
            print(
                "  ✓ baseline_to_baseline_aggregation_without_confidence: "
                f"accuracy={result_entry['accuracy']:.4f}, "
                f"correct={result_entry['correct']}/{result_entry['total']}"
            )
        
        if "baseline_to_aggllm_aggregation" in checkpoint_summary:
            summary_data = checkpoint_summary["baseline_to_aggllm_aggregation"]
            result_entry = {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "group_size": group_size,
                "aggregation_type": "baseline_to_aggllm_aggregation",
                "accuracy": summary_data.get("accuracy", 0.0),
                "correct": summary_data.get("correct", 0),
                "total": summary_data.get("total", 0)
            }
            results.append(result_entry)
            print(
                f"  ✓ baseline_to_aggllm_aggregation: accuracy={result_entry['accuracy']:.4f}, "
                f"correct={result_entry['correct']}/{result_entry['total']}"
            )
    
    # 결과를 jsonl 파일에 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 결과 저장 완료: {output_path}")
    print(f"   총 {len(results)}개의 결과 항목이 저장되었습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregation summary 병합 스크립트")
    parser.add_argument("--base_dir", required=True, help="group_size_* 폴더들이 있는 디렉토리")
    parser.add_argument("--output_file", required=True, help="저장할 jsonl 경로")
    parser.add_argument("--dataset_name", default="AIME24", help="데이터셋 이름")
    parser.add_argument("--dataset_path", default="math-ai/aime24", help="데이터셋 경로 (파일명 생성에 사용)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_group_results(
        base_dir=args.base_dir,
        output_file=args.output_file,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path
    )


