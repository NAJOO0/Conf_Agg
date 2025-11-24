"""
Stage 4-3 (Baseline Confidence Variants): Aggregation 평가
Baseline 생성 결과만을 활용하여 다양한 Confidence 스킴을 비교

주요 변경사항:
- 각 instance당 4번씩 aggregation 생성
- 4개 결과의 평균으로 최종 accuracy 계산
"""

import os
import sys
import json
import logging
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple
import tempfile

import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from peft import PeftModel

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = (
    "Given the following problem:\n"
    "{problem}\n"
    "and these solution attempts with their confidence scores:\n"
    "{solutions}\n"
    "It is possible that any, all, or none of these solutions are correct or complete. "
    "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
    "filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, "
    "and correct solution to the problem."
)


PROMPT_WITHOUT_CONFIDENCE = (
    "Given the following problem:\n"
    "{problem}\n"
    "and these solution attempts:\n"
    "{solutions}\n"
    "It is possible that any, all, or none of these solutions are correct or complete. "
    "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
    "filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, "
    "and correct solution to the problem."
)


PROMPT_NO_GUIDANCE = (
    "Given the following problem:\n"
    "{problem}\n"
    "and these solution attempts with their confidence scores:\n"
    "{solutions}\n"
    "It is possible that any, all, or none of these solutions are correct or complete. "
    "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
    "filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, "
    "and correct solution to the problem."
)


PROMPT_FULL_GUIDANCE = (
    "Given the following problem:\n"
    "{problem}\n"
    "and these solution attempts with their confidence scores:\n"
    "{solutions}\n"
    "It is possible that any, all, or none of these solutions are correct or complete. "
    "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
    "filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, "
    "and correct solution to the problem."
)


PROMPT_VARIANT_TEMPLATES: Dict[str, str] = {
    "default": DEFAULT_PROMPT_TEMPLATE,
    "without_confidence": PROMPT_WITHOUT_CONFIDENCE,
    "confidence_no_guidance": PROMPT_NO_GUIDANCE,
    "confidence_full_guidance": PROMPT_FULL_GUIDANCE,
}


def sanitize_prompt_variant_name(variant: str) -> str:
    """디렉터리 이름용 프롬프트 variant 문자열 정규화"""
    sanitized = variant.strip().lower().replace(" ", "_")
    return sanitized or "default"


def filter_solutions_by_token_length(
    solutions: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """content가 max_tokens 이상인 solution 필터링"""
    filtered = []
    token_counts = []
    for sol in solutions:
        content = sol.get("content", "")
        if not content:
            continue

        tokens = tokenizer.encode(content, add_special_tokens=False)
        token_count = len(tokens)
        token_counts.append(token_count)

        if token_count < max_tokens:
            filtered.append(sol)
        else:
            logger.debug("Solution 필터링됨: %d 토큰 (최대: %d)", token_count, max_tokens)

    return filtered, token_counts


def format_solutions_for_aggregation(
    solutions: List[Dict[str, Any]],
    confidence_keys: List[str],
    confidence_labels: Dict[str, str],
    prompt_variant: str,
) -> str:
    """Aggregation 프롬프트를 위한 solution 텍스트 생성"""
    variant_key = prompt_variant if prompt_variant in PROMPT_VARIANT_TEMPLATES else "default"
    solution_blocks: List[str] = []
    for idx, sol in enumerate(solutions, start=1):
        solution_content = sol.get("content", "")
        block_lines: List[str] = [
            f"Solution {idx}:",
            solution_content,
            f"final_answer: {sol.get('final_answer', '')}",
        ]

        if confidence_keys and variant_key != "without_confidence":
            conf_scores = sol.get("confidence_scores", {}) or {}
            conf_lines: List[str] = []

            if variant_key == "confidence_no_guidance":
                if len(confidence_keys) == 1:
                    key = confidence_keys[0]
                    value = conf_scores.get(key)
                    conf_lines.append(
                        f"Confidence score: {value:.3f}" if value is not None else "Confidence score: N/A"
                    )
                else:
                    for order, key in enumerate(confidence_keys, start=1):
                        value = conf_scores.get(key)
                        conf_lines.append(
                            f"Confidence score {order}: {value:.3f}"
                            if value is not None
                            else f"Confidence score {order}: N/A"
                        )
            elif variant_key == "confidence_full_guidance":
                for key in confidence_keys:
                    label = confidence_labels.get(key, key)
                    value = conf_scores.get(key)
                    conf_lines.append(
                        f"{label} Confidence score: {value:.3f}" if value is not None else f"{label} Confidence score: N/A"
                    )
            else:
                for key in confidence_keys:
                    label = confidence_labels.get(key, key)
                    value = conf_scores.get(key)
                    conf_lines.append(
                        f"- {label}: {value:.4f}" if value is not None else f"- {label}: N/A"
                    )

            if conf_lines:
                if variant_key in {"default"}:
                    block_lines.append("confidence scores:")
                block_lines.extend(conf_lines)

        solution_blocks.append("\n".join(block_lines))

    return "\n\n".join(solution_blocks)


def extract_content(text: str) -> str:
    """</think> 토큰 이후 값들 추출"""
    if not text:
        return ""

    text_str = str(text)
    marker = "</think>"
    marker_pos = text_str.find(marker)
    if marker_pos == -1:
        return text_str.strip()

    return text_str[marker_pos + len(marker):].strip()


def prepare_aggregation_prompts_batch(
    tokenizer: AutoTokenizer,
    aggregation_requests: List[Dict[str, Any]],
    aggregation_prompt_template: str,
    enable_thinking: bool,
    confidence_keys: List[str],
    confidence_labels: Dict[str, str],
    prompt_variant: str,
) -> Tuple[List[str], List[str]]:
    """
    배치 처리를 위한 aggregation 프롬프트 준비

    Returns:
        (formatted_prompts, prompt_texts) - formatted_prompts는 토크나이저 적용된 것, prompt_texts는 원본 프롬프트 텍스트
    """
    formatted_prompts: List[str] = []
    prompt_texts: List[str] = []

    for req in aggregation_requests:
        problem_text = req["problem_text"]
        solutions = req["solutions"]

        solutions_text = format_solutions_for_aggregation(
            solutions,
            confidence_keys=confidence_keys,
            confidence_labels=confidence_labels,
            prompt_variant=prompt_variant,
        )

        prompt = aggregation_prompt_template.format(
            problem=problem_text,
            solutions=solutions_text,
        )
        prompt_texts.append(prompt)

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        formatted_prompts.append(formatted_prompt)

    return formatted_prompts, prompt_texts


def count_correct_solutions(
    solutions: List[Dict[str, Any]],
    ground_truth: str,
    math_verifier: MathVerifier,
) -> int:
    """주어진 solutions 중 정답인 것의 개수 계산"""
    correct_count = 0
    for sol in solutions:
        final_answer = sol.get("final_answer", "")
        if final_answer and math_verifier.verify_answer(final_answer, ground_truth):
            correct_count += 1
    return correct_count


def group_solutions_for_aggregation(
    filtered_solutions: List[Dict[str, Any]],
    target_group_size: int,
    max_groups: int | None,
) -> List[List[Dict[str, Any]]]:
    """필터링된 solutions를 그룹으로 묶기"""
    groups: List[List[Dict[str, Any]]] = []
    for i in range(0, len(filtered_solutions), target_group_size):
        if max_groups and len(groups) >= max_groups:
            break
        group = filtered_solutions[i:i + target_group_size]
        if len(group) == target_group_size:
            groups.append(group)
    return groups


def group_results_by_problem_id(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    결과를 problem_id별로 그룹화
    
    변경사항: 4개의 생성 결과를 저장하도록 수정
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        problem_id = result["problem_id"]
        if problem_id not in grouped:
            grouped[problem_id] = {
                "problem_id": problem_id,
                "problem_text": result["problem_text"],
                "ground_truth": result["ground_truth"],
                "prompts": [],
            }

        prompt_info = {
            "prompt_text": result.get("prompt_text", ""),
            "num_solutions": result.get("num_solutions", 0),
            "correct_solutions_count": result.get("correct_solutions_count", 0),
            "generated_texts": result.get("generated_texts", []),  # 4개
            "parsed_contents": result.get("parsed_contents", []),  # 4개
            "final_answers": result.get("final_answers", []),  # 4개
            "is_corrects": result.get("is_corrects", []),  # 4개
            "accuracy": result.get("accuracy", 0.0),  # 4개 중 정답 비율
            "solution_group_sizes": result.get("solution_group_sizes"),
        }
        grouped[problem_id]["prompts"].append(prompt_info)

    return grouped


def build_aggregation_requests(
    problem_entries: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    group_size: int,
    max_groups: int,
    filter_max_tokens: int,
    math_verifier: MathVerifier,
) -> List[Dict[str, Any]]:
    aggregation_requests: List[Dict[str, Any]] = []
    for problem_data in problem_entries:
        problem_text = problem_data["problem_text"]
        solutions = problem_data["solutions"]
        ground_truth = problem_data["ground_truth"]
        problem_id = problem_data["problem_id"]

        filtered_solutions, _ = filter_solutions_by_token_length(
            solutions,
            tokenizer,
            max_tokens=filter_max_tokens,
        )

        solution_groups = group_solutions_for_aggregation(
            filtered_solutions,
            target_group_size=group_size,
            max_groups=max_groups,
        )

        for group in solution_groups:
            correct_count = count_correct_solutions(group, ground_truth, math_verifier)
            aggregation_requests.append(
                {
                    "problem_id": problem_id,
                    "problem_text": problem_text,
                    "ground_truth": ground_truth,
                    "solutions": group,
                    "correct_count": correct_count,
                    "solution_group_sizes": ",".join(str(len(g)) for g in solution_groups),
                }
            )

    return aggregation_requests


def run_aggregation_for_variants(
    aggregation_requests: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    llm: LLM,
    sampling_params: SamplingParams,
    math_verifier: MathVerifier,
    eval_config: Dict[str, Any],
    aggregation_prompt_template: str,
    confidence_variants: List[Dict[str, Any]],
    confidence_labels: Dict[str, str],
    prompt_variant: str,
    aggregator_name: str,
) -> Dict[str, Any]:
    """
    Confidence variant별로 aggregation 수행
    
    변경사항: 각 요청당 4번씩 생성
    """
    if not aggregation_requests:
        return {}

    variant_outputs: Dict[str, Any] = {}
    enable_thinking = eval_config.get("enable_thinking", False)
    do_evaluation = eval_config.get("aggregation_do_evaluation", True)
    num_generations_per_instance = eval_config.get("num_generations_per_instance", 4)

    for variant in confidence_variants:
        variant_id = variant["variant_id"]
        confidence_keys = variant["confidence_keys"]
        include_confidence = bool(confidence_keys)

        logger.info("[%s] Variant 생성: %s (confidence_keys=%s)", aggregator_name, variant_id, confidence_keys or "None")

        formatted_prompts, prompt_texts = prepare_aggregation_prompts_batch(
            tokenizer,
            aggregation_requests,
            aggregation_prompt_template,
            enable_thinking=enable_thinking,
            confidence_keys=confidence_keys if include_confidence else [],
            confidence_labels=confidence_labels,
            prompt_variant=prompt_variant,
        )

        # n=4로 설정하여 각 프롬프트당 4개 생성
        outputs = llm.generate(formatted_prompts, sampling_params)

        variant_results: List[Dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            req = aggregation_requests[idx]
            
            # 4개의 생성 결과 처리
            generated_texts = []
            parsed_contents = []
            final_answers = []
            is_corrects = []
            
            for gen_output in output.outputs:
                agg_text = gen_output.text
                parsed_content = extract_content(agg_text)
                agg_answer = math_verifier.extract_final_answer_from_content(parsed_content)
                
                is_correct = None
                if agg_answer and do_evaluation:
                    is_correct = math_verifier.verify_answer(agg_answer, req["ground_truth"])
                
                generated_texts.append(agg_text)
                parsed_contents.append(parsed_content)
                final_answers.append(agg_answer)
                is_corrects.append(is_correct)
            
            # 4개 결과의 평균 accuracy 계산
            if do_evaluation:
                accuracy = sum(1 for correct in is_corrects if correct) / len(is_corrects)
            else:
                accuracy = None

            variant_results.append(
                {
                    "problem_id": req["problem_id"],
                    "problem_text": req["problem_text"],
                    "ground_truth": req["ground_truth"],
                    "prompt_text": prompt_texts[idx],
                    "num_solutions": len(req["solutions"]),
                    "correct_solutions_count": req["correct_count"],
                    "generated_texts": generated_texts,
                    "parsed_contents": parsed_contents,
                    "final_answers": final_answers,
                    "is_corrects": is_corrects,
                    "accuracy": accuracy,
                    "solution_group_sizes": req["solution_group_sizes"],
                }
            )

        grouped_results = group_results_by_problem_id(variant_results)
        summary = compute_accuracy_summary(grouped_results, num_generations_per_instance)

        variant_outputs[variant_id] = {
            "variant_name": variant.get("display_name", variant_id),
            "confidence_keys": confidence_keys,
            "confidence_labels": [confidence_labels.get(k, k) for k in confidence_keys],
            "results": grouped_results,
            "summary": summary,
        }

        logger.info(
            "[%s] Variant %s 요약: avg_accuracy=%.3f (total_prompts=%d, total_generations=%d)",
            aggregator_name,
            variant_id,
            summary["accuracy"],
            summary["total_prompts"],
            summary["total_generations"],
        )

    return variant_outputs


def compute_accuracy_summary(
    grouped_results: Dict[str, Any],
    num_generations_per_instance: int = 4
) -> Dict[str, Any]:
    """
    정확도 요약 계산
    
    변경사항: 각 prompt의 accuracy(4개 평균)를 전체 평균
    """
    total_accuracy = 0.0
    total_prompts = 0
    
    for problem_data in grouped_results.values():
        prompts = problem_data.get("prompts", [])
        for prompt in prompts:
            accuracy = prompt.get("accuracy", 0.0)
            if accuracy is not None:
                total_accuracy += accuracy
                total_prompts += 1
    
    avg_accuracy = total_accuracy / total_prompts if total_prompts > 0 else 0.0
    
    return {
        "total_prompts": total_prompts,
        "total_generations": total_prompts * num_generations_per_instance,
        "accuracy": avg_accuracy,
    }


def update_summary_file(
    summary_path: str,
    dataset_name: str,
    group_size: int,
    aggregator_name: str,
    variant_outputs: Dict[str, Any],
) -> None:
    """요약 전용 파일에 aggregator 결과 집계"""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
    else:
        summary_payload = {
            "dataset_name": dataset_name,
            "group_size": group_size,
            "aggregators": {},
        }

    aggregator_summary = {
        variant_id: {
            "variant_name": data.get("variant_name", variant_id),
            "summary": data.get("summary", {}),
        }
        for variant_id, data in variant_outputs.items()
    }

    summary_payload.setdefault("aggregators", {})
    summary_payload["aggregators"][aggregator_name] = aggregator_summary

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)


def load_baseline_results(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        logger.warning("Baseline 결과 파일 없음: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    logger.info("Baseline 결과 로드: %d개 문제", len(baseline_data.get("generated_solutions", [])))
    return baseline_data


def build_confidence_variants(conf_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    default_metrics = [
        {"key": "mean_group_confidence", "label": "Mean"},
        {"key": "lowest_group_confidence", "label": "Lowest"},
        {"key": "bottom_10_percent_confidence", "label": "Bottom10%"},
        {"key": "tail_confidence", "label": "Tail"},
    ]
    metrics_cfg = conf_cfg.get("metrics", default_metrics)
    metric_labels = {m["key"]: m.get("label", m["key"]) for m in metrics_cfg}
    metric_keys = [m["key"] for m in metrics_cfg]

    default_subsets = [
        {"name": "without_confidence", "include_confidence": False, "subset_size": 0},
        {"name": "single_confidence", "include_confidence": True, "subset_size": 1},
        {"name": "double_confidence", "include_confidence": True, "subset_size": 2},
        {"name": "triple_confidence", "include_confidence": True, "subset_size": 3},
        {"name": "all_confidence", "include_confidence": True, "subset_size": len(metric_keys)},
    ]
    subsets_cfg = conf_cfg.get("subsets", default_subsets)

    variants: List[Dict[str, Any]] = []
    for subset in subsets_cfg:
        name = subset.get("name", "variant")
        display_name = subset.get("display_name", name)
        include_conf = subset.get("include_confidence", True)

        if not include_conf or subset.get("subset_size", 0) == 0:
            variants.append(
                {
                    "variant_id": name,
                    "display_name": display_name,
                    "confidence_keys": [],
                    "include_confidence": False,
                }
            )
            continue

        combinations = subset.get("combinations")
        if combinations:
            combo_lists = [list(combo) for combo in combinations]
        else:
            subset_size = subset.get("subset_size", len(metric_keys))
            if subset_size > len(metric_keys):
                logger.warning(
                    "subset_size=%d가 metrics 개수보다 커 기본 metrics 모두 사용합니다. (subset=%s)",
                    subset_size,
                    name,
                )
                subset_size = len(metric_keys)
            combo_lists = [list(combo) for combo in itertools.combinations(metric_keys, subset_size)]

        for combo in combo_lists:
            combo_suffix = "__".join(combo) if combo else "none"
            variant_id = f"{name}__{combo_suffix}" if combo else name
            variants.append(
                {
                    "variant_id": variant_id,
                    "display_name": display_name,
                    "confidence_keys": combo,
                    "include_confidence": True,
                }
            )

    if not variants:
        raise ValueError("Confidence variant 구성이 비어 있습니다. config를 확인하세요.")

    logger.info("Confidence variant %d개 생성", len(variants))
    for var in variants:
        logger.info(
            "  - %s (keys=%s)",
            var["variant_id"],
            ", ".join(var["confidence_keys"]) if var["confidence_keys"] else "None",
        )

    return variants, metric_labels


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-3: Baseline Aggregation (Confidence Variants) 메인 함수"""

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

    log_file = os.path.join(cfg.paths.log_dir, "stage4_3_baseline_confidence.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags,
    )

    if torch.cuda.is_available():
        logger.info("GPU 설정: device=0, GPU=%s", torch.cuda.get_device_name(0))

    logger.info("🚀 Stage 4-3 (Baseline Confidence Variants) 시작")
    
    eval_config = cfg.evaluation.benchmarks.evaluation
    
    # 생성 횟수 설정
    num_generations_per_instance = eval_config.get("num_generations_per_instance", 4)
    logger.info("📊 Instance당 생성 횟수: %d", num_generations_per_instance)
    
    prompt_variant_cfg = eval_config.get("aggregation_prompt_variant", "default")
    prompt_variant_key = prompt_variant_cfg if prompt_variant_cfg in PROMPT_VARIANT_TEMPLATES else "default"
    prompt_template_override = eval_config.get("aggregation_prompt_template")
    aggregation_prompt_template = (
        prompt_template_override
        if prompt_template_override
        else PROMPT_VARIANT_TEMPLATES.get(prompt_variant_key, DEFAULT_PROMPT_TEMPLATE)
    )
    prompt_variant_dir = sanitize_prompt_variant_name(prompt_variant_key)

    if not eval_config.get("aggregation_do_generation", True):
        logger.error("aggregation_do_generation 옵션이 False입니다. 이 스크립트는 generation이 필요합니다.")
        return

    # 출력 디렉토리 설정
    baseline_results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    baseline_results_dir = os.path.join(baseline_results_dir, "Qwen_Qwen3-1.7B_think_True")
    
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results", prompt_variant_dir)
    results_dir = os.path.join(results_dir, "Qwen_Qwen3-1.7B_think_True")
    
    logger.info("사용 프롬프트 variant: %s", prompt_variant_key)
    logger.info("baseline_results_dir: %s", baseline_results_dir)
    logger.info("results_dir: %s", results_dir)
    if prompt_template_override:
        logger.info("aggregation_prompt_template override 사용")

    math_verifier = MathVerifier(timeout=eval_config.timeout)

    confidence_cfg = eval_config.get("aggregation_confidence", {})
    confidence_variants, confidence_labels = build_confidence_variants(confidence_cfg)

    # 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]

    baseline_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=True,
    )
    if baseline_tokenizer.pad_token is None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

    # SamplingParams에 n 추가
    sampling_params = SamplingParams(
        temperature=eval_config.temperature,
        max_tokens=eval_config.max_tokens,
        top_p=eval_config.get("top_p", 0.8),
        top_k=eval_config.get("top_k", 20),
        min_p=eval_config.get("min_p", 0.0),
        n=num_generations_per_instance,  # 각 프롬프트당 4개 생성
    )

    baseline_llm = LLM(
        model=cfg.model.base_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=eval_config.get("gpu_memory_utilization", 0.8),
        max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384),
        dtype="bfloat16",
        trust_remote_code=True,
    )
    logger.info("Baseline 모델 로드 완료")

    group_sizes = eval_config.get("aggregation_group_sizes")
    if not group_sizes:
        group_sizes = [4]
    max_groups = eval_config.get("aggregation_max_groups", None)

    baseline_dataset_cache: Dict[str, Dict[str, Any]] = {}
    requests_cache_by_dataset: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        dataset_safe_name = dataset_path.replace("/", "_")

        logger.info("=" * 60)
        logger.info("데이터셋: %s", dataset_name)
        logger.info("=" * 60)

        baseline_path = os.path.join(
            baseline_results_dir,
            f"{dataset_safe_name}_baseline_generated.json",
        )
        baseline_data = load_baseline_results(baseline_path)
        if not baseline_data:
            logger.warning("%s 데이터셋에 대한 baseline 결과 없음, 건너뜁니다.", dataset_name)
            continue

        baseline_dataset_cache[dataset_safe_name] = baseline_data
        requests_cache_by_dataset[dataset_safe_name] = {}
        all_group_summaries: Dict[str, Dict[str, Any]] = {}

        for group_size in group_sizes:
            logger.info("-" * 40)
            logger.info("그룹 크기 %d 처리 시작 (Baseline)", group_size)
            logger.info("-" * 40)

            aggregation_requests = build_aggregation_requests(
                baseline_data["generated_solutions"],
                baseline_tokenizer,
                group_size=group_size,
                max_groups=max_groups,
                filter_max_tokens=eval_config.filter_max_tokens,
                math_verifier=math_verifier,
            )

            if not aggregation_requests:
                logger.warning("그룹 크기 %d에 대한 aggregation 요청이 없습니다.", group_size)
                continue

            logger.info("총 %d개 aggregation 요청 준비", len(aggregation_requests))
            requests_cache_by_dataset[dataset_safe_name][group_size] = aggregation_requests

            variant_outputs = run_aggregation_for_variants(
                aggregation_requests,
                tokenizer=baseline_tokenizer,
                llm=baseline_llm,
                sampling_params=sampling_params,
                math_verifier=math_verifier,
                eval_config=eval_config,
                aggregation_prompt_template=aggregation_prompt_template,
                confidence_variants=confidence_variants,
                confidence_labels=confidence_labels,
                prompt_variant=prompt_variant_key,
                aggregator_name="baseline",
            )

            group_results_dir = os.path.join(results_dir, f"group_size_{group_size}")
            os.makedirs(group_results_dir, exist_ok=True)

            output_path = os.path.join(
                group_results_dir,
                f"{dataset_safe_name}_baseline_confidence_results.json",
            )

            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    output_payload = json.load(f)
            else:
                output_payload = {
                    "dataset_name": dataset_path,
                    "group_size": group_size,
                    "num_generations_per_instance": num_generations_per_instance,
                    "aggregators": {},
                }

            output_payload.setdefault("aggregators", {})
            output_payload["aggregators"]["baseline"] = variant_outputs
            output_payload["num_generations_per_instance"] = num_generations_per_instance

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, ensure_ascii=False, indent=2)

            logger.info("그룹 크기 %d Baseline 결과 저장: %s", group_size, output_path)

            summary_path = output_path.replace("_results.json", "_summary.json")
            update_summary_file(
                summary_path,
                dataset_name=dataset_path,
                group_size=group_size,
                aggregator_name="baseline",
                variant_outputs=variant_outputs,
            )

            all_group_summaries.setdefault(group_size, {})
            all_group_summaries[group_size]["baseline"] = {
                variant_id: data["summary"] for variant_id, data in variant_outputs.items()
            }

        logger.info("데이터셋 %s Baseline 요약: %s", dataset_name, json.dumps(all_group_summaries, indent=2))

    logger.info("Baseline 모델 unload 중...")
    del baseline_llm
    torch.cuda.empty_cache()
    logger.info("Baseline 모델 unload 완료")

    # AggLLM 단계 (동일한 방식으로 수정)
    checkpoint_num = eval_config.get("checkpoint_num", None)
    if checkpoint_num is not None:
        aggllm_model_path = os.path.join(cfg.paths.model_dir, "enable_think_True_20251117",f"checkpoint-{checkpoint_num}")
    else:
        aggllm_model_path = os.path.join(cfg.paths.model_dir, "enable_think_True_20251117", "checkpoint-final")

    if not os.path.exists(aggllm_model_path):
        logger.warning("AggLLM 모델을 찾을 수 없습니다: %s (AggLLM 단계 건너뜀)", aggllm_model_path)
    else:
        logger.info("AggLLM 모델 로드 준비 중...")
        aggllm_tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=True,
        )
        if aggllm_tokenizer.pad_token is None:
            aggllm_tokenizer.pad_token = aggllm_tokenizer.eos_token

        logger.info("LoRA 가중치 병합 중... (%s)", aggllm_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, aggllm_model_path)
        merged_model = peft_model.merge_and_unload()

        merged_model_path = cfg.paths.get("merged_model_cache_dir", None)
        if not merged_model_path:
            merged_model_path = tempfile.mkdtemp(prefix="aggllm_merged_")

        os.makedirs(merged_model_path, exist_ok=True)
        config_path = os.path.join(merged_model_path, "config.json")

        if not os.path.exists(config_path):
            merged_model.save_pretrained(merged_model_path, safe_serialization=True)
            aggllm_tokenizer.save_pretrained(merged_model_path)

        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()

        aggllm_llm = LLM(
            model=merged_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=eval_config.get(
                "aggllm_gpu_memory_utilization",
                eval_config.get("gpu_memory_utilization", 0.9),
            ),
            max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384),
            dtype="bfloat16",
            trust_remote_code=True,
            kv_cache_dtype="fp8",
        )
        logger.info("AggLLM 모델 로드 완료")

        # AggLLM도 동일한 로직으로 처리 (Baseline과 거의 동일)
        for benchmark in benchmark_datasets:
            dataset_name = benchmark["name"]
            dataset_path = benchmark["path"]
            dataset_safe_name = dataset_path.replace("/", "_")

            if dataset_safe_name not in baseline_dataset_cache:
                logger.warning("Baseline 데이터가 없어 AggLLM 단계 건너뜀: %s", dataset_name)
                continue

            baseline_data = baseline_dataset_cache[dataset_safe_name]
            requests_cache = requests_cache_by_dataset.get(dataset_safe_name, {})
            aggllm_group_summaries: Dict[str, Dict[str, Any]] = {}

            for group_size in group_sizes:
                logger.info("-" * 40)
                logger.info("그룹 크기 %d 처리 시작 (AggLLM)", group_size)
                logger.info("-" * 40)

                aggregation_requests = requests_cache.get(group_size)
                if aggregation_requests is None:
                    logger.info("캐시된 aggregation 요청 없음, 재생성 시도")
                    aggregation_requests = build_aggregation_requests(
                        baseline_data["generated_solutions"],
                        aggllm_tokenizer,
                        group_size=group_size,
                        max_groups=max_groups,
                        filter_max_tokens=eval_config.filter_max_tokens,
                        math_verifier=math_verifier,
                    )
                    if aggregation_requests:
                        requests_cache[group_size] = aggregation_requests

                if not aggregation_requests:
                    logger.warning("그룹 크기 %d에 대한 AggLLM aggregation 요청이 없습니다.", group_size)
                    continue

                variant_outputs = run_aggregation_for_variants(
                    aggregation_requests,
                    tokenizer=aggllm_tokenizer,
                    llm=aggllm_llm,
                    sampling_params=sampling_params,
                    math_verifier=math_verifier,
                    eval_config=eval_config,
                    aggregation_prompt_template=aggregation_prompt_template,
                    confidence_variants=confidence_variants,
                    confidence_labels=confidence_labels,
                    prompt_variant=prompt_variant_key,
                    aggregator_name="aggllm",
                )

                group_results_dir = os.path.join(results_dir, f"group_size_{group_size}")
                os.makedirs(group_results_dir, exist_ok=True)

                output_path = os.path.join(
                    group_results_dir,
                    f"{dataset_safe_name}_baseline_confidence_results.json",
                )

                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        output_payload = json.load(f)
                else:
                    output_payload = {
                        "dataset_name": dataset_path,
                        "group_size": group_size,
                        "num_generations_per_instance": num_generations_per_instance,
                        "aggregators": {},
                    }

                output_payload.setdefault("aggregators", {})
                output_payload["aggregators"]["aggllm"] = variant_outputs
                output_payload["num_generations_per_instance"] = num_generations_per_instance

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_payload, f, ensure_ascii=False, indent=2)

                logger.info("그룹 크기 %d AggLLM 결과 저장: %s", group_size, output_path)

                summary_path = output_path.replace("_results.json", "_summary.json")
                update_summary_file(
                    summary_path,
                    dataset_name=dataset_path,
                    group_size=group_size,
                    aggregator_name="aggllm",
                    variant_outputs=variant_outputs,
                )

                aggllm_group_summaries.setdefault(group_size, {})
                aggllm_group_summaries[group_size]["aggllm"] = {
                    variant_id: data["summary"] for variant_id, data in variant_outputs.items()
                }

            if aggllm_group_summaries:
                logger.info("데이터셋 %s AggLLM 요약: %s", dataset_name, json.dumps(aggllm_group_summaries, indent=2))

        logger.info("AggLLM 모델 unload 중...")
        del aggllm_llm
        torch.cuda.empty_cache()
        logger.info("AggLLM 모델 unload 완료")

    logger.info("✅ Stage 4-3 (Baseline + AggLLM Confidence Variants) 완료")


if __name__ == "__main__":
    main()