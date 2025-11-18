"""
Stage 4-3: Aggregation 평가
저장된 Solution 결과를 바탕으로 Prompt Aggregation과 AggLLM Aggregation 수행
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
import tempfile
from typing import Tuple, List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def filter_solutions_by_token_length(
    solutions: list,
    tokenizer: AutoTokenizer,
    max_tokens: int = 2000
) -> Tuple[list, list]:
    """
    content가 max_tokens 이상인 solution 필터링
    
    Args:
        solutions: solution 리스트
        tokenizer: 토크나이저
        max_tokens: 최대 토큰 수 (기본 2000)
    
    Returns:
        (필터링된 solution 리스트, 토큰 수 리스트)
    """
    filtered = []
    token_counts = []
    for sol in solutions:
        content = sol.get("content", "")
        if not content:
            continue
        
        # 토큰 수 계산
        tokens = tokenizer.encode(content, add_special_tokens=False)
        token_count = len(tokens)
        token_counts.append(token_count)
        
        if token_count < max_tokens:
            filtered.append(sol)
        else:
            logger.debug(f"Solution 필터링됨: {token_count} 토큰 (최대: {max_tokens})")
    
    return filtered, token_counts


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
            conf_value = sol["confidence_scores"].get("tail_confidence", None)
            conf_str = f"{conf_value:.4f}" if conf_value is not None else "N/A"
            lines[-1] += f"confidence: {conf_str}\n"
    return "\n".join(lines)


def extract_content(text: str) -> str:
    """
    </think> 토큰 이후 값들 추출
    
    Args:
        text: generated_text
        
    Returns:
        </think> 이후 내용
    """
    if not text:
        return ""
    
    text_str = str(text)
    marker = "</think>"
    
    marker_pos = text_str.find(marker)
    if marker_pos == -1:
        return text_str.strip()
    
    # 마커 이후 텍스트 추출
    content = text_str[marker_pos + len(marker):].strip()
    return content


def prepare_aggregation_prompts_batch(
    tokenizer: AutoTokenizer,
    aggregation_requests: List[Dict[str, Any]],
    aggregation_prompt_template: str,
    enable_thinking: bool,
    include_confidence: bool = True
) -> Tuple[List[str], List[str]]:
    """배치 처리를 위한 aggregation 프롬프트 준비
    
    Returns:
        (formatted_prompts, prompt_texts) - formatted_prompts는 토크나이저 적용된 것, prompt_texts는 원본 프롬프트 텍스트
    """
    formatted_prompts = []
    prompt_texts = []
    for req in aggregation_requests:
        problem_text = req["problem_text"]
        solutions = req["solutions"]
        
        # Solution 텍스트 포맷팅
        solutions_text = format_solutions_for_aggregation(solutions, include_confidence=include_confidence)
        
        # 프롬프트 구성
        prompt = aggregation_prompt_template.format(
            problem=problem_text,
            solutions=solutions_text
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


def count_correct_solutions(solutions: list, ground_truth: str, math_verifier: MathVerifier) -> int:
    """주어진 solutions 중 정답인 것의 개수 계산"""
    correct_count = 0
    for sol in solutions:
        final_answer = sol.get("final_answer", "")
        if final_answer and math_verifier.verify_answer(final_answer, ground_truth):
            correct_count += 1
    return correct_count


def group_solutions_for_aggregation(
    filtered_solutions: list,
    target_group_size: int = 4,
    max_groups: int = None
) -> List[list]:
    """
    필터링된 solutions를 그룹으로 묶기
    
    Args:
        filtered_solutions: 필터링된 solution 리스트
        target_group_size: 각 그룹의 크기
        max_groups: 최대 그룹 수 (None이면 제한 없음)
    
    Returns:
        그룹 리스트 (각 그룹은 solution 리스트)
    """
    groups = []
    for i in range(0, len(filtered_solutions), target_group_size):
        if max_groups and len(groups) >= max_groups:
            break
        group = filtered_solutions[i:i+target_group_size]
        # target_group_size보다 적으면 제외
        if len(group) == target_group_size:
            groups.append(group)
    return groups


def group_results_by_problem_id(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    결과를 problem_id별로 그룹화
    
    Args:
        results: aggregation 결과 리스트
        
    Returns:
        problem_id를 키로 하는 딕셔너리
    """
    grouped = {}
    for result in results:
        problem_id = result["problem_id"]
        if problem_id not in grouped:
            grouped[problem_id] = {
                "problem_id": problem_id,
                "problem_text": result["problem_text"],
                "ground_truth": result["ground_truth"],
                "prompts": []
            }
        
        # prompt 정보 추가
        prompt_info = {
            "prompt_text": result.get("prompt_text", ""),
            "num_solutions": result.get("num_solutions", 0),
            "correct_solutions_count": result.get("correct_solutions_count", 0),
            "generated_text": result.get("generated_text", ""),
            "parsed_content": result.get("parsed_content", ""),
            "final_answer": result.get("final_answer", ""),
            "is_correct": result.get("is_correct", None)
        }
        grouped[problem_id]["prompts"].append(prompt_info)
    
    return grouped


def save_accuracy_jsonl(
    summary: Dict[str, Dict[str, Any]],
    dataset_name: str,
    dataset_path: str,
    group_size: int,
    results_dir: str,
    dataset_safe_name: str,
    checkpoint_num: int = None
) -> None:
    """
    정확도만 저장하는 jsonl 파일 생성
    
    Args:
        summary: 정확도 요약 딕셔너리
        dataset_name: 데이터셋 이름
        dataset_path: 데이터셋 경로
        group_size: 그룹 크기
        results_dir: 결과 디렉토리
        dataset_safe_name: 파일명에 사용할 안전한 데이터셋 이름
        checkpoint_num: 체크포인트 번호 (aggllm 결과에만 사용, 파일명에 포함)
    """
    # checkpoint_num이 있으면 파일명에 추가
    if checkpoint_num is not None:
        accuracy_jsonl_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_accuracy_checkpoint_{checkpoint_num}.jsonl"
        )
    else:
        accuracy_jsonl_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_accuracy.jsonl"
        )
    
    # jsonl 파일에 추가 (각 aggregation 타입별로 한 줄씩)
    with open(accuracy_jsonl_path, 'a', encoding='utf-8') as f:
        for aggregation_type, metrics in summary.items():
            accuracy_record = {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "group_size": group_size,
                "aggregation_type": aggregation_type,
                "accuracy": metrics.get("accuracy", 0.0),
                "correct": metrics.get("correct", 0),
                "total": metrics.get("total", 0)
            }
            f.write(json.dumps(accuracy_record, ensure_ascii=False) + "\n")
    
    logger.info(f"정확도 결과 저장: {accuracy_jsonl_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-3: Aggregation 평가 메인 함수"""
    
    # GPU 설정 (CUDA_VISIBLE_DEVICES가 설정된 경우를 대비)
    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES가 설정되어 있으면 0번이 실제 GPU
        # 명시적으로 GPU를 설정하여 프로세스 간 격리 보장
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_3_evaluate_aggregation.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    if torch.cuda.is_available():
        logger.info(f"GPU 설정: device=0, GPU={torch.cuda.get_device_name(0)}")
    
    logger.info("🚀 Stage 4-3: Aggregation 평가 시작")
    
    # 옵션 확인
    eval_config = cfg.evaluation.benchmarks.evaluation
    do_analysis = eval_config.get("aggregation_do_analysis", True)
    do_generation = eval_config.get("aggregation_do_generation", True)
    do_evaluation = eval_config.get("aggregation_do_evaluation", True)
    
    if not do_analysis and not do_generation and not do_evaluation:
        logger.error("analysis, generation, evaluation 중 하나는 True여야 합니다.")
        return
    
    logger.info(f"옵션: analysis={do_analysis}, generation={do_generation}, evaluation={do_evaluation}")
    
    # Baseline 건너뛰기 옵션
    skip_baseline = eval_config.get("skip_baseline", False)
    logger.info(f"Baseline 건너뛰기: {skip_baseline}")
    
    # 디렉토리 설정
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    # results_dir = os.path.join(results_dir, "think" if eval_config.get("enable_thinking", False) else "no_think")
    results_dir = os.path.join(results_dir, "no_think_no_think")
    logger.info(f"results_dir: {results_dir}")
    # Math Verifier 초기화
    math_verifier = MathVerifier(
        timeout=eval_config.timeout
    )
    
    # Aggregation 프롬프트 템플릿
    # aggregation_prompt_template = (
    #     "You are an expert mathematician and critical analyst.\n"
    #     "Your task is to synthesize multiple, potentially flawed, solution attempts "
    #     "into a single, correct, and comprehensive final answer.\n\n"
    #     "You will be given a problem, followed by several solution attempts.\n"
    #     "Solution attempts include confidence scores to help estimate their quality.\n\n"
    #     "Carefully review all the provided information. It is possible that any, all, or none "
    #     "of the solutions are correct or complete.\n"
    #     "Use them as starting points—correcting mistakes, filling in gaps, and/or combining "
    #     "useful ideas—to produce your final solution.\n\n"
    #     "---\n"
    #     "GIVEN THE FOLLOWING PROBLEM:\n{problem}\n\n"
    #     "AND THESE SOLUTION ATTEMPTS:\n{solutions}\n\n"
    #     "---\n"
    #     "Now, provide the final, comprehensive, and correct solution to the problem."
    # )
    aggregation_prompt_template = (
        "Given the following problem:\n{problem}\n"
        "and these solution attempts:\n{solutions}\n"
        "It is possible that any, all, or none of these solutions are correct or complete. Carefully review the\n"
        "provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining\n"
        "useful ideas—to produce a final, comprehensive, and correct solution to the problem."
    )
    # base_instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    
    # 벤치마크 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]
    
    # 모델 경로 확인
    checkpoint_num = eval_config.checkpoint_num
    if checkpoint_num is not None:
        # aggllm_model_path = os.path.join(cfg.paths.model_dir,f"enable_think_True_20251117/checkpoint-{checkpoint_num}")
        aggllm_model_path = os.path.join(cfg.paths.model_dir,f"checkpoint-{checkpoint_num}")
    else:
        # aggllm_model_path = os.path.join(cfg.paths.model_dir, "enable_think_True_20251117/checkpoint-final")
        aggllm_model_path = os.path.join(cfg.paths.model_dir, "checkpoint-final")
    
    if not os.path.exists(aggllm_model_path):
        logger.warning(f"AggLLM 모델을 찾을 수 없습니다: {aggllm_model_path}")
        aggllm_model_path = None
    
    # 각 데이터셋에 대해 평가
    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        dataset_safe_name = dataset_path.replace('/', '_')
        
        logger.info("=" * 60)
        logger.info(f"데이터셋: {dataset_name}")
        logger.info("=" * 60)
        
        # 정확도 jsonl 파일 초기화 (데이터셋별로 새로 시작)
        accuracy_jsonl_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_accuracy.jsonl"
        )
        if os.path.exists(accuracy_jsonl_path):
            os.remove(accuracy_jsonl_path)
            logger.info(f"기존 정확도 파일 삭제: {accuracy_jsonl_path}")
        
        # 그룹 크기별로 실험 (4, 2, 1)
        group_sizes = eval_config.aggregation_group_sizes
        all_group_results = {}  # {group_size: aggregation_results}
        
        # Baseline 결과 로드
        baseline_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_baseline_generated.json"
        )
        
        baseline_data = None
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            logger.info(f"Baseline 결과 로드: {len(baseline_data['generated_solutions'])}개 문제")
        else:
            logger.warning(f"Baseline 결과 파일 없음: {baseline_path}")
        
        # AggLLM 결과 로드
        aggllm_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_aggllm_generated.json"
        )
        
        aggllm_data = None
        if os.path.exists(aggllm_path):
            with open(aggllm_path, 'r', encoding='utf-8') as f:
                aggllm_data = json.load(f)
            logger.info(f"AggLLM 결과 로드: {len(aggllm_data['generated_solutions'])}개 문제")
        else:
            logger.warning(f"AggLLM 결과 파일 없음: {aggllm_path}")
        
        if not baseline_data and not aggllm_data:
            logger.warning(f"{dataset_name}에 대한 결과 파일이 없어 건너뜁니다.")
            continue
        
        # Analysis 단계: 토큰 수 분포 분석
        if do_analysis:
            logger.info("=" * 60)
            logger.info("Analysis 단계 시작: 토큰 수 분포 분석")
            logger.info("=" * 60)
            
            analysis_results = {
                "dataset_name": dataset_path,
                "baseline_token_distribution": {},
                "aggllm_token_distribution": {}
            }
            
            # Baseline 토큰 수 분석
            if baseline_data:
                logger.info("Baseline 토큰 수 분석 중...")
                baseline_tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.base_model,
                    trust_remote_code=True
                )
                if baseline_tokenizer.pad_token is None:
                    baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
                
                instance_token_counts = []
                for problem_data in baseline_data["generated_solutions"]:
                    solutions = problem_data["solutions"]
                    problem_id = problem_data["problem_id"]
                    
                    problem_token_counts = []
                    for sol in solutions:
                        content = sol.get("content", "")
                        if content:
                            tokens = baseline_tokenizer.encode(content, add_special_tokens=False)
                            token_count = len(tokens)
                            problem_token_counts.append(token_count)
                    
                    # 16384을 넘는 토큰 제거
                    filtered_token_counts = [t for t in problem_token_counts if t <= 16384]
                    total_instance_tokens = sum(filtered_token_counts)
                    instance_token_counts.append({
                        "problem_id": problem_id,
                        "token_counts": filtered_token_counts,
                        "total_solutions": len(solutions),
                        "filtered_solutions": len(filtered_token_counts),
                        "total_instance_tokens": total_instance_tokens,
                        "avg_tokens": sum(filtered_token_counts) / len(filtered_token_counts) if filtered_token_counts else 0,
                        "max_tokens": max(filtered_token_counts) if filtered_token_counts else 0,
                        "min_tokens": min(filtered_token_counts) if filtered_token_counts else 0
                    })
                
                # 16384 토큰을 넘는 instance는 평균 계산에서 제외
                filtered_instances = [inst for inst in instance_token_counts if inst.get("total_instance_tokens", 0) <= 16384]
                # 각 solution의 토큰 수가 16384을 넘는 경우도 제외
                all_tokens = [t for inst in filtered_instances for t in inst["token_counts"] if t <= 16384]

                # 검증: all_tokens에 16384을 넘는 값이 있는지 확인
                if all_tokens:
                    max_token_value = max(all_tokens)
                    if max_token_value > 16384:
                        logger.warning(f"경고: all_tokens에 16384을 넘는 값이 있습니다: {max_token_value}")
                        # 16384을 넘는 값 제거
                        all_tokens = [t for t in all_tokens if t <= 16384]
                
                analysis_results["baseline_token_distribution"] = {
                    "instance_level": instance_token_counts,
                    "dataset_level": {
                        "total_instances": len(instance_token_counts),
                        "filtered_instances": len(filtered_instances),
                        "excluded_instances": len(instance_token_counts) - len(filtered_instances),
                        "total_solutions": len(all_tokens),
                        "avg_tokens": sum(all_tokens) / len(all_tokens) if all_tokens else 0,
                        "max_tokens": max(all_tokens) if all_tokens else 0,
                        "min_tokens": min(all_tokens) if all_tokens else 0,
                        "median_tokens": sorted(all_tokens)[len(all_tokens)//2] if all_tokens else 0
                    }
                }
                
                # 최종 검증
                if all_tokens and analysis_results["baseline_token_distribution"]["dataset_level"]["max_tokens"] > 16384:
                    logger.error(f"오류: dataset_level max_tokens가 16384을 넘습니다: {analysis_results['baseline_token_distribution']['dataset_level']['max_tokens']}")
                    analysis_results["baseline_token_distribution"]["dataset_level"]["max_tokens"] = max([t for t in all_tokens if t <= 16384]) if all_tokens else 0
                logger.info(f"Baseline 분석 완료: {len(instance_token_counts)}개 인스턴스, 평균 {analysis_results['baseline_token_distribution']['dataset_level']['avg_tokens']:.1f} 토큰")
            
            # AggLLM 토큰 수 분석
            if aggllm_data:
                logger.info("AggLLM 토큰 수 분석 중...")
                aggllm_tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.base_model,
                    trust_remote_code=True
                )
                if aggllm_tokenizer.pad_token is None:
                    aggllm_tokenizer.pad_token = aggllm_tokenizer.eos_token
                
                instance_token_counts = []
                for problem_data in aggllm_data["generated_solutions"]:
                    solutions = problem_data["solutions"]
                    problem_id = problem_data["problem_id"]
                    
                    problem_token_counts = []
                    for sol in solutions:
                        content = sol.get("content", "")
                        if content:
                            tokens = aggllm_tokenizer.encode(content, add_special_tokens=False)
                            token_count = len(tokens)
                            problem_token_counts.append(token_count)
                    
                    # 16384을 넘는 토큰 제거
                    filtered_token_counts = [t for t in problem_token_counts if t <= 16384]
                    total_instance_tokens = sum(filtered_token_counts)
                    instance_token_counts.append({
                        "problem_id": problem_id,
                        "token_counts": filtered_token_counts,
                        "total_solutions": len(solutions),
                        "filtered_solutions": len(filtered_token_counts),
                        "total_instance_tokens": total_instance_tokens,
                        "avg_tokens": sum(filtered_token_counts) / len(filtered_token_counts) if filtered_token_counts else 0,
                        "max_tokens": max(filtered_token_counts) if filtered_token_counts else 0,
                        "min_tokens": min(filtered_token_counts) if filtered_token_counts else 0
                    })
                
                # 16384 토큰을 넘는 instance는 평균 계산에서 제외
                filtered_instances = [inst for inst in instance_token_counts if inst.get("total_instance_tokens", 0) <= 16384]
                # 각 solution의 토큰 수가 16384을 넘는 경우도 제외
                all_tokens = [t for inst in filtered_instances for t in inst["token_counts"] if t <= 16384]
                
                # 검증: all_tokens에 16384을 넘는 값이 있는지 확인
                if all_tokens:
                    max_token_value = max(all_tokens)
                    if max_token_value > 16384:
                        logger.warning(f"경고: all_tokens에 16384을 넘는 값이 있습니다: {max_token_value}")
                        # 16384을 넘는 값 제거
                        all_tokens = [t for t in all_tokens if t <= 16384]
                
                analysis_results["aggllm_token_distribution"] = {
                    "instance_level": instance_token_counts,
                    "dataset_level": {
                        "total_instances": len(instance_token_counts),
                        "filtered_instances": len(filtered_instances),
                        "excluded_instances": len(instance_token_counts) - len(filtered_instances),
                        "total_solutions": len(all_tokens),
                        "avg_tokens": sum(all_tokens) / len(all_tokens) if all_tokens else 0,
                        "max_tokens": max(all_tokens) if all_tokens else 0,
                        "min_tokens": min(all_tokens) if all_tokens else 0,
                        "median_tokens": sorted(all_tokens)[len(all_tokens)//2] if all_tokens else 0
                    }
                }
                
                # 최종 검증
                if all_tokens and analysis_results["aggllm_token_distribution"]["dataset_level"]["max_tokens"] > 16384:
                    logger.error(f"오류: dataset_level max_tokens가 16384을 넘습니다: {analysis_results['aggllm_token_distribution']['dataset_level']['max_tokens']}")
                    analysis_results["aggllm_token_distribution"]["dataset_level"]["max_tokens"] = max([t for t in all_tokens if t <= 16384]) if all_tokens else 0
                logger.info(f"AggLLM 분석 완료: {len(instance_token_counts)}개 인스턴스, 평균 {analysis_results['aggllm_token_distribution']['dataset_level']['avg_tokens']:.1f} 토큰")
            
            # 분석 결과 저장
            analysis_path = os.path.join(
                results_dir,
                f"{dataset_safe_name}_token_analysis.json"
            )
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            logger.info(f"토큰 분석 결과 저장: {analysis_path}")
        
        # Generation 단계는 모델별로 실행 (데이터셋 루프는 모델 루프 안에)
        # 이 부분은 아래 모델별 루프에서 처리됨
    
    # Generation 단계 - 모델별로 실행
    if do_generation:
        enable_thinking = eval_config.get("enable_thinking", False)
        group_sizes = eval_config.aggregation_group_sizes
        
        # Baseline 모델 처리
        if not skip_baseline:
            logger.info("=" * 60)
            logger.info("Baseline 모델 로드 중...")
            logger.info("=" * 60)
            baseline_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.base_model,
                trust_remote_code=True
            )
            if baseline_tokenizer.pad_token is None:
                baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
            
            baseline_llm = LLM(
                model=cfg.model.base_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=eval_config.get("gpu_memory_utilization", 0.9),
                max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384),
                dtype="bfloat16",
                trust_remote_code=True,
            )
            logger.info("Baseline 모델 로드 완료")
            
            # 데이터셋별로 Baseline 관련 inference 실행
            for benchmark in benchmark_datasets:
                dataset_name = benchmark["name"]
                dataset_path = benchmark["path"]
                dataset_safe_name = dataset_path.replace('/', '_')
                
                logger.info("=" * 60)
                logger.info(f"데이터셋: {dataset_name} (Baseline)")
                logger.info("=" * 60)
                
                # Baseline 결과 로드
                baseline_path = os.path.join(
                    results_dir,
                    f"{dataset_safe_name}_baseline_generated.json"
                )
                
                baseline_data = None
                if os.path.exists(baseline_path):
                    with open(baseline_path, 'r', encoding='utf-8') as f:
                        baseline_data = json.load(f)
                    logger.info(f"Baseline 결과 로드: {len(baseline_data['generated_solutions'])}개 문제")
                else:
                    logger.warning(f"Baseline 결과 파일 없음: {baseline_path}")
                    continue
                
                # 각 그룹 크기별로 Baseline 관련 inference 실행
                for group_size in group_sizes:
                    max_groups = None  # 각 그룹 크기별로 최대 4개 그룹만 사용
                    
                    logger.info("=" * 60)
                    logger.info(f"Baseline 관련 inference - 그룹 크기 {group_size}")
                    logger.info("=" * 60)
                    
                    # 이 그룹 크기에 대한 결과 초기화 (Baseline 관련만)
                    aggregation_results = {
                        "dataset_name": dataset_path,
                        "baseline_to_baseline_aggregation": [],
                        "baseline_to_baseline_aggregation_without_confidence": [],
                        "baseline_to_aggllm_aggregation": [],
                        "aggllm_to_aggllm_aggregation": []
                    }
                    
                    # Baseline → Baseline Aggregation
                    logger.info(f"Baseline → Baseline Aggregation 생성 중 (그룹 크기: {group_size})...")
                    
                    # 먼저 모든 문제에 대해 필터링 및 그룹화
                    aggregation_requests = []
                    for problem_data in baseline_data["generated_solutions"]:
                        problem_text = problem_data["problem_text"]
                        solutions = problem_data["solutions"]
                        ground_truth = problem_data["ground_truth"]
                        problem_id = problem_data["problem_id"]
                        
                        # 전체 16개를 먼저 필터링
                        filtered_solutions, _ = filter_solutions_by_token_length(
                            solutions,
                            baseline_tokenizer,
                            max_tokens=eval_config.filter_max_tokens
                        )
                        
                        # 그룹화 (group_size개씩, 최대 max_groups개)
                        solution_groups = group_solutions_for_aggregation(
                            filtered_solutions, 
                            target_group_size=group_size,
                            max_groups=max_groups
                        )
                        
                        # 각 그룹에 대해 요청 생성
                        for group in solution_groups:
                            correct_count = count_correct_solutions(group, ground_truth, math_verifier)
                            group_sizes_str = ",".join(str(len(g)) for g in solution_groups)
                            
                            aggregation_requests.append({
                                "problem_id": problem_id,
                                "problem_text": problem_text,
                                "ground_truth": ground_truth,
                                "solutions": group,
                                "correct_count": correct_count,
                                "group_sizes": group_sizes_str
                            })
                    
                    if aggregation_requests:
                            logger.info(f"총 {len(aggregation_requests)}개 aggregation 요청 준비 완료")
                            
                            # 배치로 프롬프트 준비 (confidence 포함)
                            formatted_prompts, prompt_texts = prepare_aggregation_prompts_batch(
                                baseline_tokenizer,
                                aggregation_requests,
                                aggregation_prompt_template,
                                enable_thinking,
                                include_confidence=True
                            )
                            
                            # SamplingParams 설정
                            sampling_params = SamplingParams(
                                temperature=eval_config.temperature,
                                max_tokens=eval_config.max_tokens,
                                top_p=eval_config.get("top_p", 0.8),
                                top_k=eval_config.get("top_k", 20),
                                min_p=eval_config.get("min_p", 0.0),
                            )
                            
                            # 배치로 생성
                            logger.info("배치 생성 시작...")
                            outputs = baseline_llm.generate(formatted_prompts, sampling_params)
                            logger.info("배치 생성 완료")
                            
                            # 결과 처리
                            for idx, output in enumerate(outputs):
                                req = aggregation_requests[idx]
                                agg_text = output.outputs[0].text
                                parsed_content = extract_content(agg_text)
                                agg_answer = math_verifier.extract_final_answer_from_content(parsed_content)
                                
                                aggregation_results["baseline_to_baseline_aggregation"].append({
                                    "problem_id": req["problem_id"],
                                    "problem_text": req["problem_text"],
                                    "ground_truth": req["ground_truth"],
                                    "prompt_text": prompt_texts[idx],
                                    "num_solutions": len(req["solutions"]),
                                    "correct_solutions_count": req["correct_count"],
                                    "generated_text": agg_text,
                                    "parsed_content": parsed_content,
                                    "final_answer": agg_answer,
                                    "is_correct": math_verifier.verify_answer(agg_answer, req["ground_truth"]) if do_evaluation else None,
                                    "solution_group_sizes": req["group_sizes"]
                                })
                            
                            # Baseline → Baseline Aggregation (without confidence)
                            logger.info("Baseline → Baseline Aggregation (without confidence) 생성 중...")
                            
                            # 배치로 프롬프트 준비 (confidence 제외)
                            formatted_prompts, prompt_texts = prepare_aggregation_prompts_batch(
                                baseline_tokenizer,
                                aggregation_requests,
                                aggregation_prompt_template,
                                enable_thinking,
                                include_confidence=False
                            )
                            
                            # 배치로 생성
                            logger.info("배치 생성 시작...")
                            outputs = baseline_llm.generate(formatted_prompts, sampling_params)
                            logger.info("배치 생성 완료")
                            
                            # 결과 처리
                            for idx, output in enumerate(outputs):
                                req = aggregation_requests[idx]
                                agg_text = output.outputs[0].text
                                parsed_content = extract_content(agg_text)
                                agg_answer = math_verifier.extract_final_answer_from_content(parsed_content)
                                
                                aggregation_results["baseline_to_baseline_aggregation_without_confidence"].append({
                                    "problem_id": req["problem_id"],
                                    "problem_text": req["problem_text"],
                                    "ground_truth": req["ground_truth"],
                                    "prompt_text": prompt_texts[idx],
                                    "num_solutions": len(req["solutions"]),
                                    "correct_solutions_count": req["correct_count"],
                                    "generated_text": agg_text,
                                    "parsed_content": parsed_content,
                                    "final_answer": agg_answer,
                                    "is_correct": math_verifier.verify_answer(agg_answer, req["ground_truth"]) if do_evaluation else None,
                                    "solution_group_sizes": req["group_sizes"]
                                })
                    
                    # Baseline 관련 결과 임시 저장
                    group_results_dir = os.path.join(results_dir, f"group_size_{group_size}")
                    os.makedirs(group_results_dir, exist_ok=True)
                    aggregation_path = os.path.join(
                        group_results_dir,
                        f"{dataset_safe_name}_aggregation_results.json"
                    )
                    
                    # Baseline 결과만 먼저 저장
                    formatted_results = {
                        "dataset_name": dataset_path,
                        "group_size": group_size,
                        "baseline_to_baseline_aggregation": {},
                        "baseline_to_baseline_aggregation_without_confidence": {},
                        "baseline_to_aggllm_aggregation": {},
                        "aggllm_to_aggllm_aggregation": {}
                    }
                    
                    # Baseline 결과 그룹화 및 저장
                    for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence"]:
                        results = aggregation_results.get(key, [])
                        if results:
                            grouped = group_results_by_problem_id(results)
                            formatted_results[key] = grouped
                    
                    # Evaluation 단계 (이 그룹 크기에 대해)
                    if do_evaluation:
                        # 평가 수행 (is_correct가 None인 경우에만)
                        for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence"]:
                            results_dict = formatted_results.get(key, {})
                            if isinstance(results_dict, dict):
                                for problem_id, problem_data in results_dict.items():
                                    prompts = problem_data.get("prompts", [])
                                    for prompt in prompts:
                                        if prompt.get("is_correct") is None:
                                            final_answer = prompt.get("final_answer", "")
                                            if final_answer:
                                                prompt["is_correct"] = math_verifier.verify_answer(
                                                    final_answer,
                                                    problem_data["ground_truth"]
                                                )
                    
                    # 정확도 계산 및 저장
                    summary = {}
                    for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence"]:
                        results_dict = formatted_results.get(key, {})
                        if isinstance(results_dict, dict):
                            total = 0
                            correct = 0
                            for problem_id, problem_data in results_dict.items():
                                prompts = problem_data.get("prompts", [])
                                for prompt in prompts:
                                    total += 1
                                    if prompt.get("is_correct", False):
                                        correct += 1
                            summary[key] = {
                                "correct": correct,
                                "total": total,
                                "accuracy": correct / total if total > 0 else 0.0
                            }
                    
                    formatted_results["summary"] = summary
                    
                    with open(aggregation_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"그룹 크기 {group_size} Baseline 결과 저장 완료")
                    
                    # 정확도만 저장하는 jsonl 파일 생성
                    save_accuracy_jsonl(
                        summary=summary,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        group_size=group_size,
                        results_dir=results_dir,
                        dataset_safe_name=dataset_safe_name,
                        checkpoint_num=None  # Baseline 결과에는 checkpoint_num 없음
                    )
            
            # Baseline 모델 unload
            logger.info("Baseline 모델 unload 중...")
            del baseline_llm
            torch.cuda.empty_cache()
            logger.info("Baseline 모델 unload 완료")
        
        # AggLLM 모델 처리
        if aggllm_model_path:
            logger.info("=" * 60)
            logger.info("AggLLM 모델 로드 중...")
            logger.info("=" * 60)
            aggllm_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.base_model,
                trust_remote_code=True
            )
            if aggllm_tokenizer.pad_token is None:
                aggllm_tokenizer.pad_token = aggllm_tokenizer.eos_token
            
            # LoRA 병합
            logger.info("LoRA 가중치 병합 중...")
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
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
                gpu_memory_utilization=eval_config.get("aggllm_gpu_memory_utilization", 0.9),
                max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384),
                dtype="bfloat16",
                trust_remote_code=True,
            )
            logger.info("AggLLM 모델 로드 완료")
            
            # Baseline tokenizer 로드 (Baseline → AggLLM Aggregation에 필요, skip_baseline이어도 필요)
            baseline_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.base_model,
                trust_remote_code=True
            )
            if baseline_tokenizer.pad_token is None:
                baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
            
            # 데이터셋별로 AggLLM 관련 inference 실행
            for benchmark in benchmark_datasets:
                dataset_name = benchmark["name"]
                dataset_path = benchmark["path"]
                dataset_safe_name = dataset_path.replace('/', '_')
                
                logger.info("=" * 60)
                logger.info(f"데이터셋: {dataset_name} (AggLLM)")
                logger.info("=" * 60)
                
                # Baseline 결과 로드
                baseline_path = os.path.join(
                    results_dir,
                    f"{dataset_safe_name}_baseline_generated.json"
                )
                
                baseline_data = None
                if os.path.exists(baseline_path):
                    with open(baseline_path, 'r', encoding='utf-8') as f:
                        baseline_data = json.load(f)
                    logger.info(f"Baseline 결과 로드: {len(baseline_data['generated_solutions'])}개 문제")
                
                # AggLLM 결과 로드
                aggllm_path = os.path.join(
                    results_dir,
                    f"{dataset_safe_name}_aggllm_generated.json"
                )
                
                aggllm_data = None
                if os.path.exists(aggllm_path):
                    with open(aggllm_path, 'r', encoding='utf-8') as f:
                        aggllm_data = json.load(f)
                    logger.info(f"AggLLM 결과 로드: {len(aggllm_data['generated_solutions'])}개 문제")
                
                if not baseline_data and not aggllm_data:
                    logger.warning(f"{dataset_name}에 대한 결과 파일이 없어 건너뜁니다.")
                    continue
                
                # 각 그룹 크기별로 AggLLM 관련 inference 실행
                for group_size in group_sizes:
                    max_groups = None
                    
                    logger.info("=" * 60)
                    logger.info(f"AggLLM 관련 inference - 그룹 크기 {group_size}")
                    logger.info("=" * 60)
                    
                    # 기존 결과 로드 (Baseline 결과가 있으면)
                    group_results_dir = os.path.join(results_dir, f"group_size_{group_size}")
                    # 디렉토리 생성 (Baseline을 건너뛴 경우를 대비)
                    os.makedirs(group_results_dir, exist_ok=True)
                    
                    baseline_aggregation_path = os.path.join(
                        group_results_dir,
                        f"{dataset_safe_name}_aggregation_results.json"
                    )
                    
                    # checkpoint_num이 있으면 다른 파일명으로 저장
                    if checkpoint_num is not None:
                        aggregation_path = os.path.join(
                            group_results_dir,
                            f"{dataset_safe_name}_aggregation_results_checkpoint_{checkpoint_num}.json"
                        )
                    else:
                        aggregation_path = baseline_aggregation_path
                    
                    # 기존 Baseline 결과 로드 (있으면)
                    if os.path.exists(baseline_aggregation_path):
                        with open(baseline_aggregation_path, 'r', encoding='utf-8') as f:
                            formatted_results = json.load(f)
                    else:
                        formatted_results = {
                            "dataset_name": dataset_path,
                            "group_size": group_size,
                            "baseline_to_baseline_aggregation": {},
                            "baseline_to_baseline_aggregation_without_confidence": {},
                            "baseline_to_aggllm_aggregation": {},
                            "aggllm_to_aggllm_aggregation": {}
                        }
                    
                    # 결과 초기화 (AggLLM 관련만)
                    aggregation_results = {
                        "baseline_to_aggllm_aggregation": [],
                        "aggllm_to_aggllm_aggregation": []
                    }
                    
                    # Baseline → AggLLM Aggregation
                    if baseline_data:
                        logger.info(f"Baseline → AggLLM Aggregation 생성 중 (그룹 크기: {group_size})...")
                        
                        # 먼저 모든 문제에 대해 필터링 및 그룹화
                        aggregation_requests = []
                        for problem_data in baseline_data["generated_solutions"]:
                            problem_text = problem_data["problem_text"]
                            solutions = problem_data["solutions"]
                            ground_truth = problem_data["ground_truth"]
                            problem_id = problem_data["problem_id"]
                            
                            # 전체 16개를 먼저 필터링 (baseline tokenizer 사용)
                            filtered_solutions, _ = filter_solutions_by_token_length(
                                solutions,
                                baseline_tokenizer,
                                max_tokens=eval_config.filter_max_tokens
                            )
                            
                            # 그룹화 (group_size개씩, 최대 max_groups개)
                            solution_groups = group_solutions_for_aggregation(
                                filtered_solutions, 
                                target_group_size=group_size,
                                max_groups=max_groups
                            )
                            
                            # 각 그룹에 대해 요청 생성
                            for group in solution_groups:
                                correct_count = count_correct_solutions(group, ground_truth, math_verifier)
                                group_sizes_str = ",".join(str(len(g)) for g in solution_groups)
                                
                                aggregation_requests.append({
                                    "problem_id": problem_id,
                                    "problem_text": problem_text,
                                    "ground_truth": ground_truth,
                                    "solutions": group,
                                    "correct_count": correct_count,
                                    "group_sizes": group_sizes_str
                                })
                        
                        if aggregation_requests:
                            logger.info(f"총 {len(aggregation_requests)}개 aggregation 요청 준비 완료")
                            
                            # 배치로 프롬프트 준비 (confidence 포함)
                            formatted_prompts, prompt_texts = prepare_aggregation_prompts_batch(
                                aggllm_tokenizer,
                                aggregation_requests,
                                aggregation_prompt_template,
                                enable_thinking,
                                include_confidence=True
                            )
                            
                            # SamplingParams 설정
                            sampling_params = SamplingParams(
                                temperature=eval_config.temperature,
                                max_tokens=eval_config.max_tokens,
                                top_p=eval_config.get("top_p", 0.8),
                                top_k=eval_config.get("top_k", 20),
                                min_p=eval_config.get("min_p", 0.0),
                            )
                            
                            # 배치로 생성
                            logger.info("배치 생성 시작...")
                            outputs = aggllm_llm.generate(formatted_prompts, sampling_params)
                            logger.info("배치 생성 완료")
                            
                            # 결과 처리
                            for idx, output in enumerate(outputs):
                                req = aggregation_requests[idx]
                                agg_text = output.outputs[0].text
                                parsed_content = extract_content(agg_text)
                                agg_answer = math_verifier.extract_final_answer_from_content(parsed_content)
                                
                                aggregation_results["baseline_to_aggllm_aggregation"].append({
                                    "problem_id": req["problem_id"],
                                    "problem_text": req["problem_text"],
                                    "ground_truth": req["ground_truth"],
                                    "prompt_text": prompt_texts[idx],
                                    "num_solutions": len(req["solutions"]),
                                    "correct_solutions_count": req["correct_count"],
                                    "generated_text": agg_text,
                                    "parsed_content": parsed_content,
                                    "final_answer": agg_answer,
                                    "is_correct": math_verifier.verify_answer(agg_answer, req["ground_truth"]) if do_evaluation else None,
                                    "solution_group_sizes": req["group_sizes"]
                                })
                    
                    # AggLLM → AggLLM Aggregation
                    if aggllm_data:
                        logger.info(f"AggLLM → AggLLM Aggregation 생성 중 (그룹 크기: {group_size})...")
                        
                        # 먼저 모든 문제에 대해 필터링 및 그룹화
                        aggregation_requests = []
                        for problem_data in aggllm_data["generated_solutions"]:
                            problem_text = problem_data["problem_text"]
                            solutions = problem_data["solutions"]
                            ground_truth = problem_data["ground_truth"]
                            problem_id = problem_data["problem_id"]
                            
                            # 전체 16개를 먼저 필터링
                            filtered_solutions, _ = filter_solutions_by_token_length(
                                solutions,
                                aggllm_tokenizer,
                                max_tokens=eval_config.filter_max_tokens
                            )
                            
                            # 그룹화 (group_size개씩, 최대 max_groups개)
                            solution_groups = group_solutions_for_aggregation(
                                filtered_solutions, 
                                target_group_size=group_size,
                                max_groups=max_groups
                            )
                            
                            # 각 그룹에 대해 요청 생성
                            for group in solution_groups:
                                correct_count = count_correct_solutions(group, ground_truth, math_verifier)
                                group_sizes_str = ",".join(str(len(g)) for g in solution_groups)
                                
                                aggregation_requests.append({
                                    "problem_id": problem_id,
                                    "problem_text": problem_text,
                                    "ground_truth": ground_truth,
                                    "solutions": group,
                                    "correct_count": correct_count,
                                    "group_sizes": group_sizes_str
                                })
                        
                        if aggregation_requests:
                            logger.info(f"총 {len(aggregation_requests)}개 aggregation 요청 준비 완료")
                            
                            # 배치로 프롬프트 준비
                            formatted_prompts, prompt_texts = prepare_aggregation_prompts_batch(
                                aggllm_tokenizer,
                                aggregation_requests,
                                aggregation_prompt_template,
                                enable_thinking,
                                include_confidence=True
                            )
                            
                            # SamplingParams 설정
                            sampling_params = SamplingParams(
                                temperature=eval_config.temperature,
                                max_tokens=eval_config.max_tokens,
                                top_p=eval_config.get("top_p", 0.8),
                                top_k=eval_config.get("top_k", 20),
                                min_p=eval_config.get("min_p", 0.0),
                            )
                            
                            # 배치로 생성
                            logger.info("배치 생성 시작...")
                            outputs = aggllm_llm.generate(formatted_prompts, sampling_params)
                            logger.info("배치 생성 완료")
                            
                            # 결과 처리
                            for idx, output in enumerate(outputs):
                                req = aggregation_requests[idx]
                                agg_text = output.outputs[0].text
                                parsed_content = extract_content(agg_text)
                                agg_answer = math_verifier.extract_final_answer_from_content(parsed_content)
                                
                                aggregation_results["aggllm_to_aggllm_aggregation"].append({
                                    "problem_id": req["problem_id"],
                                    "problem_text": req["problem_text"],
                                    "ground_truth": req["ground_truth"],
                                    "prompt_text": prompt_texts[idx],
                                    "num_solutions": len(req["solutions"]),
                                    "correct_solutions_count": req["correct_count"],
                                    "generated_text": agg_text,
                                    "parsed_content": parsed_content,
                                    "final_answer": agg_answer,
                                    "is_correct": math_verifier.verify_answer(agg_answer, req["ground_truth"]) if do_evaluation else None,
                                    "solution_group_sizes": req["group_sizes"]
                                })
                    
                    # AggLLM 관련 결과를 기존 결과에 추가
                    for key in ["baseline_to_aggllm_aggregation", "aggllm_to_aggllm_aggregation"]:
                        results = aggregation_results.get(key, [])
                        if results:
                            grouped = group_results_by_problem_id(results)
                            formatted_results[key] = grouped
                    
                    # Evaluation 단계 (이 그룹 크기에 대해)
                    if do_evaluation:
                        # 평가 수행 (is_correct가 None인 경우에만)
                        for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence",
                                   "baseline_to_aggllm_aggregation", "aggllm_to_aggllm_aggregation"]:
                            results_dict = formatted_results.get(key, {})
                            if isinstance(results_dict, dict):
                                for problem_id, problem_data in results_dict.items():
                                    prompts = problem_data.get("prompts", [])
                                    for prompt in prompts:
                                        if prompt.get("is_correct") is None:
                                            final_answer = prompt.get("final_answer", "")
                                            if final_answer:
                                                prompt["is_correct"] = math_verifier.verify_answer(
                                                    final_answer,
                                                    problem_data["ground_truth"]
                                                )
                    
                    # 정확도 계산 및 최종 저장
                    summary = {}
                    for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence",
                               "baseline_to_aggllm_aggregation", "aggllm_to_aggllm_aggregation"]:
                        results_dict = formatted_results.get(key, {})
                        if isinstance(results_dict, dict):
                            total = 0
                            correct = 0
                            for problem_id, problem_data in results_dict.items():
                                prompts = problem_data.get("prompts", [])
                                for prompt in prompts:
                                    total += 1
                                    if prompt.get("is_correct", False):
                                        correct += 1
                            summary[key] = {
                                "correct": correct,
                                "total": total,
                                "accuracy": correct / total if total > 0 else 0.0
                            }
                    
                    formatted_results["summary"] = summary
                    
                    with open(aggregation_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"그룹 크기 {group_size} Aggregation 결과 저장: {aggregation_path}")
                    logger.info(f"그룹 크기 {group_size} 요약: {summary}")
                    
                    # 정확도만 저장하는 jsonl 파일 생성
                    save_accuracy_jsonl(
                        summary=summary,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        group_size=group_size,
                        results_dir=results_dir,
                        dataset_safe_name=dataset_safe_name,
                        checkpoint_num=checkpoint_num
                    )
            
            # AggLLM 모델 unload
            logger.info("AggLLM 모델 unload 중...")
            del aggllm_llm
            torch.cuda.empty_cache()
            logger.info("AggLLM 모델 unload 완료")
        
        # Evaluation 단계 (저장된 결과가 있으면 평가) - 기존 형식 지원
        if do_evaluation and not do_generation:
            logger.info("=" * 60)
            logger.info("Evaluation 단계 시작")
            logger.info("=" * 60)
            
            # 저장된 aggregation 결과가 있으면 로드
            aggregation_path = os.path.join(
                results_dir,
                f"{dataset_safe_name}_aggregation_results.json"
            )
            
            if os.path.exists(aggregation_path) and not do_generation:
                # Generation 없이 Evaluation만 하는 경우
                with open(aggregation_path, 'r', encoding='utf-8') as f:
                    aggregation_results = json.load(f)
                logger.info(f"저장된 Aggregation 결과 로드: {aggregation_path}")
            
            # 기존 형식의 결과 파일이 있으면 평가 (그룹 크기별로)
            for group_size in group_sizes:
                group_results_dir = os.path.join(results_dir, f"group_size_{group_size}")
                aggregation_path = os.path.join(
                    group_results_dir,
                    f"{dataset_safe_name}_aggregation_results.json"
                )
                
                if os.path.exists(aggregation_path):
                    with open(aggregation_path, 'r', encoding='utf-8') as f:
                        formatted_results = json.load(f)
                    
                    # 평가 수행 (is_correct가 None인 경우에만)
                    for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence",
                               "baseline_to_aggllm_aggregation", "aggllm_to_aggllm_aggregation"]:
                        results_dict = formatted_results.get(key, {})
                        if isinstance(results_dict, dict):
                            for problem_id, problem_data in results_dict.items():
                                prompts = problem_data.get("prompts", [])
                                for prompt in prompts:
                                    if prompt.get("is_correct") is None:
                                        final_answer = prompt.get("final_answer", "")
                                        if final_answer:
                                            prompt["is_correct"] = math_verifier.verify_answer(
                                                final_answer,
                                                problem_data["ground_truth"]
                                            )
                    
                    # 정확도 재계산
                    summary = {}
                    for key in ["baseline_to_baseline_aggregation", "baseline_to_baseline_aggregation_without_confidence",
                               "baseline_to_aggllm_aggregation", "aggllm_to_aggllm_aggregation"]:
                        results_dict = formatted_results.get(key, {})
                        if isinstance(results_dict, dict):
                            total = 0
                            correct = 0
                            for problem_id, problem_data in results_dict.items():
                                prompts = problem_data.get("prompts", [])
                                for prompt in prompts:
                                    total += 1
                                    if prompt.get("is_correct", False):
                                        correct += 1
                            summary[key] = {
                                "correct": correct,
                                "total": total,
                                "accuracy": correct / total if total > 0 else 0.0
                            }
                    
                    formatted_results["summary"] = summary
                    
                    with open(aggregation_path, 'w', encoding='utf-8') as f:
                        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"그룹 크기 {group_size} 평가 완료 및 저장: {aggregation_path}")
                    
                    # 정확도만 저장하는 jsonl 파일 생성
                    save_accuracy_jsonl(
                        summary=summary,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        group_size=group_size,
                        results_dir=results_dir,
                        dataset_safe_name=dataset_safe_name
                    )
    
    logger.info("=" * 60)
    logger.info("✅ Stage 4-3: Aggregation 평가 완료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


