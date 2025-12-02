"""
Stage 4-1: Baseline과 AggLLM Solution 생성
모든 데이터셋에 대해 배치로 solution 생성 후 저장
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
from datasets import load_dataset
from vllm import LLM, SamplingParams
import tempfile

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.confidence import ConfidenceCalculator
from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging
from src.evaluation.comprehensive_benchmark import (
    extract_reasoning_content,
    extract_content,
    simple_extract_topk
)

logger = logging.getLogger(__name__)


def load_dataset_data(dataset_name: str) -> list:
    """데이터셋 로드 및 전처리"""
    logger.info(f"벤치마크 데이터셋 로드 중: {dataset_name}")
    
    try:
        # aime는 test, hmmt는 train으로 split을 다르게 로드
        if "aime" in dataset_name.lower():
            dataset = load_dataset(dataset_name, split="test")
        elif "hmmt" in dataset_name.lower():
            dataset = load_dataset(dataset_name, split="train")
        else:
            dataset = load_dataset(dataset_name, split="test")
        
        data = []
        for item in dataset:
            problem_text = item.get("problem", item.get("question", ""))
            ground_truth = item.get("answer", item.get("solution", ""))
            
            # aime24의 answer가 \boxed{xxx} 형태라면 중괄호 안 값만 저장
            if "aime24" in dataset_name.lower() and isinstance(ground_truth, str):
                import re
                match = re.search(r"\\boxed\{([^{}]+)\}", ground_truth)
                if match:
                    ground_truth = match.group(1).strip()
            
            data.append({
                "problem_id": len(data),
                "problem_text": problem_text,
                "ground_truth": ground_truth
            })
        
        logger.info(f"데이터셋 로드 완료: {len(data)}개 문제")
        return data
        
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        return []


def generate_solutions_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    problems: list,
    num_solutions: int,
    base_instruction: str,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    logprobs: int,
    confidence_calculator: ConfidenceCalculator,
    math_verifier: MathVerifier
) -> list:
    """
    전체 문제에 대해 배치로 solution 생성
    
    Returns:
        각 문제별로 16개 solution을 포함한 리스트
    """
    logger.info(f"배치 생성 시작: {len(problems)}개 문제, 문제당 {num_solutions}개 solution")
    
    # 모든 프롬프트 준비 (문제당 num_solutions개씩)
    all_prompts = []
    problem_indices = []  # 각 프롬프트가 어느 문제에 속하는지
    
    for problem in problems:
        prompt = f"{problem['problem_text']}\n\n{base_instruction}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        
        # 각 문제당 num_solutions개씩 추가
        for _ in range(num_solutions):
            all_prompts.append(formatted_prompt)
            problem_indices.append(problem['problem_id'])
    
    logger.info(f"총 {len(all_prompts)}개 프롬프트 생성 완료")
    
    # SamplingParams 설정
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        logprobs=logprobs,
    )
    
    # vLLM으로 배치 생성
    logger.info("vLLM 배치 생성 시작...")
    outputs = llm.generate(all_prompts, sampling_params)
    logger.info("vLLM 배치 생성 완료")
    
    # 결과를 문제별로 그룹화
    problem_solutions = {pid: [] for pid in range(len(problems))}
    
    for idx, output in enumerate(outputs):
        problem_id = problem_indices[idx]
        generated_text = output.outputs[0].text
        
        # logprobs 추출
        logprobs_list = []
        if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs:
            logprobs_list = simple_extract_topk(output.outputs[0].logprobs, logprobs)
        
        # 신뢰도 점수 계산
        if logprobs_list:
            confidence_scores = confidence_calculator.calculate_all_confidence_scores(logprobs_list)
        else:
            confidence_scores = {
                "mean_group_confidence": 0.0,
                "bottom_10_percent_confidence": 0.0,
                "tail_confidence": 0.0,
                "lowest_group_confidence": 0.0
            }
        
        # enable_thinking에 따라 파싱
        if enable_thinking:
            reasoning_content = extract_reasoning_content(generated_text)
            content = extract_content(generated_text)
            if not content:
                content = generated_text
        else:
            reasoning_content = ""
            content = generated_text
        
        # final_answer 추출
        final_answer = math_verifier.extract_final_answer_from_content(content)
        
        solution = {
            "generated_text": generated_text,
            "reasoning_content": reasoning_content,
            "content": content,
            "final_answer": final_answer,
            "confidence_scores": confidence_scores
        }
        
        problem_solutions[problem_id].append(solution)
    
    # 문제별로 리스트로 변환
    results = []
    for problem in problems:
        problem_id = problem['problem_id']
        results.append({
            "problem_id": problem_id,
            "problem_text": problem['problem_text'],
            "ground_truth": problem['ground_truth'],
            "solutions": problem_solutions[problem_id]
        })
    
    return results


def load_baseline_model(
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int
) -> tuple:
    """Baseline 모델과 토크나이저 로드"""
    logger.info("=" * 60)
    logger.info("Baseline 모델 로드 시작")
    logger.info("=" * 60)
    
    # 토크나이저 로드
    logger.info(f"Baseline 토크나이저 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # vLLM 모델 로드
    logger.info(f"Baseline vLLM 모델 로드 중...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        kv_cache_dtype="fp8"
    )
    logger.info("Baseline 모델 로드 완료")
    
    return llm, tokenizer


def generate_baseline_for_dataset(
    llm: LLM,
    tokenizer: AutoTokenizer,
    problems: list,
    output_path: str,
    num_solutions: int,
    base_instruction: str,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    logprobs: int,
    confidence_calculator: ConfidenceCalculator,
    math_verifier: MathVerifier
):
    """이미 로드된 Baseline 모델로 특정 데이터셋에 대해 solution 생성"""
    dataset_name = problems[0].get("dataset_name", "unknown")
    logger.info(f"Baseline 모델로 {dataset_name} 데이터셋 생성 시작")
    
    # 배치 생성
    results = generate_solutions_batch(
        llm=llm,
        tokenizer=tokenizer,
        problems=problems,
        num_solutions=num_solutions,
        base_instruction=base_instruction,
        enable_thinking=enable_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        logprobs=logprobs,
        confidence_calculator=confidence_calculator,
        math_verifier=math_verifier
    )
    
    # 저장
    output_data = {
        "dataset_name": dataset_name,
        "total_problems": len(problems),
        "generated_solutions": results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Baseline 결과 저장 완료: {output_path}")


def load_aggllm_model(
    model_name: str,
    aggllm_model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    merged_model_cache_dir: str
) -> tuple:
    """AggLLM 모델과 토크나이저 로드"""
    logger.info("=" * 60)
    logger.info("AggLLM 모델 로드 시작")
    logger.info("=" * 60)
    
    # 토크나이저 로드
    logger.info(f"AggLLM 토크나이저 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA 병합 및 저장
    logger.info("LoRA 가중치를 base 모델에 병합 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    peft_model = PeftModel.from_pretrained(base_model, aggllm_model_path)
    merged_model = peft_model.merge_and_unload()
    
    # 병합된 모델 저장 경로 결정
    if merged_model_cache_dir:
        merged_model_path = merged_model_cache_dir
    else:
        merged_model_path = tempfile.mkdtemp(prefix="aggllm_merged_")
    
    os.makedirs(merged_model_path, exist_ok=True)
    
    # 병합된 모델이 이미 저장되어 있는지 확인
    config_path = os.path.join(merged_model_path, "config.json")
    if not os.path.exists(config_path):
        logger.info(f"병합된 모델 저장 중: {merged_model_path}")
        merged_model.save_pretrained(merged_model_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_path)
        logger.info("병합된 모델 저장 완료")
    else:
        logger.info(f"기존 병합된 모델 사용: {merged_model_path}")
    
    # 메모리 정리
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()
    
    # vLLM으로 로드
    logger.info(f"vLLM으로 AggLLM 모델 로드 중... (GPU memory utilization: {gpu_memory_utilization})")
    llm = LLM(
        model=merged_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",  # FP8 KV cache 사용 시 BF16 필요
        trust_remote_code=True,
        kv_cache_dtype="fp8"
    )
    logger.info("AggLLM 모델 로드 완료")
    
    return llm, tokenizer


def generate_aggllm_for_dataset(
    llm: LLM,
    tokenizer: AutoTokenizer,
    problems: list,
    output_path: str,
    num_solutions: int,
    base_instruction: str,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    logprobs: int,
    confidence_calculator: ConfidenceCalculator,
    math_verifier: MathVerifier
):
    """이미 로드된 AggLLM 모델로 특정 데이터셋에 대해 solution 생성"""
    dataset_name = problems[0].get("dataset_name", "unknown")
    logger.info(f"AggLLM 모델로 {dataset_name} 데이터셋 생성 시작")
    
    # 배치 생성
    results = generate_solutions_batch(
        llm=llm,
        tokenizer=tokenizer,
        problems=problems,
        num_solutions=num_solutions,
        base_instruction=base_instruction,
        enable_thinking=enable_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        logprobs=logprobs,
        confidence_calculator=confidence_calculator,
        math_verifier=math_verifier
    )
    
    # 저장
    output_data = {
        "dataset_name": dataset_name,
        "total_problems": len(problems),
        "generated_solutions": results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"AggLLM 결과 저장 완료: {output_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-1: Solution 생성 메인 함수"""
    
    # GPU 설정 (CUDA_VISIBLE_DEVICES가 설정된 경우를 대비)
    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES가 설정되어 있으면 0번이 실제 GPU
        # 명시적으로 GPU를 설정하여 프로세스 간 격리 보장
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_1_generate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    if torch.cuda.is_available():
        logger.info(f"GPU 설정: device=0, GPU={torch.cuda.get_device_name(0)}")
    
    logger.info("🚀 Stage 4-1: Solution 생성 시작")

    eval_config = cfg.evaluation.benchmarks.evaluation
    enable_thinking = eval_config.get("enable_thinking", True)

    # 디렉토리 생성
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    results_dir = os.path.join(results_dir, cfg.model.base_model.replace('/', '_') + f"_think_{enable_thinking}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 모델 경로 확인
    # checkpoint_num = cfg.evaluation.benchmarks.evaluation.checkpoint_num
    checkpoint_num = None
    aggllm_model_path = None
    # if checkpoint_num is not None:
    #     aggllm_model_path = os.path.join(cfg.paths.model_dir, f"checkpoint-{checkpoint_num}")
    # else:
    #     aggllm_model_path = None
    #     logger.warning("AggLLM 모델을 찾을 수 없습니다. Baseline 모델만 사용하여 생성합니다.")
    
    # 평가 구성 요소 초기화
    
    confidence_group_size = eval_config.get("confidence_group_size", 512)
    
    confidence_calculator = ConfidenceCalculator(group_size=confidence_group_size)
    math_verifier = MathVerifier(timeout=eval_config.timeout)
    
    base_instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    
    # 벤치마크 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]
    
    # 모든 데이터셋 미리 로드
    all_datasets = []
    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        
        logger.info(f"데이터셋 로드 중: {dataset_name}")
        problems = load_dataset_data(dataset_path)
        if not problems:
            logger.warning(f"{dataset_name} 데이터셋 로드 실패, 건너뜀")
            continue
        
        # dataset_name 추가
        for p in problems:
            p["dataset_name"] = dataset_name
        
        dataset_safe_name = dataset_path.replace('/', '_')
        all_datasets.append({
            "name": dataset_name,
            "path": dataset_path,
            "safe_name": dataset_safe_name,
            "problems": problems
        })
    
    logger.info(f"총 {len(all_datasets)}개 데이터셋 로드 완료")
    
    # Stage 1: Baseline 모델로 모든 데이터셋 생성
    logger.info("=" * 60)
    logger.info("Baseline 모델로 모든 데이터셋 생성 시작")
    logger.info("=" * 60)
    
    baseline_llm = None
    baseline_tokenizer = None
    
    try:
        # Baseline 모델 로드
        baseline_llm, baseline_tokenizer = load_baseline_model(
            model_name=cfg.model.base_model,
            gpu_memory_utilization=eval_config.get("gpu_memory_utilization", 0.9),
            max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384)
        )
        
        # 각 데이터셋에 대해 생성
        for dataset_info in all_datasets:
            dataset_name = dataset_info["name"]
            dataset_safe_name = dataset_info["safe_name"]
            problems = dataset_info["problems"]
            
            logger.info("=" * 60)
            logger.info(f"Baseline - 데이터셋: {dataset_name}")
            logger.info("=" * 60)
            
            baseline_output_path = os.path.join(
                results_dir, 
                f"{dataset_safe_name}_baseline_generated.json"
            )
            
            try:
                generate_baseline_for_dataset(
                    llm=baseline_llm,
                    tokenizer=baseline_tokenizer,
                    problems=problems,
                    output_path=baseline_output_path,
                    num_solutions=64,
                    base_instruction=base_instruction,
                    enable_thinking=enable_thinking,
                    temperature=eval_config.temperature,
                    max_tokens=eval_config.max_tokens,
                    top_p=eval_config.get("top_p", 0.95),
                    top_k=eval_config.get("top_k", 20),
                    min_p=eval_config.get("min_p", 0.0),
                    logprobs=eval_config.get("logprobs", 5),
                    confidence_calculator=confidence_calculator,
                    math_verifier=math_verifier
                )
            except Exception as e:
                logger.error(f"{dataset_name} Baseline 생성 실패: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
    finally:
        # Baseline 모델 unload
        if baseline_llm is not None:
            logger.info("Baseline 모델 unload 중...")
            del baseline_llm
            del baseline_tokenizer
            torch.cuda.empty_cache()
            logger.info("Baseline 모델 unload 완료")
    
    # Stage 2: AggLLM 모델로 모든 데이터셋 생성
    if aggllm_model_path:
        logger.info("=" * 60)
        logger.info("AggLLM 모델로 모든 데이터셋 생성 시작")
        logger.info("=" * 60)
        
        aggllm_llm = None
        aggllm_tokenizer = None
        
        try:
            # AggLLM 모델 로드
            aggllm_llm, aggllm_tokenizer = load_aggllm_model(
                model_name=cfg.model.base_model,
                aggllm_model_path=aggllm_model_path,
                gpu_memory_utilization=eval_config.get("aggllm_gpu_memory_utilization", 0.9),
                max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 8192),
                merged_model_cache_dir=cfg.paths.get("merged_model_cache_dir", None)
            )
            
            # 각 데이터셋에 대해 생성
            for dataset_info in all_datasets:
                dataset_name = dataset_info["name"]
                dataset_safe_name = dataset_info["safe_name"]
                problems = dataset_info["problems"]
                
                logger.info("=" * 60)
                logger.info(f"AggLLM - 데이터셋: {dataset_name}")
                logger.info("=" * 60)
                
                aggllm_output_path = os.path.join(
                    results_dir,
                    f"{dataset_safe_name}_aggllm_generated.json"
                )
                
                try:
                    generate_aggllm_for_dataset(
                        llm=aggllm_llm,
                        tokenizer=aggllm_tokenizer,
                        problems=problems,
                        output_path=aggllm_output_path,
                        num_solutions=64,
                        base_instruction=base_instruction,
                        enable_thinking=enable_thinking,
                        temperature=eval_config.temperature,
                        max_tokens=eval_config.max_tokens,
                        top_p=eval_config.get("top_p", 0.8),
                        top_k=eval_config.get("top_k", 20),
                        min_p=eval_config.get("min_p", 0.0),
                        logprobs=eval_config.get("logprobs", 5),
                        confidence_calculator=confidence_calculator,
                        math_verifier=math_verifier
                    )
                except Exception as e:
                    logger.error(f"{dataset_name} AggLLM 생성 실패: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        finally:
            # AggLLM 모델 unload
            if aggllm_llm is not None:
                logger.info("AggLLM 모델 unload 중...")
                del aggllm_llm
                del aggllm_tokenizer
                torch.cuda.empty_cache()
                logger.info("AggLLM 모델 unload 완료")
    
    logger.info("=" * 60)
    logger.info("✅ Stage 4-1: Solution 생성 완료")
    logger.info(f"결과 저장 위치: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

