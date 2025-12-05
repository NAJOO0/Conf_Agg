"""
Stage 4-1 Regenerate: 불완전한 solution을 재생성하여 보완
각 문제당 원하는 개수의 완전한 solution을 확보
vLLM 배치 처리 최적화 버전
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import json
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any

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


def is_solution_complete(solution: Dict[str, Any], tokenizer=None, max_content_tokens: int = 30000) -> bool:
    """
    solution이 완전한지 확인합니다.

    Args:
        solution: solution 딕셔너리
        tokenizer: 토크나이저 (Optional, 토큰 길이 검증에 사용)
        max_content_tokens: content의 최대 토큰 길이

    Returns:
        완전하면 True, 불완전하면 False
    """
    final_answer = solution.get("final_answer", "")

    # final_answer가 None이거나 빈 문자열이면 불완전
    if final_answer is None or (isinstance(final_answer, str) and not final_answer.strip()):
        return False

    # content 파싱 검증
    if "content_parsed" in solution:
        # 새로 생성된 solution - content_parsed 플래그 사용
        if not solution["content_parsed"]:
            return False
    else:
        # 기존 solution (fallback 로직으로 생성됨) - content 직접 검증
        content = solution.get("content", "")
        generated_text = solution.get("generated_text", "")

        # content가 비어있거나, generated_text와 동일하면 파싱 실패로 간주
        # (fallback이 사용된 경우 content == generated_text)
        if not content or content == generated_text:
            # reasoning_content가 있는지 확인 (enable_thinking=True 였는지)
            reasoning_content = solution.get("reasoning_content", "")
            if reasoning_content or "</think>" in generated_text:
                # enable_thinking이 사용되었는데 content가 fallback되었음 → 불완전
                return False

    # content 토큰 길이 검증
    if tokenizer is not None:
        content = solution.get("content", "")
        if content:
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(content_tokens) > max_content_tokens:
                return False

    return True


def analyze_json_file(json_path: str, target_count: int, tokenizer=None) -> Dict[str, Any]:
    """
    JSON 파일을 분석하여 재생성이 필요한 문제를 찾습니다.

    Args:
        json_path: JSON 파일 경로
        target_count: 목표 solution 개수
        tokenizer: 토크나이저 (Optional, 토큰 길이 검증에 사용)

    Returns:
        분석 결과 딕셔너리
    """
    logger.info(f"JSON 파일 분석 중: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_name = data.get("dataset_name", "unknown")
    total_problems = data.get("total_problems", 0)
    generated_solutions = data.get("generated_solutions", [])

    problems_needing_regeneration = []

    for problem in generated_solutions:
        problem_id = problem.get("problem_id")
        solutions = problem.get("solutions", [])

        # 완전한 solution 개수 세기 (불완전한 것은 제거)
        complete_solutions = [sol for sol in solutions if is_solution_complete(sol, tokenizer)]
        complete_count = len(complete_solutions)

        if complete_count < target_count:
            missing_count = target_count - complete_count
            problems_needing_regeneration.append({
                "problem_id": problem_id,
                "problem_text": problem.get("problem_text", ""),
                "ground_truth": problem.get("ground_truth", ""),
                "complete_count": complete_count,
                "missing_count": missing_count,
                "complete_solutions": complete_solutions  # 완전한 것만 보관
            })
            logger.info(
                f"문제 {problem_id}: {complete_count}/{target_count} 완전 "
                f"({missing_count}개 재생성 필요)"
            )

    analysis = {
        "json_path": json_path,
        "dataset_name": dataset_name,
        "total_problems": total_problems,
        "problems_needing_regeneration": problems_needing_regeneration,
        "total_needing_regeneration": len(problems_needing_regeneration),
        "all_problems": generated_solutions
    }

    logger.info(
        f"분석 완료: {len(problems_needing_regeneration)}/{total_problems}개 문제가 "
        f"재생성 필요"
    )

    return analysis


def batch_regenerate_solutions(
    llm: LLM,
    tokenizer: AutoTokenizer,
    problems_info: List[Dict[str, Any]],
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
) -> Dict[int, List[Dict[str, Any]]]:
    """
    여러 문제에 대해 부족한 solution을 한 번에 배치로 재생성합니다.
    (vLLM 배치 처리 최적화)

    Args:
        llm: vLLM 모델
        tokenizer: 토크나이저
        problems_info: 문제 정보 리스트 (각각 problem_id, problem_text, missing_count 포함)
        ... (기타 파라미터)

    Returns:
        {problem_id: [new_solutions]} 형태의 딕셔너리
    """
    if not problems_info:
        return {}

    # 모든 프롬프트 준비
    all_prompts = []
    prompt_to_problem_id = []  # 각 프롬프트가 어느 문제에 속하는지

    for prob_info in problems_info:
        problem_id = prob_info["problem_id"]
        problem_text = prob_info["problem_text"]
        missing_count = prob_info["missing_count"]

        # 프롬프트 구성
        prompt = f"{problem_text}\n\n{base_instruction}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # 각 문제당 missing_count개씩 추가
        for _ in range(missing_count):
            all_prompts.append(formatted_prompt)
            prompt_to_problem_id.append(problem_id)

    total_prompts = len(all_prompts)
    logger.info(f"총 {total_prompts}개 프롬프트 배치 생성 시작...")

    # SamplingParams 설정
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        logprobs=logprobs,
    )

    # vLLM으로 배치 생성 (한 번에!)
    outputs = llm.generate(all_prompts, sampling_params)
    logger.info(f"배치 생성 완료: {len(outputs)}개 solution")

    # 결과를 문제별로 그룹화
    problem_solutions = {}

    for idx, output in enumerate(outputs):
        problem_id = prompt_to_problem_id[idx]
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
        content_parsed = True  # 파싱 성공 여부 플래그
        if enable_thinking:
            reasoning_content = extract_reasoning_content(generated_text)
            content = extract_content(generated_text)
            if not content:
                # content 파싱 실패 - 불완전한 solution으로 표시
                content_parsed = False
                content = ""  # 폴백 대신 빈 문자열로 설정
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
            "confidence_scores": confidence_scores,
            "content_parsed": content_parsed
        }

        # 문제별로 저장
        if problem_id not in problem_solutions:
            problem_solutions[problem_id] = []
        problem_solutions[problem_id].append(solution)

    return problem_solutions


def regenerate_with_batch_retry(
    llm: LLM,
    tokenizer: AutoTokenizer,
    problems_needing_regeneration: List[Dict[str, Any]],
    target_count: int,
    max_retry: int,
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
) -> Dict[int, List[Dict[str, Any]]]:
    """
    배치 재시도 로직으로 모든 문제의 solution을 재생성합니다.

    Args:
        llm: vLLM 모델
        tokenizer: 토크나이저
        problems_needing_regeneration: 재생성이 필요한 문제 리스트
        target_count: 목표 solution 개수
        max_retry: 최대 재시도 횟수
        ... (기타 파라미터)

    Returns:
        {problem_id: complete_solutions} 딕셔너리
    """
    # 문제별 완전한 solution을 저장
    problem_complete_solutions = {}
    for prob_info in problems_needing_regeneration:
        problem_id = prob_info["problem_id"]
        problem_complete_solutions[problem_id] = prob_info["complete_solutions"].copy()

    logger.info(f"배치 재생성 시작: {len(problems_needing_regeneration)}개 문제")

    for retry in range(max_retry):
        # 아직 부족한 문제들만 필터링
        problems_still_needing = []

        for prob_info in problems_needing_regeneration:
            problem_id = prob_info["problem_id"]
            current_complete = len(problem_complete_solutions[problem_id])

            if current_complete < target_count:
                missing_count = target_count - current_complete
                problems_still_needing.append({
                    "problem_id": problem_id,
                    "problem_text": prob_info["problem_text"],
                    "missing_count": missing_count
                })

        if not problems_still_needing:
            logger.info(f"모든 문제가 목표 달성! (retry {retry}회 만에)")
            break

        total_missing = sum(p["missing_count"] for p in problems_still_needing)
        logger.info(
            f"Retry {retry + 1}/{max_retry}: "
            f"{len(problems_still_needing)}개 문제, 총 {total_missing}개 solution 생성"
        )

        # 배치로 한 번에 생성
        new_solutions_by_problem = batch_regenerate_solutions(
            llm=llm,
            tokenizer=tokenizer,
            problems_info=problems_still_needing,
            base_instruction=base_instruction,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=32768,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            logprobs=logprobs,
            confidence_calculator=confidence_calculator,
            math_verifier=math_verifier
        )

        # 완전한 solution만 추가
        newly_complete_total = 0
        newly_generated_total = 0

        for problem_id, new_solutions in new_solutions_by_problem.items():
            newly_generated_total += len(new_solutions)
            complete_new = [sol for sol in new_solutions if is_solution_complete(sol, tokenizer)]
            problem_complete_solutions[problem_id].extend(complete_new)
            newly_complete_total += len(complete_new)

        logger.info(
            f"Retry {retry + 1} 결과: {newly_complete_total}/{newly_generated_total}개 완전 "
            f"(완전 비율: {newly_complete_total/newly_generated_total*100:.1f}%)"
        )

    # 최종 통계
    achieved = 0
    failed = 0
    for problem_id, complete_sols in problem_complete_solutions.items():
        if len(complete_sols) >= target_count:
            achieved += 1
        else:
            failed += 1
            logger.warning(
                f"문제 {problem_id}: 최대 재시도 후에도 목표 미달 "
                f"({len(complete_sols)}/{target_count})"
            )

    logger.info(
        f"최종 결과: {achieved}/{achieved+failed}개 문제 목표 달성 "
        f"(성공률: {achieved/(achieved+failed)*100:.1f}%)"
    )

    # target_count만큼만 잘라서 반환
    for problem_id in problem_complete_solutions:
        problem_complete_solutions[problem_id] = problem_complete_solutions[problem_id][:target_count]

    return problem_complete_solutions


def load_baseline_model(
    model_name: str,
    gpu_memory_utilization: float,
    max_model_len: int
) -> tuple:
    """Baseline 모델과 토크나이저 로드"""
    logger.info("=" * 60)
    logger.info("모델 로드 시작")
    logger.info("=" * 60)

    # 토크나이저 로드
    logger.info(f"토크나이저 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # vLLM 모델 로드
    logger.info(f"vLLM 모델 로드 중...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        kv_cache_dtype="fp8"
    )
    logger.info("모델 로드 완료")

    return llm, tokenizer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-1 Regenerate: 불완전한 solution 재생성 메인 함수"""

    # GPU 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_1_regenerate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=False,  # regenerate 스크립트에서는 wandb 비활성화
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )

    if torch.cuda.is_available():
        logger.info(f"GPU 설정: device=0, GPU={torch.cuda.get_device_name(0)}")

    logger.info("🚀 Stage 4-1 Regenerate: 불완전한 solution 재생성 시작 (배치 처리 최적화)")

    eval_config = cfg.evaluation.benchmarks.evaluation
    enable_thinking = eval_config.get("enable_thinking", True)

    # 결과 디렉토리
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    results_dir = os.path.join(results_dir, cfg.model.base_model.replace('/', '_') + f"_think_{enable_thinking}_32768")

    if not os.path.exists(results_dir):
        logger.error(f"결과 디렉토리가 존재하지 않습니다: {results_dir}")
        return

    # 재생성 설정
    target_count = 64  # 목표 solution 개수
    max_retry = 3  # 최대 재시도 횟수

    # 평가 구성 요소 초기화
    confidence_group_size = eval_config.get("confidence_group_size", 2048)
    confidence_calculator = ConfidenceCalculator(group_size=confidence_group_size)
    math_verifier = MathVerifier(timeout=eval_config.timeout)

    base_instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    # 재생성할 JSON 파일 찾기
    json_files = [
        f for f in os.listdir(results_dir)
        if f.endswith("_baseline_generated.json")
    ]

    if not json_files:
        logger.error(f"재생성할 JSON 파일을 찾을 수 없습니다: {results_dir}")
        return

    logger.info(f"총 {len(json_files)}개 JSON 파일 발견")

    # 모델 로드 (모든 파일에 대해 한 번만)
    llm = None
    tokenizer = None

    try:
        llm, tokenizer = load_baseline_model(
            model_name=cfg.model.base_model,
            gpu_memory_utilization=eval_config.get("gpu_memory_utilization", 0.9),
            max_model_len=eval_config.get("max_model_len", eval_config.max_tokens + 16384)
        )

        # 각 JSON 파일 처리
        for json_file in json_files:
            json_path = os.path.join(results_dir, json_file)

            logger.info("=" * 60)
            logger.info(f"처리 중: {json_file}")
            logger.info("=" * 60)

            # 분석
            analysis = analyze_json_file(json_path, target_count, tokenizer)

            if analysis["total_needing_regeneration"] == 0:
                logger.info(f"{json_file}: 재생성 불필요 (모든 문제가 완전함)")
                continue

            # 배치 재생성 (모든 문제를 한 번에 처리)
            problems_needing_regeneration = analysis["problems_needing_regeneration"]

            complete_solutions_by_problem = regenerate_with_batch_retry(
                llm=llm,
                tokenizer=tokenizer,
                problems_needing_regeneration=problems_needing_regeneration,
                target_count=target_count,
                max_retry=max_retry,
                base_instruction=base_instruction,
                enable_thinking=enable_thinking,
                temperature=eval_config.temperature,
                max_tokens=32768,
                top_p=eval_config.get("top_p", 0.95),
                top_k=eval_config.get("top_k", 20),
                min_p=eval_config.get("min_p", 0.0),
                logprobs=eval_config.get("logprobs", 5),
                confidence_calculator=confidence_calculator,
                math_verifier=math_verifier
            )

            # 원본 데이터 업데이트
            all_problems = analysis["all_problems"]
            for problem in all_problems:
                problem_id = problem["problem_id"]
                if problem_id in complete_solutions_by_problem:
                    # 완전한 solution으로 덮어쓰기 (불완전한 것은 제거됨)
                    problem["solutions"] = complete_solutions_by_problem[problem_id]

            # 저장
            output_data = {
                "dataset_name": analysis["dataset_name"],
                "total_problems": analysis["total_problems"],
                "generated_solutions": all_problems
            }

            # 백업 생성
            backup_path = json_path.replace(".json", "_backup.json")
            if not os.path.exists(backup_path):
                logger.info(f"백업 생성: {backup_path}")
                with open(backup_path, 'w', encoding='utf-8') as f:
                    with open(json_path, 'r', encoding='utf-8') as orig:
                        f.write(orig.read())

            # 새 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 재생성 완료 및 저장: {json_path}")

    finally:
        # 모델 unload
        if llm is not None:
            logger.info("모델 unload 중...")
            del llm
            del tokenizer
            torch.cuda.empty_cache()
            logger.info("모델 unload 완료")

    logger.info("=" * 60)
    logger.info("✅ Stage 4-1 Regenerate: 모든 파일 재생성 완료")
    logger.info(f"결과 저장 위치: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
