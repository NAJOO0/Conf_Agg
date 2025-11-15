"""
Stage 4-2 추가 분석: Generated Content 토큰 수, 정확도, Confidence 상관관계 분석
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.math_verifier import MathVerifier
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Tokenizer 로드 함수
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers를 사용할 수 없습니다.")


def load_tokenizer(model_name: str):
    """
    Tokenizer 로드
    
    Args:
        model_name: 모델 이름 또는 경로
        
    Returns:
        tokenizer
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers가 필요합니다.")
    
    logger.info(f"Tokenizer 로드 중: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("✅ Tokenizer 로드 완료")
        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer 로드 실패: {e}")
        raise


def count_tokens_with_tokenizer(tokenizer, text: str) -> int:
    """
    Tokenizer를 사용하여 정확한 토큰 수 계산
    
    Args:
        tokenizer: tokenizer 인스턴스
        text: 텍스트 문자열
        
    Returns:
        토큰 수
    """
    if not text or not isinstance(text, str):
        return 0
    
    try:
        # tokenizer.encode 사용
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        elif hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text)
            return len(tokens)
        else:
            # Fallback: 단어 수 추정
            return len(text.split())
    except Exception as e:
        logger.warning(f"토큰 수 계산 실패: {e}")
        # Fallback: 단어 수 추정
        return len(str(text).split())


def extract_content_from_generated_text(generated_text: str) -> str:
    """
    </think> 토큰 이후 값들 추출
    
    Args:
        generated_text: generated_text
        
    Returns:
        </think> 이후 내용 (마커가 없으면 전체 텍스트)
    """
    if not generated_text or not isinstance(generated_text, str):
        return ""
    
    marker = "</think>"
    marker_pos = generated_text.find(marker)
    
    if marker_pos == -1:
        # 마커가 없으면 전체 텍스트 반환 (enable_thinking=False인 경우)
        return generated_text.strip()
    
    # 마커 이후 텍스트 추출
    content = generated_text[marker_pos + len(marker):].strip()
    return content


def analyze_token_confidence_correlation(
    baseline_path: str,
    aggllm_path: str,
    tokenizer,
    math_verifier: MathVerifier,
    output_dir: str,
    max_tokens: int = 16384
) -> dict:
    """
    Generated content의 토큰 수, 정확도, 각 confidence의 상관관계 분석
    
    Args:
        baseline_path: Baseline 결과 파일 경로
        aggllm_path: AggLLM 결과 파일 경로
        tokenizer: tokenizer 인스턴스
        math_verifier: MathVerifier 인스턴스
        output_dir: 출력 디렉토리
        
    Returns:
        분석 결과 딕셔너리
    """
    results = {}
    
    # Baseline 분석
    if os.path.exists(baseline_path):
        logger.info(f"Baseline 결과 분석: {baseline_path}")
        baseline_results = analyze_single_file(
            baseline_path, tokenizer, math_verifier, "baseline", max_tokens
        )
        results["baseline"] = baseline_results
    else:
        logger.warning(f"Baseline 결과 파일 없음: {baseline_path}")
    
    # AggLLM 분석
    if os.path.exists(aggllm_path):
        logger.info(f"AggLLM 결과 분석: {aggllm_path}")
        aggllm_results = analyze_single_file(
            aggllm_path, tokenizer, math_verifier, "aggllm", max_tokens
        )
        results["aggllm"] = aggllm_results
    else:
        logger.warning(f"AggLLM 결과 파일 없음: {aggllm_path}")
    
    # 결과 저장
    output_path = os.path.join(output_dir, "token_confidence_correlation_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"분석 결과 저장: {output_path}")
    
    return results


def analyze_single_file(
    file_path: str,
    tokenizer,
    math_verifier: MathVerifier,
    method_name: str,
    max_tokens: int = 16384
) -> dict:
    """
    단일 파일에 대한 분석 수행
    
    Args:
        file_path: JSON 파일 경로
        tokenizer: tokenizer 인스턴스
        math_verifier: MathVerifier 인스턴스
        method_name: 방법 이름 (baseline 또는 aggllm)
        max_tokens: 최대 토큰 수 (기본값: 32768)
        
    Returns:
        분석 결과 딕셔너리
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {method_name.upper()} 분석 ===")
    logger.info(f"{'='*60}")
    logger.info(f"Max tokens 설정: {max_tokens}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터 수집 (instance별로 그룹화)
    all_data = []
    instance_data = defaultdict(lambda: {
        "problem_id": None,
        "ground_truth": None,
        "solutions": []
    })
    
    for problem_data in data["generated_solutions"]:
        problem_id = problem_data.get("problem_id", len(instance_data))
        ground_truth = problem_data["ground_truth"]
        
        instance_data[problem_id]["problem_id"] = problem_id
        instance_data[problem_id]["ground_truth"] = ground_truth
        
        for solution in problem_data["solutions"]:
            # Generated text 전체 토큰 수 계산 (max_tokens 도달 여부 확인용)
            generated_text = solution.get("generated_text", "")
            generated_text_token_count = count_tokens_with_tokenizer(tokenizer, generated_text)
            
            # Content 추출
            content = extract_content_from_generated_text(generated_text)
            
            # Content 토큰 수 계산
            content_token_count = count_tokens_with_tokenizer(tokenizer, content)
            
            # Max tokens 도달 여부 확인 (99% 이상이면 도달한 것으로 간주)
            reached_max_tokens = generated_text_token_count >= (max_tokens * 0.99)
            
            # 정답 여부 확인
            final_answer = solution.get("final_answer", "")
            is_correct = False
            if final_answer:
                try:
                    is_correct = math_verifier.verify_answer(final_answer, ground_truth)
                except Exception as e:
                    logger.warning(f"정답 검증 실패: {e}")
                    is_correct = False
            
            # Confidence scores 추출
            confidence_scores = solution.get("confidence_scores", {})
            
            # 데이터 저장
            solution_data = {
                "content_token_count": content_token_count,
                "generated_text_token_count": generated_text_token_count,
                "reached_max_tokens": reached_max_tokens,
                "is_correct": is_correct,
                "confidence_scores": confidence_scores
            }
            all_data.append(solution_data)
            instance_data[problem_id]["solutions"].append(solution_data)
    
    logger.info(f"총 {len(all_data)}개 solution 분석")
    
    # Instance별 max_tokens 도달 통계
    instance_stats = []
    for problem_id, instance_info in instance_data.items():
        solutions = instance_info["solutions"]
        max_tokens_count = sum(1 for s in solutions if s["reached_max_tokens"])
        correct_count = sum(1 for s in solutions if s["is_correct"])
        total_count = len(solutions)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # 평균 토큰 수 계산 (content_token_count 기준)
        token_counts_list = [s["content_token_count"] for s in solutions]
        avg_token_count = np.mean(token_counts_list) if token_counts_list else 0.0
        
        instance_stats.append({
            "problem_id": problem_id,
            "max_tokens_reached_count": max_tokens_count,
            "total_solutions": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "avg_token_count": float(avg_token_count)
        })
    
    logger.info(f"\n=== {method_name.upper()} Instance별 Max Tokens 도달 통계 ===")
    for stat in instance_stats:
        logger.info(f"  Problem {stat['problem_id']}: {stat['max_tokens_reached_count']}/{stat['total_solutions']}개 도달, 정확도: {stat['accuracy']:.3f}, 평균 토큰 수: {stat['avg_token_count']:.1f}")
    
    # Max tokens 도달한 instance 목록
    max_tokens_reached_instances = set(
        stat["problem_id"] for stat in instance_stats 
        if stat["max_tokens_reached_count"] > 0
    )
    
    logger.info(f"\nMax tokens 도달한 instance 수: {len(max_tokens_reached_instances)}/{len(instance_stats)}")
    
    # Max tokens 도달한 solution 개수 합계
    total_max_tokens_reached_solutions = sum(
        stat["max_tokens_reached_count"] for stat in instance_stats
    )
    
    logger.info(f"Max tokens 도달한 solution 개수 (실제 도달): {total_max_tokens_reached_solutions}개")
    
    # Max tokens 도달하지 않은 inference만 필터링 (instance 전체가 아닌 개별 inference만)
    filtered_data = []
    excluded_data = []
    for d in all_data:
        if d["reached_max_tokens"]:
            excluded_data.append(d)
        else:
            filtered_data.append(d)
    
    logger.info(f"Max tokens 도달한 solution 수 (제외됨): {len(excluded_data)}개")
    logger.info(f"Max tokens 도달하지 않은 solution 수 (분석에 포함): {len(filtered_data)}/{len(all_data)}")
    
    # 배열로 변환 (전체 데이터)
    token_counts_all = np.array([d["content_token_count"] for d in all_data])
    is_corrects_all = np.array([d["is_correct"] for d in all_data], dtype=float)
    
    # 배열로 변환 (필터링된 데이터 - max_tokens 도달한 instance 제외)
    token_counts = np.array([d["content_token_count"] for d in filtered_data]) if filtered_data else np.array([])
    is_corrects = np.array([d["is_correct"] for d in filtered_data], dtype=float) if filtered_data else np.array([])
    
    # Confidence scores 추출 (전체 데이터)
    confidence_metrics = [
        "bottom_10_percent_confidence",
        "tail_confidence",
        "mean_group_confidence",
        "lowest_group_confidence",
        "top_10_percent_confidence",
        "highest_group_confidence"
    ]
    
    confidence_dict_all = {}
    for metric in confidence_metrics:
        values = []
        for d in all_data:
            conf = d["confidence_scores"].get(metric)
            if conf is not None:
                values.append(float(conf))
            else:
                values.append(np.nan)
        confidence_dict_all[metric] = np.array(values)
    
    # Confidence scores 추출 (필터링된 데이터)
    confidence_dict = {}
    for metric in confidence_metrics:
        values = []
        for d in filtered_data:
            conf = d["confidence_scores"].get(metric)
            if conf is not None:
                values.append(float(conf))
            else:
                values.append(np.nan)
        confidence_dict[metric] = np.array(values) if filtered_data else np.array([])
    
    # 분석 결과 저장
    analysis_results = {
        "total_solutions": len(all_data),
        "max_tokens": max_tokens,
        "max_tokens_reached_instances": {
            "count": len(max_tokens_reached_instances),
            "total_instances": len(instance_stats),
            "instance_ids": sorted(list(max_tokens_reached_instances))
        },
        "instance_statistics": instance_stats,
        "filtered_solutions": {
            "count": len(filtered_data),
            "excluded_count": len(excluded_data),
            "excluded_max_tokens_reached_count": total_max_tokens_reached_solutions
        },
        "token_statistics": {
            "all_solutions": {
                "mean": float(np.mean(token_counts_all)) if len(token_counts_all) > 0 else 0.0,
                "median": float(np.median(token_counts_all)) if len(token_counts_all) > 0 else 0.0,
                "std": float(np.std(token_counts_all)) if len(token_counts_all) > 0 else 0.0,
                "min": int(np.min(token_counts_all)) if len(token_counts_all) > 0 else 0,
                "max": int(np.max(token_counts_all)) if len(token_counts_all) > 0 else 0,
            },
            "filtered_solutions": {
                "mean": float(np.mean(token_counts)) if len(token_counts) > 0 else 0.0,
                "median": float(np.median(token_counts)) if len(token_counts) > 0 else 0.0,
                "std": float(np.std(token_counts)) if len(token_counts) > 0 else 0.0,
                "min": int(np.min(token_counts)) if len(token_counts) > 0 else 0,
                "max": int(np.max(token_counts)) if len(token_counts) > 0 else 0,
                "percentiles": {
                    "25": float(np.percentile(token_counts, 25)) if len(token_counts) > 0 else 0.0,
                    "50": float(np.percentile(token_counts, 50)) if len(token_counts) > 0 else 0.0,
                    "75": float(np.percentile(token_counts, 75)) if len(token_counts) > 0 else 0.0,
                    "90": float(np.percentile(token_counts, 90)) if len(token_counts) > 0 else 0.0,
                    "95": float(np.percentile(token_counts, 95)) if len(token_counts) > 0 else 0.0,
                    "99": float(np.percentile(token_counts, 99)) if len(token_counts) > 0 else 0.0
                }
            }
        },
        "accuracy": {
            "all_solutions": {
                "overall": float(np.mean(is_corrects_all)) if len(is_corrects_all) > 0 else 0.0,
                "correct_count": int(np.sum(is_corrects_all)),
                "total_count": len(is_corrects_all)
            },
            "filtered_solutions": {
                "overall": float(np.mean(is_corrects)) if len(is_corrects) > 0 else 0.0,
                "correct_count": int(np.sum(is_corrects)),
                "total_count": len(is_corrects)
            }
        },
        "correlations": {}
    }
    
    # 토큰 수와 정확도 상관관계
    valid_mask = ~np.isnan(token_counts) & ~np.isnan(is_corrects)
    if valid_mask.sum() > 0:
        valid_tokens = token_counts[valid_mask]
        valid_corrects = is_corrects[valid_mask]
        
        if len(valid_tokens) > 1 and len(np.unique(valid_tokens)) > 1:
            pearson_corr, pearson_p = pearsonr(valid_tokens, valid_corrects)
            spearman_corr, spearman_p = spearmanr(valid_tokens, valid_corrects)
            
            analysis_results["correlations"]["token_count_vs_accuracy"] = {
                "pearson": {
                    "correlation": float(pearson_corr),
                    "p_value": float(pearson_p)
                },
                "spearman": {
                    "correlation": float(spearman_corr),
                    "p_value": float(spearman_p)
                }
            }
    
    # 각 Confidence와 정확도 상관관계
    for metric in confidence_metrics:
        conf_values = confidence_dict[metric]
        valid_mask = ~np.isnan(conf_values) & ~np.isnan(is_corrects)
        
        if valid_mask.sum() > 0:
            valid_conf = conf_values[valid_mask]
            valid_corrects = is_corrects[valid_mask]
            
            if len(valid_conf) > 1 and len(np.unique(valid_conf)) > 1:
                pearson_corr, pearson_p = pearsonr(valid_conf, valid_corrects)
                spearman_corr, spearman_p = spearmanr(valid_conf, valid_corrects)
                
                analysis_results["correlations"][f"{metric}_vs_accuracy"] = {
                    "pearson": {
                        "correlation": float(pearson_corr),
                        "p_value": float(pearson_p)
                    },
                    "spearman": {
                        "correlation": float(spearman_corr),
                        "p_value": float(spearman_p)
                    }
                }
    
    # 각 Confidence와 토큰 수 상관관계
    for metric in confidence_metrics:
        conf_values = confidence_dict[metric]
        valid_mask = ~np.isnan(conf_values) & ~np.isnan(token_counts)
        
        if valid_mask.sum() > 0:
            valid_conf = conf_values[valid_mask]
            valid_tokens = token_counts[valid_mask]
            
            if len(valid_conf) > 1 and len(np.unique(valid_conf)) > 1:
                pearson_corr, pearson_p = pearsonr(valid_conf, valid_tokens)
                spearman_corr, spearman_p = spearmanr(valid_conf, valid_tokens)
                
                analysis_results["correlations"][f"{metric}_vs_token_count"] = {
                    "pearson": {
                        "correlation": float(pearson_corr),
                        "p_value": float(pearson_p)
                    },
                    "spearman": {
                        "correlation": float(spearman_corr),
                        "p_value": float(spearman_p)
                    }
                }
    
    # 구간별 정답률 분석 (토큰 수 기준) - 필터링된 데이터만 사용
    interval_analysis = []
    if len(token_counts) > 0:
        logger.info(f"\n=== {method_name.upper()} 구간별 정답률 분석 (토큰 수 기준, max_tokens 도달 instance 제외) ===")
        percentiles = [0, 25, 50, 75, 90, 100]
        percentile_values = [np.percentile(token_counts, p) for p in percentiles]
        
        for i in range(len(percentiles) - 1):
            pct_start = percentiles[i]
            pct_end = percentiles[i+1]
            val_start = percentile_values[i]
            val_end = percentile_values[i+1] if i+1 < len(percentile_values) else np.inf
            
            if i < len(percentiles) - 2:
                mask = (token_counts >= val_start) & (token_counts < val_end)
            else:
                mask = (token_counts >= val_start) & (token_counts <= val_end)
            
            if mask.sum() > 0:
                interval_accuracy = is_corrects[mask].mean()
                interval_count = mask.sum()
                interval_analysis.append({
                    "percentile_range": f"{pct_start}-{pct_end}%",
                    "token_range": f"{val_start:.0f}-{val_end:.0f}",
                    "accuracy": float(interval_accuracy),
                    "count": int(interval_count)
                })
                logger.info(f"  {pct_start}%-{pct_end}% 구간 ({val_start:.0f} ~ {val_end:.0f} 토큰): 정답률 {interval_accuracy:.3f} ({interval_count}개)")
    
    analysis_results["interval_analysis"] = interval_analysis
    
    # 로그 출력
    logger.info(f"\n=== {method_name.upper()} 전체 통계 ===")
    logger.info(f"총 solution 수: {len(all_data)}")
    logger.info(f"Max tokens 도달 instance 제외 후 solution 수: {len(filtered_data)}")
    if len(token_counts) > 0:
        logger.info(f"평균 토큰 수 (필터링 후): {np.mean(token_counts):.2f}")
        logger.info(f"중앙값 토큰 수 (필터링 후): {np.median(token_counts):.2f}")
        logger.info(f"전체 정답률 (필터링 후): {np.mean(is_corrects):.3f}")
    
    logger.info(f"\n=== {method_name.upper()} 상관관계 요약 ===")
    for key, value in analysis_results["correlations"].items():
        if "pearson" in value:
            logger.info(f"  {key}:")
            logger.info(f"    피어슨 상관계수: {value['pearson']['correlation']:.4f} (p-value: {value['pearson']['p_value']:.4e})")
            logger.info(f"    스피어만 상관계수: {value['spearman']['correlation']:.4f} (p-value: {value['spearman']['p_value']:.4e})")
    
    return analysis_results


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 4-2 추가 분석: 토큰 수, 정확도, Confidence 상관관계 분석"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage4_2_analyze_token_confidence_correlation.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 4-2 추가 분석: 토큰 수, 정확도, Confidence 상관관계 분석 시작")
    
    # 디렉토리 설정
    results_dir = os.path.join(cfg.paths.output_dir, "comprehensive_results")
    results_dir = os.path.join(results_dir, "Qwen_Qwen3-4B-Instruct-2507")
    
    # Tokenizer 초기화
    model_name = cfg.model.base_model
    logger.info(f"Tokenizer 로드: {model_name}")
    tokenizer = load_tokenizer(model_name)
    
    # Math Verifier 초기화
    math_verifier = MathVerifier(
        timeout=cfg.evaluation.benchmarks.evaluation.timeout
    )
    
    # 벤치마크 데이터셋 설정
    benchmark_datasets = [
        {"name": "AIME24", "path": "math-ai/aime24"},
        {"name": "AIME25", "path": "math-ai/aime25"},
        {"name": "HMMT24", "path": "MathArena/hmmt_feb_2024"},
        {"name": "HMMT25", "path": "MathArena/hmmt_feb_2025"},
    ]
    
    all_results = {}
    
    # 각 데이터셋에 대해 분석
    for benchmark in benchmark_datasets:
        dataset_name = benchmark["name"]
        dataset_path = benchmark["path"]
        dataset_safe_name = dataset_path.replace('/', '_')
        
        logger.info("=" * 60)
        logger.info(f"데이터셋 분석: {dataset_name}")
        logger.info("=" * 60)
        
        # Baseline 경로
        baseline_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_baseline_generated.json"
        )
        
        # AggLLM 경로
        aggllm_path = os.path.join(
            results_dir,
            f"{dataset_safe_name}_aggllm_generated.json"
        )
        
        # Max tokens 설정 가져오기
        max_tokens = cfg.evaluation.benchmarks.evaluation.max_tokens
        
        # 분석 수행
        dataset_results = analyze_token_confidence_correlation(
            baseline_path,
            aggllm_path,
            tokenizer,
            math_verifier,
            results_dir,
            max_tokens
        )
        
        all_results[dataset_name] = dataset_results
    
    # 전체 요약 저장
    summary_path = os.path.join(results_dir, "token_confidence_correlation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"전체 요약 저장: {summary_path}")
    logger.info("\n✅ Stage 4-2 추가 분석 완료")


if __name__ == "__main__":
    main()

