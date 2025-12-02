"""
Stage 2: 데이터 큐레이션 스크립트
"""
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.data.curation import DataCurator
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Stage 2 메인 함수"""
    
    # 로깅 설정
    log_file = os.path.join(cfg.paths.log_dir, "stage2_curate.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file,
        wandb_enabled=cfg.experiment.wandb.enabled,
        wandb_project=cfg.experiment.wandb.project,
        wandb_tags=cfg.experiment.wandb.tags
    )
    
    logger.info("🚀 Stage 2: 데이터 큐레이션 시작")
    logger.info(f"설정: {cfg.data.curation}")
    
    # 디렉토리 생성
    os.makedirs(os.path.join(cfg.paths.data_dir, "curated"), exist_ok=True)
    
    # 입력 파일 경로
    generated_data_path = os.path.join(cfg.paths.data_dir, "generated", "dataset_4000_think.parquet")
    
    if not os.path.exists(generated_data_path):
        logger.error(f"Stage 1 결과 파일을 찾을 수 없습니다: {generated_data_path}")
        logger.error("먼저 Stage 1을 실행해주세요: python scripts/stage1_generate.py")
        return
    
    # baseline 전략인 경우 별도의 프롬프트 템플릿 사용
    if cfg.data.curation.strategy == "baseline":
        default_prompt_template = (
            "Given the following problem:\n"
            "{problem}\n"
            "and these solution attempts:\n"
            "{solutions}\n"
            "It is possible that any, all, or none of these solutions are correct or complete. "
            "Carefully review the provided solutions, using them as starting points—correcting mistakes, "
            "filling in gaps, and/or combining useful ideas--to produce a final, comprehensive, and correct solution to the problem."
        )
    else:
        default_prompt_template = (
            "Given the following problem:\n"
            "{problem}\n"
            "and these solution attempts with their confidence scores:\n"
            "{solutions}\n"
            "It is possible that any, all, or none of these solutions are correct or complete. "
            "Carefully review the provided solutions and their confidence scores, using them as starting points—correcting mistakes, "
            "filling in gaps, and/or combining useful ideas--to produce a final, comprehensive, and correct solution to the problem."
        )

    # 데이터 큐레이션 실행
    curator = DataCurator(
        strategy=cfg.data.curation.strategy,
        easy_sample_percentage=cfg.data.curation.easy_sample_percentage,
        num_sets_per_problem=cfg.data.curation.num_sets_per_problem,
        set_size=cfg.data.curation.set_size,
        enable_thinking=cfg.data.curation.enable_thinking,
        timeout=cfg.data.curation.verification.timeout,
        confidence_key=getattr(cfg.data.curation, "confidence_key", "bottom_10_percent_confidence"),
        fill_insufficient_with_sampling=getattr(cfg.data.curation, "fill_insufficient_with_sampling", False),
        order_strategy=getattr(cfg.data.curation, "order_strategy", "hard_first"),
        prompt_template=getattr(
            cfg.data.curation,
            "prompt_template",
            default_prompt_template
        ),
    )
    
    output_dir = os.path.join(cfg.paths.data_dir, f"curated_4000_32_{cfg.data.curation.strategy}")
    result_paths = curator.curate_data(
        generated_data_path, 
        output_dir,
        train_split=cfg.data.curation.output.get("train_split", 0.95)
    )
    
    logger.info("✅ Stage 2 완료")

    if cfg.data.curation.strategy == "curriculum":
        logger.info("Curriculum 데이터셋 생성 완료:")
        for key, path in result_paths.items():
            logger.info(f"  {key}: {path}")
    elif cfg.data.curation.strategy == "multitask":
        logger.info("Multitask 데이터셋 생성 완료:")
        logger.info(f"  Train: {result_paths['train']}")
        logger.info(f"  Validation: {result_paths['validation']}")
    elif cfg.data.curation.strategy == "baseline":
        logger.info("Baseline 데이터셋 생성 완료 (confidence 없이 aggregation):")
        logger.info(f"  Train: {result_paths['train']}")
        logger.info(f"  Validation: {result_paths['validation']}")
    else:
        logger.info(f"훈련 데이터: {result_paths['train']}")
        logger.info(f"검증 데이터: {result_paths['validation']}")


if __name__ == "__main__":
    main()

