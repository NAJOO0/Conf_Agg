"""
로깅 유틸리티 모듈
"""
import logging
import os
from typing import Optional
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    wandb_enabled: bool = False,
    wandb_project: str = "conf-agg-llm",
    wandb_tags: Optional[list] = None
) -> logging.Logger:
    """
    로깅 시스템을 설정합니다.
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_file: 로그 파일 경로 (선택사항)
        wandb_enabled: WandB 로깅 활성화 여부
        wandb_project: WandB 프로젝트 이름
        wandb_tags: WandB 태그 리스트
    
    Returns:
        설정된 로거
    """
    # 로거 생성
    logger = logging.getLogger("conf_agg_llm")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # WandB 초기화 (선택사항)
    # TRL이 wandb를 초기화하므로, 여기서는 run_name만 환경 변수로 설정
    if wandb_enabled:
        # 환경 변수 WANDB_NAME이 있으면 설정 (TRL이 이를 사용할 수 있도록)
        run_name = os.environ.get("WANDB_NAME", None)
        if run_name:
            # TRL이 wandb를 초기화할 때 이 환경 변수를 사용하도록 설정
            os.environ["WANDB_NAME"] = run_name
            logger.info(f"WandB run_name 환경 변수 설정: {run_name}")
        else:
            # 환경 변수가 없으면 Hydra 설정에서 가져오기
            try:
                hydra_cfg = HydraConfig.get()
                job_name = getattr(hydra_cfg.job, "name", "train")
                job_num = getattr(hydra_cfg.job, "num", None)
                if job_num is not None:
                    run_name = f"{job_name}_{job_num}"
                else:
                    run_name = job_name
                os.environ["WANDB_NAME"] = run_name
                logger.info(f"WandB run_name 환경 변수 설정 (Hydra): {run_name}")
            except Exception:
                pass
    
    logger.info(f"로깅 시스템 초기화 완료 (레벨: {log_level})")
    return logger


def get_logger(name: str = "conf_agg_llm") -> logging.Logger:
    """로거를 가져옵니다."""
    return logging.getLogger(name)

