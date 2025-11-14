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
    if wandb_enabled:
        # WandB API 키 확인
        if not wandb.api.api_key:
            logger.warning("WandB API 키가 설정되지 않았습니다. 환경 변수 WANDB_API_KEY를 설정하거나 'wandb login'을 실행해주세요.")
        else:
            try:
                hydra_cfg = HydraConfig.get()
                # job.num이 없을 수 있으므로 안전하게 처리
                job_name = getattr(hydra_cfg.job, "name", "train")
                job_num = getattr(hydra_cfg.job, "num", None)
                if job_num is not None:
                    run_name = f"{job_name}_{job_num}"
                else:
                    run_name = job_name
                
                # hydra_cfg.cfg 대신 OmegaConf를 사용하여 안전하게 변환
                # 이렇게 하면 job.num 검증 에러를 피할 수 있습니다
                try:
                    config_dict = OmegaConf.to_container(hydra_cfg.cfg, resolve=True)
                except Exception:
                    # Hydra config를 가져올 수 없으면 빈 dict 사용
                    config_dict = {}
                
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    tags=wandb_tags or [],
                    config=config_dict
                )
                logger.info(f"WandB 초기화 완료: {wandb_project}")
            except Exception as e:
                logger.warning(f"WandB 초기화 실패: {e}")
    
    logger.info(f"로깅 시스템 초기화 완료 (레벨: {log_level})")
    return logger


def get_logger(name: str = "conf_agg_llm") -> logging.Logger:
    """로거를 가져옵니다."""
    return logging.getLogger(name)

