#!/usr/bin/env python3
"""
Hugging Face Hub에 단일 Parquet 파일을 업로드하는 스크립트

기능
- 지정한 dataset 리포지토리에 Parquet 파일 업로드
- 리포지토리가 없으면 옵션에 따라 생성
- 커밋 메시지, 업로드 경로, 브랜치, 공개/비공개 설정 지원
- 토큰은 우선순위: --token 인자 > 환경변수 HF_TOKEN

사용 예시
  # 리포지토리가 존재할 때
  python scripts/upload_to_hf.py \
    --repo-id your-username/your-dataset \
    --file /abs/path/to/raw_generated.parquet \
    --path-in-repo data/raw_generated.parquet \
    --commit-message "Add raw_generated.parquet"

  # 리포지토리가 없으면 생성하여 업로드
  python scripts/upload_to_hf.py \
    --repo-id your-username/your-dataset \
    --file /abs/path/to/raw_generated.parquet \
    --create \
    --private
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, create_repo
from huggingface_hub.utils import HfHubHTTPError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token
    # 마지막 수단: 로컬 캐시
    cached = HfFolder.get_token()
    return cached


def ensure_repo_exists(api: HfApi, repo_id: str, private: bool, token: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        logger.info(f"리포지토리 확인됨: {repo_id}")
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.info(f"리포지토리가 없어 생성합니다: {repo_id} (private={private})")
            create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        else:
            raise


def upload_parquet(
    repo_id: str,
    file_path: str,
    path_in_repo: Optional[str] = None,
    commit_message: str = "Add parquet",
    branch: str = "main",
    private: bool = False,
    create_if_missing: bool = False,
    token: Optional[str] = None,
) -> None:
    token_resolved = resolve_token(token)
    if not token_resolved:
        logger.error("Hugging Face 토큰이 필요합니다. --token 또는 환경변수 HF_TOKEN을 설정하세요.")
        sys.exit(1)

    abs_file = os.path.abspath(file_path)
    if not os.path.exists(abs_file):
        logger.error(f"파일을 찾을 수 없습니다: {abs_file}")
        sys.exit(1)
    if not abs_file.endswith(".parquet"):
        logger.error(f"Parquet 파일이 아닙니다: {abs_file}")
        sys.exit(1)

    size_mb = os.path.getsize(abs_file) / (1024 ** 2)
    logger.info(f"업로드 대상: {abs_file} ({size_mb:.2f} MB)")

    if path_in_repo is None:
        path_in_repo = f"data/{Path(abs_file).name}"

    api = HfApi()

    if create_if_missing:
        ensure_repo_exists(api, repo_id, private, token_resolved)

    logger.info(
        f"업로드 시작 -> repo: {repo_id}, repo_path: {path_in_repo}, branch: {branch}, private={private}"
    )

    try:
        api.upload_file(
            path_or_fileobj=abs_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token_resolved,
            commit_message=commit_message,
            revision=branch,
        )
        logger.info("✅ 업로드 완료")
        logger.info(f"https://huggingface.co/datasets/{repo_id}/tree/{branch}/{Path(path_in_repo).parent}")
    except HfHubHTTPError as e:
        logger.error(f"업로드 실패: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a Parquet file to Hugging Face Hub dataset repo")
    parser.add_argument("--repo-id", type=str, required=True, help="대상 dataset 리포지토리 (e.g., user/dataset)")
    parser.add_argument("--file", type=str, required=True, help="업로드할 Parquet 파일 경로 (절대 경로 권장)")
    parser.add_argument("--path-in-repo", type=str, default=None, help="리포지토리 내 저장 경로 (기본: data/<파일명>)")
    parser.add_argument("--commit-message", type=str, default="Add parquet", help="커밋 메시지")
    parser.add_argument("--branch", type=str, default="main", help="업로드할 브랜치/리비전")
    parser.add_argument("--private", action="store_true", help="리포지토리를 private으로 설정 (생성 시 적용)")
    parser.add_argument("--create", action="store_true", help="리포지토리가 없으면 생성")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face 액세스 토큰 (옵션)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_parquet(
        repo_id=args.repo_id,
        file_path=args.file,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        branch=args.branch,
        private=args.private,
        create_if_missing=args.create,
        token=args.token,
    )


if __name__ == "__main__":
    main()


