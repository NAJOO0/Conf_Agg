# Dockerfile for Conf_Agg with uv package management
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Avoid interactive tzdata prompts and set timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

# Python 3.12 설치
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       tzdata \
       software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 의존성 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       wget \
       curl \
       build-essential \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (최신 버전)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH=/root/.local/bin:${PATH}

# uv 설정
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 프로젝트 파일 복사 (uv가 의존성을 해결할 수 있도록)
COPY pyproject.toml ./
COPY README.md ./

# uv sync는 컨테이너 실행 시에 하도록 주석 처리
# RUN uv sync --no-dev

# 환경 변수 설정
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# uv 가상환경 활성화를 위한 스크립트
RUN echo '#!/bin/bash\nsource .venv/bin/activate\nexec "$@"' > /usr/local/bin/uv-run \
    && chmod +x /usr/local/bin/uv-run

# 기본 명령어
CMD ["uv-run", "python", "--version"]
