FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9

# PYTHONPATH=/app/src: enables importing formula_foundry (src-layout) without
# an editable pip install. In containers we prefer explicit path over install
# to keep the image deterministic and avoid pip metadata writes.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md DESIGN_DOCUMENT.md /app/
COPY src/ /app/src/
COPY tools/ /app/tools/
COPY config/ /app/config/

RUN uv venv /opt/venv
RUN uv sync --frozen

ENTRYPOINT ["python3", "-m", "tools.m0"]
CMD ["doctor"]
