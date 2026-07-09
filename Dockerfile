ARG API_BASE_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime
FROM ${API_BASE_IMAGE}
ARG DOCS_INIT_BUILD_ID=auto

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=20

WORKDIR /app

COPY requirements.api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.api.txt

COPY app ./app
COPY scripts ./scripts
COPY docker ./docker

RUN mkdir -p data/docs
RUN if [ "$DOCS_INIT_BUILD_ID" = "auto" ]; then \
      date -u +%Y%m%d%H%M%S%N > /app/.image_build_id; \
    else \
      printf '%s\n' "$DOCS_INIT_BUILD_ID" > /app/.image_build_id; \
    fi

RUN chmod +x docker/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/docker/entrypoint.sh"]
