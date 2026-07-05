# Training image, pushed to the `training-images` Artifact Registry repo.
# CUDA base for l4/h100; the v5e (TPU) variant lands with the XLA/JAX adapter.
#
# Build context must be the `ml/` parent directory, not `mcm-enoki/` alone --
# pyproject.toml resolves mcm via [tool.uv.sources] path = "../mcm-reishi",
# a sibling repo outside mcm-enoki's own tree that a plain `docker build mcm-enoki`
# can never see. From ml/:
#   docker build -f mcm-enoki/jobs/Dockerfile -t <tag> .
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY mcm-reishi /mcm-reishi
WORKDIR /app
COPY mcm-enoki .
RUN uv venv && uv pip install -e . --group cluster
ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint comes from the RayJob spec (python -m enoki.driver <recipe>).
