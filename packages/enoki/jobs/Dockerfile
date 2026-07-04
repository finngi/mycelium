# Training image, pushed to the `training-images` Artifact Registry repo.
# CUDA base for l4/h100; the v5e (TPU) variant lands with the XLA/JAX adapter.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .
RUN uv venv && uv pip install -e . --group cluster
ENV PATH="/app/.venv/bin:$PATH"

# Entrypoint comes from the RayJob spec (python -m hyphae.driver <recipe>).
