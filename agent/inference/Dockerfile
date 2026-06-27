# Servicio de inferencia Mamba — imagen Linux con CUDA + nvcc.
#
# mamba-ssm / causal-conv1d compilan kernels contra CUDA, por eso se usa la
# variante -devel (trae nvcc). La versión de torch/CUDA coincide con el entorno
# WSL2 donde se entrenó (torch 2.5.1 + cu121, triton 3.1), que es el rango que
# fija pyproject.toml ("mamba-ssm>=2.2.0,<2.3.0").
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MAMBA_SERVICE_HOST=0.0.0.0 \
    MAMBA_SERVICE_PORT=8077 \
    MAMBA_RUN_DIR=experiments/2026-05-28_01-44-54_mamba_small_seed789 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=4 \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

WORKDIR /app

# git para instalar los kernels desde fuente (los sdists de PyPI no traen csrc/).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install packaging ninja setuptools wheel

# Kernels Mamba: compilar contra el torch 2.5.1+cu121 de la imagen (mismo entorno
# que el WSL2 del proyecto). Se instalan desde GIT (tags v1.4.0 / v2.2.2) porque
# los sdists de PyPI no incluyen los fuentes CUDA (csrc/*.cpp) → "missing rule".
# --no-build-isolation evita el CUDA mismatch (usa el torch de la imagen, no baja
# torch 2.12). *_FORCE_BUILD + TORCH_CUDA_ARCH_LIST=8.6 compilan para la RTX 3050.
RUN pip install --no-build-isolation "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0"
RUN pip install --no-build-isolation "git+https://github.com/state-spaces/mamba.git@v2.2.2"

# mamba-ssm 2.2.x importa utilidades de generación de transformers que fueron
# removidas en transformers>=4.45 (GreedySearchDecoderOnlyOutput). Pin a la
# última versión compatible para que `from mamba_ssm import Mamba` no truene.
RUN pip install "transformers==4.44.2"

# Stack del servicio + deps mínimas del paquete `exoplanet` que toca el predictor
# (torch ya viene en la imagen; NO traemos astropy/lightkurve/sklearn).
COPY agent/inference/requirements.txt /tmp/req-inf.txt
RUN pip install -r /tmp/req-inf.txt && pip install "pyyaml>=6.0" "numpy>=1.26" "einops>=0.8"

# Código del proyecto. data/ y experiments/ NO van en la imagen: se montan como
# volúmenes read-only en runtime (ver docker-compose.yml).
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY agent /app/agent
RUN pip install -e . --no-deps

EXPOSE 8077
CMD ["python", "-m", "agent.inference.server"]
