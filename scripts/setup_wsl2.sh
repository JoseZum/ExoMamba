#!/usr/bin/env bash
# Setup del entorno WSL2 para entrenar Mamba (Fase 8).
#
# Pre-requisitos (HACER EN WINDOWS, no acá):
#   1. WSL2 instalado: en PowerShell admin → wsl --install -d Ubuntu-24.04
#   2. Driver NVIDIA instalado en Windows host (NO dentro de WSL2).
#      Verificar: nvidia-smi.exe en PowerShell host.
#   3. WSL2 reiniciado tras instalar driver: wsl --shutdown (en PowerShell).
#
# Después de eso, abrir la terminal de Ubuntu, clonar el repo y correr este script.
#
# Uso (dentro de WSL2):
#   chmod +x scripts/setup_wsl2.sh
#   ./scripts/setup_wsl2.sh
#
# Es idempotente: se puede correr varias veces sin romper nada.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo "  Setup WSL2 para mamba-exoplanet"
echo "  Repo: $REPO_ROOT"
echo "=================================================================="

# --- 1) Sanity check: estamos en Linux WSL ---
if ! grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    echo "[ERROR] Este script es solo para WSL2. Estás en Linux nativo o macOS?"
    exit 1
fi
echo "[OK] Estás en WSL2."

# --- 2) Sistema: paquetes base ---
echo
echo "--- Actualizando APT y dependencias del sistema ---"
sudo apt update
sudo apt install -y \
    build-essential \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    nvidia-cuda-toolkit

# --- 3) Verificar nvcc ---
echo
echo "--- Verificando nvcc ---"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc no se instaló bien. Probá manual:"
    echo "         sudo apt install nvidia-cuda-toolkit"
    exit 1
fi
nvcc --version | grep release || true

# --- 4) Verificar GPU visible ---
echo
echo "--- Verificando GPU NVIDIA (debe verse desde WSL2) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -15
else
    echo "[WARN] nvidia-smi no encontrado en WSL2."
    echo "       Asegurate de tener el driver NVIDIA instalado en Windows host."
fi

# --- 5) Crear venv si no existe ---
echo
echo "--- Setup venv Python 3.11 ---"
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    echo "[OK] venv creado en .venv/"
else
    echo "[OK] venv ya existe."
fi

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip

# --- 6) Instalar PyTorch CUDA primero (mamba-ssm necesita torch ya instalado) ---
echo
echo "--- Instalando PyTorch con CUDA 12.x ---"
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# --- 7) Instalar el paquete con extras dev + mamba ---
echo
echo "--- Instalando exoplanet + dev + mamba (esto compila mamba-ssm, puede tardar 5-15 min) ---"
pip install -e ".[dev]"

# Instalar mamba aparte porque a veces falla y querés ver el error claro
echo
echo "--- Instalando causal-conv1d ---"
pip install --no-build-isolation causal-conv1d>=1.4.0

echo
echo "--- Instalando mamba-ssm ---"
pip install --no-build-isolation mamba-ssm>=2.2.0

# --- 8) Verificación final ---
echo
echo "--- Verificación del entorno ---"
python scripts/verify_wsl2_env.py

echo
echo "=================================================================="
echo "  Setup completo."
echo
echo "  Próximos pasos:"
echo "    source .venv/bin/activate"
echo "    python scripts/smoke_train_mamba.py"
echo "    python scripts/train.py --config configs/mamba_small.yaml"
echo "=================================================================="
