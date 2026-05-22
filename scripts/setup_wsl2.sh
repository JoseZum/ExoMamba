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
    python3 \
    python3-venv \
    python3-dev \
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
    # head cierra el pipe antes de que nvidia-smi termine de escribir, lo que provoca
    # SIGPIPE. Con `set -euo pipefail`, eso mataría el script. `|| true` lo evita.
    nvidia-smi | head -15 || true
else
    echo "[WARN] nvidia-smi no encontrado en WSL2."
    echo "       Asegurate de tener el driver NVIDIA instalado en Windows host."
fi

# --- 5) Crear venv si no existe ---
echo
echo "--- Setup venv Python ---"
# Si existe un .venv/ pero sin bin/activate, es de Windows (tiene Scripts/activate).
# Lo borramos para crear uno nativo de Linux.
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo "[WARN] Detectado .venv/ de Windows (sin bin/activate). Recreando para Linux..."
    rm -rf .venv
fi
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[OK] venv creado en .venv/ (usando $(python3 --version))"
else
    echo "[OK] venv ya existe y es de Linux."
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
echo "--- Instalando herramientas de build (necesarias para --no-build-isolation) ---"
# Sin esto, causal-conv1d y mamba-ssm fallan con "ModuleNotFoundError: No module named 'wheel'"
# porque --no-build-isolation les hace usar el venv actual en vez de un entorno aislado.
pip install --upgrade pip setuptools wheel packaging ninja

echo
echo "--- Instalando causal-conv1d ---"
pip install --no-build-isolation "causal-conv1d>=1.4.0"

echo
echo "--- Instalando mamba-ssm (sin deps, pinned <2.3 por compat con triton 3.1) ---"
# CRÍTICO #1: mamba-ssm reciente arrastra transformers 5.x, que requiere torch>=2.12.
# Eso desinstalaría nuestro torch 2.5.1+cu121 y rompería el ABI compilado de
# causal-conv1d (undefined symbol al importar). --no-deps evita esto.
# CRÍTICO #2: mamba-ssm 2.3+ requiere triton>=3.5, que solo viene con torch 2.12.
# Con torch 2.5.1 (triton 3.1) tenemos que quedarnos en la rama 2.2.x.
pip install --no-build-isolation --no-deps "mamba-ssm>=2.2.0,<2.3.0"

# Aunque usemos la clase Mamba directamente en nuestro modelo, el __init__.py del
# paquete sí importa MambaLMHeadModel (que necesita transformers). Por eso lo
# instalamos manualmente, pinned a 4.x para no traer la 5.x que rompe torch.
echo
echo "--- Instalando transformers <5 (necesario para el import de mamba_ssm) ---"
pip install "transformers<5"

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
