"""Verificador del entorno WSL2 antes de entrenar Mamba.

Chequea en orden:
  1. Sistema operativo Linux (no Windows).
  2. Python ≥ 3.10.
  3. PyTorch instalado, con CUDA compilada.
  4. GPU NVIDIA visible y accesible.
  5. nvcc disponible (para compilar mamba-ssm si hace falta).
  6. causal-conv1d importable.
  7. mamba-ssm importable.
  8. Test funcional: forward pass de un bloque Mamba en GPU.

Si falla algún paso, imprime cómo arreglarlo. Devuelve exit code 0 si todo OK,
1 si alguna verificación falló.

Uso (dentro de WSL2 con el venv activado):
  python scripts/verify_wsl2_env.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"{mark} {name}" + (f"  ({detail})" if detail else ""))
    if not ok and fix:
        print(f"       Fix: {fix}")
    return ok


def main() -> int:
    failures = 0

    section("1) Sistema operativo")
    is_linux = sys.platform.startswith("linux")
    failures += not check(
        "Linux",
        is_linux,
        sys.platform,
        "Estás en Windows? Mamba solo corre en WSL2 con Ubuntu.",
    )

    # Detectar si efectivamente es WSL
    is_wsl = False
    if is_linux:
        try:
            with open("/proc/version") as f:
                proc_ver = f.read().lower()
            is_wsl = "microsoft" in proc_ver or "wsl" in proc_ver
        except FileNotFoundError:
            pass
    print(f"       (WSL detectado: {is_wsl})")

    section("2) Python")
    py_ok = sys.version_info >= (3, 10)
    failures += not check(
        "Python ≥ 3.10",
        py_ok,
        sys.version.split()[0],
        "sudo apt install python3.11 python3.11-venv python3.11-dev",
    )

    section("3) PyTorch + CUDA")
    try:
        import torch

        torch_ok = True
        print(f"       torch {torch.__version__}")
        print(f"       compilado con CUDA: {torch.version.cuda}")
    except ImportError:
        torch_ok = False
        torch = None
    failures += not check(
        "torch importable",
        torch_ok,
        fix="pip install -e '.[dev]'",
    )

    if torch_ok:
        cuda_ok = torch.cuda.is_available()
        failures += not check(
            "CUDA disponible para torch",
            cuda_ok,
            fix="Verificar que el driver NVIDIA está instalado en Windows host (no en WSL2).",
        )
        if cuda_ok:
            print(f"       GPU: {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"       VRAM total: {total / 1e9:.2f} GB | libre: {free / 1e9:.2f} GB")

    section("4) nvcc (compilador CUDA — necesario para compilar mamba-ssm)")
    nvcc_path = shutil.which("nvcc")
    failures += not check(
        "nvcc en PATH",
        nvcc_path is not None,
        nvcc_path or "no encontrado",
        "sudo apt install nvidia-cuda-toolkit  (o bien instalar CUDA Toolkit desde nvidia.com/cuda-downloads)",
    )
    if nvcc_path:
        try:
            out = subprocess.check_output(["nvcc", "--version"], text=True)
            for line in out.splitlines():
                if "release" in line.lower():
                    print(f"       {line.strip()}")
        except subprocess.CalledProcessError:
            pass

    section("5) causal-conv1d")
    try:
        import causal_conv1d  # noqa: F401

        cc1d_ok = True
        print(f"       causal_conv1d {causal_conv1d.__version__}")
    except ImportError as e:
        cc1d_ok = False
        print(f"       Error: {e}")
    failures += not check(
        "causal-conv1d importable",
        cc1d_ok,
        fix="pip install causal-conv1d  (debe compilar con nvcc, puede tardar 5-10 min la primera vez)",
    )

    section("6) mamba-ssm")
    try:
        import mamba_ssm

        mamba_ok = True
        print(f"       mamba_ssm {mamba_ssm.__version__}")
    except ImportError as e:
        mamba_ok = False
        print(f"       Error: {e}")
    failures += not check(
        "mamba-ssm importable",
        mamba_ok,
        fix="pip install -e '.[dev,mamba]'  (después de instalar nvcc y causal-conv1d)",
    )

    section("7) Test funcional: forward de un bloque Mamba")
    if torch_ok and mamba_ok and torch.cuda.is_available():
        try:
            from mamba_ssm import Mamba

            device = torch.device("cuda")
            block = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
            x = torch.randn(2, 1000, 64, device=device)
            with torch.no_grad():
                y = block(x)
            assert y.shape == (2, 1000, 64)
            test_ok = True
            print(f"       forward OK | shape: {tuple(y.shape)}")
        except Exception as e:
            test_ok = False
            print(f"       Error: {type(e).__name__}: {e}")
        failures += not check("Forward pass de Mamba en GPU", test_ok)
    else:
        print("       (saltado: dependencias previas fallaron)")
        failures += 1

    section("Veredicto")
    if failures == 0:
        print("TODO PASA. El entorno WSL2 está listo para entrenar Mamba.")
        print()
        print("Próximos pasos:")
        print("  1) python scripts/smoke_train_mamba.py     # smoke con tensores random")
        print("  2) python scripts/train.py --config configs/mamba_small.yaml   # entrenamiento real")
        return 0
    print(f"FALLARON {failures} chequeos. NO empieces a entrenar hasta arreglar.")
    print()
    print("Si seguiste los pasos de docs/WSL2_SETUP.md y siguen fallando,")
    print("revisá la sección 'Troubleshooting' de ese mismo doc.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
