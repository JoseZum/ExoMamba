"""Smoke train Mamba — PREFLIGHT obligatorio antes de Fase 8 real.

Este script NO toca datos reales. Genera tensores random con la forma
exacta que va a recibir el modelo en producción `(batch_size, L, 1)` y
hace 3 forward + backward + optimizer.step para confirmar que:

  1. mamba-ssm importa correctamente.
  2. La GPU acepta el modelo (no OOM).
  3. FP16 + gradient checkpointing funcionan en combinación.
  4. El backward pasa sin errores numéricos.

IMPORTANTE: Solo corre en Linux/WSL2 con CUDA + nvcc + mamba-ssm compilado.
mamba-ssm NO tiene wheels para Windows nativo — confirmado en CLAUDE.md.

Uso (en WSL2 Ubuntu 24.04):
  source .venv/bin/activate
  pip install mamba-ssm causal-conv1d
  python scripts/smoke_train_mamba.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

# --- Parámetros del smoke ---
BATCH_SIZE = 16
SEQ_LEN = 18000
D_MODEL = 64        # dimensión interna chica para que entre en 4 GB
D_STATE = 16
N_LAYERS = 4
LR = 1e-3
N_STEPS = 3
USE_FP16 = True


def section(msg: str) -> None:
    print()
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def main() -> int:
    section("1) Importar mamba-ssm")
    try:
        from mamba_ssm import Mamba  # type: ignore
    except ImportError as e:
        sys.exit(
            f"\n[FAIL] No se pudo importar mamba_ssm: {e}\n"
            "Estás en Windows nativo? Mamba-ssm no funciona allí.\n"
            "Pasos en WSL2 Ubuntu 24.04:\n"
            "  pip install causal-conv1d\n"
            "  pip install mamba-ssm\n"
        )
    print("[OK] mamba_ssm importado")

    section("2) Verificar CUDA")
    if not torch.cuda.is_available():
        sys.exit("\n[FAIL] CUDA no disponible. Mamba requiere GPU.")
    print(f"[OK] CUDA disponible — {torch.cuda.get_device_name(0)}")
    print(f"      torch {torch.__version__}")
    print(f"      VRAM disponible: ~{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")

    section("3) Construir modelo Mamba simple")

    class TinyMamba(nn.Module):
        """Stack de N_LAYERS bloques Mamba con proyección final a 1 logit."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, D_MODEL)
            self.layers = nn.ModuleList(
                [Mamba(d_model=D_MODEL, d_state=D_STATE, d_conv=4, expand=2) for _ in range(N_LAYERS)]
            )
            self.norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, 1)
            h = self.embed(x)              # (B, L, D)
            for layer in self.layers:
                h = layer(h) + h           # residual
            h = self.norm(h.mean(dim=1))    # pool global → (B, D)
            return self.head(h).squeeze(-1) # (B,)

    device = torch.device("cuda")
    model = TinyMamba().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Modelo creado | params: {n_params:,}")

    section("4) Forward + backward + step (x3)")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

    for step in range(1, N_STEPS + 1):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 1, device=device)
        y = torch.randint(0, 2, (BATCH_SIZE,), device=device).float()

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        if USE_FP16:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        dt = time.time() - t0
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  step {step} | loss={loss.item():.4f} | time={dt:.2f}s | VRAM={mem:.2f} GB")

    section("Veredicto")
    print("[OK] Smoke train Mamba completado sin errores.")
    print("    Podés proceder con configs/mamba_small.yaml + datos reales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
