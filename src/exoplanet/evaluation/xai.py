"""
Explicabilidad (XAI) sobre curvas de luz (Fase 9).

Tres técnicas explícitas: **gradient saliency, integrated
gradients y occlusion sensitivity**. 
incorrecto.

Convenciones:

  * Todos los modelos del proyecto reciben un `batch` dict con clave
    `global_view` de shape `(B, 1, L)` (ver `src/exoplanet/models/base.py`).
    Las funciones de este módulo aceptan x de shape `(B, L, 1)` o `(L, 1)`
    porque esa es la forma natural "per-sample, channels-last" que conviene
    visualizar; internamente las transponemos a `(B, 1, L)` antes de invocar
    el modelo.

  * Las funciones NO cambian el modo del modelo (asume `model.eval()` por
    el caller). Tampoco mueven el modelo entre devices: el caller decide.

  * Los tensores devueltos están en CPU.

  * Loss/logit usado: para `target_class=1` (positivo) usamos el logit crudo
    de la cabeza (que `forward` ya devuelve squeezed). Para `target_class=0`,
    usamos `-logit` — equivalente al logit de la clase negativa en BCE.

  * Estas funciones operan en FP32 (el caller debe convertir si entrenó FP16).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

# Estilo consistente con plots.py
_PALETTE = sns.color_palette("colorblind")
_FIGSIZE = (10, 6)
_DPI = 120


def _ensure_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Acepta `(L, 1)` o `(B, L, 1)`. Devuelve `(x_batched, was_squeezed)`.

    Esto facilita uso por sample (un científico que mira UNA curva) sin
    obligar al caller a hacer unsqueeze a mano.
    """
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError(f"x debe ser (L, 1) o (B, L, 1); recibí shape={tuple(x.shape)}.")


def _to_model_input(x_blc: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convierte (B, L, 1) → batch dict que espera el modelo: `global_view` (B, 1, L)."""
    x_bcl = x_blc.transpose(1, 2).contiguous()  # (B, L, 1) -> (B, 1, L)
    return {"global_view": x_bcl}


def _signed_logit(model: nn.Module, x_blc: torch.Tensor, target_class: int) -> torch.Tensor:
    """Devuelve el logit "firmado" según target_class.

    BCE binaria: logit > 0 favorece clase positiva (1). Para atribuir hacia la
    clase negativa basta con invertir el signo.
    """
    logits = model(_to_model_input(x_blc))  # (B,)
    if target_class == 1:
        return logits
    if target_class == 0:
        return -logits
    raise ValueError(f"target_class debe ser 0 o 1; recibí {target_class}.")


def gradient_saliency(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    abs_value: bool = True,
) -> torch.Tensor:
    """Saliency = ∂logit/∂input.

    Args:
        model: en eval(). Recibe dict `{"global_view": (B, 1, L)}`.
        x: (B, L, 1) o (L, 1). Float.
        target_class: 0 o 1.
        abs_value: si True devuelve |grad| (default; para visualización). Si False
            devuelve grad firmado (útil si interesa dirección de la atribución).

    Returns:
        Tensor `(B, L)` en CPU.
    """
    x_batched, _ = _ensure_batched(x.detach().clone().float())
    x_batched.requires_grad_(True)

    # Importante: NO ponemos model.train(). Se asume model.eval() por el caller.
    logit = _signed_logit(model, x_batched, target_class).sum()
    # backward respecto a x. retain_graph=False: no necesitamos el grafo después.
    grad = torch.autograd.grad(logit, x_batched, retain_graph=False, create_graph=False)[0]
    # (B, L, 1) → (B, L)
    grad = grad.squeeze(-1)
    if abs_value:
        grad = grad.abs()
    return grad.detach().cpu()


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrated Gradients (Sundararajan et al., 2017).

    IG(x_i) = (x_i - baseline_i) · ∫₀¹ ∂F(baseline + α (x - baseline))/∂x_i dα

    Aproximamos la integral por riemann sum con `n_steps` pasos.

    Args:
        model: en eval().
        x: (B, L, 1) o (L, 1). Float.
        target_class: 0 o 1.
        n_steps: pasos de la suma de Riemann (más → más fiel, más costo).
        baseline: tensor del mismo shape que x. Default: ceros. Para curvas de
            luz normalizadas por mediana (~1.0) un baseline más natural sería
            `torch.ones_like(x)`, pero ceros es el default canónico de la lit.

    Returns:
        Tensor `(B, L)` con atribución firmada, en CPU.
    """
    x_batched, _ = _ensure_batched(x.detach().clone().float())
    if baseline is None:
        baseline = torch.zeros_like(x_batched)
    else:
        baseline_batched, _ = _ensure_batched(baseline.detach().clone().float())
        if baseline_batched.shape != x_batched.shape:
            raise ValueError(
                f"baseline shape {tuple(baseline_batched.shape)} != x shape {tuple(x_batched.shape)}"
            )
        baseline = baseline_batched

    # Riemann midpoint: α en (1/(2n), 3/(2n), ..., (2n-1)/(2n)). Midpoint reduce
    # error sistemático respecto al rectángulo izquierdo o derecho.
    alphas = (torch.arange(n_steps, dtype=torch.float32, device=x_batched.device) + 0.5) / n_steps

    total_grad = torch.zeros_like(x_batched)
    for alpha in alphas:
        interp = baseline + alpha * (x_batched - baseline)
        interp.requires_grad_(True)
        logit = _signed_logit(model, interp, target_class).sum()
        grad = torch.autograd.grad(logit, interp, retain_graph=False, create_graph=False)[0]
        total_grad = total_grad + grad.detach()

    avg_grad = total_grad / float(n_steps)
    ig = (x_batched - baseline) * avg_grad
    # (B, L, 1) → (B, L)
    return ig.squeeze(-1).detach().cpu()


def occlusion_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    window_size: int = 100,
    stride: int = 50,
    target_class: int = 1,
) -> torch.Tensor:
    """Sensibilidad por oclusión: desliza una ventana de ceros y mide el cambio de logit.

    La atribución por timestep se obtiene promediando el `drop` sobre todas las
    ventanas que lo cubrieron (un timestep puede estar dentro de varias ventanas
    si `stride < window_size`).

    Args:
        model: en eval().
        x: (B, L, 1) o (L, 1).
        window_size: tamaño de la ventana ocluida.
        stride: paso entre ventanas. Si stride < window_size → solape.
        target_class: 0 o 1.

    Returns:
        Tensor `(B, L)` en CPU. Valores positivos = al ocluir esa región el
        logit cae → la región contribuía a favor de target_class. Valores
        negativos = ocluir la región AUMENTÓ el logit → región contribuía
        en contra.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size y stride deben ser > 0.")

    x_batched, _ = _ensure_batched(x.detach().clone().float())
    b, length, _ = x_batched.shape

    with torch.no_grad():
        base_logit = _signed_logit(model, x_batched, target_class).detach()  # (B,)

        attribution = torch.zeros((b, length), dtype=torch.float32, device=x_batched.device)
        coverage = torch.zeros((b, length), dtype=torch.float32, device=x_batched.device)

        # Generar starts asegurando cubrir el final aunque (L - window) % stride != 0
        starts = list(range(0, max(length - window_size + 1, 1), stride))
        if starts[-1] + window_size < length:
            starts.append(length - window_size)

        for start in starts:
            end = min(start + window_size, length)
            occluded = x_batched.clone()
            occluded[:, start:end, :] = 0.0
            new_logit = _signed_logit(model, occluded, target_class).detach()  # (B,)
            drop = (base_logit - new_logit).unsqueeze(1)  # (B, 1)
            attribution[:, start:end] += drop
            coverage[:, start:end] += 1.0

        coverage = coverage.clamp(min=1.0)
        attribution = attribution / coverage

    return attribution.detach().cpu()


def plot_xai_overlay(
    curve: np.ndarray | torch.Tensor,
    attribution: np.ndarray | torch.Tensor,
    output_path: Path | str,
    title: str | None = None,
) -> Path:
    """Figura de dos paneles: curva original arriba + heatmap de atribución abajo.

    Args:
        curve: 1-D (L,) o (1, L). Valores de flujo.
        attribution: 1-D (L,) o (1, L). Mismo length que curve.
        output_path: dónde guardar el PNG.
        title: suptitle opcional.

    Returns:
        Path absoluto del PNG generado.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(curve):
        curve = curve.detach().cpu().numpy()
    if torch.is_tensor(attribution):
        attribution = attribution.detach().cpu().numpy()
    curve = np.asarray(curve).squeeze()
    attribution = np.asarray(attribution).squeeze()
    if curve.ndim != 1 or attribution.ndim != 1:
        raise ValueError(
            f"curve y attribution deben ser 1-D después de squeeze; "
            f"shapes={curve.shape}, {attribution.shape}"
        )
    if curve.shape[0] != attribution.shape[0]:
        raise ValueError(
            f"length mismatch: curve={curve.shape[0]} vs attribution={attribution.shape[0]}"
        )

    length = curve.shape[0]
    t = np.arange(length)

    fig, (ax_curve, ax_attr) = plt.subplots(
        2, 1, figsize=_FIGSIZE, dpi=_DPI, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_curve.plot(t, curve, color=_PALETTE[0], lw=0.8)
    ax_curve.set_ylabel("Flujo normalizado")
    ax_curve.grid(True, alpha=0.3)

    # Heatmap por imshow sobre matriz 1xL para tener colorbar consistente.
    abs_max = float(np.abs(attribution).max()) if attribution.size else 1.0
    abs_max = abs_max if abs_max > 0 else 1.0
    im = ax_attr.imshow(
        attribution[np.newaxis, :],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        extent=(0, length, 0, 1),
    )
    ax_attr.set_yticks([])
    ax_attr.set_xlabel("Timestep")
    ax_attr.set_ylabel("Atribución")
    fig.colorbar(im, ax=ax_attr, orientation="horizontal", pad=0.25, fraction=0.5)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


__all__ = [
    "gradient_saliency",
    "integrated_gradients",
    "occlusion_sensitivity",
    "plot_xai_overlay",
]
