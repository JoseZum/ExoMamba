"""CLI de XAI sobre runs entrenados (Fase 5 - AMBICIOSO).

Carga un run (`config.yaml` + `checkpoints/best.pt`) y un `predictions.csv` ya
generado por `scripts/evaluate.py`. Identifica los cuadrantes de confusión
(TP, TN, FN, FP) con un threshold dado y, por cada cuadrante, selecciona los
top-K casos más "confidentes" (mayor `|y_prob - 0.5|`). Para cada caso corre
las tres técnicas XAI obligatorias (CLAUDE.md):

  - `gradient_saliency`
  - `integrated_gradients`
  - `occlusion_sensitivity`

Cada caso produce N PNGs (uno por método) vía `plot_xai_overlay`, más un
overview `_summary.png` que muestra los 8 casos (4 cuadrantes x 2 ranks) con
la curva + atribución por occlusion (suele ser la más interpretable visualmente).

Limitaciones conocidas:

  - Las funciones en `src/exoplanet/evaluation/xai.py` están escritas para
    modelos single-view: convierten internamente `(B, L, 1)` -> batch dict
    `{"global_view": (B, 1, L)}` y NO incluyen `local_view` ni `scalar_features`.
    Para Mamba single-view y CNN single-branch eso funciona sin cambios.
    Para modelos multi-view (ExoMamba V1, AstroNet multibranch) hay que
    extender via wrappers en este script (no en xai.py). Se proveen dos
    wrappers documentados pero por ahora SOLO se ha validado el flujo
    single-view; multi-view se marca como TODO y se levanta un error claro
    si el caller intenta correrlo.

Uso:

  python scripts/run_xai.py \\
      --run experiments/2026-05-22_14-32-51_mamba_small \\
      --split test \\
      --output paper/figures/xai/mamba_small

  python scripts/run_xai.py \\
      --run experiments/<run> \\
      --split test \\
      --output paper/figures/xai/<model> \\
      --top-k 2 \\
      --methods saliency,integrated_gradients,occlusion
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

# Inserta src/ en path como hace el resto de scripts cuando el paquete no está
# instalado editable. Cinturón y tirantes.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from exoplanet.data import LightCurveDataset  # noqa: E402
from exoplanet.evaluation.xai import (  # noqa: E402
    gradient_saliency,
    integrated_gradients,
    occlusion_sensitivity,
    plot_xai_overlay,
)
from exoplanet.training.config import load_config  # noqa: E402
from exoplanet.training.registry import build_model  # noqa: E402

QUADRANTS = ("TP", "TN", "FN", "FP")
DEFAULT_METHODS = ("saliency", "integrated_gradients", "occlusion")


# ---------------------------------------------------------------------------
# Wrappers multi-view (TODO)
# ---------------------------------------------------------------------------
def _multiview_global_saliency_wrapper(
    model: nn.Module,
    x_global: torch.Tensor,
    local_view: torch.Tensor | None,
    scalar_features: torch.Tensor | None,
    method: str,
    **method_kwargs: Any,
) -> torch.Tensor:
    """Wrapper para atribuir SOLO sobre `global_view` manteniendo `local_view`
    fijo. NO usado por ahora - todos los modelos del catalogue actual (CNN
    single-branch, Mamba single) ignoran local_view / scalar_features.

    Implementación bocetada para el momento en que entren los modelos
    multi-view (ExoMamba V1, AstroNet multibranch). Idea:

      1. Construir una `nn.Module` envoltorio que reciba SOLO un tensor
         `global_view` y, dentro del forward, lo combine con un `local_view`
         FIJO (registrado como buffer) antes de llamar al modelo real.
      2. Pasar ese envoltorio a las funciones XAI estándar.

    Hasta que esos modelos existan en el registry, este wrapper levanta
    `NotImplementedError` para no dar resultados engañosos.
    """
    raise NotImplementedError(
        "Multi-view XAI wrapper aún no implementado. "
        "Los modelos actualmente soportados son single-view (CNN, Mamba). "
        "Cuando se añadan ExoMamba V1 / AstroNet multibranch, "
        "construir un nn.Module envoltorio que mantenga local_view fijo y "
        "pasarlo a las funciones XAI estándar."
    )


# ---------------------------------------------------------------------------
# Carga de modelo y datos
# ---------------------------------------------------------------------------
def _resolve_split_csv(data_cfg: dict[str, Any], split: str) -> str:
    """Replica la lógica de scripts/evaluate.py para resolver el CSV del split."""
    key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[split]
    if key in data_cfg and data_cfg[key]:
        return str(data_cfg[key])
    if split == "test":
        return "data/splits/test_tics.csv"
    raise KeyError(f"El config no define '{key}' y no hay default para split='{split}'.")


def _load_checkpoint(ckpt_path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Carga state dict; misma lógica que evaluate.py (formato CheckpointManager)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible. Missing: {missing}. Unexpected: {unexpected}."
        )
    return ckpt


def _build_dataset(cfg: dict[str, Any], split: str) -> LightCurveDataset:
    data_cfg = cfg["data"]
    split_csv = _resolve_split_csv(data_cfg, split)
    return LightCurveDataset(
        split_csv,
        processed_dir=data_cfg.get("processed_dir", "data/processed/global"),
        augment=None,  # XAI sobre datos crudos
    )


def _index_by_tic(dataset: LightCurveDataset) -> dict[int, int]:
    """Mapa tic_id -> idx dentro del dataset, para lookup O(1)."""
    return {int(tid): i for i, tid in enumerate(dataset.tids)}


# ---------------------------------------------------------------------------
# Selección de casos
# ---------------------------------------------------------------------------
def _split_quadrants(
    preds: pd.DataFrame, threshold: float
) -> dict[str, pd.DataFrame]:
    """Divide predictions en 4 cuadrantes según (y_true, y_pred)."""
    if "y_pred" not in preds.columns:
        preds = preds.assign(y_pred=(preds["y_prob"] >= threshold).astype(int))
    q = {
        "TP": preds[(preds["y_true"] == 1) & (preds["y_pred"] == 1)],
        "TN": preds[(preds["y_true"] == 0) & (preds["y_pred"] == 0)],
        "FN": preds[(preds["y_true"] == 1) & (preds["y_pred"] == 0)],
        "FP": preds[(preds["y_true"] == 0) & (preds["y_pred"] == 1)],
    }
    return q


def _top_k_by_confidence(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-K más confidentes = los de mayor |y_prob - 0.5|.

    El threshold operacional es 0.5; la distancia al 0.5 es una proxy directa
    de cuán seguro estuvo el modelo al asignar la clase.
    """
    if df.empty:
        return df
    return (
        df.assign(_conf=(df["y_prob"] - 0.5).abs())
        .sort_values("_conf", ascending=False)
        .drop(columns="_conf")
        .head(k)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Ejecución de los métodos XAI
# ---------------------------------------------------------------------------
def _make_method_dispatch(
    model: nn.Module,
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Devuelve un dict {nombre_método: callable(x_blc) -> attribution (B, L)}.

    Cada callable está cerrado sobre `model` y los hiperparámetros default.
    Esto permite añadir métodos nuevos sin tocar el caller.
    """
    return {
        "saliency": lambda x: gradient_saliency(model, x, target_class=1, abs_value=True),
        "integrated_gradients": lambda x: integrated_gradients(
            model, x, target_class=1, n_steps=32
        ),
        "occlusion": lambda x: occlusion_sensitivity(
            model, x, window_size=200, stride=100, target_class=1
        ),
    }


def _to_xai_input(global_view: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convierte la `global_view` del dataset (1, L) a (L, 1) para xai.py.

    El Dataset devuelve `global_view` de shape `(1, L)` (channels-first, batch
    NO incluido - el collate lo añadiría). Las funciones XAI esperan
    `(L, 1)` o `(B, L, 1)` (channels-last), así que transponemos.
    """
    if global_view.dim() != 2 or global_view.shape[0] != 1:
        raise ValueError(
            f"global_view inesperada: shape={tuple(global_view.shape)}, esperaba (1, L)."
        )
    x = global_view.transpose(0, 1).contiguous().float().to(device)  # (L, 1)
    return x


# ---------------------------------------------------------------------------
# Plot del overview
# ---------------------------------------------------------------------------
def _plot_summary_grid(
    cases: list[dict[str, Any]],
    output_path: Path,
    method_for_overlay: str = "occlusion",
) -> Path:
    """Grid 4x2 (filas = cuadrantes TP/TN/FN/FP, columnas = rank 1/2).

    Cada celda muestra la curva en eje y normalizado + la atribución del
    método `method_for_overlay` con shading rojo proporcional a la atribución.

    Si `cases` tiene menos elementos que casillas, se dejan en blanco.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), dpi=120, sharex=True)

    # Index cases por (cuadrante, rank) para lookup robusto
    by_qr: dict[tuple[str, int], dict[str, Any]] = {
        (c["quadrant"], c["rank"]): c for c in cases
    }

    for row, quadrant in enumerate(QUADRANTS):
        for col in range(2):
            ax = axes[row, col]
            case = by_qr.get((quadrant, col))
            if case is None:
                ax.set_visible(False)
                continue
            curve = np.asarray(case["curve"]).squeeze()
            attr = case["attributions"].get(method_for_overlay)
            tic = case["tic_id"]
            y_prob = case["y_prob"]
            t = np.arange(curve.shape[0])

            ax.plot(t, curve, color="steelblue", lw=0.6)
            if attr is not None:
                attr = np.asarray(attr).squeeze()
                # Twin axis para el overlay de atribución
                ax2 = ax.twinx()
                abs_max = float(np.abs(attr).max()) or 1.0
                ax2.fill_between(
                    t,
                    0,
                    attr / abs_max,
                    color="crimson",
                    alpha=0.25,
                    linewidth=0,
                )
                ax2.set_ylim(-1.05, 1.05)
                ax2.set_yticks([])
            ax.set_title(
                f"{quadrant} #{col + 1} - TIC {tic} (p={y_prob:.3f})", fontsize=9
            )
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"XAI overview - overlay = {method_for_overlay}", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Corre XAI (saliency, integrated gradients, occlusion) sobre los "
            "casos más confidentes por cuadrante de confusión."
        )
    )
    p.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run dir generado por scripts/train.py (debe contener config.yaml y best.pt).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split del que tomar los casos (default: test).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directorio donde guardar los PNGs (se crea si no existe).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Casos por cuadrante (default: 2 -> 8 casos x N metodos).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help=(
            "Lista separada por comas. Opciones: "
            f"{', '.join(DEFAULT_METHODS)}. Default: las tres."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold para clasificar y_pred si predictions.csv no lo trae.",
    )
    p.add_argument(
        "--predictions",
        type=str,
        default=None,
        help=(
            "Path explícito a predictions.csv. Default: "
            "<run>/eval_<split>/predictions.csv (lo que escribe evaluate.py)."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Forzar device. Default: cuda si está disponible.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in methods if m not in DEFAULT_METHODS]
    if invalid:
        print(
            f"ERROR: métodos no reconocidos: {invalid}. "
            f"Disponibles: {DEFAULT_METHODS}",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"ERROR: run dir no existe: {run_dir}", file=sys.stderr)
        return 2
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    cfg = load_config(cfg_path)

    # predictions.csv
    if args.predictions is not None:
        preds_path = Path(args.predictions)
    else:
        preds_path = run_dir / f"eval_{args.split}" / "predictions.csv"
    if not preds_path.exists():
        print(
            f"ERROR: predictions.csv no encontrado en {preds_path}.\n"
            f"Corré primero: python scripts/evaluate.py --run {run_dir} --split {args.split}",
            file=sys.stderr,
        )
        return 2

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda pedido pero CUDA no está disponible; usando CPU.")
        device = torch.device("cpu")
    elif args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Modelo
    model = build_model(cfg["model"]).to(device)
    _load_checkpoint(ckpt_path, model, device)
    model.eval()
    print(f"Modelo: {cfg['model']['type']} | checkpoint: {ckpt_path}")

    # Dataset (mismo split que predictions)
    dataset = _build_dataset(cfg, args.split)
    tic_to_idx = _index_by_tic(dataset)
    print(f"Dataset: {len(dataset)} muestras del split '{args.split}'")

    # Predictions + cuadrantes + top-K
    preds = pd.read_csv(preds_path)
    required_cols = {"tic_id", "y_true", "y_prob"}
    if not required_cols.issubset(preds.columns):
        print(
            f"ERROR: predictions.csv debe tener {required_cols}; trae {set(preds.columns)}",
            file=sys.stderr,
        )
        return 2
    quadrants = _split_quadrants(preds, threshold=args.threshold)
    print(
        "Cuadrantes (n): "
        + ", ".join(f"{q}={len(df)}" for q, df in quadrants.items())
    )

    selection: dict[str, pd.DataFrame] = {
        q: _top_k_by_confidence(df, args.top_k) for q, df in quadrants.items()
    }

    # Methods dispatch
    method_fns = _make_method_dispatch(model)

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline por caso
    summary_cases: list[dict[str, Any]] = []
    n_emitted = 0
    for quadrant, top_df in selection.items():
        if top_df.empty:
            print(f"  {quadrant}: 0 casos disponibles, salto.")
            continue
        for rank, row in enumerate(top_df.itertuples(index=False)):
            tic = int(row.tic_id)
            y_prob = float(row.y_prob)
            idx = tic_to_idx.get(tic)
            if idx is None:
                print(
                    f"  WARNING: TIC {tic} no está en el dataset {args.split}; "
                    "salto.",
                )
                continue
            sample = dataset[idx]
            global_view = sample["global_view"]  # (1, L)
            x_blc = _to_xai_input(global_view, device)  # (L, 1)

            attributions: dict[str, np.ndarray] = {}
            for method_name in methods:
                fn = method_fns[method_name]
                attr = fn(x_blc)  # (1, L) o (L,)
                attr_np = attr.detach().cpu().numpy().squeeze()
                attributions[method_name] = attr_np
                out_png = (
                    output_dir
                    / f"{quadrant}_{rank + 1}_tic{tic}_{method_name}.png"
                )
                plot_xai_overlay(
                    curve=global_view.squeeze().detach().cpu().numpy(),
                    attribution=attr_np,
                    output_path=out_png,
                    title=(
                        f"{quadrant} rank {rank + 1} - TIC {tic} "
                        f"(y_prob={y_prob:.3f}) - {method_name}"
                    ),
                )
                n_emitted += 1
                print(f"  -> {out_png.name}")

            summary_cases.append(
                {
                    "quadrant": quadrant,
                    "rank": rank,
                    "tic_id": tic,
                    "y_prob": y_prob,
                    "curve": global_view.squeeze().detach().cpu().numpy(),
                    "attributions": attributions,
                }
            )

    # Summary grid (usa occlusion si está, si no el primero disponible)
    if summary_cases:
        method_for_overlay = (
            "occlusion" if "occlusion" in methods else methods[0]
        )
        summary_path = output_dir / "_summary.png"
        _plot_summary_grid(summary_cases, summary_path, method_for_overlay)
        print(f"\nOverview: {summary_path}")
    else:
        print("\nNo se generó overview porque no hubo casos válidos.")

    print(f"\nTotal PNGs generados: {n_emitted} en {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
