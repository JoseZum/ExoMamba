"""CLI de evaluación final (Fase 9).

Carga el checkpoint `best.pt` de una corrida y evalúa sobre un split (default test).
Genera un dump completo bajo `<run_dir>/eval_<split>/`:

  * metrics.json     — métricas + threshold + n_samples + timestamp + meta
  * predictions.csv  — (tic_id, y_true, y_prob, y_pred)
  * roc_curve.png
  * pr_curve.png
  * confusion_matrix.png
  * calibration.png

Restricción operativa (CLAUDE.md §"Restricciones operativas críticas"):

  * El test sellado se evalúa UNA SOLA VEZ por modelo. Re-evaluar invalida
    el reporte final. Por eso este script imprime un warning explícito y
    además guarda timestamp en metrics.json para auditar.
  * Augmentation NUNCA en eval — se fuerza `augment=None` aunque el config
    del entrenamiento tuviera `augment.enabled=true`.

Uso:

  python scripts/evaluate.py --run experiments/<run_dir>
  python scripts/evaluate.py --run experiments/<run_dir> --split val
  python scripts/evaluate.py --run experiments/<run_dir> --split test --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Imports del paquete. Mamba se importa lazy desde MambaBaseline.__init__,
# así que importar el registry en Windows no truena aunque mamba-ssm falte:
# solo truena si el config pide `mamba_baseline` en runtime.
from exoplanet.data import LightCurveDataset
from exoplanet.evaluation.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from exoplanet.training.collate import collate_lightcurves
from exoplanet.training.config import load_config
from exoplanet.training.loop import evaluate_one_epoch
from exoplanet.training.losses import build_loss
from exoplanet.training.registry import build_model


SPLIT_CSV_KEYS = {
    "train": "train_csv",
    "val": "val_csv",
    "test": "test_csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evalúa el best.pt de una corrida sobre un split (default test)."
    )
    p.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run dir generado por scripts/train.py (debe contener config.yaml y checkpoints/best.pt).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split a evaluar. Default: test (CUIDADO: test es sellado).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de decisión para y_pred binario y la matriz de confusión.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override de batch_size. Por defecto usa el del config de entrenamiento.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override de num_workers del DataLoader.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Forzar device. Default: cuda si está disponible.",
    )
    return p.parse_args()


def _resolve_split_csv(data_cfg: dict[str, Any], split: str) -> str:
    """Resuelve la ruta del CSV del split.

    Convención del proyecto: el config del train referencia train_csv y val_csv.
    Para test asumimos `data/splits/test_tics.csv` salvo que el config lo
    declare explícitamente como `test_csv`.
    """
    key = SPLIT_CSV_KEYS[split]
    if key in data_cfg and data_cfg[key]:
        return str(data_cfg[key])
    if split == "test":
        return "data/splits/test_tics.csv"
    raise KeyError(
        f"El config no define '{key}' y no hay default para split='{split}'. "
        f"Agregalo al config o pasá un --split con default conocido."
    )


def _count_labels(csv_path: str | Path) -> tuple[int, int]:
    """Cuenta positivos/negativos sin cargar el dataset completo en memoria."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return pos, neg


def _load_checkpoint(ckpt_path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Carga state dict del modelo. Devuelve el dict del checkpoint para metadata."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    # weights_only=False porque el checkpoint del proyecto guarda metadata adicional
    # (epoch, optimizer_state, metrics) — formato del CheckpointManager.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatible. Missing: {missing}. Unexpected: {unexpected}."
        )
    return ckpt


def _gather_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Pasada de inferencia adicional para recolectar tic_id por sample.

    `evaluate_one_epoch` calcula loss + métricas pero no devuelve tic_ids,
    porque su firma fue diseñada para usarse en train (donde no necesitamos
    auditar por sample). Acá los necesitamos para escribir predictions.csv.
    """
    model.eval()
    tic_ids: list[int] = []
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_on_dev = {}
            for k, v in batch.items():
                batch_on_dev[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            logits = model(batch_on_dev)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.append(probs)
            y_true.append(batch["label"].cpu().numpy())
            tic_ids.extend(batch["tic_id"].cpu().tolist())

    y_true_arr = np.concatenate(y_true) if y_true else np.array([])
    y_prob_arr = np.concatenate(y_prob) if y_prob else np.array([])
    y_pred_arr = (y_prob_arr >= threshold).astype(int)
    return {
        "tic_id": np.asarray(tic_ids, dtype=np.int64),
        "y_true": y_true_arr.astype(int),
        "y_prob": y_prob_arr.astype(float),
        "y_pred": y_pred_arr,
    }


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true.astype(float)) ** 2))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _write_predictions_csv(path: Path, preds: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tic_id", "y_true", "y_prob", "y_pred"])
        for tic, yt, yp, yh in zip(
            preds["tic_id"], preds["y_true"], preds["y_prob"], preds["y_pred"], strict=True
        ):
            w.writerow([int(tic), int(yt), float(yp), int(yh)])


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"ERROR: run dir no existe: {run_dir}", file=sys.stderr)
        return 2
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    cfg = load_config(cfg_path)

    # WARNING explícito sobre el test sellado.
    if args.split == "test":
        print(
            "⚠️  EVALUANDO CONTRA TEST SELLADO. "
            "Esto debe correrse UNA SOLA VEZ por modelo al final.",
            flush=True,
        )

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda pedido pero CUDA no está disponible; usando CPU.")
        device = torch.device("cpu")
    elif args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Modelo desde registry (mamba-ssm se importa lazy en MambaBaseline.__init__).
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {cfg['model']['type']} | params: {n_params:,}")

    ckpt_meta = _load_checkpoint(ckpt_path, model, device)
    print(f"Checkpoint cargado: {ckpt_path.name} (epoch {ckpt_meta.get('epoch', '?')})")

    # Data — augment FORZADO a None (restricción operativa).
    data_cfg = cfg["data"]
    split_csv = _resolve_split_csv(data_cfg, args.split)
    batch_size = int(args.batch_size if args.batch_size is not None else data_cfg.get("batch_size", 16))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    dataset = LightCurveDataset(
        split_csv,
        processed_dir=data_cfg.get("processed_dir", "data/processed/global"),
        augment=None,  # NUNCA augmentation en eval
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_lightcurves,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Split: {args.split} | csv: {split_csv} | n_samples: {len(dataset)}")

    # Loss reproducida a partir del config original (mismo pos_weight balanced
    # del train para que el `loss` reportado sea comparable). Usamos counts del
    # CSV de TRAIN porque la BCE pos_weight es propiedad del modelo entrenado.
    train_pos, train_neg = _count_labels(data_cfg["train_csv"])
    loss_fn = build_loss(cfg["training"]["loss"], train_pos, train_neg).to(device)

    # 1) evaluate_one_epoch para loss + auc/f1/recall/precision (no duplicar código).
    eval_metrics = evaluate_one_epoch(model, loader, loss_fn, device)
    # eval_metrics ya trae auc_roc, auc_pr, f1, recall, precision, threshold (0.5), loss.

    # 2) Pasada extra para recolectar tic_id por sample + métricas extra (brier, acc).
    preds = _gather_predictions(model, loader, device, threshold=args.threshold)

    accuracy = _accuracy(preds["y_true"], preds["y_pred"])
    brier = _brier_score(preds["y_true"], preds["y_prob"])

    out_dir = run_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "run_dir": str(run_dir),
        "split": args.split,
        "split_csv": str(split_csv),
        "n_samples": int(len(dataset)),
        "n_pos": int((preds["y_true"] == 1).sum()),
        "n_neg": int((preds["y_true"] == 0).sum()),
        "threshold": float(args.threshold),
        "model_type": cfg["model"]["type"],
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "best_val_metrics_at_ckpt": ckpt_meta.get("metrics"),
        "device": str(device),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "loss": float(eval_metrics["loss"]),
            "auc_roc": float(eval_metrics["auc_roc"]),
            "auc_pr": float(eval_metrics["auc_pr"]),
            "f1": float(eval_metrics["f1"]),
            "recall": float(eval_metrics["recall"]),
            "precision": float(eval_metrics["precision"]),
            "accuracy": float(accuracy),
            "brier_score": float(brier),
        },
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    _write_predictions_csv(out_dir / "predictions.csv", preds)

    # Plots (si y_true tiene una sola clase, plot_roc/pr/calibration pueden fallar;
    # con N=237 y dataset balanceado a propósito esto no debería ocurrir).
    label = cfg["model"]["type"]
    plot_roc_curve(preds["y_true"], preds["y_prob"], out_dir / "roc_curve.png", label=label)
    plot_pr_curve(preds["y_true"], preds["y_prob"], out_dir / "pr_curve.png", label=label)
    plot_confusion_matrix(preds["y_true"], preds["y_pred"], out_dir / "confusion_matrix.png")
    plot_calibration(preds["y_true"], preds["y_prob"], out_dir / "calibration.png")

    print(f"\nResultados guardados en: {out_dir}")
    m = metrics_payload["metrics"]
    print(
        f"AUC-ROC={m['auc_roc']:.4f} | AUC-PR={m['auc_pr']:.4f} | "
        f"F1={m['f1']:.4f} | Recall={m['recall']:.4f} | Precision={m['precision']:.4f} | "
        f"Acc={m['accuracy']:.4f} | Brier={m['brier_score']:.4f} | loss={m['loss']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
