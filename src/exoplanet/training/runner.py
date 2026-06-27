"""Orquestador: corre un experimento completo desde un config."""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from exoplanet.data import LightCurveDataset, build_augment_pipeline
from exoplanet.training.checkpoint import CheckpointManager
from exoplanet.training.collate import collate_lightcurves
from exoplanet.training.config import dump_config
from exoplanet.training.early_stopping import EarlyStopping
from exoplanet.training.loop import evaluate_one_epoch, train_one_epoch
from exoplanet.training.losses import build_loss
from exoplanet.training.optimizers import build_optimizer
from exoplanet.training.registry import build_model
from exoplanet.training.schedulers import build_scheduler
from exoplanet.utils.git_info import git_summary
from exoplanet.utils.logging import TensorBoardWriter, setup_logger
from exoplanet.utils.paths import make_experiment_dir
from exoplanet.utils.seeds import set_seed


def _count_labels(csv_path: str | Path) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return pos, neg


def _count_labels_from_dataset(dataset) -> tuple[int, int]:
    """Cuenta labels iterando el dataset realmente cargado. Respeta subset/filtros."""
    pos = neg = 0
    for i in range(len(dataset)):
        label = int(dataset[i]["label"])
        if label == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def _write_env_info(path: Path) -> None:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with path.open("w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")


def _write_git_info(path: Path) -> None:
    summary = git_summary()
    with path.open("w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def _build_loader(
    split_csv: str | Path,
    processed_dir: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    subset: int | None = None,
    augment_cfg: dict | None = None,
    local_dir: str | Path | None = None,
) -> DataLoader:
    """Construye un DataLoader sobre LightCurveDataset.

    Si `augment_cfg` está presente y `augment_cfg["enabled"]` es True, se
    construye un Compose desde `augment_cfg["pipeline"]` y se pasa al dataset.
    El caller es responsable de invocar este builder con `augment_cfg=None`
    para val/test (restricción operativa: augmentation solo en train).

    `local_dir` (opcional): si se pasa, el dataset cargará `<local_dir>/<tic>.pt`
    como `local_view`. Necesario para modelos Tier 2 (ExoMamba V1, AstroNet).
    """
    augment = None
    if augment_cfg is not None and augment_cfg.get("enabled", False):
        pipeline_spec = augment_cfg.get("pipeline", [])
        if pipeline_spec:
            augment = build_augment_pipeline(pipeline_spec)

    ds = LightCurveDataset(
        split_csv, processed_dir=processed_dir, augment=augment, local_dir=local_dir
    )
    if subset is not None and subset < len(ds):
        ds = Subset(ds, list(range(subset)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lightcurves,
        pin_memory=torch.cuda.is_available(),
    )


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    """Corre un experimento completo. Devuelve resumen con mejor métrica + run_dir."""
    # 1) Reproducibilidad y dir de salida
    exp_cfg = cfg["experiment"]
    set_seed(int(exp_cfg.get("seed", 42)), deterministic=bool(exp_cfg.get("deterministic", False)))
    run_dir = make_experiment_dir(exp_cfg.get("output_dir", "experiments"), exp_cfg["name"])

    # 2) Snapshots de reproducibilidad
    dump_config(cfg, run_dir / "config.yaml")
    _write_env_info(run_dir / "env_info.txt")
    _write_git_info(run_dir / "git_info.txt")

    # 3) Logger
    logger = setup_logger("train", log_file=run_dir / "train.log")
    logger.info(f"Experimento: {exp_cfg['name']}")
    logger.info(f"Run dir: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 4) Datos
    data_cfg = cfg["data"]
    subset = data_cfg.get("subset")  # None o int (para smoke tests)
    augment_cfg = data_cfg.get("augment")  # solo se aplica al train loader
    local_dir = data_cfg.get("local_dir")  # None para Tier 1; ruta para Tier 2
    train_loader = _build_loader(
        data_cfg["train_csv"], data_cfg["processed_dir"],
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        shuffle=True, subset=subset,
        augment_cfg=augment_cfg,
        local_dir=local_dir,
    )
    val_loader = _build_loader(
        data_cfg["val_csv"], data_cfg["processed_dir"],
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        shuffle=False, subset=subset,
        augment_cfg=None,  # val NUNCA con augmentation
        local_dir=local_dir,
    )
    if augment_cfg is not None and augment_cfg.get("enabled", False):
        aug_repr = getattr(train_loader.dataset, "augment", None)
        if aug_repr is None and hasattr(train_loader.dataset, "dataset"):
            aug_repr = getattr(train_loader.dataset.dataset, "augment", None)
        logger.info(f"Augmentation (train-only): {aug_repr}")
    # Contamos sobre el dataset realmente cargado (respeta subset si aplica),
    # no sobre el CSV crudo - antes mezclábamos los dos y el log mentía con subset.
    pos_count, neg_count = _count_labels_from_dataset(train_loader.dataset)
    val_pos, val_neg = _count_labels_from_dataset(val_loader.dataset)
    logger.info(f"Train: {len(train_loader.dataset)} samples (pos={pos_count}, neg={neg_count})")
    logger.info(f"Val:   {len(val_loader.dataset)} samples (pos={val_pos}, neg={val_neg})")

    # 5) Modelo, loss, optimizer, scheduler
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo: {cfg['model']['type']} | params entrenables: {n_params:,}")

    if cfg["training"].get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ACTIVADO")

    loss_fn = build_loss(cfg["training"]["loss"], pos_count, neg_count).to(device)
    optimizer = build_optimizer(model, cfg["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["training"].get("scheduler"))

    fp16 = bool(cfg["training"].get("fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    if fp16:
        logger.info("Mixed precision (FP16): ACTIVADO")

    # 6) Early stopping + checkpoints + TensorBoard
    es_cfg = cfg.get("early_stopping", {"enabled": False})
    es = (
        EarlyStopping(
            metric=es_cfg.get("metric", "val_auc"),
            patience=int(es_cfg.get("patience", 10)),
            mode=es_cfg.get("mode", "max"),
        )
        if es_cfg.get("enabled", False)
        else None
    )
    ckpt_mgr = CheckpointManager(
        run_dir / "checkpoints",
        metric=es_cfg.get("metric", "val_auc"),
        mode=es_cfg.get("mode", "max"),
    )
    tb_writer = TensorBoardWriter(run_dir / "tensorboard") if cfg.get("logging", {}).get("tensorboard", True) else TensorBoardWriter(None)

    # 7) Loop de epochs
    epochs = int(cfg["training"].get("epochs", 50))
    log_every = int(cfg.get("logging", {}).get("log_every_n_steps", 10))
    metrics_csv = run_dir / "metrics.csv"
    metrics_header = [
        "epoch", "train_loss", "val_loss", "val_auc_roc", "val_auc_pr",
        "val_f1", "val_recall", "val_precision", "lr",
    ]
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(metrics_header)

    # grad_clip: defensa contra NaN en Mamba+FP16. Default 1.0 (norma global).
    # Para desactivar pasar null en el YAML.
    grad_clip_cfg = cfg["training"].get("grad_clip", 1.0)
    grad_clip = float(grad_clip_cfg) if grad_clip_cfg is not None else None
    if grad_clip is not None:
        logger.info(f"Gradient clipping: max_norm={grad_clip}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        logger.info(f"=== Epoch {epoch}/{epochs} ===")
        train_loss, global_step = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            fp16=fp16, scaler=scaler, grad_clip=grad_clip,
            log_every_n_steps=log_every,
            logger=logger, tb_writer=tb_writer, global_step_start=global_step,
        )
        val = evaluate_one_epoch(model, val_loader, loss_fn, device)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_auc={val['auc_roc']:.4f} | "
            f"val_f1={val['f1']:.4f} | val_recall={val['recall']:.4f} | "
            f"val_prec={val['precision']:.4f} | lr={lr_now:.2e}"
        )

        # Loggear a TB
        tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
        tb_writer.add_scalars("epoch/val", {
            "loss": val["loss"],
            "auc_roc": val["auc_roc"],
            "auc_pr": val["auc_pr"],
            "f1": val["f1"],
            "recall": val["recall"],
            "precision": val["precision"],
        }, epoch)
        tb_writer.add_scalar("epoch/lr", lr_now, epoch)

        # Persistir métricas
        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val["loss"], val["auc_roc"], val["auc_pr"],
                val["f1"], val["recall"], val["precision"], lr_now,
            ])

        # Checkpoints
        # Si train_loss o val_loss son NaN/inf, NO consideramos este epoch como "best":
        # el modelo está envenenado (típicamente FP16 overflow) y aunque val_auc parezca
        # alto, el checkpoint guardado sería el de un estado roto que no se reproduce.
        import math as _m
        train_ok = _m.isfinite(train_loss)
        val_ok = _m.isfinite(val["loss"])
        if not train_ok or not val_ok:
            logger.warning(
                f"  [SKIP-BEST] Epoch {epoch}: train_loss={train_loss}, val_loss={val['loss']}. "
                "No se considera como candidato a best (estado inestable)."
            )
        else:
            renamed = ckpt_mgr.maybe_save_best(
                model, optimizer, epoch, {"val_auc": val["auc_roc"], **val}
            )
            if renamed:
                logger.info(f"  [BEST] Nuevo mejor val_auc={val['auc_roc']:.4f}")
        ckpt_mgr.save_last(model, optimizer, epoch, val)

        # Scheduler step (cosine se actualiza por epoch)
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if es is not None:
            es.step(val["auc_roc"])
            if es.stopped:
                logger.info(f"Early stopping disparado (patience={es.patience} sin mejora).")
                break

    tb_writer.close()
    summary = {
        "run_dir": str(run_dir),
        "best_val_auc": ckpt_mgr.best_value,
        "best_epoch": ckpt_mgr.best_epoch,
    }
    logger.info(f"=== Fin del entrenamiento ===")
    logger.info(f"Mejor val_auc: {summary['best_val_auc']:.4f} (epoch {summary['best_epoch']})")
    return summary
