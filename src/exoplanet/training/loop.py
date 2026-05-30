"""
Funciones de entrenamiento y evaluación por epoch.

Separamos train_one_epoch y evaluate_one_epoch del orquestador
(runner.py) para que cada pieza la podamos testear y reusar por separado.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
    log_every_n_steps: int = 10,
    logger=None,
    tb_writer=None,
    global_step_start: int = 0,
) -> tuple[float, int]:
    """
    Una epoch de entrenamiento. Devuelve (loss promedio, global_step final).

    grad_clip: norma máxima de gradientes (clip_grad_norm_). Crítico para Mamba
    sobre secuencias largas con FP16 pOrque sin esto, los activations del SSM
    pueden explotar y producir NaN en el backward. None desactiva el clipping.
    """
    model.train()
    losses: list[float] = []
    global_step = global_step_start
    use_amp = fp16 and device.type == "cuda" and scaler is not None

    for i, batch in enumerate(loader):
        batch = _move_to_device(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(batch)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                # Hay que des-escalar antes de clipear: scaler.scale() multiplicó los grads
                # por un factor grande para evitar underflow en FP16.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        global_step += 1

        if log_every_n_steps and (i + 1) % log_every_n_steps == 0:
            if logger:
                logger.info(f"  step {i + 1}/{len(loader)} | loss={loss_val:.4f}")
            if tb_writer:
                tb_writer.add_scalar("train/step_loss", loss_val, global_step)

    avg = float(np.mean(losses)) if losses else float("nan")
    return avg, global_step


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evalúa el modelo. Devuelve dict con loss + arrays y_true / y_prob."""
    from exoplanet.training.metrics import compute_classification_metrics

    model.eval()
    losses: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for batch in loader:
        batch = _move_to_device(batch, device)
        labels = batch["label"]
        logits = model(batch)
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob_all.append(probs)
        y_true_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])

    metrics = compute_classification_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics
