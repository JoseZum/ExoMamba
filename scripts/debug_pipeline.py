"""Diagnóstico mínimo del pipeline de entrenamiento.

Verifica antes de cualquier sanity overfit:
  1. Que el Dataset carga curvas con la forma y rango esperados.
  2. Que los labels son sensatos (mix de 0s y 1s).
  3. Que el modelo hace forward sin crashear y devuelve logits con la forma correcta.
  4. Que la loss se calcula sin NaN ni inf.
  5. Que un solo step del optimizer ACTUALIZA los pesos (delta != 0).

Si cualquiera de estos falla, hay un bug que NO es de "el modelo no aprende"
— es de pipeline/configuración. Hay que arreglar eso antes de seguir.

Uso:
  python scripts/debug_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from exoplanet.data import LightCurveDataset
from exoplanet.models import CNNBaseline
from exoplanet.training import collate_lightcurves

TRAIN_CSV = Path("data/splits/train_tics.csv")
PROCESSED = Path("data/processed/global")


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "[PASS]" if ok else "[FAIL]"
    print(f"{mark} {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    if not TRAIN_CSV.exists() or not PROCESSED.exists():
        sys.exit("Faltan splits o .pt — corré make_splits y preprocess_global antes.")

    failures = 0

    # ---- 1) Dataset ----
    section("1) Dataset: forma, rango, label")
    ds = LightCurveDataset(TRAIN_CSV, processed_dir=PROCESSED)
    print(f"len(dataset) = {len(ds)}")
    s0 = ds[0]
    gv = s0["global_view"]
    print(f"sample[0] keys = {sorted(s0.keys())}")
    print(f"global_view shape = {tuple(gv.shape)}, dtype = {gv.dtype}")
    print(f"global_view min = {gv.min().item():.6f}")
    print(f"global_view max = {gv.max().item():.6f}")
    print(f"global_view mean = {gv.mean().item():.6f}")
    print(f"global_view std = {gv.std().item():.6f}")
    print(f"label[0] = {s0['label']}  (tipo: {type(s0['label']).__name__})")

    failures += not check("shape correcta (1, 18000)", tuple(gv.shape) == (1, 18000))
    failures += not check("dtype float32", gv.dtype == torch.float32)
    failures += not check(
        "valores cerca de 1.0",
        0.5 < gv.mean().item() < 1.5,
        f"mean={gv.mean().item():.4f}",
    )
    failures += not check("sin NaN", not torch.isnan(gv).any().item())
    failures += not check("sin inf", not torch.isinf(gv).any().item())

    # ---- 2) Labels variados ----
    section("2) Labels: mix de 0 y 1 en el dataset")
    labels_all = [ds.labels[i] for i in range(min(64, len(ds)))]
    n0 = labels_all.count(0)
    n1 = labels_all.count(1)
    print(f"primeros 64 labels: {n0} ceros, {n1} unos")
    failures += not check("hay ambas clases en los primeros 64", n0 > 0 and n1 > 0)

    # ---- 3) Collate + DataLoader ----
    section("3) DataLoader + collate")
    subset = Subset(ds, list(range(8)))
    loader = DataLoader(
        subset, batch_size=4, shuffle=False, collate_fn=collate_lightcurves
    )
    batch = next(iter(loader))
    print(f"batch keys = {sorted(batch.keys())}")
    print(f"batch[global_view] shape = {tuple(batch['global_view'].shape)}")
    print(f"batch[label] = {batch['label'].tolist()}  dtype={batch['label'].dtype}")
    print(f"batch[tic_id] = {batch['tic_id'].tolist()}")

    failures += not check(
        "global_view batched (4,1,18000)", tuple(batch["global_view"].shape) == (4, 1, 18000)
    )
    failures += not check("label dtype float32", batch["label"].dtype == torch.float32)
    failures += not check("local_view es None (Tier 1)", batch["local_view"] is None)

    # ---- 4) Forward + loss ----
    section("4) Modelo: forward + loss")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model = CNNBaseline(dropout=0.0).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    batch_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }
    model.eval()
    with torch.no_grad():
        logits = model(batch_dev)
    print(f"logits shape = {tuple(logits.shape)}, dtype = {logits.dtype}")
    print(f"logits valores = {logits.tolist()}")
    loss = loss_fn(logits, batch_dev["label"])
    print(f"loss inicial (sin entrenar) = {loss.item():.4f}")

    failures += not check("logits shape (B,)", logits.shape == (4,))
    failures += not check("loss finita", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    failures += not check(
        "loss razonable (cerca de log(2) ~ 0.69)",
        0.3 < loss.item() < 1.5,
        f"loss={loss.item():.4f}",
    )

    # ---- 5) Gradient + update ----
    section("5) Step del optimizer: ¿los pesos cambian?")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Tomamos snapshot de los pesos de la última Linear
    last_layer = model.head[-1]
    weight_before = last_layer.weight.detach().clone()
    bias_before = last_layer.bias.detach().clone()

    optimizer.zero_grad()
    logits = model(batch_dev)
    loss = loss_fn(logits, batch_dev["label"])
    loss.backward()

    # Magnitud de gradientes
    grad_norm = 0.0
    n_params_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            n_params_with_grad += 1
    grad_norm = grad_norm ** 0.5
    print(f"loss backward = {loss.item():.4f}")
    print(f"# params con gradiente = {n_params_with_grad}")
    print(f"norma total del gradiente = {grad_norm:.6f}")

    optimizer.step()

    weight_after = last_layer.weight.detach().clone()
    bias_after = last_layer.bias.detach().clone()
    delta_w = (weight_after - weight_before).abs().max().item()
    delta_b = (bias_after - bias_before).abs().max().item()
    print(f"delta máx en último Linear.weight = {delta_w:.6e}")
    print(f"delta máx en último Linear.bias   = {delta_b:.6e}")

    failures += not check(
        "gradiente no-cero",
        grad_norm > 1e-8,
        f"||grad||={grad_norm:.6e}",
    )
    failures += not check(
        "pesos cambian tras un step",
        delta_w > 1e-8,
        f"delta_w={delta_w:.6e}",
    )

    # ---- Veredicto ----
    section("Veredicto")
    if failures == 0:
        print("TODO PASA. El pipeline está sano. Podés correr el sanity overfit.")
        return 0
    print(f"FALLARON {failures} chequeos. NO sigas al sanity overfit hasta arreglar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
