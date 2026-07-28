"""
Descarga archivos _lc.fits de curvas de luz TESS/SPOC 2-min desde MAST.

Posteriormente, el pipeline de preprocesamiento (Fase 3) extraerá la
serie PDCSAP_FLUX desde estos FITS. Este script solo descarga.

Lee data/splits/tics_labeled.csv y consulta MAST por cada TIC ID
buscando curvas SPOC de cadencia 120s. Mantiene un manifest CSV con
el resultado por TIC. El script es idempotente: en corridas sucesivas
saltea TICs en estado terminal (ok, no_data) y reintenta los que
fallaron (error, download_failed) salvo que se pase --no-retry-failed.

Sobre --max-sectors: capar sectores sesga el dataset (toma siempre los
primeros). Es útil para piloto/desarrollo. Para entrenamiento final del
paper se recomienda descargar todos los sectores o aplicar una política
explícita de selección documentada.

Salida:
  data/raw/lightcurves/mastDownload/TESS/...   archivos .fits (gitignored)
  data/splits/manifest.csv                      versionado, una fila por TIC

Uso:
  python scripts/download_lightcurves.py --limit 5            # piloto
  python scripts/download_lightcurves.py --limit 50           # muestra inicial
  python scripts/download_lightcurves.py                      # dataset completo
  python scripts/download_lightcurves.py --max-sectors 3      # capear sectores (piloto)
  python scripts/download_lightcurves.py --shuffle            # orden aleatorio (seed 42)
  python scripts/download_lightcurves.py --no-retry-failed    # no reintentar fallos

"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
import lightkurve as lk  # noqa: E402

LABELED_PATH = Path("data/splits/tics_labeled.csv")
RAW_DIR = Path("data/raw/lightcurves")
MANIFEST_PATH = Path("data/splits/manifest.csv")

# Estados terminales: TIC no se reintenta.
# 'ok' = descarga exitosa; 'no_data' = MAST confirmó que no hay SPOC 2-min para ese TIC.
TERMINAL_STATUSES = {"ok", "no_data"}

MANIFEST_COLS = [
    "tid", "label", "n_sectors_found", "n_sectors_downloaded",
    "sectors", "total_size_mb", "status", "error", "duration_s",
    "downloaded_at",
]


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame(columns=MANIFEST_COLS)


def tic_size_mb(tid: int) -> float:
    pattern = f"**/*{tid:016d}*_lc.fits"
    total = sum(p.stat().st_size for p in RAW_DIR.glob(pattern))
    return total / 1e6


def download_tic(tid: int, label: int, max_sectors: int | None) -> dict:
    t0 = time.time()
    row = {c: "" for c in MANIFEST_COLS}
    row.update({
        "tid": tid, "label": label,
        "n_sectors_found": 0, "n_sectors_downloaded": 0,
        "total_size_mb": 0.0, "status": "pending", "duration_s": 0.0,
    })
    try:
        search = lk.search_lightcurve(
            f"TIC {tid}", mission="TESS", author="SPOC", exptime=120,
        )
        n_found = len(search)
        row["n_sectors_found"] = n_found
        if n_found == 0:
            row["status"] = "no_data"
            return row
        if max_sectors is not None and n_found > max_sectors:
            search = search[:max_sectors]
        coll = search.download_all(download_dir=str(RAW_DIR))
        if coll is None or len(coll) == 0:
            row["status"] = "download_failed"
            return row
        sectors = sorted({int(lc.meta.get("SECTOR", -1)) for lc in coll})
        row["n_sectors_downloaded"] = len(coll)
        row["sectors"] = ";".join(str(s) for s in sectors)
        row["total_size_mb"] = round(tic_size_mb(tid), 2)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)[:500]
    finally:
        row["duration_s"] = round(time.time() - t0, 2)
        row["downloaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N TICs (piloto)")
    parser.add_argument("--max-sectors", type=int, default=None,
                        help="Máximo de sectores a descargar por TIC")
    parser.add_argument("--shuffle", action="store_true",
                        help="Mezclar el orden con seed=42 antes de aplicar --limit")
    parser.add_argument("--no-retry-failed", action="store_true",
                        help="Saltear también los TICs en estado 'error' o 'download_failed' "
                             "(por defecto se reintentan)")
    args = parser.parse_args()

    if not LABELED_PATH.exists():
        sys.exit(f"No existe {LABELED_PATH}. Corré antes scripts/get_data.py.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LABELED_PATH)
    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    manifest = load_manifest()
    if len(manifest):
        skip_statuses = TERMINAL_STATUSES | ({"error", "download_failed"} if args.no_retry_failed else set())
        skip_tids = set(
            manifest.loc[manifest["status"].isin(skip_statuses), "tid"].astype(int).tolist()
        )
        # Quitamos del manifest las filas de TICs que vamos a reintentar para no duplicar.
        retry_tids = set(manifest["tid"].astype(int).tolist()) - skip_tids
        if retry_tids:
            manifest = manifest[~manifest["tid"].astype(int).isin(retry_tids)].reset_index(drop=True)
    else:
        skip_tids = set()
    todo = df[~df["tid"].isin(skip_tids)].reset_index(drop=True)

    n_skip = len(df) - len(todo)
    retry_msg = " (modo: reintenta error/download_failed)" if not args.no_retry_failed else ""
    print(f"Total seleccionados:    {len(df):,}")
    print(f"En estado terminal:     {n_skip:,}")
    print(f"Por descargar/reintentar: {len(todo):,}{retry_msg}\n")

    if len(todo) == 0:
        print("Nada que hacer.")
        return 0

    new_rows: list[dict] = []
    for i, r in todo.iterrows():
        tid = int(r["tid"])
        label = int(r["label"])
        print(f"[{i + 1}/{len(todo)}] TIC {tid} (label={label}) ... ", end="", flush=True)
        result = download_tic(tid, label, args.max_sectors)
        print(
            f"{result['status']} | "
            f"{result['n_sectors_downloaded']}/{result['n_sectors_found']} sec | "
            f"{result['total_size_mb']:.1f} MB | "
            f"{result['duration_s']:.1f} s"
        )
        new_rows.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            combined = pd.concat([manifest, pd.DataFrame(new_rows, columns=MANIFEST_COLS)], ignore_index=True)
            combined.to_csv(MANIFEST_PATH, index=False)

    manifest = pd.concat([manifest, pd.DataFrame(new_rows, columns=MANIFEST_COLS)], ignore_index=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print("\n=== Resumen ===")
    print(manifest["status"].value_counts().to_string())
    ok = manifest[manifest["status"] == "ok"]
    if len(ok):
        print(f"\nTICs ok: {len(ok):,}")
        print(f"Sectores descargados (total): {int(ok['n_sectors_downloaded'].sum()):,}")
        print(f"Tamaño total: {ok['total_size_mb'].sum() / 1024:.2f} GB")
        print(f"Promedio sectores/TIC: {ok['n_sectors_downloaded'].mean():.2f}")
    print(f"\nManifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
