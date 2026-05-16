"""
Preprocesamiento base (vista global) para Tier 1.

Por cada TIC con status=ok en data/splits/manifest.csv:

  1. Lee todos sus archivos _lc.fits en los diferentes sectores.

  2. Elige el sector con mayor fracción de puntos validos, 
    donde un punto es valido si QUALITY=0 y PDCSAP_FLUX (columna de brillo) es finito.

  3. Enmascara puntos con QUALITY != 0 (los pone NaN).

  4. Interpola linealmente gaps de NaN de longitud <= MAX_GAP puntos.
    Ejm: [1.00, 1.01, NaN, 1.03, 1.04] se puede rellenar como: [1.00, 1.01, 1.02, 1.03, 1.04]
    Si MAX_GAP=n, entonces runs de NaN de longitud <= n se rellenan, y runs > n quedan como NaN.

  5. Descarta el TIC si la fraccion de puntos validos finales < MIN_VALID_FRACTION.
    Si después de limpiar la curva menos del 50% de los puntos son válidos, ese TIC se descarta,
    debido a que puede llegar a ser más ruido que señal.

  6. Normaliza dividiendo por la mediana de la propia curva (sin estadistica global,para evitar data leakage).

    Las estrellas no tienen todas el mismo brillo. Una estrella puede tener flujo promedio de 1000, otra de 50000, otra de 200000.
    Pero al modelo no le interesa el brillo absoluto. Le interesa la forma relativa de la curva, especialmente las bajadas pequeñas.

    Entonces se hace:

    flux_normalizado = flux / mediana_de_esa_misma_curva

  7. Recorta la longitud de la curva a L=18000 (o padea con 1.0 si len < L) y guarda valid_mask.
    valid_mask es una máscara que indica qué partes de la curva son datos confiables y qué partes son relleno o huecos largos

Salida:
  data/processed/global/<tid>.pt | tensor por TIC (gitignored)
  data/splits/processed_manifest.csv | versionado, una fila por TIC

Uso:
  python scripts/preprocess_global.py --limit 10    # piloto
  python scripts/preprocess_global.py               # dataset completo

"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

MANIFEST_PATH = Path("data/splits/manifest.csv")
LABELED_PATH = Path("data/splits/tics_labeled.csv")
RAW_DIR = Path("data/raw/lightcurves")
OUT_DIR = Path("data/processed/global")
OUT_MANIFEST = Path("data/splits/processed_manifest.csv")

L = 18000                  # longitud fija de salida
MIN_VALID_FRACTION = 0.5   # debajo de esto se descarta el TIC
MAX_GAP = 5                # gaps de NaN <= MAX_GAP se interpolan; mas largos quedan

PROC_COLS = [
    "tid", "label", "sector_chosen", "valid_fraction", "n_points_raw",
    "status", "error", "duration_s", "processed_at",
]


# 1) Buscar todos los FITS de un TIC en los sectores disponibles.
def find_fits_for_tic(tid: int) -> list[Path]:
    return sorted(RAW_DIR.glob(f"**/*{tid:016d}*_lc.fits"))


# Leer un sector y recuperar flujo, calidad y sector.
def load_sector(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        cols = set(data.columns.names)
        if "PDCSAP_FLUX" not in cols or "QUALITY" not in cols:
            raise ValueError(f"FITS sin PDCSAP_FLUX/QUALITY: {path.name}")
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
        quality = np.asarray(data["QUALITY"], dtype=np.int32)
        sector = int(hdul[0].header.get("SECTOR", -1))
    return flux, quality, sector


# 2) Medir la fraccion de puntos validos antes de elegir el mejor sector.
def valid_fraction(flux: np.ndarray, quality: np.ndarray) -> float:
    if len(flux) == 0:
        return 0.0
    return float(((quality == 0) & np.isfinite(flux)).mean())


# 4) Interpolar linealmente gaps cortos de NaN.
def interpolate_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola linealmente secuencias de NaN de longitud <= max_gap. Las más largas quedan NaN."""
    x = x.copy()
    n = len(x)
    isnan = ~np.isfinite(x)
    if not isnan.any():
        return x
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                x[i:j] = np.linspace(x[i - 1], x[j], gap_len + 2)[1:-1]
            i = j
        else:
            i += 1
    return x


# 7) Recortar o paddear la curva a longitud fija L.
def to_fixed_length(flux: np.ndarray, mask: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Recorta centrado o padea con 1.0 (mediana post-normalizacion)."""
    n = len(flux)
    if n == target:
        return flux, mask
    if n > target:
        start = (n - target) // 2
        return flux[start:start + target], mask[start:start + target]
    pad_total = target - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    out = np.concatenate([np.ones(pad_left), flux, np.ones(pad_right)])
    m = np.concatenate([np.zeros(pad_left, bool), mask, np.zeros(pad_right, bool)])
    return out, m


def process_tic(tid: int, label: int) -> dict:
    t0 = time.time()
    row = {c: "" for c in PROC_COLS}
    row.update({
        "tid": tid, "label": label, "sector_chosen": -1,
        "valid_fraction": 0.0, "n_points_raw": 0, "status": "pending",
        "duration_s": 0.0,
    })
    try:
        paths = find_fits_for_tic(tid)
        if not paths:
            row["status"] = "no_fits"
            return row

        # 1) Elegir mejor sector (mayor fraccion valida en crudo).
        best = None
        for p in paths:
            flux, quality, sector = load_sector(p)
            vf = valid_fraction(flux, quality)
            if best is None or vf > best[3]:
                best = (flux, quality, sector, vf)
        flux, quality, sector, _ = best
        row["sector_chosen"] = sector
        row["n_points_raw"] = len(flux)

        # 3) Enmascarar puntos con QUALITY != 0.
        flux_masked = flux.astype(np.float64).copy()
        flux_masked[quality != 0] = np.nan

        # 4) Interpolar gaps cortos de NaN y dejar los largos como NaN.
        flux_interp = interpolate_short_gaps(flux_masked, MAX_GAP)

        # 5) Descartar TICs con fraccion final de puntos validos muy baja.
        final_valid = float(np.isfinite(flux_interp).mean())
        row["valid_fraction"] = round(final_valid, 4)
        if final_valid < MIN_VALID_FRACTION:
            row["status"] = "dropped_low_quality"
            return row

        # 6) Normalizar por la mediana de la propia curva.
        median = np.nanmedian(flux_interp)
        if not np.isfinite(median) or median == 0:
            row["status"] = "dropped_bad_median"
            return row
        flux_norm = flux_interp / median

        # 6) Marcar como valido lo que el modelo puede usar.
        #    False = gap largo no interpolado o padding posterior.
        #    No distingue original vs interpolado; para Tier 1 no hace falta.
        valid_mask = np.isfinite(flux_norm)
        flux_norm[~valid_mask] = 1.0  # gaps largos -> mediana normalizada

        # 7) Recortar o paddear a longitud fija.
        flux_L, mask_L = to_fixed_length(flux_norm, valid_mask, L)

        # 7) Forma (1, L): canal explicito para CNN 1D / Mamba.
        global_view = torch.from_numpy(flux_L.astype(np.float32)).unsqueeze(0)
        valid_mask_t = torch.from_numpy(mask_L).unsqueeze(0)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tid": tid,
            "label": label,
            "global_view": global_view,
            "valid_mask": valid_mask_t,
            "sector": sector,
            "valid_fraction": final_valid,
        }, OUT_DIR / f"{tid}.pt")
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)[:500]
    finally:
        row["duration_s"] = round(time.time() - t0, 2)
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N TICs (piloto)")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"No existe {MANIFEST_PATH}. Corre antes scripts/download_lightcurves.py.")
    if not LABELED_PATH.exists():
        sys.exit(f"No existe {LABELED_PATH}. Corre antes scripts/get_data.py.")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["status"] = manifest["status"].astype(str).str.strip().str.lower()
    labels = pd.read_csv(LABELED_PATH)[["tid", "label"]]
    todo = (
        manifest[manifest["status"] == "ok"][["tid"]]
        .merge(labels, on="tid")
        .reset_index(drop=True)
    )
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Por procesar: {len(todo):,}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, r in enumerate(todo.itertuples(index=False), start=1):
        tid = int(r.tid)
        label = int(r.label)
        print(f"[{i}/{len(todo)}] TIC {tid} ... ", end="", flush=True)
        result = process_tic(tid, label)
        print(
            f"{result['status']} | sec={result['sector_chosen']} | "
            f"vf={result['valid_fraction']:.3f} | {result['duration_s']:.2f}s"
        )
        rows.append(result)
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(rows, columns=PROC_COLS).to_csv(OUT_MANIFEST, index=False)

    pd.DataFrame(rows, columns=PROC_COLS).to_csv(OUT_MANIFEST, index=False)
    print("\nRESUMEN:")
    df = pd.DataFrame(rows)
    print(df["status"].value_counts().to_string())
    ok = df[df["status"] == "ok"]
    if len(ok):
        print(f"\nTICs ok: {len(ok):,}")
        print(f"Distribucion label: {ok['label'].value_counts().to_dict()}")
        print(f"Valid fraction promedio: {ok['valid_fraction'].mean():.3f}")
    print(f"\nManifest: {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
