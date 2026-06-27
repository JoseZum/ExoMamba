"""
Preprocesamiento de la vista local phase-folded (Fase 3.b / Tier 2).

Por cada TIC con `data/processed/global/<tid>.pt` válido y metadata
(period, epoch, duration) presente en `data/splits/toi_summary.csv`:

  1. Lee el FITS del sector elegido en preprocess_global (sector_chosen del
     manifest) para recuperar TIME y PDCSAP_FLUX alineados.
  2. Calcula la fase centrada en el tránsito:
        phase = ((time - T0_tess) / period + 0.5) mod 1.0 - 0.5
  3. Recorta a la ventana ±2.5 * duration_dias / period en fase.
  4. Bin a 201 puntos uniformes vía np.histogram (sum y count) → media por bin.
  5. Interpolación lineal sobre bins vacíos.
  6. Normalización por la mediana del propio vector binneado.
  7. Tensor float32 (1, 201). Guarda en data/processed/local/<tid>.pt.

Decisiones documentadas en data/processed/local/DECISIONES.md.

Salida:
  data/processed/local/<tid>.pt           tensor + metadata por TIC (gitignored)
  data/splits/processed_local_manifest.csv  versionado, una fila por TIC

Uso:
  python scripts/preprocess_local.py --limit 10           # piloto
  python scripts/preprocess_local.py                      # dataset completo
  python scripts/preprocess_local.py --tics-csv data/splits/tics_labeled.csv \
                                     --output-dir data/processed/local
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

warnings.filterwarnings("ignore", category=UserWarning)

# --- Paths por defecto -------------------------------------------------------

GLOBAL_DIR = Path("data/processed/global")
RAW_DIR = Path("data/raw/lightcurves")
TOI_SUMMARY_PATH = Path("data/splits/toi_summary.csv")
PROCESSED_MANIFEST_PATH = Path("data/splits/processed_manifest.csv")
DEFAULT_TICS_CSV = Path("data/splits/tics_labeled.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/local")
DEFAULT_MANIFEST_PATH = Path("data/splits/processed_local_manifest.csv")

# --- Hiperparámetros (ver DECISIONES.md) -------------------------------------

N_BINS = 201
WINDOW_DURATIONS = 2.5
PERIOD_MAX_DAYS = 27.4
MAX_EMPTY_FRACTION = 0.5  # si > 50% bins vacíos, descartar
BJDREFI = 2457000.0       # TESS time offset

MANIFEST_COLS = [
    "tid", "label", "status", "reason", "n_transits_seen",
    "period", "epoch_tess", "duration_h", "duration_s", "processed_at",
]


# Cache global: tid (int) -> { sector (int): Path }. Se llena perezosamente
# en build_fits_index() para evitar globs O(N) por TIC.
_FITS_INDEX: dict[int, dict[int, Path]] | None = None


def build_fits_index() -> dict[int, dict[int, Path]]:
    """Construye un índice tid → {sector → Path} recorriendo RAW_DIR una sola vez."""
    global _FITS_INDEX
    if _FITS_INDEX is not None:
        return _FITS_INDEX
    index: dict[int, dict[int, Path]] = {}
    # Nombre típico: tessYYYYDDDHHMMSS-sSSSS-TIIIIIIIIIIIIIIII-XXXX-s_lc.fits
    import re
    pat = re.compile(r"-s(\d{4})-(\d{16})-")
    for p in RAW_DIR.rglob("*_lc.fits"):
        m = pat.search(p.name)
        if not m:
            continue
        sector = int(m.group(1))
        tid = int(m.group(2))
        index.setdefault(tid, {})[sector] = p
    _FITS_INDEX = index
    return index


def find_fits_for_tic_sector(tid: int, sector: int) -> Path | None:
    """Devuelve el FITS de `tid` correspondiente a `sector`. None si no existe."""
    index = build_fits_index()
    by_sector = index.get(tid)
    if not by_sector:
        return None
    if sector in by_sector:
        return by_sector[sector]
    # Fallback: cualquier sector del TIC.
    return next(iter(by_sector.values()))


def load_time_flux(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (time, flux, quality). El time está en BJD - BJDREFI (TESS)."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        cols = set(data.columns.names)
        for needed in ("TIME", "PDCSAP_FLUX", "QUALITY"):
            if needed not in cols:
                raise ValueError(f"FITS {path.name} sin columna {needed}")
        time_arr = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
        quality = np.asarray(data["QUALITY"], dtype=np.int32)
    return time_arr, flux, quality


def phase_fold(
    time_arr: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch_tess: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula fase centrada en el tránsito (∈ [-0.5, 0.5)) y devuelve (fase, flujo).

    Se descartan puntos con flux no finito antes del folding.
    """
    valid = np.isfinite(time_arr) & np.isfinite(flux)
    t = time_arr[valid]
    f = flux[valid]
    phase = ((t - epoch_tess) / period + 0.5) % 1.0 - 0.5
    return phase, f


def bin_local_view(
    phase: np.ndarray,
    flux: np.ndarray,
    window_frac: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Binnea el flujo en `n_bins` bins uniformes sobre [-window_frac, +window_frac].

    Devuelve (mean_por_bin, count_por_bin). Bins sin puntos quedan en NaN.
    """
    in_window = np.abs(phase) <= window_frac
    p = phase[in_window]
    f = flux[in_window]
    edges = np.linspace(-window_frac, +window_frac, n_bins + 1)
    sum_f, _ = np.histogram(p, bins=edges, weights=f)
    cnt, _ = np.histogram(p, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_f = np.where(cnt > 0, sum_f / np.maximum(cnt, 1), np.nan)
    return mean_f, cnt


def interpolate_empty_bins(view: np.ndarray) -> np.ndarray:
    """Interpola linealmente bins NaN; rellena extremos con primer/último válido."""
    out = view.copy()
    n = len(out)
    valid = np.isfinite(out)
    if not valid.any():
        return out  # totalmente vacío; el caller debe filtrar antes
    idx = np.arange(n)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def count_transits(
    time_arr: np.ndarray,
    period: float,
    epoch_tess: float,
    duration_days: float,
) -> int:
    """Cuenta cuántos tránsitos (al menos parciales) caen dentro del rango de TIME."""
    t = time_arr[np.isfinite(time_arr)]
    if len(t) == 0 or period <= 0:
        return 0
    t_min, t_max = float(t.min()), float(t.max())
    # Índice del tránsito que cae justo a la izquierda de t_min.
    n_min = int(np.floor((t_min - epoch_tess) / period))
    n_max = int(np.ceil((t_max - epoch_tess) / period))
    transit_times = epoch_tess + np.arange(n_min, n_max + 1) * period
    # Tránsito visible si el centro ± duración/2 toca el rango observado.
    half = duration_days / 2.0
    visible = (transit_times + half >= t_min) & (transit_times - half <= t_max)
    return int(visible.sum())


def build_catalog_lookup() -> dict[int, dict[str, float]]:
    """TIC → {period, epoch, duration_h} usando la TOI con duración más estrecha."""
    df = pd.read_csv(TOI_SUMMARY_PATH)
    valid_mask = (
        df["pl_orbper"].notna()
        & df["pl_tranmid"].notna()
        & df["pl_trandurh"].notna()
        & (df["pl_orbper"] > 0)
        & (df["pl_trandurh"] > 0)
    )
    valid = df.loc[valid_mask].copy()
    valid = valid.sort_values(["tid", "pl_trandurh"], ascending=[True, True])
    chosen = valid.drop_duplicates(subset=["tid"], keep="first")
    return {
        int(row.tid): {
            "period": float(row.pl_orbper),
            "epoch": float(row.pl_tranmid),
            "duration_h": float(row.pl_trandurh),
        }
        for row in chosen.itertuples(index=False)
    }


def build_sector_lookup() -> dict[int, int]:
    """TIC → sector_chosen del preprocesamiento global."""
    if not PROCESSED_MANIFEST_PATH.exists():
        return {}
    pm = pd.read_csv(PROCESSED_MANIFEST_PATH)
    pm = pm[pm["status"].astype(str).str.lower() == "ok"]
    return {int(r.tid): int(r.sector_chosen) for r in pm.itertuples(index=False)}


def process_tic(
    tid: int,
    label: int,
    catalog: dict[int, dict[str, float]],
    sector_lookup: dict[int, int],
    output_dir: Path,
) -> dict:
    t0 = time.time()
    row = {c: "" for c in MANIFEST_COLS}
    row.update({
        "tid": tid,
        "label": label,
        "status": "pending",
        "reason": "",
        "n_transits_seen": 0,
        "period": np.nan,
        "epoch_tess": np.nan,
        "duration_h": np.nan,
        "duration_s": 0.0,
    })

    try:
        if not (GLOBAL_DIR / f"{tid}.pt").exists():
            row["status"] = "no_global_pt"
            row["reason"] = f"no existe {GLOBAL_DIR}/{tid}.pt"
            return row

        meta = catalog.get(tid)
        if meta is None:
            row["status"] = "no_metadata"
            row["reason"] = "TIC no tiene TOI con (period, epoch, duration) válidos"
            return row

        period = meta["period"]
        epoch_tess = meta["epoch"] - BJDREFI
        duration_h = meta["duration_h"]
        duration_d = duration_h / 24.0

        row["period"] = period
        row["epoch_tess"] = epoch_tess
        row["duration_h"] = duration_h

        if period > PERIOD_MAX_DAYS:
            row["status"] = "period_too_long"
            row["reason"] = f"period={period:.3f}d > {PERIOD_MAX_DAYS}d"
            return row
        if duration_d >= period / 2.0:
            row["status"] = "duration_degenerate"
            row["reason"] = f"duration_d={duration_d:.3f} >= period/2={period/2:.3f}"
            return row

        # Sector elegido en preprocess_global; si no hay, intentamos cualquiera.
        sector = sector_lookup.get(tid, -1)
        fits_path = find_fits_for_tic_sector(tid, sector) if sector > 0 else None
        if fits_path is None:
            fits_path = find_fits_for_tic_sector(tid, -1)
        if fits_path is None:
            row["status"] = "no_fits_time"
            row["reason"] = "no se encontró FITS para recuperar TIME"
            return row

        time_arr, flux, quality = load_time_flux(fits_path)
        # Enmascarar puntos con QUALITY != 0 antes del folding.
        bad = quality != 0
        flux = flux.astype(np.float64)
        flux[bad] = np.nan

        n_tr = count_transits(time_arr, period, epoch_tess, duration_d)
        row["n_transits_seen"] = n_tr

        phase, f = phase_fold(time_arr, flux, period, epoch_tess)
        window_frac = WINDOW_DURATIONS * duration_d / period
        # Clamp por seguridad: la ventana en fase no debe exceder [-0.5, 0.5].
        window_frac = float(min(window_frac, 0.5))

        view, cnt = bin_local_view(phase, f, window_frac, N_BINS)
        empty_frac = float((cnt == 0).mean())
        if empty_frac > MAX_EMPTY_FRACTION:
            row["status"] = "too_many_empty_bins"
            row["reason"] = f"empty_frac={empty_frac:.2f} > {MAX_EMPTY_FRACTION}"
            return row

        view = interpolate_empty_bins(view)
        if not np.isfinite(view).all():
            row["status"] = "processing_error"
            row["reason"] = "NaN residual tras interpolación"
            return row

        median = float(np.median(view))
        if not np.isfinite(median) or median == 0:
            row["status"] = "processing_error"
            row["reason"] = f"mediana no finita o cero ({median})"
            return row
        view = view / median

        local_view = torch.from_numpy(view.astype(np.float32)).unsqueeze(0)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tid": tid,
            "label": label,
            "local_view": local_view,
            "period": period,
            "epoch_tess": epoch_tess,
            "duration_h": duration_h,
            "n_transits_seen": n_tr,
        }, output_dir / f"{tid}.pt")
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "processing_error"
        row["reason"] = str(e)[:300]
    finally:
        row["duration_s"] = round(time.time() - t0, 2)
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tics-csv", type=Path, default=DEFAULT_TICS_CSV,
                        help="CSV con columnas (tid, label) - qué TICs procesar.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directorio destino para los .pt locales.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                        help="Ruta del manifest CSV de salida.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N TICs (piloto).")
    parser.add_argument("--resume", action="store_true",
                        help="Saltar TICs ya presentes (con cualquier status) en el manifest existente.")
    args = parser.parse_args()

    if not args.tics_csv.exists():
        sys.exit(f"No existe {args.tics_csv}.")
    if not TOI_SUMMARY_PATH.exists():
        sys.exit(f"No existe {TOI_SUMMARY_PATH}. Corre scripts/get_data.py.")

    todo = pd.read_csv(args.tics_csv)
    if not {"tid", "label"}.issubset(todo.columns):
        sys.exit("El CSV de TICs debe tener columnas (tid, label).")

    catalog = build_catalog_lookup()
    sector_lookup = build_sector_lookup()
    print(f"TICs en catálogo con metadata válida: {len(catalog):,}")
    print(f"TICs con sector_chosen en processed_manifest: {len(sector_lookup):,}")

    existing_rows: list[dict] = []
    if args.resume and args.manifest.exists():
        prev = pd.read_csv(args.manifest)
        done_tids = set(prev["tid"].astype(int))
        existing_rows = prev.to_dict("records")
        before = len(todo)
        todo = todo[~todo["tid"].astype(int).isin(done_tids)].reset_index(drop=True)
        print(f"--resume: omitiendo {before - len(todo):,} TICs ya presentes en {args.manifest}.")

    if args.limit:
        todo = todo.head(args.limit)
    print(f"Por procesar: {len(todo):,}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = list(existing_rows)
    for i, r in enumerate(todo.itertuples(index=False), start=1):
        tid = int(r.tid)
        label = int(r.label)
        print(f"[{i}/{len(todo)}] TIC {tid} ... ", end="", flush=True)
        result = process_tic(tid, label, catalog, sector_lookup, args.output_dir)
        print(
            f"{result['status']:<22} | n_tr={result['n_transits_seen']:>2} | "
            f"reason={result['reason'][:40]}"
        )
        rows.append(result)
        if i % 50 == 0 or i == len(todo):
            pd.DataFrame(rows, columns=MANIFEST_COLS).to_csv(args.manifest, index=False)

    pd.DataFrame(rows, columns=MANIFEST_COLS).to_csv(args.manifest, index=False)
    print("\nRESUMEN:")
    df = pd.DataFrame(rows)
    print(df["status"].value_counts().to_string())
    ok = df[df["status"] == "ok"]
    if len(ok):
        print(f"\nTICs ok: {len(ok):,}")
        print(f"Distribución label: {ok['label'].value_counts().to_dict()}")
        print(f"n_transits_seen - mean={ok['n_transits_seen'].mean():.1f}, "
              f"min={ok['n_transits_seen'].min()}, max={ok['n_transits_seen'].max()}")
    print(f"\nManifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
