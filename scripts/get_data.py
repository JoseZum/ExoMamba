"""
Descarga el TOI Catalog completo desde el NASA Exoplanet Archive (TAP).

Salida:
  data/raw/toi_catalog.csv | tabla completa, gitignored
  data/splits/toi_summary.csv | resumen ligero (tid, tfopwg_disp, sectors), versionado

Uso:
  python scripts/get_data.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+toi"
    "&format=csv"
)

RAW_PATH = Path("data/raw/toi_catalog.csv")
SPLITS_PATH = Path("data/splits/toi_summary.csv")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {url} …")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="\r")
    print(f"\nGuardado en {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def summarize(df: pd.DataFrame) -> None:
    print(f"\nTotal de filas: {len(df):,}")
    if "tfopwg_disp" in df.columns:
        print("\nDistribución por disposición (tfopwg_disp):")
        counts = df["tfopwg_disp"].value_counts(dropna=False)
        for disp, n in counts.items():
            print(f"  {str(disp):6s}  {n:>5,}")
    else:
        print("Columna 'tfopwg_disp' no encontrada, columnas disponibles:")
        print(df.columns.tolist())


def save_summary(df: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "tid",
            "toi",
            "tfopwg_disp",
            "st_tmag",
            "pl_orbper",
            "pl_trandep",
            "pl_tranmid",
            "pl_trandurh",
        ]
        if c in df.columns
    ]
    summary = df[cols].copy()
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SPLITS_PATH, index=False)
    print(f"\nResumen versionado guardado en {SPLITS_PATH}  ({len(summary):,} filas, {len(cols)} cols)")


def main() -> None:
    download(TAP_URL, RAW_PATH)
    df = pd.read_csv(RAW_PATH, low_memory=False)
    summarize(df)
    save_summary(df)


if __name__ == "__main__":
    sys.exit(main())
