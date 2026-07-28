"""Construye la tabla de etiquetas por TIC a partir del catálogo TOI.

POR QUÉ EXISTE ESTE SCRIPT
--------------------------
Hasta ahora la definición de la tarea vivía en una celda de
`notebooks/01_toi_eda.ipynb`:

    labeled = df[df["tfopwg_disp"].isin(["CP", "FP"])]

Es decir, la decisión más consecuente de todo el proyecto -- qué cuenta como
positivo y qué como negativo -- no era reproducible desde la línea de comandos,
no estaba cubierta por tests y no quedaba registrada en ningún artefacto. Este
script la convierte en un paso explícito, parametrizado y auditable.

ESQUEMAS DE ETIQUETADO
----------------------
`cp_fp` es la tarea del paper actual: planeta confirmado contra falso positivo.
Es la formulación más limpia, pero también la más fácil: los planetas confirmados
están mejor estudiados y sus curvas son más nítidas que las de un candidato
cualquiera.

`pc_fp` y `triage` son las formulaciones operacionalmente relevantes, donde el
clasificador ve lo que ve un astrónomo en el momento del vetting: candidatos sin
confirmar. Son más difíciles y mucho más comparables con la literatura de triage
(Yu et al. 2019, DART-Vetter).

CONFLICTOS DE ETIQUETA POR ESTRELLA
-----------------------------------
Un TIC puede alojar varios TOIs con disposiciones distintas: son sistemas
multiplanetarios donde un candidato se confirmó y otro resultó falso positivo.
Como el modelo clasifica la curva completa de la estrella (no un TOI individual),
hay que decidir qué etiqueta lleva esa estrella. La regla por defecto es
`positive-wins`, y no es arbitraria: si la estrella tiene un planeta confirmado,
su curva de luz contiene un tránsito planetario real, así que etiquetarla como
negativa sería sencillamente incorrecto.

Uso:

  python scripts/make_labels.py --task cp_fp --dry-run
  python scripts/make_labels.py --task pc_fp
  python scripts/make_labels.py --task cp_fp --verify-against data/splits/tics_labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "data" / "splits" / "toi_summary.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Disposiciones del TESS Follow-up Observing Program Working Group:
#   CP  planeta confirmado          KP  planeta ya conocido, recuperado por TESS
#   PC  candidato a planeta         APC candidato ambiguo
#   FP  falso positivo              FA  falsa alarma (artefacto instrumental)
SCHEMES: dict[str, dict[str, object]] = {
    "cp_fp": {
        "positive": ("CP",),
        "negative": ("FP",),
        "summary": "Planeta confirmado contra falso positivo (la tarea del paper actual).",
    },
    "pc_fp": {
        "positive": ("PC",),
        "negative": ("FP",),
        "summary": "Candidato sin confirmar contra falso positivo (triage operacional estricto).",
    },
    "triage": {
        "positive": ("CP", "KP", "PC"),
        "negative": ("FP", "FA"),
        "summary": (
            "Todo lo que sigue vivo como planeta contra todo lo descartado. "
            "APC queda fuera por ser ambiguo por definición."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera data/splits/tics_labeled_<task>.csv desde el catálogo TOI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:8s} pos={v['positive']} neg={v['negative']}\n           {v['summary']}"
            for k, v in SCHEMES.items()
        ),
    )
    p.add_argument("--task", choices=sorted(SCHEMES), default="cp_fp", help="Esquema de etiquetado.")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="CSV del catálogo TOI.")
    p.add_argument("--out", type=Path, default=None, help="Destino. Default: tics_labeled_<task>.csv")
    p.add_argument(
        "--on-conflict",
        choices=["positive-wins", "drop"],
        default="positive-wins",
        help=(
            "Qué hacer con una estrella que aloja TOIs positivos y negativos a la vez. "
            "positive-wins: la curva contiene un tránsito real, así que gana el positivo. "
            "drop: descartar la estrella por ambigua."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Sólo reporta estadísticas, no escribe.")
    p.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Compara el resultado contra un CSV de etiquetas existente y reporta diferencias.",
    )
    return p.parse_args()


def build_labels(
    catalog: pd.DataFrame, task: str, on_conflict: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Colapsa el catálogo (una fila por TOI) a una tabla de una fila por TIC."""
    scheme = SCHEMES[task]
    positive = list(scheme["positive"])  # type: ignore[arg-type]
    negative = list(scheme["negative"])  # type: ignore[arg-type]

    usable = catalog[catalog["tfopwg_disp"].isin(positive + negative)].copy()
    usable["label"] = usable["tfopwg_disp"].isin(positive).astype(int)

    # Una estrella es positiva si CUALQUIERA de sus TOIs lo es.
    por_tic = usable.groupby("tid")["label"]
    etiqueta_max = por_tic.max()
    etiqueta_min = por_tic.min()
    conflictivos = etiqueta_max[etiqueta_max != etiqueta_min].index

    if on_conflict == "drop":
        usable = usable[~usable["tid"].isin(conflictivos)]
        resueltos = usable.groupby("tid")["label"].max()
    else:
        resueltos = etiqueta_max

    # Nos quedamos con la fila representativa: la del TOI que define la etiqueta,
    # para que tfopwg_disp del CSV sea coherente con label y no una disposición suelta.
    usable = usable.merge(resueltos.rename("label_tic"), on="tid")
    representativa = (
        usable[usable["label"] == usable["label_tic"]]
        .sort_values(["tid", "pl_orbper"])
        .drop_duplicates(subset="tid", keep="first")
    )

    columnas = [c for c in ("tid", "tfopwg_disp", "st_tmag", "pl_orbper") if c in representativa]
    salida = representativa[columnas].copy()
    salida["label"] = representativa["label_tic"].to_numpy()
    salida = salida.sort_values("tid").reset_index(drop=True)

    stats = {
        "toi_en_catalogo": len(catalog),
        "toi_usables": int(len(usable)),
        "tics_resultantes": int(len(salida)),
        "positivos": int((salida["label"] == 1).sum()),
        "negativos": int((salida["label"] == 0).sum()),
        "tics_en_conflicto": int(len(conflictivos)),
    }
    return salida, stats


def main() -> int:
    args = parse_args()

    if not args.catalog.exists():
        print(f"ERROR: no existe el catálogo {args.catalog}", file=sys.stderr)
        return 2

    catalog = pd.read_csv(args.catalog)
    scheme = SCHEMES[args.task]
    labels, stats = build_labels(catalog, args.task, args.on_conflict)

    print(f"Tarea            : {args.task} -- {scheme['summary']}")
    print(f"  positivos      : {scheme['positive']}")
    print(f"  negativos      : {scheme['negative']}")
    print(f"  conflictos     : {args.on_conflict}")
    print(f"TOIs en catálogo : {stats['toi_en_catalogo']}")
    print(f"TICs resultantes : {stats['tics_resultantes']}")
    print(f"  positivos      : {stats['positivos']}")
    print(f"  negativos      : {stats['negativos']}")
    ratio = stats["negativos"] / max(stats["positivos"], 1)
    print(f"  balance        : 1:{ratio:.2f} (pos:neg)")
    print(f"TICs en conflicto: {stats['tics_en_conflicto']} (alojan positivos y negativos a la vez)")

    if args.verify_against is not None:
        if not args.verify_against.exists():
            print(f"ERROR: no existe {args.verify_against}", file=sys.stderr)
            return 2
        previo = pd.read_csv(args.verify_against).drop_duplicates(subset="tid")
        comparado = labels.merge(previo[["tid", "label"]], on="tid", how="outer",
                                 suffixes=("_nuevo", "_previo"), indicator=True)
        solo_nuevo = int((comparado["_merge"] == "left_only").sum())
        solo_previo = int((comparado["_merge"] == "right_only").sum())
        ambos = comparado[comparado["_merge"] == "both"]
        difieren = ambos[ambos["label_nuevo"] != ambos["label_previo"]]
        print(f"\nComparación contra {args.verify_against.name}:")
        print(f"  sólo en el nuevo : {solo_nuevo}")
        print(f"  sólo en el previo: {solo_previo}")
        print(f"  etiqueta distinta: {len(difieren)}")
        if len(difieren):
            print(difieren[["tid", "label_previo", "label_nuevo"]].to_string(index=False))

    if args.dry_run:
        print("\n(dry-run: no se escribió nada)")
        return 0

    destino = args.out or SPLITS_DIR / f"tics_labeled_{args.task}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(destino, index=False)
    print(f"\nEscrito: {destino} ({len(labels)} TICs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
