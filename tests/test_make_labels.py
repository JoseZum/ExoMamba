"""La definición de la tarea (qué es positivo y qué negativo) tiene que estar testeada.

`scripts/make_labels.py` reemplaza la celda de notebook que decidía esto. Estos
tests fijan dos cosas: que el esquema `cp_fp` reproduce el etiquetado histórico
del paper, y que la regla de desempate para estrellas multiplanetarias hace lo
que dice hacer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "data" / "splits" / "toi_summary.csv"
HISTORICO = REPO_ROOT / "data" / "splits" / "tics_labeled.csv"

# Sistemas multiplanetarios donde un TOI se confirmó y otro resultó falso positivo.
# El etiquetado histórico se quedó con el FP; `positive-wins` los corrige, porque
# la curva de esas estrellas sí contiene un tránsito planetario real.
TICS_EN_CONFLICTO = {207468071, 441738827}


@pytest.fixture(scope="module")
def make_labels():
    path = REPO_ROOT / "scripts" / "make_labels.py"
    spec = importlib.util.spec_from_file_location("make_labels_under_test", path)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def catalogo() -> pd.DataFrame:
    if not CATALOG.exists():
        pytest.skip(f"No existe {CATALOG}")
    return pd.read_csv(CATALOG)


def test_cp_fp_reproduce_el_etiquetado_del_paper(make_labels, catalogo) -> None:
    """Salvo los conflictos conocidos, debe coincidir TIC a TIC con tics_labeled.csv."""
    if not HISTORICO.exists():
        pytest.skip(f"No existe {HISTORICO}")

    nuevo, _ = make_labels.build_labels(catalogo, "cp_fp", "positive-wins")
    previo = pd.read_csv(HISTORICO).drop_duplicates(subset="tid")

    assert set(nuevo["tid"]) == set(previo["tid"]), "el conjunto de TICs cambió"

    comparado = nuevo.merge(previo[["tid", "label"]], on="tid", suffixes=("_nuevo", "_previo"))
    difieren = comparado[comparado["label_nuevo"] != comparado["label_previo"]]
    assert set(difieren["tid"]) == TICS_EN_CONFLICTO, (
        f"diferencias inesperadas contra el etiquetado histórico: {sorted(difieren['tid'])}"
    )


def test_positive_wins_corrige_las_estrellas_en_conflicto(make_labels, catalogo) -> None:
    etiquetas, _ = make_labels.build_labels(catalogo, "cp_fp", "positive-wins")
    en_conflicto = etiquetas[etiquetas["tid"].isin(TICS_EN_CONFLICTO)]
    assert len(en_conflicto) == len(TICS_EN_CONFLICTO)
    assert (en_conflicto["label"] == 1).all(), "una estrella con planeta confirmado quedó como FP"


def test_drop_descarta_las_estrellas_en_conflicto(make_labels, catalogo) -> None:
    etiquetas, stats = make_labels.build_labels(catalogo, "cp_fp", "drop")
    assert not set(etiquetas["tid"]) & TICS_EN_CONFLICTO
    assert stats["tics_en_conflicto"] == len(TICS_EN_CONFLICTO)


def test_una_fila_por_tic_y_etiquetas_binarias(make_labels, catalogo) -> None:
    """El split es por estrella, así que el etiquetado debe colapsar los TOIs por TIC."""
    for tarea in ("cp_fp", "pc_fp", "triage"):
        etiquetas, _ = make_labels.build_labels(catalogo, tarea, "positive-wins")
        assert etiquetas["tid"].duplicated().sum() == 0, f"{tarea} devolvió TICs repetidos"
        assert set(etiquetas["label"].unique()) <= {0, 1}, f"{tarea} tiene etiquetas no binarias"
        assert len(etiquetas) > 0


def test_pc_fp_amplia_el_dataset(make_labels, catalogo) -> None:
    """El sentido de pc_fp es tener bastante más señal que cp_fp."""
    cp_fp, _ = make_labels.build_labels(catalogo, "cp_fp", "positive-wins")
    pc_fp, _ = make_labels.build_labels(catalogo, "pc_fp", "positive-wins")
    assert len(pc_fp) > 2 * len(cp_fp)


def test_los_esquemas_no_solapan_positivos_y_negativos(make_labels) -> None:
    """Una disposición no puede contar como positiva y negativa en el mismo esquema."""
    for tarea, esquema in make_labels.SCHEMES.items():
        solapan = set(esquema["positive"]) & set(esquema["negative"])
        assert not solapan, f"{tarea} usa {solapan} en ambas clases"
