"""Garantías metodológicas de los splits y del test sellado.

Estos tests son la contraparte ejecutable de lo que el paper afirma en la sección
de reproducibilidad. Hasta ahora esas garantías vivían sólo en `scripts/make_splits.py`
(código de generación que corre una vez) y en prosa, así que nada impedía que una
edición de los CSV versionados rompiera el split sin que fallara ningún test.

Corren sobre `data/splits/*.csv`, que SÍ están versionados, así que funcionan en un
clon limpio sin necesidad de descargar ni preprocesar nada.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"

TIER1 = ("train", "val", "test")
TIER2 = ("tier2_train", "tier2_val", "tier2_test")

# Conteos reportados en el paper (Table I). Si un cambio de pipeline los mueve,
# estos tests fallan y obligan a actualizar el paper en el mismo commit.
EXPECTED_TIER1 = {"train": 1103, "val": 236, "test": 237}
EXPECTED_TEST_COMPOSITION = {"pos": 91, "neg": 146}


def _load(split: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{split}_tics.csv"
    if not path.exists():
        pytest.skip(f"No existe {path}; corré scripts/make_splits.py primero.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def tier1() -> dict[str, pd.DataFrame]:
    return {s: _load(s) for s in TIER1}


@pytest.fixture(scope="module")
def tier2() -> dict[str, pd.DataFrame]:
    return {s: _load(s) for s in TIER2}


# --- Anti-leakage --------------------------------------------------------------


def test_tier1_splits_no_comparten_tics(tier1: dict[str, pd.DataFrame]) -> None:
    """Ningún TIC puede aparecer en dos splits: sería leakage estrella a estrella."""
    for a, b in itertools.combinations(TIER1, 2):
        overlap = set(tier1[a]["tid"]) & set(tier1[b]["tid"])
        assert not overlap, (
            f"{len(overlap)} TICs compartidos entre {a} y {b}: {sorted(overlap)[:5]}"
        )


def test_tier2_splits_no_comparten_tics(tier2: dict[str, pd.DataFrame]) -> None:
    for a, b in itertools.combinations(TIER2, 2):
        overlap = set(tier2[a]["tid"]) & set(tier2[b]["tid"])
        assert not overlap, f"{len(overlap)} TICs compartidos entre {a} y {b}"


def test_tier2_es_subconjunto_de_tier1(
    tier1: dict[str, pd.DataFrame], tier2: dict[str, pd.DataFrame]
) -> None:
    """Tier 2 filtra Tier 1, no lo rebaraja.

    Si un TIC que estaba en train Tier 1 apareciera en test Tier 2, el modelo Tier 2
    se evaluaría sobre una estrella que el pipeline ya usó para entrenar.
    """
    for split in TIER1:
        parent = set(tier1[split]["tid"])
        child = set(tier2[f"tier2_{split}"]["tid"])
        assert child <= parent, f"tier2_{split} tiene {len(child - parent)} TICs fuera de {split}"


def test_ningun_split_tiene_tics_duplicados(
    tier1: dict[str, pd.DataFrame], tier2: dict[str, pd.DataFrame]
) -> None:
    """Un TIC repetido dentro de un split infla su peso efectivo en la métrica."""
    for name, df in {**tier1, **tier2}.items():
        dups = df["tid"].duplicated().sum()
        assert dups == 0, f"{name} tiene {dups} TIC duplicados"


# --- Integridad de etiquetas y estratificación ---------------------------------


def test_etiquetas_son_binarias(
    tier1: dict[str, pd.DataFrame], tier2: dict[str, pd.DataFrame]
) -> None:
    """CP=1, FP=0. Cualquier otra disposición (PC, KP, APC, FA) debe estar excluida."""
    for name, df in {**tier1, **tier2}.items():
        valores = set(df["label"].unique())
        assert valores <= {0, 1}, f"{name} tiene etiquetas fuera de (0, 1): {valores - {0, 1}}"


def test_estratificacion_preservada(tier1: dict[str, pd.DataFrame]) -> None:
    """La proporción de positivos debe ser la misma en los tres splits.

    `make_splits.py` estratifica por label; si alguien regenera los splits sin
    estratificar, la métrica del test deja de ser comparable con la de val.
    """
    fracs = {s: float((tier1[s]["label"] == 1).mean()) for s in TIER1}
    assert max(fracs.values()) - min(fracs.values()) < 0.02, f"estratificación rota: {fracs}"


def test_conteos_coinciden_con_el_paper(tier1: dict[str, pd.DataFrame]) -> None:
    """Los N del paper tienen que salir de los CSV versionados, no de una nota a mano."""
    actual = {s: len(tier1[s]) for s in TIER1}
    assert actual == EXPECTED_TIER1, f"esperado {EXPECTED_TIER1}, encontrado {actual}"
    assert sum(actual.values()) == 1576


def test_composicion_del_test_coincide_con_el_paper(tier1: dict[str, pd.DataFrame]) -> None:
    test = tier1["test"]
    composicion = {
        "pos": int((test["label"] == 1).sum()),
        "neg": int((test["label"] == 0).sum()),
    }
    assert composicion == EXPECTED_TEST_COMPOSITION


# --- Guard del test sellado ----------------------------------------------------


@pytest.fixture
def evaluate_mod(tmp_path: Path):
    """Carga scripts/evaluate.py con la bitácora redirigida a un temporal.

    `scripts/` no es un paquete importable, así que lo cargamos por ruta. La
    bitácora se redirige para que los tests nunca escriban el archivo real del repo.
    """
    path = REPO_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TEST_SEAL_LEDGER = tmp_path / "test_seal_ledger.json"
    return module


def test_guard_permite_la_primera_evaluacion(evaluate_mod) -> None:
    assert evaluate_mod._check_test_seal("experiments/run_a", force=False) is True


def test_guard_bloquea_la_segunda_evaluacion(evaluate_mod) -> None:
    """El corazón del sellado: una corrida ya evaluada no se vuelve a evaluar."""
    evaluate_mod._check_test_seal("experiments/run_a", force=False)
    evaluate_mod._record_test_seal("experiments/run_a", auc_roc=0.81, n_samples=237, forced=False)
    assert evaluate_mod._check_test_seal("experiments/run_a", force=False) is False


def test_guard_no_afecta_a_otras_corridas(evaluate_mod) -> None:
    evaluate_mod._record_test_seal("experiments/run_a", auc_roc=0.81, n_samples=237, forced=False)
    assert evaluate_mod._check_test_seal("experiments/run_b", force=False) is True


def test_reevaluacion_forzada_queda_registrada(evaluate_mod) -> None:
    """--force-reeval-test desbloquea, pero deja rastro auditable."""
    evaluate_mod._record_test_seal("experiments/run_a", auc_roc=0.81, n_samples=237, forced=False)
    assert evaluate_mod._check_test_seal("experiments/run_a", force=True) is True
    evaluate_mod._record_test_seal("experiments/run_a", auc_roc=0.81, n_samples=237, forced=True)

    historial = evaluate_mod._load_seal_ledger()["evaluations"]["experiments/run_a"]
    assert len(historial) == 2
    assert historial[0]["forced_reevaluation"] is False
    assert historial[1]["forced_reevaluation"] is True
    assert all(e["git_commit"] for e in historial)


def test_default_del_cli_no_es_test(evaluate_mod) -> None:
    """El split por defecto debe ser `val`: tocar el test tiene que ser deliberado."""
    import argparse

    parser_defaults = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        for action in self._actions:
            parser_defaults[action.dest] = action.default
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            evaluate_mod.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original

    assert parser_defaults["split"] == "val"
    assert parser_defaults["force_reeval_test"] is False
