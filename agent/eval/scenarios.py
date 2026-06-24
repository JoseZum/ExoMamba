"""Escenarios de validación S1–S6 (ETAPA3.md).

Los TICs de S1/S2/S3/S5 se seleccionan del catálogo TOI real de forma
**determinista** (orden por TIC ID + clasificador mock determinista), así que la
suite es reproducible. S4 (entradas inválidas) y S6 (off-topic) son fijos.

| ID | Qué evalúa |
|----|------------|
| S1 | 10 CP confirmados → clasificación + reporte consistente con NASA |
| S2 | 10 FP conocidos → ídem clase negativa |
| S3 | 5 casos donde el modelo discrepa con NASA → el agente DEBE flaggear |
| S4 | TIC inexistente/malformado → no alucina |
| S5 | período > 27 d (límite físico) → el agente declara la limitación |
| S6 | off-topic → responde dentro de scope |
"""

from __future__ import annotations

from agent import mock


def _unique_tids():
    cat = mock.load_catalog()
    return sorted(cat.index.unique())


def build_scenarios(max_scan: int | None = None) -> dict:
    """Construye las listas de casos por escenario a partir del catálogo real."""
    cps: list[int] = []
    fps: list[int] = []
    discrep: list[dict] = []
    longp: list[int] = []

    for i, tid in enumerate(_unique_tids()):
        if max_scan and i >= max_scan:
            break
        info = mock.get_toi_info(int(tid))
        if info is None:
            continue
        disp = info["disposition"]
        cls = mock.classify(int(tid), info)
        model_planet = cls["prob_planeta"] >= 0.5
        nasa_planet = info["is_planet_truth"]

        if disp == "CP" and len(cps) < 10:
            cps.append(int(tid))
        elif disp == "FP" and len(fps) < 10:
            fps.append(int(tid))

        if (disp in ("CP", "KP", "FP") and model_planet != nasa_planet
                and len(discrep) < 5):
            discrep.append({"tic_id": int(tid), "disposition": disp,
                            "model_label": cls["label"]})

        if (info.get("period_days") and info["period_days"] > 27
                and len(longp) < 5):
            longp.append(int(tid))

        if (len(cps) >= 10 and len(fps) >= 10 and len(discrep) >= 5
                and len(longp) >= 5):
            break

    return {
        "S1": {"desc": "10 CP confirmados", "kind": "analysis",
               "cases": [{"query": f"Analiza TIC {t}", "tic_id": t,
                          "expect_planet": True} for t in cps]},
        "S2": {"desc": "10 FP conocidos", "kind": "analysis",
               "cases": [{"query": f"Analiza TIC {t}", "tic_id": t,
                          "expect_planet": False} for t in fps]},
        "S3": {"desc": "5 discrepancias modelo vs NASA", "kind": "analysis",
               "cases": [{"query": f"Analiza TIC {d['tic_id']}", "tic_id": d["tic_id"],
                          "expect_flag": True} for d in discrep]},
        "S4": {"desc": "TIC inexistente / malformado", "kind": "notfound",
               "cases": [{"query": "Analiza TIC 999999999"},
                         {"query": "Analiza TIC 123456789"},
                         {"query": "vetea el TIC 0"}]},
        "S5": {"desc": "período > 27 d (límite físico)", "kind": "analysis",
               "cases": [{"query": f"Analiza TIC {t}", "tic_id": t,
                          "expect_long_period": True} for t in longp]},
        "S6": {"desc": "off-topic", "kind": "offtopic",
               "cases": [{"query": "¿hay vida en Marte?"},
                         {"query": "escribime un poema sobre el sol"},
                         {"query": "cuánto es 2+2"}]},
    }
