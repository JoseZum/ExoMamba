"""Las 5 métricas específicas del agente (ETAPA3.md).

Operan sobre el dict de resultado que devuelve `Agent.run` (más el caso esperado
del escenario). NO miden el modelo (eso es Etapa 2); miden el comportamiento del
agente: que llame las tools correctas, que no invente números, que flaggee
discrepancias y que responda en latencia razonable.
"""

from __future__ import annotations

import re

# Tools mínimas que un análisis correcto debe invocar.
_REQUIRED_ANALYSIS_TOOLS = {
    "get_toi_info", "load_light_curve", "classify",
    "verify_prediction", "compare_with_disposition", "explain",
}
_PROB_RE = re.compile(r"Probabilidad de planeta:\*\*\s*([0-9]*\.?[0-9]+)")


def _called_tools(result: dict) -> set:
    return {c["tool"] for c in result["tool_calls"]}


def tool_call_accuracy(result: dict) -> bool:
    """¿Llamó el conjunto de tools correcto para el tipo de consulta?"""
    called = _called_tools(result)
    kind = result["kind"]
    if kind == "analysis":
        return _REQUIRED_ANALYSIS_TOOLS.issubset(called) and any(
            t == "visualize" for t in called)
    if kind == "notfound":
        return called == {"get_toi_info"}
    if kind == "offtopic":
        return len(called) == 0
    return False


def faithfulness(result: dict) -> bool:
    """¿Los números del informe coinciden con lo que devolvieron las tools?"""
    if result["kind"] != "analysis":
        # En notfound/offtopic no debe haber números de análisis.
        return _PROB_RE.search(result["report_md"]) is None
    m = _PROB_RE.search(result["report_md"])
    if not m or result.get("classification") is None:
        return False
    reported = float(m.group(1))
    actual = round(result["classification"]["prob_planeta"], 3)
    return abs(reported - actual) < 1e-3


def non_hallucination(result: dict) -> bool:
    """Para notfound/offtopic: no inventa análisis. Para analysis: es fiel."""
    if result["kind"] in ("notfound", "offtopic"):
        return _PROB_RE.search(result["report_md"]) is None
    return faithfulness(result)


def flag_present(result: dict) -> bool:
    """¿El informe declara la discrepancia con NASA cuando corresponde?"""
    return "DISCREPANCIA" in result["report_md"]


def declares_long_period(result: dict) -> bool:
    """¿El informe declara la limitación de período > 27 d?"""
    return "27 d" in result["report_md"] or "no confiable" in result["report_md"].lower()


def total_latency_ms(result: dict) -> float:
    return sum(c["latency_ms"] for c in result["tool_calls"])


def aggregate(results_by_scenario: dict) -> dict:
    """Calcula las métricas agregadas por escenario y globales.

    `results_by_scenario`: {scenario_id: [(case, result), ...]}
    """
    out = {"per_scenario": {}, "global": {}}
    all_results = []

    for sid, pairs in results_by_scenario.items():
        results = [r for _, r in pairs]
        all_results.extend(results)
        n = len(results)
        tca = sum(tool_call_accuracy(r) for r in results)
        faith = sum(faithfulness(r) for r in results)
        nohall = sum(non_hallucination(r) for r in results)
        lat = [total_latency_ms(r) for r in results]
        row = {
            "desc": pairs[0][0].get("_desc", "") if pairs else "",
            "n": n,
            "tool_call_accuracy": f"{tca}/{n}",
            "faithfulness": f"{faith}/{n}",
            "non_hallucination": f"{nohall}/{n}",
            "avg_latency_ms": round(sum(lat) / n, 1) if n else 0.0,
        }
        if sid == "S3":
            flagged = sum(flag_present(r) for _, r in pairs)
            row["flag_recall"] = f"{flagged}/{n}"
        if sid == "S5":
            declared = sum(declares_long_period(r) for _, r in pairs)
            row["long_period_declared"] = f"{declared}/{n}"
        if sid in ("S1", "S2"):
            correct = sum(
                (r["classification"]["prob_planeta"] >= 0.5) == case["expect_planet"]
                for case, r in pairs if r["kind"] == "analysis")
            row["model_accuracy_vs_nasa"] = f"{correct}/{n}"
        out["per_scenario"][sid] = row

    N = len(all_results)
    out["global"] = {
        "n_sessions": N,
        "tool_call_accuracy": round(sum(tool_call_accuracy(r) for r in all_results) / N, 3),
        "faithfulness": round(sum(faithfulness(r) for r in all_results) / N, 3),
        "non_hallucination_rate": round(sum(non_hallucination(r) for r in all_results) / N, 3),
        "avg_latency_ms": round(sum(total_latency_ms(r) for r in all_results) / N, 1),
    }
    return out
