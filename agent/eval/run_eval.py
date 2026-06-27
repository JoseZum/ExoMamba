"""Corre la suite de validación completa y guarda resultados en eval/results/.

Uso:
    python -m agent.eval.run_eval            # modo mock (sin API key)
    python -m agent.eval.run_eval --claude   # modo Claude (requiere ANTHROPIC_API_KEY)

Genera:
    eval/results/eval_<timestamp>.json   - métricas + detalle por caso
    eval/results/SUMMARY.md              - tabla legible para el paper
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from agent.eval import metrics, scenarios
from agent.llm import Agent

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run(mode: str = "mock") -> dict:
    agent = Agent(mode=mode)
    scen = scenarios.build_scenarios()
    results_by_scenario: dict = {}
    detail: dict = {}

    for sid, block in scen.items():
        pairs = []
        cases_detail = []
        for case in block["cases"]:
            case = {**case, "_desc": block["desc"]}
            result = agent.run(case["query"])
            pairs.append((case, result))
            cases_detail.append({
                "query": case["query"],
                "kind": result["kind"],
                "prob": (result["classification"]["prob_planeta"]
                         if result.get("classification") else None),
                "flagged": "DISCREPANCIA" in result["report_md"],
                "n_tools": len(result["tool_calls"]),
                "log": Path(result["log_path"]).name,
            })
        results_by_scenario[sid] = pairs
        detail[sid] = {"desc": block["desc"], "cases": cases_detail}

    agg = metrics.aggregate(results_by_scenario)
    return {"mode": agent.mode, "aggregate": agg, "detail": detail}


def _markdown(report: dict) -> str:
    agg = report["aggregate"]
    g = agg["global"]
    lines = [
        "# Resultados de validación del agente - suite S1–S6",
        "",
        f"- **Modo:** {report['mode']}",
        f"- **Sesiones totales:** {g['n_sessions']}",
        f"- **Tool-call accuracy (global):** {g['tool_call_accuracy']:.3f}",
        f"- **Faithfulness (global):** {g['faithfulness']:.3f}",
        f"- **Tasa de no-alucinación (global):** {g['non_hallucination_rate']:.3f}",
        f"- **Latencia media:** {g['avg_latency_ms']:.1f} ms",
        "",
        "## Por escenario",
        "",
        "| ID | Escenario | N | Tool-call acc | Faithfulness | No-alucina | Lat (ms) | Extra |",
        "|----|-----------|---|---------------|--------------|-----------|----------|-------|",
    ]
    for sid, row in agg["per_scenario"].items():
        extra = ""
        if "flag_recall" in row:
            extra = f"flag recall {row['flag_recall']}"
        elif "long_period_declared" in row:
            extra = f"límite declarado {row['long_period_declared']}"
        elif "model_accuracy_vs_nasa" in row:
            extra = f"acc vs NASA {row['model_accuracy_vs_nasa']}"
        lines.append(
            f"| {sid} | {row['desc']} | {row['n']} | {row['tool_call_accuracy']} | "
            f"{row['faithfulness']} | {row['non_hallucination']} | "
            f"{row['avg_latency_ms']:.0f} | {extra} |")
    lines += ["", "_Generado por `python -m agent.eval.run_eval`. "
              "El modo mock usa el orquestador determinista; con ANTHROPIC_API_KEY "
              "se reejecuta contra Claude._"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", action="store_true",
                    help="usar modo Claude (requiere ANTHROPIC_API_KEY)")
    args = ap.parse_args()
    mode = "claude" if args.claude else "mock"

    report = run(mode=mode)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = RESULTS_DIR / f"eval_{report['mode']}_{ts}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = RESULTS_DIR / "SUMMARY.md"
    md_path.write_text(_markdown(report), encoding="utf-8")

    g = report["aggregate"]["global"]
    print(f"Modo: {report['mode']} · sesiones: {g['n_sessions']}")
    print(f"  tool-call accuracy : {g['tool_call_accuracy']:.3f}")
    print(f"  faithfulness       : {g['faithfulness']:.3f}")
    print(f"  no-alucinación     : {g['non_hallucination_rate']:.3f}")
    print(f"  latencia media     : {g['avg_latency_ms']:.1f} ms")
    print(f"\nGuardado: {json_path.name}  +  SUMMARY.md")


if __name__ == "__main__":
    main()
