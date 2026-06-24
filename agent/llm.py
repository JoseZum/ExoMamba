"""Orquestador del agente — dos modos intercambiables.

- `mock`   : orquestación determinista (llama las tools en orden fijo). Funciona
             HOY, sin API key. Es lo que mueve la demo y la suite de validación.
- `claude` : loop real de tool calling contra la API de Anthropic (Claude Haiku 4.5).
             Se activa automáticamente cuando hay ANTHROPIC_API_KEY y el SDK
             `anthropic` está instalado.

Ambos modos producen el MISMO dict de salida (mismo informe, mismas figuras, mismo
log), así que el frontend y la evaluación no distinguen el modo. Cambiar de mock a
Claude no toca ni la UI ni la suite.

Uso:
    from agent.llm import Agent
    agent = Agent()            # 'auto': claude si hay key, si no mock
    result = agent.run("Analiza TIC 79748331")
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agent import tools
from agent.logs import SessionLogger

_PROMPTS = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS / "system.md").read_text(encoding="utf-8")

# Claude Haiku 4.5 — barato y con tool calling de primera. Solo se usa con API key.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Secuencia determinista de análisis (modo mock). get_toi_info se llama aparte.
_ANALYSIS_SEQUENCE = [
    ("load_light_curve", lambda tic: {"tic_id": tic}),
    ("classify", lambda tic: {"tic_id": tic}),
    ("verify_prediction", lambda tic: {"tic_id": tic}),
    ("visualize", lambda tic: {"tic_id": tic, "kind": "sky_map"}),
    ("visualize", lambda tic: {"tic_id": tic, "kind": "orbit_diagram"}),
    ("explain", lambda tic: {"tic_id": tic}),
    ("compare_with_disposition", lambda tic: {"tic_id": tic}),
]


# --------------------------------------------------------------------------- #
# Routing / utilidades compartidas
# --------------------------------------------------------------------------- #
def parse_tic(text: str) -> int | None:
    """Extrae un TIC ID del texto. Acepta 'TIC 123', 'tic123', '12345', etc."""
    m = re.search(r"tic[\s_-]*?(\d{3,})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b(\d{4,})\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def build_report_md(info: dict, cls: dict, ver: dict, cmp: dict,
                    session_id: str = "", mode: str = "mock") -> str:
    """Informe markdown (fuente única; lo usan app y eval)."""
    toi = info.get("toi")
    title = f"TIC {info['tic_id']}" + (f" · TOI-{toi}" if toi else "")
    rec_map = {"trust": "confiar", "review": "revisar", "reject": "rechazar"}
    ver_line = ("✅ todos los chequeos pasan" if ver["all_passed"]
                else f"⚠️ falla(n): {', '.join(ver['failed_checks'])}")
    agree_line = ("✅ coincide con la disposición oficial NASA" if cmp["agree"]
                  else f"🚩 **DISCREPANCIA**: el modelo dice {cmp['model_label']}, "
                       f"NASA dice {cmp['nasa_label']}")
    long_period = info.get("period_days") is not None and info["period_days"] > 27

    lines = [
        f"### Informe — {title}",
        "",
        f"**Veredicto del modelo:** {cls['label']}",
        f"**Probabilidad de planeta:** {cls['prob_planeta']:.3f} · confianza {cls['confianza']}",
        "",
        f"**Verificador físico:** {ver_line} → recomendación: *{rec_map[ver['recommendation']]}*",
        f"**Contraste con NASA:** {agree_line}",
        "",
        "**Explicabilidad:** la saliency se concentra en el dip central de la curva "
        "phase-folded (ver panel). En un planeta real la atribución se alinea con el "
        "tránsito; en un falso positivo tiende a dispersarse.",
    ]
    if long_period:
        lines += ["", f"> ⚠️ Período = {info['period_days']:.1f} d (> 27 d): un sector TESS "
                  "no captura una órbita completa. Predicción **no confiable** para este caso."]
    if cmp["flag_for_review"]:
        lines += ["", "> 🚩 Caso marcado para **revisión humana** por discrepancia "
                  "modelo/NASA. El agente no emite veredicto final solo."]
    if session_id:
        lines += ["", f"<span style='color:#7d8aa0;font-size:0.85rem'>Sesión {session_id} · "
                  f"modo {mode}</span>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(self, mode: str = "auto", model: str = DEFAULT_CLAUDE_MODEL):
        if mode == "auto":
            mode = "claude" if (has_api_key() and _anthropic_available()) else "mock"
        self.mode = mode
        self.model = model

    # -- punto de entrada único ------------------------------------------- #
    def run(self, user_query: str, progress_cb=None) -> dict:
        """Ejecuta una sesión. `progress_cb(tool, args, latency_ms)` se llama tras
        cada tool (opcional, para animar la UI en vivo)."""
        log = SessionLogger(user_query, mode=self.mode)
        tic = parse_tic(user_query)
        if tic is None:
            return self._offtopic(user_query, log)

        # Chequeo de existencia (es la 1.ª tool real del agente).
        first = tools.dispatch("get_toi_info", {"tic_id": tic})
        log.add_tool_call("get_toi_info", {"tic_id": tic}, first["result"], first["latency_ms"])
        if progress_cb:
            progress_cb("get_toi_info", {"tic_id": tic}, first["latency_ms"])
        if "error" in first["result"]:
            return self._notfound(tic, log)

        if self.mode == "claude":
            return self._run_claude(tic, user_query, first["result"], log, progress_cb)
        return self._run_mock(tic, first["result"], log, progress_cb)

    # -- guardrails (idénticos en ambos modos) ---------------------------- #
    def _offtopic(self, query: str, log: SessionLogger) -> dict:
        report = ("Solo analizo **TOIs por su TIC ID**. Pasame algo como "
                  "`Analiza TIC 79748331` y corro el modelo + los chequeos físicos. "
                  "No tengo herramientas para responder fuera de ese alcance.")
        log.finish(report)
        return {"session_id": log.session_id, "mode": self.mode, "kind": "offtopic",
                "report_md": report, "tic_id": None, "info": None, "classification": None,
                "verify": None, "comparison": None, "figures": {},
                "tool_calls": log.tool_calls, "log_path": str(log.save())}

    def _notfound(self, tic: int, log: SessionLogger) -> dict:
        n = len(__import__("agent.mock", fromlist=["load_catalog"]).load_catalog())
        report = (f"No tengo registro de **TIC {tic}** en el catálogo TOI cargado "
                  f"(~{n:,} objetos con disposición). Verificá el ID o pasame otro "
                  "candidato.")
        log.finish(report)
        return {"session_id": log.session_id, "mode": self.mode, "kind": "notfound",
                "report_md": report, "tic_id": tic, "info": None, "classification": None,
                "verify": None, "comparison": None, "figures": {},
                "tool_calls": log.tool_calls, "log_path": str(log.save())}

    # -- modo mock (determinista) ----------------------------------------- #
    def _run_mock(self, tic: int, info: dict, log: SessionLogger,
                  progress_cb=None) -> dict:
        collected = {"info": info}
        figures: dict[str, str] = {}
        for name, argfn in _ANALYSIS_SEQUENCE:
            args = argfn(tic)
            out = tools.dispatch(name, args)
            log.add_tool_call(name, args, out["result"], out["latency_ms"])
            if progress_cb:
                progress_cb(name, args, out["latency_ms"])
            res = out["result"]
            if name == "classify":
                collected["classification"] = res
            elif name == "verify_prediction":
                collected["verify"] = res
            elif name == "compare_with_disposition":
                collected["comparison"] = res
            elif name == "visualize" and "png_path" in res:
                figures[res["kind"]] = res["png_path"]
            elif name == "explain" and "png_path" in res:
                figures["explain"] = res["png_path"]
        return self._finalize(tic, collected, figures, log)

    # -- modo claude (LLM real con tool calling) -------------------------- #
    def _run_claude(self, tic: int, user_query: str, info: dict,
                    log: SessionLogger, progress_cb=None) -> dict:
        import anthropic

        client = anthropic.Anthropic()
        collected = {"info": info}
        figures: dict[str, str] = {}
        total_tokens = 0
        # Sembramos el resultado de get_toi_info ya ejecutado.
        messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "seed_get_toi_info",
                 "name": "get_toi_info", "input": {"tic_id": tic}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "seed_get_toi_info",
                 "content": json.dumps(info, ensure_ascii=False)}]},
        ]
        for _ in range(12):  # cota dura de iteraciones
            resp = client.messages.create(
                model=self.model, system=SYSTEM_PROMPT, tools=tools.TOOL_SCHEMAS,
                messages=messages, max_tokens=2000,
            )
            total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                final = "".join(b.text for b in resp.content if b.type == "text")
                log.finish(final, total_tokens=total_tokens)
                # Completar el panel con lo recolectado (puede faltar si el LLM no llamó algo).
                return self._finalize(tic, collected, figures, log,
                                      override_report=final)
            results_block = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                out = tools.dispatch(block.name, dict(block.input))
                log.add_tool_call(block.name, dict(block.input), out["result"],
                                  out["latency_ms"])
                if progress_cb:
                    progress_cb(block.name, dict(block.input), out["latency_ms"])
                res = out["result"]
                if block.name == "classify":
                    collected["classification"] = res
                elif block.name == "verify_prediction":
                    collected["verify"] = res
                elif block.name == "compare_with_disposition":
                    collected["comparison"] = res
                elif block.name == "visualize" and "png_path" in res:
                    figures[res["kind"]] = res["png_path"]
                elif block.name == "explain" and "png_path" in res:
                    figures["explain"] = res["png_path"]
                results_block.append({"type": "tool_result", "tool_use_id": block.id,
                                      "content": json.dumps(res, ensure_ascii=False)})
            messages.append({"role": "user", "content": results_block})
        # Si se agotaron las iteraciones, cerrar con lo que haya.
        return self._finalize(tic, collected, figures, log)

    # -- cierre común ----------------------------------------------------- #
    def _finalize(self, tic: int, collected: dict, figures: dict,
                  log: SessionLogger, override_report: str | None = None) -> dict:
        info = collected["info"]
        cls = collected.get("classification") or tools.classify(tic)
        ver = collected.get("verify") or tools.verify_prediction(tic)
        cmp = collected.get("comparison") or tools.compare_with_disposition(tic)
        report = override_report or build_report_md(
            info, cls, ver, cmp, session_id=log.session_id, mode=self.mode)
        if not log.final_report_md:
            log.finish(report)
        return {
            "session_id": log.session_id, "mode": self.mode, "kind": "analysis",
            "report_md": report, "tic_id": tic, "info": info,
            "classification": cls, "verify": ver, "comparison": cmp,
            "figures": figures, "tool_calls": log.tool_calls,
            "log_path": str(log.save()),
        }
