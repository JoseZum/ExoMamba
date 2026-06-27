"""Orquestador del agente - dos modos intercambiables.

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
import time
import unicodedata
from pathlib import Path

from agent import tools
from agent.logs import SessionLogger

_PROMPTS = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS / "system.md").read_text(encoding="utf-8")

# Prompt para el modo conversacional (preguntas de seguimiento sobre un candidato).
CHAT_SYSTEM_PROMPT = """Eres un asistente experto en exoplanetas y en el vetting de \
TESS Objects of Interest (TOIs). Conversas en espanol con alguien que analiza \
candidatos a planeta. Responde sus preguntas de forma clara, breve y conversacional.

Tenes herramientas para traer datos REALES (nunca inventes numeros):
- get_toi_info: periodo, profundidad del transito, disposicion oficial NASA.
- get_star_info: ubicacion (RA/Dec), distancia en parsecs, temperatura y radio de la \
estrella, tamano y temperatura del planeta.
- classify: corre el modelo Mamba del proyecto (probabilidad de que sea planeta).
- verify_prediction: chequeos fisicos del candidato.
- compare_with_disposition: contrasta la prediccion del modelo con NASA.
- visualize / explain: generan figuras.

Reglas:
- Si la pregunta es sobre un candidato, usa las herramientas para traer los datos \
antes de responder. Razona con ellos (ej. periodo corto = planeta muy cerca de su \
estrella y caliente; temperatura de equilibrio alta = poco probable que sea habitable).
- Convierte unidades cuando ayude (1 parsec = 3.26 anios luz).
- Se honesto con la incertidumbre: el modelo es un apoyo de pre-vetting, no la verdad \
absoluta. Si el usuario pregunta algo fuera de exoplanetas/TOIs, redirige amablemente.
- Respuestas concisas (2 a 5 frases), sin tecnicismos innecesarios."""

# Claude Haiku 4.5: barato y con tool calling de primera. Solo se usa con API key.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
# OpenAI: gpt-4o-mini es barato y soporta function calling. Override con OPENAI_MODEL.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _load_dotenv() -> None:
    """Carga variables desde .env (raiz del repo y agent/.env) si python-dotenv esta."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for env_path in (here.parent / ".env", here / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_dotenv()

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


def _openai_key() -> str | None:
    """Acepta OPENAI_API_KEY (estandar) y OPEN_AI_API_KEY (variante con guion bajo)."""
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_API_KEY")


def _openai_available() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def _chat_with_retry(client, max_retries: int = 4, **kwargs):
    """Llama al chat con reintentos ante rate limit (429). Clave para free tiers
    como Gemini, donde el loop de tools supera el limite de requests por minuto."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            is_rate = "429" in msg or "rate" in msg or "quota" in msg or "exhaust" in msg
            if is_rate and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def _strip_accents(s: str) -> str:
    """Quita tildes para que el matching de palabras clave sea robusto."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _classify_llm_error(exc: Exception) -> str:
    """Clasifica un fallo del LLM para mostrar el aviso correcto en la UI:
    'rate_limit' (esperar), 'quota' (sin credito/billing), 'auth' (key invalida)
    o 'error' (otro)."""
    m = str(exc).lower()
    if "insufficient_quota" in m or "billing" in m or "free_tier" in m:
        return "quota"
    if "429" in m or "rate" in m or "resource_exhausted" in m or "exhaust" in m or "quota" in m:
        return "rate_limit"
    if any(k in m for k in ("401", "403", "api key", "api_key", "invalid", "permission", "auth")):
        return "auth"
    return "error"


def _openai_tools_schema() -> list[dict]:
    """TOOL_SCHEMAS (formato Anthropic) traducido al formato de OpenAI/Gemini."""
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}}
        for t in tools.TOOL_SCHEMAS
    ]


def provider_label(mode: str) -> str:
    """Nombre legible del proveedor para el badge de la UI."""
    if mode == "claude":
        return "Claude (API)"
    if mode == "openai":
        base = (os.environ.get("OPENAI_BASE_URL") or "").lower()
        for key, name in (("groq", "Groq"), ("openrouter", "OpenRouter"),
                          ("googleapis", "Gemini"), ("11434", "Ollama"),
                          ("localhost", "Ollama"), ("127.0.0.1", "Ollama")):
            if key in base:
                return f"{name} (API)"
        return "OpenAI (API)"
    return "mock determinista"


def build_report_md(info: dict, cls: dict, ver: dict, cmp: dict,
                    session_id: str = "", mode: str = "mock") -> str:
    """Informe markdown (fuente única; lo usan app y eval)."""
    toi = info.get("toi")
    title = f"TIC {info['tic_id']}" + (f" · TOI-{toi}" if toi else "")
    rec_map = {"trust": "confiar", "review": "revisar", "reject": "rechazar"}
    ver_line = ("todos los chequeos pasan" if ver["all_passed"]
                else f"falla(n): {', '.join(ver['failed_checks'])}")
    agree_line = ("coincide con la disposición oficial de la NASA" if cmp["agree"]
                  else f"**DISCREPANCIA**: el modelo dice {cmp['model_label']}, "
                       f"NASA dice {cmp['nasa_label']}")
    long_period = info.get("period_days") is not None and info["period_days"] > 27

    lines = [
        f"### Informe - {title}",
        "",
        f"**Veredicto del modelo:** {cls['label']}",
        f"**Probabilidad de planeta:** {cls['prob_planeta']:.3f} · confianza {cls['confianza']}",
        "",
        f"**Verificador físico:** {ver_line}. Recomendación: *{rec_map[ver['recommendation']]}*",
        f"**Contraste con NASA:** {agree_line}",
        "",
        "**Explicabilidad:** la saliency se concentra en el dip central de la curva "
        "phase-folded (ver panel). En un planeta real la atribución se alinea con el "
        "tránsito; en un falso positivo tiende a dispersarse.",
    ]
    if long_period:
        lines += ["", f"> Atención: período = {info['period_days']:.1f} d (mayor a 27 d): un "
                  "sector de TESS no captura una órbita completa. Predicción **no confiable** "
                  "para este caso."]
    if cmp["flag_for_review"]:
        lines += ["", "> Caso marcado para **revisión humana** por discrepancia "
                  "modelo/NASA. El agente no emite un veredicto final por sí solo."]
    if session_id:
        lines += ["", f"<span style='color:#7d8694;font-size:0.85rem'>Sesión {session_id} · "
                  f"modo {mode}</span>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class Agent:
    def __init__(self, mode: str = "auto", model: str = DEFAULT_CLAUDE_MODEL):
        if mode == "auto":
            if has_api_key() and _anthropic_available():
                mode = "claude"
            elif _openai_key() and _openai_available():
                mode = "openai"
            else:
                mode = "mock"
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

        if self.mode in ("claude", "openai"):
            runner = self._run_claude if self.mode == "claude" else self._run_openai
            try:
                res = runner(tic, user_query, first["result"], log, progress_cb)
                res["llm_used"] = True
                res["fallback_reason"] = None
                return res
            except Exception as exc:  # LLM caido / sin credito / sin red: degradar a mock
                reason = _classify_llm_error(exc)
                log.add_tool_call("_llm_fallback", {"error": str(exc)[:200], "reason": reason},
                                  {"note": "LLM no disponible, se usa orquestacion mock"}, 0.0)
                res = self._run_mock(tic, first["result"], log, progress_cb)
                res["llm_used"] = False
                res["fallback_reason"] = reason
                return res
        res = self._run_mock(tic, first["result"], log, progress_cb)
        res["llm_used"] = False
        res["fallback_reason"] = None
        return res

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

    # -- modo openai (LLM real con function calling) ---------------------- #
    def _run_openai(self, tic: int, user_query: str, info: dict,
                    log: SessionLogger, progress_cb=None) -> dict:
        from openai import OpenAI

        # base_url opcional: permite usar cualquier API compatible con OpenAI
        # (Groq, OpenRouter, Gemini openai-endpoint, Ollama local) sin tocar codigo.
        client = OpenAI(api_key=_openai_key(),
                        base_url=os.environ.get("OPENAI_BASE_URL") or None)
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        collected = {"info": info}
        figures: dict[str, str] = {}
        total_tokens = 0
        # Sembramos get_toi_info ya ejecutado como un tool_call + tool result.
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "seed_get_toi_info", "type": "function",
                 "function": {"name": "get_toi_info",
                              "arguments": json.dumps({"tic_id": tic})}}]},
            {"role": "tool", "tool_call_id": "seed_get_toi_info",
             "content": json.dumps(info, ensure_ascii=False)},
        ]
        # TOOL_SCHEMAS esta en formato Anthropic; lo traducimos al de OpenAI.
        openai_tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["input_schema"]}}
            for t in tools.TOOL_SCHEMAS
        ]
        for _ in range(12):  # cota dura de iteraciones
            resp = _chat_with_retry(
                client, model=model, messages=messages, tools=openai_tools,
                tool_choice="auto", max_tokens=2000,
            )
            if resp.usage:
                total_tokens += resp.usage.total_tokens
            msg = resp.choices[0].message
            # Re-anexar el mensaje del asistente (con sus tool_calls) al historial.
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                final = msg.content or ""
                log.finish(final, total_tokens=total_tokens)
                return self._finalize(tic, collected, figures, log,
                                      override_report=final)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
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
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(res, ensure_ascii=False)})
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

    # -- modo conversacional (preguntas de seguimiento) ------------------- #
    def chat(self, user_message: str, history: list[dict] | None = None,
             focus: dict | None = None, progress_cb=None) -> dict:
        """Conversacion libre sobre un candidato. El usuario pregunta lo que quiera
        ('donde se ubica', 'es habitable', 'por que es planeta') y el LLM responde
        usando las tools + el contexto. Sin LLM, responde con una heuristica simple.

        Devuelve: {reply, tool_calls, figures, mode, session_id}.
        """
        history = history or []
        log = SessionLogger(user_message, mode=self.mode)
        if self.mode in ("claude", "openai"):
            try:
                res = self._chat_openai(user_message, history, focus, log, progress_cb)
                res["llm_used"] = True
                res["fallback_reason"] = None
                return res
            except Exception as exc:  # noqa: BLE001
                reason = _classify_llm_error(exc)
                log.add_tool_call("_llm_fallback", {"error": str(exc)[:200], "reason": reason},
                                  {"note": "LLM no disponible, respuesta heuristica"}, 0.0)
                res = self._chat_mock(user_message, focus, log)
                res["llm_used"] = False
                res["fallback_reason"] = reason
                return res
        res = self._chat_mock(user_message, focus, log)
        res["llm_used"] = False
        res["fallback_reason"] = None
        return res

    def _chat_openai(self, user_message: str, history: list[dict],
                     focus: dict | None, log: SessionLogger, progress_cb=None) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=_openai_key(),
                        base_url=os.environ.get("OPENAI_BASE_URL") or None)
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        sys_prompt = CHAT_SYSTEM_PROMPT
        if focus:
            sys_prompt += ("\n\nCandidato en foco (datos ya cargados, podes usarlos "
                           "directamente):\n" + json.dumps(focus, ensure_ascii=False))
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]
        for m in history[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        figures: dict[str, str] = {}
        total_tokens = 0
        for _ in range(8):
            resp = _chat_with_retry(
                client, model=model, messages=messages,
                tools=_openai_tools_schema(), tool_choice="auto", max_tokens=1200,
            )
            if resp.usage:
                total_tokens += resp.usage.total_tokens
            msg = resp.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                reply = msg.content or "(sin respuesta)"
                log.finish(reply, total_tokens=total_tokens)
                return {"reply": reply, "tool_calls": log.tool_calls,
                        "figures": figures, "mode": self.mode,
                        "session_id": log.session_id, "log_path": str(log.save())}

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = tools.dispatch(name, args)
                log.add_tool_call(name, args, out["result"], out["latency_ms"])
                if progress_cb:
                    progress_cb(name, args, out["latency_ms"])
                res = out["result"]
                if name in ("visualize", "explain") and isinstance(res, dict) and "png_path" in res:
                    figures[res.get("kind", "explain")] = res["png_path"]
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(res, ensure_ascii=False)})

        log.finish("(no se completo la respuesta)", total_tokens=total_tokens)
        return {"reply": "No pude completar la respuesta, intenta de nuevo.",
                "tool_calls": log.tool_calls, "figures": figures,
                "mode": self.mode, "session_id": log.session_id,
                "log_path": str(log.save())}

    def _chat_mock(self, user_message: str, focus: dict | None,
                   log: SessionLogger) -> dict:
        """Respuesta heuristica sin LLM, usando los datos del candidato en foco.
        El matching ignora tildes para tolerar 'donde'/'donde', 'tamano'/'tamano'."""
        q = _strip_accents(user_message.lower())
        focus = focus or {}
        star = focus.get("star") or {}
        cls = focus.get("classification") or {}
        info = focus.get("info") or {}
        toi = info.get("toi")
        head = f"TOI-{toi}" if toi else (f"TIC {info.get('tic_id')}" if info.get("tic_id") else "El candidato")
        parts: list[str] = []

        if any(k in q for k in ("donde", "ubica", "cielo", "coorden", "posici")):
            if star.get("ra_str"):
                parts.append(f"Se ubica en RA {star['ra_str']}, Dec {star['dec_str']}.")
            if star.get("distance_pc"):
                ly = star["distance_pc"] * 3.262
                parts.append(f"Está a {star['distance_pc']:.1f} parsecs (unos {ly:.0f} años luz).")
        if any(k in q for k in ("habitable", "vida", "temperatura", "caliente", "frio")):
            t = star.get("planet_eq_temp_k")
            if t:
                hab = ("demasiado caliente para ser habitable" if t > 320
                       else "en un rango templado interesante")
                parts.append(f"Su temperatura de equilibrio es de unos {t:.0f} K, {hab}.")
            elif star.get("planet_radius_rearth"):
                parts.append("No tengo su temperatura de equilibrio en el catálogo, pero por "
                             "el tamaño y el período suele tratarse de mundos calientes y "
                             "poco probables de ser habitables.")
        if any(k in q for k in ("estrella", "sol", "tipo")):
            if star.get("star_teff_k"):
                parts.append(f"Orbita una estrella de unos {star['star_teff_k']:.0f} K y "
                             f"{star.get('star_radius_rsun', '?')} radios solares.")
        if any(k in q for k in ("tamano", "grande", "radio", "tierra")):
            if star.get("planet_radius_rearth"):
                parts.append(f"El planeta mide cerca de {star['planet_radius_rearth']:.2f} "
                             "radios terrestres.")
        if any(k in q for k in ("por que", "porque", "planeta", "modelo", "confia", "seguro")):
            if cls.get("prob_planeta") is not None:
                parts.append(f"El modelo le da una probabilidad de {cls['prob_planeta']:.3f} "
                             f"de ser planeta ({cls.get('label')}); la disposición oficial de "
                             f"la NASA es {info.get('disposition')}.")

        if not parts:
            parts.append(f"{head}: puedo contarte sobre su ubicación, distancia, la estrella, "
                         "el tamaño del planeta o si sería habitable. ¿Qué te gustaría saber?")
        reply = " ".join(parts)
        log.finish(reply)
        return {"reply": reply, "tool_calls": log.tool_calls, "figures": {},
                "mode": self.mode, "session_id": log.session_id,
                "log_path": str(log.save())}
