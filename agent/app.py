"""TOI Vetting Assistant, frontend Streamlit (Etapa 3).

UI con selector de candidatos reales + panel de análisis + chat conversacional.
Consume el Agent (agent/llm.py), que orquesta las herramientas y registra cada
sesión en agent/logs/. La predicción la da el modelo Mamba real si el servicio de
inferencia está arriba; si no, cae al mock. El LLM es Claude u OpenAI/Gemini según
la API key del .env; sin key, orquestación determinista.

Correr desde la raíz del repo:
    python -m streamlit run agent/app.py --server.address 127.0.0.1
    # abrir http://127.0.0.1:8501
"""

from __future__ import annotations

import html
import random
import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit ejecuta este archivo con agent/ en sys.path, no la raíz del repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import mock, model_client
from agent.llm import Agent, parse_tic, provider_label

# --------------------------------------------------------------------------- #
# Candidatos del selector: se construyen del catálogo real cruzado con las curvas
# .pt disponibles (para que la predicción sea del Mamba real). El usuario elige
# uno con un clic; no hace falta conocer ningún ID. La etiqueta indica la
# disposición oficial NASA para ver si el modelo acierta. Hay cientos de planetas
# confirmados disponibles, así que se muestran muchos (no solo un puñado).
# --------------------------------------------------------------------------- #
CANDIDATES = [
    {"tic": 79748331, "disp": "Planeta confirmado", "toi": "TOI-1064"},
    {"tic": 158297421, "disp": "Planeta confirmado", "toi": "TOI-1073"},
    {"tic": 317060587, "disp": "Planeta confirmado", "toi": "TOI-1052"},
    {"tic": 182943944, "disp": "Falso positivo", "toi": "TOI-1017"},
    {"tic": 291555748, "disp": "Falso positivo", "toi": "TOI-1018"},
    {"tic": 140706664, "disp": "Falso positivo", "toi": "TOI-1020"},
    {"tic": 309257814, "disp": "Candidato sin confirmar", "toi": "TOI-171"},
    {"tic": 142087638, "disp": "Candidato sin confirmar", "toi": "TOI-2404"},
    {"tic": 34077285, "disp": "Candidato sin confirmar", "toi": "TOI-880"},
]
for _c in CANDIDATES:
    _c["label"] = f"{_c['disp']}  ·  {_c['toi']}  (TIC {_c['tic']})"


# --------------------------------------------------------------------------- #
# Configuración + estilo
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="TOI Vetting Assistant", layout="wide",
                   initial_sidebar_state="collapsed")

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  html, body, .stApp, [class^="st-"], [class*=" st-"], .stMarkdown,
  h1, h2, h3, h4, p, div, span, label, button, input, textarea {
      font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif !important;
  }
  code, .mono, .toolchip, .num { font-family: 'IBM Plex Mono', monospace !important; }
  .stApp { background: #0b0e14; }
  .card { background:#11151d; border:1px solid #222a36; border-radius:6px;
          padding:16px 18px; margin-bottom:14px; }
  .verdict { border-left:3px solid #3a4250; }
  .verdict-planet { border-left-color:#3fb950; }
  .verdict-fp { border-left-color:#d9534f; }
  .verdict-label { font-size:1.4rem; font-weight:600; letter-spacing:0.4px; margin:0; }
  .verdict-sub { color:#7d8694; font-size:0.82rem; margin-top:3px; }
  .probbar-track { height:8px; background:#1b2230; border-radius:4px; overflow:hidden;
                   margin:12px 0 6px 0; }
  .probbar-fill { height:100%; border-radius:4px; }
  .toolchip { display:inline-block; background:#161b24; border:1px solid #2a3340;
              color:#9db4cc; border-radius:4px; padding:3px 9px; margin:2px 4px 2px 0;
              font-size:0.76rem; }
  .pill-ok { color:#3fb950; font-weight:600; }
  .pill-no { color:#d9534f; font-weight:600; }
  .pill-na { color:#7d8694; }
  .badge { background:#161b24; border:1px solid #2a3340; color:#9db4cc;
           border-radius:4px; padding:3px 10px; font-size:0.78rem; font-weight:500; }
  .muted { color:#7d8694; font-size:0.84rem; }
  h1, h2, h3 { letter-spacing:0.2px; font-weight:600; }
  div.block-container { padding-top:2.2rem; }
  /* Terminal del servicio de inferencia (evidencia de la llamada al Mamba real) */
  .term { background:#06080c; border:1px solid #1c2530; border-radius:6px;
          margin:6px 0 14px 0; overflow:hidden; }
  .term-bar { background:#0e131b; border-bottom:1px solid #1c2530; color:#7d8694;
              font-family:'IBM Plex Mono',monospace; font-size:0.74rem; font-weight:500;
              padding:5px 12px; letter-spacing:0.3px; }
  .term-bar::before { content:'\\25cf \\25cf \\25cf'; color:#2a3340; margin-right:9px;
              letter-spacing:2px; }
  .term-body { font-family:'IBM Plex Mono',monospace; font-size:0.76rem; line-height:1.55;
               padding:10px 12px; max-height:240px; overflow-y:auto; white-space:pre-wrap; }
  .term-ts { color:#3a4250; }
  /* No romper los iconos Material (avatares del chat) con el font override */
  [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
  span[translate="no"] {
      font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
  }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Estado de sesión
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_agent() -> Agent:
    return Agent(mode="auto")


@st.cache_data(show_spinner=False)
def available_tics() -> list[int]:
    """TICs del catálogo que tienen curva preprocesada (para el botón al azar)."""
    proc = _REPO_ROOT / "data" / "processed" / "global"
    cat = mock.load_catalog()
    have = {int(p.stem) for p in proc.glob("*.pt")} if proc.exists() else set()
    return [int(t) for t in cat.index.unique() if int(t) in have]


def _init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("focus", None)
    st.session_state.setdefault("selected_tic", CANDIDATES[0]["tic"])


def is_analysis_command(text: str) -> bool:
    """True si el mensaje pide un análisis estructurado (llena el panel); False si
    es una pregunta conversacional sobre el candidato en foco."""
    t = text.lower()
    has_tic = parse_tic(text) is not None
    wants = any(w in t for w in ("analiz", "vetea", "veta", "clasific"))
    only_tic = has_tic and len(t.split()) <= 3
    return has_tic and (wants or only_tic)


def _focus_from_tic(tic: int) -> dict:
    """Contexto ligero de un candidato (sin análisis previo) para el chat."""
    return {"tic_id": tic, "info": mock.get_toi_info(tic),
            "star": mock.get_star_info(tic), "classification": None}


_init_state()
agent = get_agent()


# --------------------------------------------------------------------------- #
# Render: encabezado
# --------------------------------------------------------------------------- #
def render_header():
    badge = mock.model_badge()
    mode_txt = provider_label(agent.mode)
    svc = model_client.health()
    if svc:
        model_txt = "Mamba real (GPU)"
        metric_txt = (f"AUC test {svc['test_auc']:.3f}" if svc.get("test_auc")
                      else "modelo real")
        n_params = svc.get("n_params", badge["n_params"])
    else:
        model_txt = f"{badge['name']} (simulado)"
        metric_txt = f"AUC val {badge['val_auc']:.3f}"
        n_params = badge["n_params"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("## TOI Vetting Assistant")
        st.markdown("<span class='muted'>Asistente de vetting de TESS Objects of "
                    "Interest. Consume el modelo del proyecto como herramienta.</span>",
                    unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div style='text-align:right; padding-top:14px'>"
            f"<span class='badge'>modelo: {model_txt}</span> "
            f"<span class='badge'>agente: {mode_txt}</span><br>"
            f"<span class='muted num'>{metric_txt} · {n_params:,} params</span></div>",
            unsafe_allow_html=True)
    st.divider()


# --------------------------------------------------------------------------- #
# Render: selector de candidatos
# --------------------------------------------------------------------------- #
def render_picker():
    st.markdown("#### Elige un candidato y presiona Analizar")
    labels = [c["label"] for c in CANDIDATES]
    sel_col, b1, b2 = st.columns([3, 1, 1])
    with sel_col:
        choice = st.selectbox("Candidato", labels, label_visibility="collapsed")
    sel_tic = next(c["tic"] for c in CANDIDATES if c["label"] == choice)
    st.session_state.selected_tic = sel_tic
    with b1:
        if st.button("Analizar", type="primary", width="stretch"):
            st.session_state.pending = f"Analiza TIC {sel_tic}"
            st.rerun()
    with b2:
        if st.button("Al azar", width="stretch"):
            pool = available_tics() or [c["tic"] for c in CANDIDATES]
            st.session_state.pending = f"Analiza TIC {random.choice(pool)}"
            st.rerun()
    st.caption("Planeta confirmado, falso positivo o candidato sin confirmar según la "
               "disposición oficial de la NASA. También puedes escribir un TIC ID a mano abajo.")


# --------------------------------------------------------------------------- #
# Render: panel de análisis
# --------------------------------------------------------------------------- #
def render_empty_panel():
    st.markdown("### Panel de análisis")
    st.markdown("<div class='card'><span class='muted'>Elige un candidato arriba y "
                "presiona Analizar. Vas a ver el veredicto del modelo, el verificador "
                "físico y las visualizaciones (mapa del cielo, órbita estimada y curva "
                "con saliency).</span></div>", unsafe_allow_html=True)


def render_verdict_card(cls: dict):
    is_planet = cls["prob_planeta"] >= 0.5
    klass = "verdict-planet" if is_planet else "verdict-fp"
    col = mock.C_PLANET if is_planet else mock.C_FP
    pct = cls["prob_planeta"] * 100
    src_txt = "Mamba real" if cls.get("source") == "mamba_real" else "simulado (mock)"
    st.markdown(
        f"<div class='card verdict {klass}'>"
        f"<p class='verdict-label' style='color:{col}'>{cls['label']}</p>"
        f"<p class='verdict-sub'>probabilidad de planeta · confianza {cls['confianza']} "
        f"· fuente: {src_txt}</p>"
        f"<div class='probbar-track'><div class='probbar-fill' "
        f"style='width:{pct:.0f}%; background:{col}'></div></div>"
        f"<span class='muted num'>{cls['prob_planeta']:.3f}</span></div>",
        unsafe_allow_html=True)


_VERIFY_LABELS = {
    "periodo_en_sector": "Período en el sector",
    "profundidad_plausible": "Profundidad plausible",
    "duracion_consistente": "Duración consistente",
    "estrella_observable": "Estrella observable",
}


def render_verify_card(ver: dict):
    icon = {True: "<span class='pill-ok'>OK</span>",
            False: "<span class='pill-no'>falla</span>",
            None: "<span class='pill-na'>s/d</span>"}
    rows = [f"{icon[ok]} {_VERIFY_LABELS.get(k, k.replace('_', ' '))} "
            f"<span class='muted'>· {ver['detail'][k]}</span>"
            for k, ok in ver["checks"].items()]
    rec_color = {"trust": mock.C_PLANET, "review": "#d6a334", "reject": mock.C_FP}
    rec_txt = {"trust": "CONFIAR", "review": "REVISAR", "reject": "RECHAZAR"}
    rec = ver["recommendation"]
    st.markdown("<div class='card'><b>Verificador físico</b><br>" + "<br>".join(rows)
                + f"<br><br>recomendación: <b style='color:{rec_color[rec]}'>"
                f"{rec_txt.get(rec, rec.upper())}</b></div>", unsafe_allow_html=True)


def render_panel(a: dict):
    st.markdown("### Panel de análisis")
    render_verdict_card(a["classification"])
    if a.get("term_log"):
        render_model_terminal(a["term_log"])
    render_verify_card(a["verify"])
    figs = a["figures"]
    order = [("sky_map", "Mapa del cielo"), ("orbit_diagram", "Órbita estimada"),
             ("explain", "Curva y saliency (XAI)")]
    for key, cap in order:
        if key in figs and Path(figs[key]).exists():
            st.image(figs[key], caption=cap, width="stretch")
    with st.expander("Herramientas usadas (evidencia, guardado en agent/logs/)"):
        total = sum(c["latency_ms"] for c in a["tool_calls"])
        for c in a["tool_calls"]:
            st.markdown(f"<span class='toolchip'>{c['tool']} · {c['latency_ms']:.0f} ms</span>",
                        unsafe_allow_html=True)
        st.markdown(f"<br><span class='muted num'>latencia total {total:.0f} ms · "
                    f"{len(a['tool_calls'])} herramientas · sesión {a['session_id']}</span>",
                    unsafe_allow_html=True)


_TERM_COLOR = {"req": "#5aa7ff", "ok": "#3fb950", "warn": "#d6a334", "info": "#9db4cc"}


def render_model_terminal(logs: list[dict], title: str = "Servicio de inferencia · Mamba (Docker)"):
    """Renderiza el log HTTP del model_client como una terminal (evidencia de la
    llamada real al modelo). Si el servicio está caído, muestra la caída al mock."""
    if not logs:
        return
    body = "".join(
        f"<div><span class='term-ts'>{e['ts']}</span>  "
        f"<span style='color:{_TERM_COLOR.get(e['level'], '#9db4cc')}'>"
        f"{html.escape(e['line'])}</span></div>"
        for e in logs)
    st.markdown(f"<div class='term'><div class='term-bar'>{html.escape(title)}</div>"
                f"<div class='term-body'>{body}</div></div>", unsafe_allow_html=True)


def render_llm_status(res: dict):
    """Avisa quien respondio: el LLM real o el respaldo determinista (y por que)."""
    prov = provider_label(agent.mode)
    if res.get("llm_used"):
        st.caption(f"Respondido por {prov}")
        return
    reason = res.get("fallback_reason")
    if reason == "rate_limit":
        st.warning(f"{prov} alcanzó el límite de requests por minuto. Respondió el respaldo "
                   "determinista. Espera unos segundos y vuelve a intentar.")
    elif reason == "quota":
        st.error(f"La cuenta de {prov} no tiene crédito o agotó la cuota. Respondió el "
                 "respaldo. Revisa el billing de la API.")
    elif reason == "auth":
        st.error(f"La API key de {prov} es inválida o no tiene permiso. Respondió el respaldo.")
    elif agent.mode == "mock":
        st.caption("Modo respaldo determinista (sin LLM configurado en el .env)")
    else:
        st.caption("Respondió el respaldo determinista")


# --------------------------------------------------------------------------- #
# Flujo principal
# --------------------------------------------------------------------------- #
render_header()
render_picker()

_ph = ("Pregunta sobre el candidato (por ejemplo, dónde se ubica o si es habitable) "
       "o escribe Analiza TIC 12345")
prompt = st.chat_input(_ph)
if prompt:
    st.session_state.pending = prompt


def _progress_cb(status):
    """Callback de progreso: muestra cada herramienta y, en vivo, las líneas nuevas
    de la terminal del servicio Mamba (GET /health, POST /classify, latencias)."""
    seen = {"n": 0}

    def cb(tool, args, latency_ms):
        status.write(f"**{tool}**  ·  {latency_ms:.0f} ms")
        logs = model_client.get_log()
        for e in logs[seen["n"]:]:
            status.write(f"&nbsp;&nbsp;&nbsp;`{e['ts']}`  {e['line']}")
        seen["n"] = len(logs)
        time.sleep(0.08)
    return cb


col_chat, col_panel = st.columns([0.92, 1.08], gap="large")

with col_chat:
    st.markdown("### Conversación")
    if not st.session_state.messages:
        st.markdown("<div class='card'><span class='muted'>Hola. Elige un candidato "
                    "arriba y presiona Analizar. Después puedes <b>preguntarme lo que "
                    "quieras</b> sobre él: dónde se ubica, qué tan lejos está, qué tipo "
                    "de estrella tiene o si sería habitable.</span></div>",
                    unsafe_allow_html=True)
    for msg in st.session_state.messages:
        _av = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(msg["role"], avatar=_av):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if st.session_state.pending:
        query = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(query)

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            if is_analysis_command(query):
                model_client.reset_log()
                with st.status("Analizando candidato...", expanded=True) as status:
                    result = agent.run(query, progress_cb=_progress_cb(status))
                    status.update(label=f"Listo ({len(result['tool_calls'])} herramientas)",
                                  state="complete", expanded=False)
                term_log = model_client.get_log()
                render_model_terminal(term_log)
                st.markdown(result["report_md"], unsafe_allow_html=True)
                render_llm_status(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["report_md"]})
                if result["kind"] == "analysis":
                    result["term_log"] = term_log
                    st.session_state.analysis = result
                    tic = result.get("tic_id")
                    st.session_state.focus = {
                        "tic_id": tic,
                        "info": result.get("info"),
                        "classification": result.get("classification"),
                        "verify": result.get("verify"),
                        "star": mock.get_star_info(tic) if tic else None,
                    }
            else:
                # Si no se analizó nada aún, usa el candidato elegido en el selector.
                focus = st.session_state.focus
                if focus is None:
                    focus = _focus_from_tic(st.session_state.selected_tic)
                model_client.reset_log()
                with st.status("Pensando...", expanded=True) as status:
                    chat_res = agent.chat(query, history=st.session_state.messages[:-1],
                                          focus=focus, progress_cb=_progress_cb(status))
                    status.update(label="Listo", state="complete", expanded=False)
                render_model_terminal(model_client.get_log())
                st.markdown(chat_res["reply"])
                render_llm_status(chat_res)
                for _k, _p in chat_res.get("figures", {}).items():
                    if Path(_p).exists():
                        st.image(_p, width="stretch")
                st.session_state.messages.append(
                    {"role": "assistant", "content": chat_res["reply"]})

with col_panel:
    if st.session_state.analysis is not None:
        render_panel(st.session_state.analysis)
    else:
        render_empty_panel()
