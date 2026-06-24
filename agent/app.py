"""TOI Vetting Assistant — frontend Streamlit (Etapa 3).

UI de 2 columnas: chat (izquierda) + panel de análisis (derecha). Consume el
`Agent` (agent/llm.py), que orquesta las 7 tools y registra cada sesión en
agent/logs/. Modo automático: usa Claude si hay ANTHROPIC_API_KEY, si no corre en
modo mock determinista (sin costo). La UI es idéntica en ambos modos.

Correr desde la raíz del repo:
    python -m streamlit run agent/app.py --server.address 127.0.0.1
    # abrir http://127.0.0.1:8501
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Streamlit ejecuta este archivo con agent/ en sys.path, no la raíz del repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import mock
from agent.llm import Agent

# --------------------------------------------------------------------------- #
# Configuración + CSS
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="TOI Vetting Assistant", page_icon="🪐",
                   layout="wide", initial_sidebar_state="collapsed")

_CSS = """
<style>
  .card { background:#121829; border:1px solid #26324a; border-radius:14px;
          padding:16px 18px; margin-bottom:14px; }
  .verdict-planet { background:linear-gradient(135deg,#0d2818 0%,#121829 60%);
                    border:1px solid #1f6b45; }
  .verdict-fp { background:linear-gradient(135deg,#2a0f16 0%,#121829 60%);
                border:1px solid #7a2433; }
  .verdict-label { font-size:1.6rem; font-weight:800; letter-spacing:0.5px; margin:0; }
  .verdict-sub { color:#7d8aa0; font-size:0.85rem; margin-top:2px; }
  .probbar-track { height:14px; background:#1c2438; border-radius:8px; overflow:hidden;
                   margin:10px 0 4px 0; }
  .probbar-fill { height:100%; border-radius:8px; }
  .toolchip { display:inline-block; background:#18203a; border:1px solid #2b3a5c;
              color:#9fd0ff; border-radius:8px; padding:3px 9px; margin:2px 4px 2px 0;
              font-family:ui-monospace,monospace; font-size:0.78rem; }
  .pill-ok { color:#00ff88; font-weight:700; }
  .pill-no { color:#ff4466; font-weight:700; }
  .pill-na { color:#7d8aa0; }
  .badge { background:#18203a; border:1px solid #2b3a5c; color:#00d9ff;
           border-radius:999px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
  .muted { color:#7d8aa0; font-size:0.85rem; }
  h1,h2,h3 { letter-spacing:0.3px; }
  div.block-container { padding-top:2.2rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Estado de sesión
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_agent() -> Agent:
    return Agent(mode="auto")


def _init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("pending", None)


_init_state()
agent = get_agent()


# --------------------------------------------------------------------------- #
# Render: header
# --------------------------------------------------------------------------- #
def render_header():
    badge = mock.model_badge()
    mode_txt = "Claude (API)" if agent.mode == "claude" else "mock determinista"
    left, right = st.columns([3, 2])
    with left:
        st.markdown("## 🪐 TOI Vetting Assistant")
        st.markdown("<span class='muted'>Asistente de vetting de TESS Objects of "
                    "Interest — consume el modelo del proyecto como herramienta.</span>",
                    unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div style='text-align:right; padding-top:14px'>"
            f"<span class='badge'>modelo: {badge['name']}</span> "
            f"<span class='badge'>agente: {mode_txt}</span><br>"
            f"<span class='muted'>AUC val {badge['val_auc']:.3f} · "
            f"{badge['n_params']:,} params</span></div>",
            unsafe_allow_html=True)
    st.divider()


# --------------------------------------------------------------------------- #
# Render: panel derecho
# --------------------------------------------------------------------------- #
def render_empty_panel():
    st.markdown("### 📊 Panel de análisis")
    st.markdown("<div class='card'><span class='muted'>Pasá un TIC ID en el chat para "
                "ver el veredicto del modelo, el verificador físico y las "
                "visualizaciones (mapa del cielo, órbita estimada y curva con "
                "saliency).</span></div>", unsafe_allow_html=True)
    st.markdown("**Ejemplos para probar:**")
    c1, c2, c3 = st.columns(3)
    examples = [
        (c1, "🟢 CP · 79748331", "Analiza TIC 79748331"),
        (c2, "🔴 FP · 182943944", "Analiza TIC 182943944"),
        (c3, "🟡 PC · 341420329", "Analiza TIC 341420329"),
    ]
    for col, label, query in examples:
        if col.button(label, width="stretch"):
            st.session_state.pending = query
            st.rerun()


def render_verdict_card(cls: dict):
    is_planet = cls["prob_planeta"] >= 0.5
    klass = "verdict-planet" if is_planet else "verdict-fp"
    col = mock.C_PLANET if is_planet else mock.C_FP
    pct = cls["prob_planeta"] * 100
    st.markdown(
        f"<div class='card {klass}'>"
        f"<p class='verdict-label' style='color:{col}'>{cls['label']}</p>"
        f"<p class='verdict-sub'>probabilidad de planeta · confianza {cls['confianza']}</p>"
        f"<div class='probbar-track'><div class='probbar-fill' "
        f"style='width:{pct:.0f}%; background:{col}'></div></div>"
        f"<span class='muted'>{cls['prob_planeta']:.3f}</span></div>",
        unsafe_allow_html=True)


def render_verify_card(ver: dict):
    icon = {True: "<span class='pill-ok'>✓</span>",
            False: "<span class='pill-no'>✗</span>",
            None: "<span class='pill-na'>–</span>"}
    rows = [f"{icon[ok]} {k.replace('_', ' ')} <span class='muted'>· {ver['detail'][k]}</span>"
            for k, ok in ver["checks"].items()]
    rec_color = {"trust": mock.C_PLANET, "review": "#ffb84d", "reject": mock.C_FP}
    rec = ver["recommendation"]
    st.markdown("<div class='card'><b>🔬 Verificador físico</b><br>" + "<br>".join(rows)
                + f"<br><br>recomendación: <b style='color:{rec_color[rec]}'>"
                f"{rec.upper()}</b></div>", unsafe_allow_html=True)


def render_panel(a: dict):
    st.markdown("### 📊 Panel de análisis")
    render_verdict_card(a["classification"])
    render_verify_card(a["verify"])
    figs = a["figures"]
    order = [("sky_map", "Mapa del cielo"), ("orbit_diagram", "Órbita estimada"),
             ("explain", "Curva + saliency (XAI)")]
    for key, cap in order:
        if key in figs and Path(figs[key]).exists():
            st.image(figs[key], caption=cap, width="stretch")
    with st.expander("🔍 Tool calls (evidencia · logueado en agent/logs/)"):
        total = sum(c["latency_ms"] for c in a["tool_calls"])
        for c in a["tool_calls"]:
            st.markdown(f"<span class='toolchip'>{c['tool']} · {c['latency_ms']} ms</span>",
                        unsafe_allow_html=True)
        st.markdown(f"<br><span class='muted'>latencia total ≈ {total:.0f} ms · "
                    f"{len(a['tool_calls'])} tools · sesión {a['session_id']}</span>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Flujo principal
# --------------------------------------------------------------------------- #
render_header()

prompt = st.chat_input("Escribí un TIC ID (ej. 'Analiza TIC 79748331') o una consulta…")
if prompt:
    st.session_state.pending = prompt

col_chat, col_panel = st.columns([0.92, 1.08], gap="large")

with col_chat:
    st.markdown("### 💬 Conversación")
    if not st.session_state.messages:
        st.markdown("<div class='card'><span class='muted'>Hola 👋 Soy un asistente de "
                    "vetting de TOIs. Pasame un TIC ID y te digo si parece un planeta o "
                    "un falso positivo, con explicación y chequeos físicos.</span></div>",
                    unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🛰️" if msg["role"] == "assistant" else "🧑‍🔬"):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if st.session_state.pending:
        query = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🛰️"):
            with st.status("Analizando candidato…", expanded=True) as status:
                def _cb(tool, args, latency_ms):
                    status.write(f"🔧 `{tool}`  ·  {latency_ms:.0f} ms")
                    time.sleep(0.12)  # animación visible
                result = agent.run(query, progress_cb=_cb)
                status.update(label=f"Listo ✓ ({len(result['tool_calls'])} tools)",
                              state="complete", expanded=False)
            st.markdown(result["report_md"], unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": result["report_md"]})
        if result["kind"] == "analysis":
            st.session_state.analysis = result

with col_panel:
    if st.session_state.analysis is not None:
        render_panel(st.session_state.analysis)
    else:
        render_empty_panel()
