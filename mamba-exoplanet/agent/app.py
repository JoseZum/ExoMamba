"""TOI Vetting Assistant — frontend Streamlit (Etapa 3).

UI de 2 columnas: chat (izquierda) + panel de análisis (derecha). Hoy corre contra
el backend MOCK (agent/mock.py): datos reales del catálogo TOI, verifier físico real,
predicción del modelo simulada. Sin API key ni modelo cargado.

Correr desde la raíz del repo:
    streamlit run agent/app.py

Conexión futura (semana 11): reemplazar `agent.mock.classify` por el forward del
Mamba locked y enchufar el loop de tool calling de Claude en agent/llm.py. La UI
no cambia.
"""

from __future__ import annotations

import re
import time

import streamlit as st

from agent import mock

# --------------------------------------------------------------------------- #
# Configuración de página + CSS
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="TOI Vetting Assistant",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CSS = """
<style>
  /* Tarjetas del panel */
  .card {
    background: #121829;
    border: 1px solid #26324a;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
  }
  .verdict-planet {
    background: linear-gradient(135deg, #0d2818 0%, #121829 60%);
    border: 1px solid #1f6b45;
  }
  .verdict-fp {
    background: linear-gradient(135deg, #2a0f16 0%, #121829 60%);
    border: 1px solid #7a2433;
  }
  .verdict-label {
    font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px; margin: 0;
  }
  .verdict-sub { color: #7d8aa0; font-size: 0.85rem; margin-top: 2px; }
  /* Barra de probabilidad */
  .probbar-track {
    height: 14px; background: #1c2438; border-radius: 8px; overflow: hidden;
    margin: 10px 0 4px 0;
  }
  .probbar-fill { height: 100%; border-radius: 8px; }
  /* Chips de tool calls */
  .toolchip {
    display: inline-block; background: #18203a; border: 1px solid #2b3a5c;
    color: #9fd0ff; border-radius: 8px; padding: 3px 9px; margin: 2px 4px 2px 0;
    font-family: ui-monospace, monospace; font-size: 0.78rem;
  }
  .pill-ok  { color: #00ff88; font-weight: 700; }
  .pill-no  { color: #ff4466; font-weight: 700; }
  .pill-na  { color: #7d8aa0; }
  .badge {
    background: #18203a; border: 1px solid #2b3a5c; color: #00d9ff;
    border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600;
  }
  .muted { color: #7d8aa0; font-size: 0.85rem; }
  h1, h2, h3 { letter-spacing: 0.3px; }
  /* Compactar el header */
  div.block-container { padding-top: 2.2rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Estado de sesión
# --------------------------------------------------------------------------- #
def _init_state():
    st.session_state.setdefault("messages", [])      # [{role, content}]
    st.session_state.setdefault("analysis", None)     # dict del último TIC analizado
    st.session_state.setdefault("tool_log", [])       # tool calls del último análisis
    st.session_state.setdefault("session_count", 0)   # nº de análisis en la sesión
    st.session_state.setdefault("pending", None)      # query a procesar este run


_init_state()


# --------------------------------------------------------------------------- #
# Parsing del input  (router de intención)
# --------------------------------------------------------------------------- #
def parse_tic(text: str) -> int | None:
    """Extrae un TIC ID del texto. Acepta 'TIC 123', 'tic123', '123', etc."""
    m = re.search(r"tic[\s_-]*?(\d{3,})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # número suelto largo (evita capturar "700" de "TOI-700" como TIC)
    nums = re.findall(r"\b(\d{4,})\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


# --------------------------------------------------------------------------- #
# Pipeline de análisis (anima tool calls y arma el informe)
# --------------------------------------------------------------------------- #
def run_analysis(tic_id: int, info: dict, status):
    """Ejecuta el pipeline con animación en `status`. Devuelve el dict de análisis."""
    classification = mock.classify(tic_id, info)
    plan = mock.pipeline_plan(tic_id, info, classification)
    tool_log = []

    for step in plan:
        status.write(
            f"🔧 `{step['tool']}` — {step['summary']}  ·  {step['latency_ms']} ms"
        )
        tool_log.append(step)
        time.sleep(min(step["latency_ms"] / 1000.0, 0.45))  # animación visible

    verify = mock.verify_prediction(info)
    comparison = mock.compare_with_disposition(info, classification)
    figs = {
        "sky": mock.make_sky_map(tic_id, info),
        "orbit": mock.make_orbit_diagram(tic_id, info),
        "lightcurve": mock.make_lightcurve_saliency(tic_id, info),
    }
    return {
        "tic_id": tic_id,
        "info": info,
        "classification": classification,
        "verify": verify,
        "comparison": comparison,
        "figs": figs,
        "tool_log": tool_log,
    }


def build_report_md(a: dict) -> str:
    """Informe markdown que el agente 'entrega' (lo que va al chat)."""
    info, cls, ver, cmp = a["info"], a["classification"], a["verify"], a["comparison"]
    toi = info.get("toi")
    title = f"TIC {a['tic_id']}" + (f" · TOI-{toi}" if toi else "")

    rec_map = {"trust": "confiar", "review": "revisar", "reject": "rechazar"}
    ver_icon = "✅ todos los chequeos pasan" if ver["all_passed"] else (
        f"⚠️ falla(n): {', '.join(ver['failed_checks'])}"
    )
    agree_line = (
        "✅ coincide con la disposición oficial NASA"
        if cmp["agree"]
        else f"🚩 **DISCREPANCIA**: el modelo dice {cmp['model_label']}, NASA dice {cmp['nasa_label']}"
    )

    lines = [
        f"### Informe — {title}",
        "",
        f"**Veredicto del modelo:** {cls['label']}",
        f"**Probabilidad de planeta:** {cls['prob_planeta']:.3f}  ·  confianza {cls['confianza']}",
        "",
        f"**Verificador físico:** {ver_icon}  → recomendación: *{rec_map[ver['recommendation']]}*",
        f"**Contraste con NASA:** {agree_line}",
        "",
        "**Explicabilidad:** la saliency se concentra en el dip central de la curva "
        "phase-folded (ver panel derecho). En un planeta real la atribución se alinea "
        "con el tránsito; en un falso positivo tiende a dispersarse.",
    ]
    if cmp["flag_for_review"]:
        lines += ["", "> 🚩 Caso marcado para **revisión humana** por discrepancia "
                  "modelo/NASA. El agente no emite veredicto final solo."]
    lines += ["", f"<span class='muted'>Sesión #{st.session_state.session_count:03d} · "
              "backend mock (predicción simulada, datos de catálogo reales)</span>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Render: header
# --------------------------------------------------------------------------- #
def render_header():
    badge = mock.model_badge()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("## 🪐 TOI Vetting Assistant")
        st.markdown(
            "<span class='muted'>Asistente de vetting de TESS Objects of Interest — "
            "consume el modelo del proyecto como herramienta.</span>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div style='text-align:right; padding-top:14px'>"
            f"<span class='badge'>modelo: {badge['name']}</span><br>"
            f"<span class='muted'>AUC val {badge['val_auc']:.3f} · "
            f"{badge['n_params']:,} params</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()


# --------------------------------------------------------------------------- #
# Render: panel derecho
# --------------------------------------------------------------------------- #
def render_empty_panel():
    st.markdown("### 📊 Panel de análisis")
    st.markdown(
        "<div class='card'><span class='muted'>Pasá un TIC ID en el chat para ver "
        "el veredicto del modelo, el verificador físico y las visualizaciones "
        "(mapa del cielo, órbita estimada y curva con saliency).</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Ejemplos para probar:**")
    c1, c2, c3 = st.columns(3)
    examples = [
        (c1, "🟢 CP · 79748331", "Analiza TIC 79748331"),
        (c2, "🔴 FP · 182943944", "Analiza TIC 182943944"),
        (c3, "🟡 PC · 341420329", "Analiza TIC 341420329"),
    ]
    for col, label, query in examples:
        if col.button(label, use_container_width=True):
            st.session_state.pending = query
            st.rerun()


def render_verdict_card(a: dict):
    cls = a["classification"]
    is_planet = cls["prob_planeta"] >= 0.5
    klass = "verdict-planet" if is_planet else "verdict-fp"
    col = mock.C_PLANET if is_planet else mock.C_FP
    pct = cls["prob_planeta"] * 100
    st.markdown(
        f"<div class='card {klass}'>"
        f"<p class='verdict-label' style='color:{col}'>{cls['label']}</p>"
        f"<p class='verdict-sub'>probabilidad de planeta · confianza {cls['confianza']}</p>"
        f"<div class='probbar-track'>"
        f"<div class='probbar-fill' style='width:{pct:.0f}%; background:{col}'></div></div>"
        f"<span class='muted'>{cls['prob_planeta']:.3f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_verify_card(a: dict):
    ver = a["verify"]
    rows = []
    icon = {True: "<span class='pill-ok'>✓</span>",
            False: "<span class='pill-no'>✗</span>",
            None: "<span class='pill-na'>–</span>"}
    for key, ok in ver["checks"].items():
        label = key.replace("_", " ")
        rows.append(f"{icon[ok]} {label} <span class='muted'>· {ver['detail'][key]}</span>")
    rec_color = {"trust": mock.C_PLANET, "review": "#ffb84d", "reject": mock.C_FP}
    rec = ver["recommendation"]
    st.markdown(
        "<div class='card'><b>🔬 Verificador físico</b><br>"
        + "<br>".join(rows)
        + f"<br><br>recomendación: <b style='color:{rec_color[rec]}'>{rec.upper()}</b></div>",
        unsafe_allow_html=True,
    )


def render_panel(a: dict):
    st.markdown("### 📊 Panel de análisis")
    render_verdict_card(a)
    render_verify_card(a)
    st.pyplot(a["figs"]["sky"], use_container_width=True)
    st.pyplot(a["figs"]["orbit"], use_container_width=True)
    st.pyplot(a["figs"]["lightcurve"], use_container_width=True)

    with st.expander("🔍 Tool calls (debug / evidencia para logs)"):
        total = sum(s["latency_ms"] for s in a["tool_log"])
        for s in a["tool_log"]:
            st.markdown(
                f"<span class='toolchip'>{s['tool']} · {s['latency_ms']} ms</span>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<br><span class='muted'>latencia total ≈ {total} ms · "
                    f"{len(a['tool_log'])} tools</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Flujo principal
# --------------------------------------------------------------------------- #
render_header()

# Capturar input (se ancla abajo visualmente, pero el valor llega arriba del run)
prompt = st.chat_input("Escribí un TIC ID (ej. 'Analiza TIC 79748331') o una consulta…")
if prompt:
    st.session_state.pending = prompt

col_chat, col_panel = st.columns([0.92, 1.08], gap="large")

with col_chat:
    st.markdown("### 💬 Conversación")
    if not st.session_state.messages:
        st.markdown(
            "<div class='card'><span class='muted'>Hola 👋 Soy un asistente de vetting "
            "de TOIs. Pasame un TIC ID y te digo si parece un planeta o un falso "
            "positivo, con explicación y chequeos físicos.</span></div>",
            unsafe_allow_html=True,
        )
    # Historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🛰️" if msg["role"] == "assistant" else "🧑‍🔬"):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Procesar pendiente (este es el único punto que muta estado)
    if st.session_state.pending:
        query = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(query)

        tic = parse_tic(query)
        with st.chat_message("assistant", avatar="🛰️"):
            if tic is None:
                # S6: off-topic / malformado → responder dentro de scope, sin alucinar
                reply = (
                    "Solo analizo **TOIs por su TIC ID**. Pasame algo como "
                    "`Analiza TIC 79748331` y corro el modelo + los chequeos físicos. "
                    "No tengo herramientas para responder fuera de ese alcance."
                )
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                info = mock.get_toi_info(tic)
                if info is None:
                    # S4: TIC inexistente → no alucina
                    reply = (
                        f"No tengo registro de **TIC {tic}** en el catálogo TOI cargado "
                        f"(~{len(mock.load_catalog()):,} objetos con disposición). "
                        "Verificá el ID o pasame otro candidato."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.session_count += 1
                    with st.status("Analizando candidato…", expanded=True) as status:
                        analysis = run_analysis(tic, info, status)
                        status.update(label="Análisis completo ✓", state="complete",
                                      expanded=False)
                    report = build_report_md(analysis)
                    st.markdown(report, unsafe_allow_html=True)
                    st.session_state.analysis = analysis
                    st.session_state.messages.append(
                        {"role": "assistant", "content": report}
                    )

with col_panel:
    if st.session_state.analysis is not None:
        render_panel(st.session_state.analysis)
    else:
        render_empty_panel()
