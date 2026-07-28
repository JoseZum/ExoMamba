"""El registro de tools del agente y su system prompt no pueden desincronizarse.

Esto no es cosmético. `TOOL_SCHEMAS` es lo que el LLM recibe como herramientas
invocables, y `prompts/system.md` es lo que le dice cuándo usar cada una. Si una
tool está en el registro pero no en el prompt, el modelo puede llamarla sin
ninguna instrucción de contexto; si está en el prompt pero no en el registro, el
modelo intentará una llamada que falla. Ya pasó una vez: `get_star_info` estuvo
registrada y ausente del prompt, degradando justo el modo conversacional.

Los archivos se leen como texto (`ast` para el .py) en vez de importarse: `agent/`
tiene dependencias propias — Streamlit, el cliente HTTP del servicio de inferencia —
que no están instaladas en el entorno de tests ni en CI.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PY = REPO_ROOT / "agent" / "tools.py"
SYSTEM_MD = REPO_ROOT / "agent" / "prompts" / "system.md"


def _tools_declaradas_en_el_registro() -> list[str]:
    """Extrae los `name` de TOOL_SCHEMAS parseando el AST, sin ejecutar el módulo."""
    if not TOOLS_PY.exists():
        pytest.skip(f"No existe {TOOLS_PY}")
    arbol = ast.parse(TOOLS_PY.read_text(encoding="utf-8"))

    for nodo in arbol.body:
        if not isinstance(nodo, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "TOOL_SCHEMAS" for t in nodo.targets):
            continue
        nombres = []
        for entrada in nodo.value.elts:  # type: ignore[attr-defined]
            for clave, valor in zip(entrada.keys, entrada.values, strict=True):
                if getattr(clave, "value", None) == "name":
                    nombres.append(valor.value)
        return nombres

    pytest.fail("No se encontró la asignación de TOOL_SCHEMAS en agent/tools.py")
    return []


def _tools_documentadas_en_el_prompt() -> list[str]:
    """Extrae los nombres de la lista numerada bajo '## Herramientas disponibles'."""
    if not SYSTEM_MD.exists():
        pytest.skip(f"No existe {SYSTEM_MD}")
    texto = SYSTEM_MD.read_text(encoding="utf-8")

    seccion = re.search(
        r"##\s*Herramientas disponibles\s*(.*?)(?=\n##\s|\Z)", texto, re.DOTALL
    )
    assert seccion, "system.md no tiene una sección '## Herramientas disponibles'"
    return re.findall(r"^\s*\d+\.\s*`(\w+)\(", seccion.group(1), re.MULTILINE)


def test_el_prompt_documenta_exactamente_las_tools_registradas() -> None:
    registro = set(_tools_declaradas_en_el_registro())
    prompt = set(_tools_documentadas_en_el_prompt())

    faltan_en_prompt = registro - prompt
    sobran_en_prompt = prompt - registro

    assert not faltan_en_prompt, (
        f"Tools invocables que el system prompt no documenta: {sorted(faltan_en_prompt)}. "
        "El LLM puede llamarlas sin saber cuándo."
    )
    assert not sobran_en_prompt, (
        f"Tools que el prompt promete pero no existen en TOOL_SCHEMAS: "
        f"{sorted(sobran_en_prompt)}. El LLM intentará llamadas que fallan."
    )


def test_no_hay_tools_duplicadas_en_el_registro() -> None:
    nombres = _tools_declaradas_en_el_registro()
    assert len(nombres) == len(set(nombres)), f"nombres repetidos en TOOL_SCHEMAS: {nombres}"


def test_cada_tool_registrada_tiene_su_funcion() -> None:
    """Cada entrada de TOOL_SCHEMAS debe corresponder a una función del módulo."""
    arbol = ast.parse(TOOLS_PY.read_text(encoding="utf-8"))
    funciones = {n.name for n in arbol.body if isinstance(n, ast.FunctionDef)}
    for nombre in _tools_declaradas_en_el_registro():
        assert nombre in funciones, f"TOOL_SCHEMAS declara '{nombre}' pero no existe la función"
