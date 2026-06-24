"""Logger estructurado de sesiones del agente.

Cada sesión se guarda como un JSON en agent/logs/<session_id>.json con el formato
de ETAPA3.md: query del usuario, secuencia de tool calls (con args, resumen y
latencia), informe final, totales. Es la **evidencia verificable** que pide el
rubric de Etapa 3 (10 pts de integración).

Uso:
    log = SessionLogger(user_query="Analiza TIC 79748331", mode="mock")
    log.add_tool_call("get_toi_info", {"tic_id": 79748331}, result, latency_ms)
    log.finish(final_report_md=report, total_tokens=0)
    path = log.save()
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _summarize(result: dict) -> str:
    """Resumen corto y legible del resultado de una tool (para el log)."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"error: {result['error']}"
    if "disposition" in result:
        return f"disp={result['disposition']}, P={result.get('period_days')}"
    if "prob_planeta" in result:
        return f"prob={result['prob_planeta']}, label={result.get('label')}"
    if "all_passed" in result:
        return f"all_passed={result['all_passed']}, rec={result.get('recommendation')}"
    if "agree" in result:
        return f"agree={result['agree']}, flag={result.get('flag_for_review')}"
    if "png_path" in result:
        return f"figura: {Path(result['png_path']).name}"
    if "available" in result:
        return f"available={result['available']} ({result.get('source')})"
    return ", ".join(f"{k}={v}" for k, v in list(result.items())[:3])


class SessionLogger:
    def __init__(self, user_query: str, mode: str = "mock",
                 logs_dir: Path | None = None):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + uuid.uuid4().hex[:6]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.user_query = user_query
        self.mode = mode
        self.tool_calls: list[dict] = []
        self.final_report_md = ""
        self.total_tokens = 0
        self._t_start = time.perf_counter()
        self._logs_dir = logs_dir or LOGS_DIR

    def add_tool_call(self, tool: str, args: dict, result: dict, latency_ms: float):
        self.tool_calls.append({
            "tool": tool,
            "args": args,
            "result_summary": _summarize(result),
            "latency_ms": latency_ms,
        })

    def finish(self, final_report_md: str, total_tokens: int = 0):
        self.final_report_md = final_report_md
        self.total_tokens = total_tokens

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "user_query": self.user_query,
            "tool_calls": self.tool_calls,
            "final_report_md": self.final_report_md,
            "total_tokens": self.total_tokens,
            "n_tool_calls": len(self.tool_calls),
            "total_latency_ms": round(sum(c["latency_ms"] for c in self.tool_calls), 1),
            "wall_time_ms": round((time.perf_counter() - self._t_start) * 1000, 1),
        }

    def save(self) -> Path:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        path = self._logs_dir / f"{self.session_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path
