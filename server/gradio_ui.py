"""
Professional operations UI for LLMFleet-SRE.

This is intentionally standalone so it does not affect evaluator-critical
server behavior. It talks to an already running OpenEnv server via HTTP.

Run:
  python -m llmfleet_sre.server.gradio_ui

Environment variables:
  UI_ENV_BASE_URL   Target OpenEnv API base URL (default: http://127.0.0.1:7860)
  UI_HOST           Gradio host (default: 0.0.0.0)
  UI_PORT           Gradio port (default: 7861)
"""

from __future__ import annotations

import json
import ast
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gradio as gr
import httpx


UI_ENV_BASE_URL = os.getenv("UI_ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "7861"))
UI_LLM_PROVIDER = os.getenv("UI_LLM_PROVIDER", "groq")
UI_LLM_API_BASE = os.getenv("UI_LLM_API_BASE", "https://api.groq.com/openai/v1")
UI_LLM_MODEL = os.getenv("UI_LLM_MODEL", "llama-3.1-8b-instant")
README_PATH = Path(__file__).resolve().parent.parent / "README.md"
HTTP_SESSION = httpx.Client(timeout=25.0)
SYNC_ENV = None
SYNC_ENV_BASE_URL = None


LLM_SYSTEM_PROMPT = """You are an SRE assistant for an LLM serving cluster.
Given the current observation, propose exactly one next action.

Return only a JSON object with this schema:
{
    "action_type": "route_batch|load_model|evict_model|restart_node|noop",
    "target_node": "node_a|node_b|node_c|null",
    "model_name": "string|null",
    "request_ids": ["req_id"]
}

Rules:
- Prefer serving premium requests quickly.
- Only restart nodes in oom_crashed status.
- Use route_batch with real request_ids present in queue.
- Avoid invalid combinations.
Do not include markdown fences or explanations.
"""

# Allow running as script: `python server/gradio_ui.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


TASK_META = [
    {
        "name": "easy",
        "difficulty": "easy",
        "description": "Route 5 queued chat requests to a node that already has the model loaded.",
        "arc": "Core",
    },
    {
        "name": "medium",
        "difficulty": "medium",
        "description": "Recover an OOM-crashed node and clear a backing-up request queue under latency pressure.",
        "arc": "Core",
    },
    {
        "name": "hard",
        "difficulty": "hard",
        "description": "Evict chat models, load a code model, and serve a mixed premium/best-effort queue.",
        "arc": "Core",
    },
    {
        "name": "loghaul",
        "difficulty": "hard",
        "description": "Sustain cluster performance across a 50-step episode with a quiet-to-spike-to-quiet traffic shift.",
        "arc": "Extended",
    },
]


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --ink: #142033;
    --muted: #4f647f;
    --canvas-a: #f7fbff;
    --canvas-b: #e7f1ff;
    --panel-bg: rgba(255, 255, 255, 0.82);
    --line: #b8cfe9;
    --brand: #0f6cbd;
    --brand-strong: #084a86;
    --accent: #0a8f68;
    --warn: #b46900;
    --danger: #c42a39;
    --shadow: 0 14px 30px rgba(15, 64, 120, 0.12);
}

.gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink);
  background:
        radial-gradient(circle at 10% 5%, rgba(25, 108, 200, 0.11), transparent 32%),
        radial-gradient(circle at 90% 15%, rgba(14, 143, 104, 0.08), transparent 30%),
        linear-gradient(140deg, var(--canvas-a), var(--canvas-b)) !important;
    min-height: 100vh;
}

.gradio-container .form, .gradio-container input, .gradio-container textarea, .gradio-container select, .gradio-container .box {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

.hero-card {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: linear-gradient(120deg, rgba(255,255,255,0.95), rgba(241,247,255,0.95));
    box-shadow: var(--shadow);
    padding: 18px 20px;
    margin-bottom: 14px;
}

.hero-title {
    letter-spacing: 0.01em;
    font-size: 34px;
    line-height: 1.15;
  margin: 0;
    color: var(--brand-strong);
    font-weight: 800;
}

.hero-sub {
  margin-top: 6px;
    font-weight: 600;
    font-size: 14px;
    color: var(--muted);
}

.panel {
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: var(--shadow);
  padding: 12px;
}

.panel-obs {
    background: var(--panel-bg);
    color: var(--ink);
}

.panel-log {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(255,255,255,0.9);
    padding: 10px;
    box-shadow: inset 0 0 0 1px rgba(15, 108, 189, 0.05);
}

.pill {
  display: inline-block;
    border: 1px solid var(--line);
  border-radius: 999px;
    padding: 4px 10px;
  font-size: 12px;
    font-weight: 600;
  margin-right: 6px;
  margin-bottom: 6px;
    background: rgba(255, 255, 255, 0.9);
    color: var(--brand-strong);
}

.pill-easy { background: #ecfcf5; border-color: #8bd9b8; color: var(--accent); }
.pill-medium { background: #fff7e6; border-color: #efd08f; color: var(--warn); }
.pill-hard { background: #fff0f2; border-color: #f1b2b8; color: var(--danger); }

.task-card {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.9);
  padding: 12px;
  margin-bottom: 10px;
    box-shadow: var(--shadow);
}

.task-title {
    font-weight: 700;
  font-size: 14px;
  margin-bottom: 6px;
    color: var(--brand-strong);
}

.task-desc {
  font-size: 12px;
  line-height: 1.6;
    color: var(--muted);
}

.logline {
  font-size: 12px;
  margin-bottom: 4px;
}

.logline-ok { color: var(--accent); }
.logline-err { color: var(--danger); }
.logline-step { color: var(--brand); }
.logline-debug { color: #5f7894; font-size: 0.9em; }

.flow-assist {
        border: 1px dashed var(--brand);
    border-radius: 12px;
        background: rgba(15, 108, 189, 0.06);
    padding: 10px;
    margin: 8px 0 14px 0;
}

.flow-title {
        font-weight: 700;
    font-size: 12px;
        color: var(--brand-strong);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.flow-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.flow-step {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255,255,255,0.9);
    color: var(--ink);
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(15, 64, 120, 0.08);
}

.flow-arrow {
    color: var(--brand);
    font-weight: 900;
    font-size: 16px;
    animation: pulse-arrow 1.2s ease-in-out infinite;
}

.readme-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.readme-card {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(255,255,255,0.9);
    padding: 10px;
    box-shadow: 0 6px 14px rgba(15, 64, 120, 0.1);
}

.readme-card h4 {
    margin: 0 0 6px 0;
    color: var(--brand-strong);
    font-size: 13px;
}

.readme-card p {
    margin: 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
}

@keyframes pulse-arrow {
    0% { opacity: 0.45; transform: translateX(0); }
    50% { opacity: 1; transform: translateX(3px); }
    100% { opacity: 0.45; transform: translateX(0); }
}

button.primary, button[class*="primary"] {
    border: 1px solid var(--brand) !important;
    box-shadow: 0 10px 20px rgba(15, 108, 189, 0.2) !important;
    font-weight: 700 !important;
    background: linear-gradient(180deg, #2185d0, var(--brand)) !important;
    color: #ffffff !important;
}
button.primary:hover, button[class*="primary"]:hover {
    background: linear-gradient(180deg, #196fb2, #0b589a) !important;
    color: #ffffff !important;
}

button.secondary, button[class*="secondary"] {
    border: 1px solid var(--line) !important;
    box-shadow: 0 8px 14px rgba(15, 64, 120, 0.1) !important;
    font-weight: 700 !important;
    background: #ffffff !important;
    color: var(--brand-strong) !important;
}
button.secondary:hover, button[class*="secondary"]:hover {
    background: #f1f7ff !important;
}

.gradio-container .tabs,
.gradio-container .tabs *,
.gradio-container .tab-nav,
.gradio-container .tab-nav *,
.gradio-container [role="tab"],
.gradio-container [role="tab"] *,
.gradio-container [data-testid="tab"],
.gradio-container [data-testid="tab"] * {
    color: var(--brand-strong) !important;
}

.gradio-container [role="tab"],
.gradio-container [data-testid="tab"] {
    background: rgba(255, 255, 255, 0.82) !important;
    border: 1px solid var(--line) !important;
    border-bottom: none !important;
}

.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [data-testid="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: var(--brand) !important;
    box-shadow: 0 -2px 0 var(--brand) inset !important;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 28px;
    }
}
"""


@dataclass
class EpisodeState:
    active: bool
    task_name: Optional[str]
    step_idx: int
    total_reward: float
    done: bool
    log_lines: List[str]
    last_obs: Dict[str, Any]


def _state_to_dict(state: EpisodeState) -> Dict[str, Any]:
    return {
        "active": state.active,
        "task_name": state.task_name,
        "step_idx": state.step_idx,
        "total_reward": state.total_reward,
        "done": state.done,
        "log_lines": state.log_lines,
        "last_obs": state.last_obs,
    }


def _dict_to_state(data: Dict[str, Any]) -> EpisodeState:
    return EpisodeState(
        active=bool(data.get("active", False)),
        task_name=data.get("task_name"),
        step_idx=int(data.get("step_idx", 0)),
        total_reward=float(data.get("total_reward", 0.0)),
        done=bool(data.get("done", False)),
        log_lines=list(data.get("log_lines", [])),
        last_obs=data.get("last_obs") if isinstance(data.get("last_obs"), dict) else {},
    )


def _default_state() -> Dict[str, Any]:
    return _state_to_dict(EpisodeState(False, None, 0, 0.0, False, [], {}))


def _difficulty_badge(difficulty: str) -> str:
    klass = {
        "easy": "pill pill-easy",
        "medium": "pill pill-medium",
        "hard": "pill pill-hard",
    }.get(difficulty, "pill")
    return f'<span class="{klass}">{difficulty}</span>'


def _task_catalog_html() -> str:
    chunks = []
    for task in TASK_META:
        chunks.append(
            """
            <div class="task-card">
              <div class="task-title">{name} <span class="pill">{arc}</span> {badge}</div>
              <div class="task-desc">{desc}</div>
            </div>
            """.format(
                name=task["name"],
                arc=task["arc"],
                badge=_difficulty_badge(task["difficulty"]),
                desc=task["description"],
            )
        )
    return "\n".join(chunks)


def _render_log(lines: List[str]) -> str:
    if not lines:
        return '<div class="panel panel-log"><div class="logline">No actions yet. Reset a task to begin.</div></div>'
    return '<div class="panel panel-log">' + "\n".join(lines[-120:]) + "</div>"


def _json_pretty(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, ensure_ascii=True)
    except Exception:
        return str(data)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    # Direct JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Handle fenced markdown blocks.
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    # Last resort: parse from first { to last }.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _extract_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                # OpenAI-style content parts: {"type":"text","text":"..."}
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return str(content)


def _extract_action_from_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # First try tool call args if the model responded with function/tool format.
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        fn = first.get("function", {}) if isinstance(first, dict) else {}
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str):
            parsed = _extract_json_object(args)
            if isinstance(parsed, dict):
                return parsed

    text = _extract_message_text(message)
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        return parsed

    # Fallback for pseudo-JSON dict output with single quotes.
    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, dict):
            return literal
    except Exception:
        return None

    return None


def _readme_markdown() -> str:
    try:
        return README_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "# README\n\n"
            "Could not load README.md from the project root.\n\n"
            "Expected path: `llmfleet_sre/README.md`"
        )


def _readme_overview_html() -> str:
        return """
        <div class="hero-card">
            <h1 class="hero-title">LLMFleet-SRE // Field Guide</h1>
            <div class="hero-sub">What this environment does, why it matters, and how to run it fast.</div>
        </div>
        <div class="flow-assist">
            <div class="flow-title">Quick Start Order</div>
            <div class="flow-row">
                <div class="flow-step">1) Start API (7860)</div>
                <div class="flow-arrow">-></div>
                <div class="flow-step">2) Open UI (7861)</div>
                <div class="flow-arrow">-></div>
                <div class="flow-step">3) Probe</div>
                <div class="flow-arrow">-></div>
                <div class="flow-step">4) Reset Task</div>
                <div class="flow-arrow">-></div>
                <div class="flow-step">5) Step Actions</div>
            </div>
        </div>
        <div class="readme-grid">
            <div class="readme-card">
                <h4>Environment</h4>
                <p>3-node simulated GPU inference fleet. Manage model loading, routing, eviction, and recovery under SLA pressure.</p>
            </div>
            <div class="readme-card">
                <h4>Tasks</h4>
                <p>Four canonical tasks: easy, medium, hard, and longhaul. Longhaul runs 50-step horizon.</p>
            </div>
            <div class="readme-card">
                <h4>Actions</h4>
                <p>route_batch, load_model, evict_model, restart_node, noop. One action per step.</p>
            </div>
            <div class="readme-card">
                <h4>Reward Shape</h4>
                <p>Throughput bonus, queue latency penalty, SLA penalty, idle/cost penalties, and OOM crash penalty.</p>
            </div>
            <div class="readme-card">
                <h4>Validation-Safe</h4>
                <p>This UI is optional for local ops. Evaluator-critical flow remains server + inference contract.</p>
            </div>
            <div class="readme-card">
                <h4>Tip</h4>
                <p>If episode is done, click Reset before next step. UI now enforces this guard automatically.</p>
            </div>
        </div>
        """


def _safe_get(url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = HTTP_SESSION.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _safe_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = HTTP_SESSION.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def _get_sync_env(base_url: str):
    """Return a persistent sync client bound to base_url."""
    global SYNC_ENV, SYNC_ENV_BASE_URL
    base = base_url.rstrip("/")

    if SYNC_ENV is not None and SYNC_ENV_BASE_URL != base:
        try:
            SYNC_ENV.close()
        except Exception:
            pass
        SYNC_ENV = None
        SYNC_ENV_BASE_URL = None

    if SYNC_ENV is None:
        module = __import__("llmfleet_sre.client", fromlist=["LLMFleetSreEnv"])
        LLMFleetSreEnv = getattr(module, "LLMFleetSreEnv")

        SYNC_ENV = LLMFleetSreEnv(base_url=base).sync()
        SYNC_ENV.connect()
        SYNC_ENV_BASE_URL = base

    return SYNC_ENV


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return {}


def _status_header(state: EpisodeState) -> str:
    status = "ACTIVE" if state.active and not state.done else ("DONE" if state.done else "IDLE")
    return (
        f'<div class="pill">task={state.task_name or "none"}</div>'
        f'<div class="pill">status={status}</div>'
        f'<div class="pill">step={state.step_idx}</div>'
        f'<div class="pill">reward={state.total_reward:.2f}</div>'
    )


def _pick_restart_node(task_name: Optional[str], obs: Dict[str, Any]) -> str:
    nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
    for node_id, node in nodes.items():
        if isinstance(node, dict) and node.get("status") == "oom_crashed":
            return str(node_id)
    if task_name == "medium":
        return "node_b"
    return "node_a"


def _pick_route_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    queue = obs.get("request_queue", []) if isinstance(obs, dict) else []
    nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
    request_ids = [str(req.get("request_id")) for req in queue if isinstance(req, dict) and req.get("request_id")]
    target_node = "node_a"

    if queue and isinstance(queue[0], dict):
        required_model = queue[0].get("required_model")
        if required_model:
            for node_id, node in nodes.items():
                if not isinstance(node, dict):
                    continue
                if node.get("status") == "healthy" and required_model in (node.get("loaded_models") or []):
                    target_node = str(node_id)
                    break

    return {
        "action_type": "route_batch",
        "target_node": target_node,
        "model_name": None,
        "request_ids": request_ids,
    }


def _find_node_for_model(obs: Dict[str, Any], model_name: str) -> Optional[str]:
    nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
    # Model sizes aligned with environment catalogue.
    model_vram = {
        "llama3-8b-chat": 18,
        "llama3-70b-chat": 45,
        "codellama-34b": 35,
        "mistral-7b-sum": 16,
        "code-lora-adapter": 8,
    }
    needed = model_vram.get(model_name, 0)

    best_node = None
    best_free = -1
    for node_id, node in nodes.items():
        if not isinstance(node, dict) or node.get("status") != "healthy":
            continue
        loaded = node.get("loaded_models") or []
        if model_name in loaded:
            return str(node_id)

        total = int(node.get("vram_total_gb", 0) or 0)
        used = int(node.get("vram_used_gb", 0) or 0)
        free = total - used
        if free >= needed and free > best_free:
            best_node = str(node_id)
            best_free = free

    return best_node


def combo_action(combo: str, task_name: str, state_dict: Dict[str, Any]) -> str:
    state = _dict_to_state(state_dict)
    obs = state.last_obs if isinstance(state.last_obs, dict) else {}
    queue = obs.get("request_queue", []) if isinstance(obs, dict) else []

    if combo == "easy":
        if queue:
            return _json_pretty(_pick_route_action(obs))
        return _json_pretty({"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []})

    if combo == "medium":
        restart_node = _pick_restart_node(task_name, obs)
        nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
        node = nodes.get(restart_node, {}) if isinstance(nodes, dict) else {}
        if isinstance(node, dict) and node.get("status") == "oom_crashed":
            return _json_pretty({"action_type": "restart_node", "target_node": restart_node, "model_name": None, "request_ids": []})
        if queue:
            return _json_pretty(_pick_route_action(obs))
        return _json_pretty({"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []})

    if combo == "hard":
        target = _find_node_for_model(obs, "codellama-34b")
        if target:
            nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
            node = nodes.get(target, {}) if isinstance(nodes, dict) else {}
            loaded = node.get("loaded_models") if isinstance(node, dict) else []
            if "codellama-34b" not in (loaded or []):
                return _json_pretty({
                    "action_type": "load_model",
                    "target_node": target,
                    "model_name": "codellama-34b",
                    "request_ids": [],
                })
        if queue:
            return _json_pretty(_pick_route_action(obs))
        return _json_pretty({"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []})

    if combo == "loghaul":
        if queue:
            return _json_pretty(_pick_route_action(obs))
        preload_node = _find_node_for_model(obs, "llama3-8b-chat")
        if preload_node:
            nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
            node = nodes.get(preload_node, {}) if isinstance(nodes, dict) else {}
            loaded = node.get("loaded_models") if isinstance(node, dict) else []
            if "llama3-8b-chat" not in (loaded or []):
                return _json_pretty({
                    "action_type": "load_model",
                    "target_node": preload_node,
                    "model_name": "llama3-8b-chat",
                    "request_ids": [],
                })
        return _json_pretty({"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []})

    return _json_pretty({"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []})


def _validate_action(task_name: Optional[str], obs: Dict[str, Any], action: Dict[str, Any]) -> Optional[str]:
    action_type = action.get("action_type")
    target_node = action.get("target_node")
    model_name = action.get("model_name")
    request_ids = action.get("request_ids")

    nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
    queue = obs.get("request_queue", []) if isinstance(obs, dict) else []
    queue_ids = {
        str(req.get("request_id"))
        for req in queue
        if isinstance(req, dict) and req.get("request_id")
    }

    if action_type in {"restart_node", "load_model", "evict_model", "route_batch"} and not target_node:
        return "target_node is required for this action type"

    if action_type == "route_batch":
        if not isinstance(request_ids, list) or not request_ids:
            return "route_batch requires at least one request_id"
        requested = {str(rid) for rid in request_ids}
        if not requested.issubset(queue_ids):
            return "route_batch request_ids must exist in current queue"

    node_state = nodes.get(target_node, {}) if isinstance(nodes, dict) else {}
    if action_type == "restart_node":
        if not isinstance(node_state, dict) or node_state.get("status") != "oom_crashed":
            return "restart_node is only allowed for nodes in oom_crashed status"
        if task_name == "medium":
            node_b = nodes.get("node_b", {}) if isinstance(nodes, dict) else {}
            if isinstance(node_b, dict) and node_b.get("status") == "oom_crashed" and target_node != "node_b":
                return "for medium task, restart node_b first while it is crashed"

    if action_type == "load_model":
        if not model_name:
            return "load_model requires model_name"
        if not isinstance(node_state, dict) or node_state.get("status") != "healthy":
            return "load_model is only allowed on healthy nodes"

    if action_type == "evict_model":
        if not model_name:
            return "evict_model requires model_name"
        loaded = node_state.get("loaded_models") if isinstance(node_state, dict) else []
        if model_name not in (loaded or []):
            return "evict_model requires model_name to be loaded on target_node"

    return None


def _repair_route_action(obs: Dict[str, Any], action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a repaired route action when IDs are stale, else None."""
    if action.get("action_type") != "route_batch":
        return None

    queue = obs.get("request_queue", []) if isinstance(obs, dict) else []
    if not isinstance(queue, list) or not queue:
        return None

    queue_ids = {
        str(req.get("request_id"))
        for req in queue
        if isinstance(req, dict) and req.get("request_id")
    }
    current_ids = action.get("request_ids")
    if not isinstance(current_ids, list):
        current_ids = []
    requested = {str(rid) for rid in current_ids}

    # Nothing to repair if current ids are already valid.
    if requested and requested.issubset(queue_ids):
        return None

    return _pick_route_action(obs)


def suggest_action_with_llm(
    provider: str,
    api_base: str,
    api_key: str,
    model_name: str,
    temperature: float,
    task_name: str,
    state_dict: Dict[str, Any],
):
    state = _dict_to_state(state_dict)
    obs = state.last_obs if isinstance(state.last_obs, dict) else {}

    if not state.active or not obs:
        return (
            '<div class="logline logline-err">LLM Copilot needs an active episode. Reset a task first.</div>',
            gr.update(),
        )

    base = (api_base or "").strip().rstrip("/")
    key = (api_key or "").strip()
    model = (model_name or "").strip()
    if not base or not key or not model:
        return (
            '<div class="logline logline-err">Missing LLM settings: provide API base, API key, and model.</div>',
            gr.update(),
        )

    user_prompt = {
        "task_name": task_name or state.task_name,
        "observation": obs,
        "instruction": "Return one best next action JSON only.",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
        ],
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("choices", [{}])[0].get("message", {})
        message = message if isinstance(message, dict) else {}
        action = _extract_action_from_message(message)
        if not isinstance(action, dict):
            raw_preview = _extract_message_text(message)
            raw_preview = (raw_preview or "")[:300]
            return (
                '<div class="logline logline-err">LLM response was not valid JSON action. Adjust model/settings and retry.</div>'
                f'<div class="logline logline-debug">Preview: {raw_preview}</div>',
                gr.update(),
            )

        validation_error = _validate_action(task_name or state.task_name, obs, action)
        if validation_error:
            return (
                f'<div class="logline logline-err">LLM proposed blocked action: {validation_error}</div>'
                '<div class="logline logline-debug">Review and adjust manually or retry with lower temperature.</div>',
                gr.update(value=_json_pretty(action)),
            )

        return (
            f'<div class="logline logline-ok">LLM suggestion generated via {provider} and validated.</div>',
            gr.update(value=_json_pretty(action)),
        )
    except Exception as exc:
        return (
            f'<div class="logline logline-err">LLM call failed: {str(exc)}</div>',
            gr.update(),
        )


def do_probe(base_url: str):
    base = base_url.rstrip("/")
    tasks_data = _safe_get(f"{base}/tasks")
    if tasks_data and isinstance(tasks_data.get("tasks"), list):
        discovered = tasks_data.get("tasks", [])
        canonical = []
        for item in discovered:
            if item.get("has_grader") and not item.get("alias_for"):
                canonical.append(str(item.get("id") or item.get("name") or "").strip())
        fallback = [t["name"] for t in TASK_META]
        choices = canonical or fallback
        default_task = choices[0] if choices else "easy"
        return (
            f'<div class="logline logline-ok">Connected to {base}</div>'
            f'<div class="logline logline-step">Sync mode: persistent env client</div>'
            f'<div class="logline logline-step">Detected {len(discovered)} tasks via /tasks</div>'
            f'<div class="logline logline-debug">Canonical graded tasks: {", ".join(choices)}</div>',
            gr.update(choices=choices, value=default_task)
        )
    return (
        f'<div class="logline logline-err">Connection check failed for {base}</div>'
        '<div class="logline logline-debug">Verify environment server is running and reachable.</div>',
        gr.update()
    )


def do_reset(base_url: str, task_name: str, seed: str, state_dict: Dict[str, Any]):
    state = _dict_to_state(state_dict)
    kwargs: Dict[str, Any] = {"task_name": task_name}
    if seed.strip():
        try:
            kwargs["seed"] = int(seed)
        except ValueError:
            kwargs["seed"] = seed.strip()

    try:
        env = _get_sync_env(base_url)
        result = env.reset(**kwargs)
        obs = _obs_to_dict(result.observation)
    except Exception as exc:
        state.log_lines.append(f'<div class="logline logline-err">[RESET] sync_client_error={str(exc)}</div>')
        return _status_header(state), _render_log(state.log_lines), "{}", state_dict

    state.active = True
    state.task_name = task_name
    state.step_idx = 0
    state.total_reward = 0.0
    state.done = False
    state.last_obs = obs if isinstance(obs, dict) else {}
    qsize = len(obs.get("request_queue", [])) if isinstance(obs, dict) else 0
    state.log_lines.append(
        f'<div class="logline logline-ok">[RESET] task={task_name} ok=true queue={qsize}</div>'
    )

    new_state = _state_to_dict(state)
    return _status_header(state), _render_log(state.log_lines), _json_pretty(obs), new_state


def do_step(base_url: str, action_json: str, state_dict: Dict[str, Any]):
    state = _dict_to_state(state_dict)
    if not state.active:
        state.log_lines.append('<div class="logline logline-err">[STEP] blocked=episode_not_started</div>')
        new_state = _state_to_dict(state)
        return _status_header(state), _render_log(state.log_lines), "{}", new_state
    if state.done:
        state.log_lines.append('<div class="logline logline-err">[STEP] blocked=episode_done_reset_required</div>')
        new_state = _state_to_dict(state)
        return _status_header(state), _render_log(state.log_lines), "{}", new_state

    try:
        action = json.loads(action_json)
        if not isinstance(action, dict):
            raise ValueError("Action must be a JSON object")
    except Exception as exc:
        state.log_lines.append(f'<div class="logline logline-err">[STEP] invalid_action_json={str(exc)}</div>')
        new_state = _state_to_dict(state)
        return _status_header(state), _render_log(state.log_lines), "{}", new_state

    repaired = _repair_route_action(state.last_obs, action)
    if repaired is not None:
        action = repaired
        state.log_lines.append('<div class="logline logline-debug">[STEP] auto_fixed=route_request_ids_refreshed_from_queue</div>')

    validation_error = _validate_action(state.task_name, state.last_obs, action)
    if validation_error:
        state.log_lines.append(f'<div class="logline logline-err">[STEP] blocked={validation_error}</div>')
        new_state = _state_to_dict(state)
        return _status_header(state), _render_log(state.log_lines), _json_pretty(state.last_obs or {}), new_state

    try:
        module = __import__("llmfleet_sre.models", fromlist=["LLMFleetAction"])
        LLMFleetAction = getattr(module, "LLMFleetAction")

        env = _get_sync_env(base_url)
        result = env.step(LLMFleetAction(**action))
        obs = _obs_to_dict(result.observation)
        reward = float(result.reward or 0.0)
        done = bool(result.done or obs.get("done", False))
    except Exception as exc:
        state.log_lines.append(f'<div class="logline logline-err">[STEP] sync_client_error={str(exc)}</div>')
        new_state = _state_to_dict(state)
        return _status_header(state), _render_log(state.log_lines), "{}", new_state

    state.step_idx += 1
    state.total_reward += reward
    state.done = done
    state.last_obs = obs if isinstance(obs, dict) else {}
    state.log_lines.append(
        f'<div class="logline logline-step">[STEP] idx={state.step_idx} reward={reward:.2f} done={str(done).lower()}</div>'
    )

    new_state = _state_to_dict(state)
    return _status_header(state), _render_log(state.log_lines), _json_pretty(obs), new_state


def preset_action(action_type: str, task_name: str, state_dict: Dict[str, Any]) -> str:
    state = _dict_to_state(state_dict)
    obs = state.last_obs if isinstance(state.last_obs, dict) else {}

    templates = {
        "noop": {"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []},
        "restart": {
            "action_type": "restart_node",
            "target_node": _pick_restart_node(task_name, obs),
            "model_name": None,
            "request_ids": [],
        },
        "load_chat": {"action_type": "load_model", "target_node": "node_a", "model_name": "llama3-8b-chat", "request_ids": []},
        "route": _pick_route_action(obs),
        "evict": {"action_type": "evict_model", "target_node": "node_a", "model_name": "llama3-70b-chat", "request_ids": []},
    }
    return _json_pretty(templates[action_type])


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="LLMFleet-SRE Operations Console", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="hero-card">
              <h1 class="hero-title">LLMFleet-SRE Operations Console</h1>
              <div class="hero-sub">Production-style control surface for task resets, action stepping, and live observation review.</div>
            </div>
            """
        )

        state = gr.State(_default_state())

        with gr.Tabs():
            with gr.Tab("README"):
                gr.HTML(_readme_overview_html())
                with gr.Accordion("Full README", open=False):
                    gr.Markdown(_readme_markdown())

            with gr.Tab("Mission Control"):
                with gr.Row():
                    base_url = gr.Textbox(label="Environment Base URL", value=UI_ENV_BASE_URL, scale=3)
                    probe_btn = gr.Button("Probe", variant="secondary", scale=1)
                probe_out = gr.HTML('<div class="logline">Press Probe to verify connectivity and refresh task options.</div>')
                gr.HTML(
                    """
                    <div class="flow-assist">
                      <div class="flow-title">Operator Flow Assistant</div>
                      <div class="flow-row">
                        <div class="flow-step">1) Probe</div>
                        <div class="flow-arrow">-></div>
                        <div class="flow-step">2) Select Task + Reset</div>
                        <div class="flow-arrow">-></div>
                        <div class="flow-step">3) Use Preset or Edit Action JSON</div>
                        <div class="flow-arrow">-></div>
                        <div class="flow-step">4) Execute Step</div>
                        <div class="flow-arrow">-></div>
                        <div class="flow-step">5) Read Observation + Log</div>
                      </div>
                    </div>
                    """
                )

                with gr.Row():
                    task_choice = gr.Dropdown(
                        label="Task",
                        choices=[t["name"] for t in TASK_META],
                        value="easy",
                        scale=2,
                    )
                    seed_input = gr.Textbox(label="Seed (optional)", value="", scale=1)
                    reset_btn = gr.Button("Reset Episode", variant="primary", scale=1)

                status_html = gr.HTML('<div class="pill">task=none</div><div class="pill">status=IDLE</div>')

                with gr.Row():
                    with gr.Column(scale=2):
                        action_json = gr.Code(
                            language="json",
                            label="Action JSON",
                            value=_json_pretty(
                                {
                                    "action_type": "noop",
                                    "target_node": None,
                                    "model_name": None,
                                    "request_ids": [],
                                }
                            ),
                        )
                        with gr.Row():
                            gr.Button("Preset: Noop", variant="secondary").click(
                                fn=lambda t, s: preset_action("noop", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            gr.Button("Preset: Restart", variant="secondary").click(
                                fn=lambda t, s: preset_action("restart", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            gr.Button("Preset: Load Chat", variant="secondary").click(
                                fn=lambda t, s: preset_action("load_chat", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                        with gr.Row():
                            gr.Button("Preset: Route", variant="secondary").click(
                                fn=lambda t, s: preset_action("route", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            gr.Button("Preset: Evict", variant="secondary").click(
                                fn=lambda t, s: preset_action("evict", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            step_btn = gr.Button("Execute Step", variant="primary")

                        with gr.Row():
                            gr.Button("Combo: Easy Fast Path", variant="secondary").click(
                                fn=lambda t, s: combo_action("easy", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            gr.Button("Combo: Medium Recovery", variant="secondary").click(
                                fn=lambda t, s: combo_action("medium", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                        with gr.Row():
                            gr.Button("Combo: Hard Code First", variant="secondary").click(
                                fn=lambda t, s: combo_action("hard", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )
                            gr.Button("Combo: Loghaul Flow", variant="secondary").click(
                                fn=lambda t, s: combo_action("loghaul", t, s),
                                inputs=[task_choice, state],
                                outputs=[action_json],
                            )

                    with gr.Column(scale=2):
                        obs_panel = gr.Code(
                            language="json",
                            label="Observation",
                            value="{}",
                            elem_classes=["panel", "panel-obs"],
                        )
                        log_panel = gr.HTML('<div class="panel panel-log">No actions yet.</div>')

                probe_btn.click(do_probe, inputs=[base_url], outputs=[probe_out, task_choice])
                reset_btn.click(
                    do_reset,
                    inputs=[base_url, task_choice, seed_input, state],
                    outputs=[status_html, log_panel, obs_panel, state],
                )
                step_btn.click(
                    do_step,
                    inputs=[base_url, action_json, state],
                    outputs=[status_html, log_panel, obs_panel, state],
                )

            with gr.Tab("LLM Copilot"):
                gr.Markdown(
                    """
### LLM Decision Copilot
Use an external LLM (Groq/OpenAI-compatible API) to propose the next action from the current observation.

Flow:
1. Reset and run at least one step in Mission Control.
2. Configure provider/model below.
3. Click `Suggest Action` to populate Mission Control `Action JSON`.
4. Review and execute in Mission Control.
                    """
                )

                with gr.Row():
                    llm_provider = gr.Dropdown(
                        label="Provider",
                        choices=["groq", "openai-compatible"],
                        value=UI_LLM_PROVIDER,
                        scale=1,
                    )
                    llm_api_base = gr.Textbox(label="API Base", value=UI_LLM_API_BASE, scale=2)
                with gr.Row():
                    llm_model = gr.Textbox(label="Model", value=UI_LLM_MODEL, scale=2)
                    llm_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.2, scale=1)
                llm_api_key = gr.Textbox(label="API Key", value="", type="password")
                llm_out = gr.HTML('<div class="logline">Configure API settings, then click Suggest Action.</div>')
                llm_btn = gr.Button("Suggest Action", variant="primary")

                llm_btn.click(
                    suggest_action_with_llm,
                    inputs=[llm_provider, llm_api_base, llm_api_key, llm_model, llm_temp, task_choice, state],
                    outputs=[llm_out, action_json],
                )

            with gr.Tab("Task Codex"):
                gr.HTML(
                    """
                    <div class="hero-sub" style="margin-bottom: 10px;">
                                            Canonical tasks with deterministic graders and production constraints.
                    </div>
                    """
                )
                gr.HTML(_task_catalog_html())

            with gr.Tab("Runbook"):
                gr.Markdown(
                    """
### Launch Sequence
1. Start environment server on port 7860.
2. Launch this UI on port 7861.
3. Probe connectivity, reset a task, then step actions.

### Notes
- This UI is optional and isolated from evaluator-critical behavior.
- Evaluator checks still rely on core endpoints and inference logs.
- Prefer canonical task ids discovered from /tasks (easy, medium, hard, loghaul).

### Demo combos

Easy combo:
1. Reset `easy`.
2. Click `Preset: Route`.
3. Execute Step once.

Medium combo:
1. Reset `medium`.
2. Click `Preset: Restart` (targets crashed node).
3. Click `Preset: Route` and execute when model is ready.

Hard combo:
1. Reset `hard`.
2. Use load-model for `codellama-34b` on a healthy node.
3. Route quickly; evict only when VRAM pressure demands it.

Loghaul combo:
1. Early phase: preload likely model.
2. Spike phase: route premium queue aggressively.
3. Cooldown phase: evict expensive idle models.

JSON template:

```json
{
    "action_type": "route_batch",
    "target_node": "node_a",
    "model_name": null,
    "request_ids": ["req_0001", "req_0002"]
}
```

### End-to-end (ditto)

```text
1) Probe
2) easy -> Reset -> Combo: Easy Fast Path -> Execute Step
3) medium -> Reset -> Combo: Medium Recovery -> Execute Step -> Combo: Medium Recovery -> Execute Step
4) hard -> Reset -> Combo: Hard Code First -> Execute Step -> Preset: Route -> Execute Step
5) loghaul -> Reset -> Combo: Loghaul Flow -> Execute Step (repeat a few times)
```

If episode is already done, reset before next step.
                    """
                )

    return demo


def main() -> None:
    app = create_ui()
    app.launch(server_name=UI_HOST, server_port=UI_PORT, share=False)


if __name__ == "__main__":
    main()