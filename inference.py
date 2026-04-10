"""
LLMFleet-SRE baseline inference script.

An LLM agent acts as an SRE for a simulated GPU inference cluster.
Reads: API_BASE_URL, MODEL_NAME, API_KEY, IMAGE_NAME, TASK_NAME.

TASK_NAME may be a single task id or a comma-separated list.
If not provided, defaults to easy,medium,hard so validators can observe
multiple task runs.

Emits exact [START], [STEP], [END] log format required by OpenEnv judges.
"""
import asyncio
import json
import os
import sys
from typing import List, Optional

from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider


class Port7860DockerProvider(LocalDockerProvider):
    """Local Docker provider that maps host ports to container port 7860."""

    def start_container(self, image: str, port=None, env_vars=None, **kwargs):
        import subprocess
        import time

        if port is None:
            port = self._find_available_port()

        self._container_name = self._generate_container_name(image)
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
            "-p",
            f"{port}:7860",
        ]

        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(image)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._container_id = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Failed to start Docker container.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"Stderr: {e.stderr}\n"
                f"Stdout: {e.stdout}"
            )
            raise RuntimeError(error_msg) from e

        time.sleep(1)
        return f"http://localhost:{port}"

#  Config 
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
# Must use evaluator-injected key only.
API_KEY      = os.environ.get("API_KEY")
IMAGE_NAME   = os.environ.get("IMAGE_NAME",   "Ajeya95/llmfleet-sre")
TASK_NAME    = os.environ.get("TASK_NAME",    "easy,medium,hard")
FALLBACK_MODELS_RAW = os.environ.get(
    "FALLBACK_MODELS",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,Qwen/Qwen2.5-Coder-3B-Instruct,google/gemma-3n-E4B-it,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-4B-Instruct-2507",
)
DEBUG_LOGS = os.environ.get("DEBUG_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}

MAX_STEPS         = 25
MAX_TOTAL_REWARD  = 1.0
SUCCESS_THRESHOLD = 0.6
STRICT_MIN_SCORE_OUTPUT = 0.01
STRICT_MAX_SCORE_OUTPUT = 0.99
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "120"))
PROMPT_QUEUE_LIMIT = int(os.environ.get("PROMPT_QUEUE_LIMIT", "12"))

BENCHMARK = "llmfleet-sre"

TASK_ALIASES = {
    "task_easy": "easy",
    "task_medium": "medium",
    "task_hard": "hard",
    "task_longhaul": "loghaul",
    "med": "medium",
}


def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(f"[DEBUG] {message}", flush=True)


def _normalize_task_name(task_name: str) -> str:
    return TASK_ALIASES.get(task_name.strip(), task_name.strip())


def _parse_tasks_to_run(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return ["easy", "medium", "hard"]
    if raw.lower() in {"all", "*"}:
        return ["easy", "medium", "hard", "loghaul"]

    tasks = [_normalize_task_name(t) for t in raw.split(",") if t.strip()]
    deduped: List[str] = []
    seen = set()
    for t in tasks:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped or ["easy", "medium", "hard"]


def _build_model_chain(primary_model: str, fallback_raw: str) -> List[str]:
    seen = set()
    chain: List[str] = []

    def _add(model: str):
        model = model.strip()
        if not model or model in seen:
            return
        seen.add(model)
        chain.append(model)

    _add(primary_model)
    for model in fallback_raw.split(","):
        _add(model)
    return chain


def _should_try_next_model(error_str: str) -> bool:
    e = error_str.lower()
    return any(
        token in e
        for token in [
            "model_not_supported",
            "not a chat model",
            "not supported by any provider",
            "failed_to_auth",
            "401",
            "402",
            "credit",
            "quota",
            "429",
            "rate limit",
        ]
    )


#  Logging helpers (guideline format)

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    done_val = str(bool(done)).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={float(reward):.2f} done={done_val} error={error_val}",
        flush=True,
    )


# FIX 3: add task= param to [END] log line
def log_end(task, success, steps, score, rewards):
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(bool(success)).lower()} steps={steps} score={float(score):.2f} rewards={rewards_str}",
        flush=True,
    )


#  Agent System Prompt — updated for NL observations 

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) managing an LLM inference cluster.

You will receive a CLUSTER STATUS REPORT in plain English each step. Read it carefully and decide the single best action.

Model VRAM requirements:
- llama3-8b-chat: 18 GB  (serves: chat requests)
- llama3-70b-chat: 45 GB (serves: chat requests, higher quality)
- codellama-34b: 35 GB   (serves: code requests)
- mistral-7b-sum: 16 GB  (serves: summarize requests)
- code-lora-adapter: 8 GB (adapter)

Node capacities:
- node_a: 80 GB VRAM
- node_b: 80 GB VRAM
- node_c: 40 GB VRAM

Available actions:
- route_batch: Send a list of request_ids to a node. Node must be healthy with the required model loaded.
- load_model: Load a model onto a node. CHECK: vram_used_gb + model_vram_gb must not exceed vram_total_gb or the node will OOM crash.
- evict_model: Remove a model from a node to free VRAM.
- restart_node: Restart a crashed (OOM) node. Clears all models. Node takes 2 steps to become healthy.
- noop: Take no action this step.

Decision rules:
1. ALWAYS check VRAM free before load_model. Free = vram_total_gb - vram_used_gb.
2. NEVER route to a node with status=loading or status=oom_crashed.
3. Premium SLA requests breach in 5 steps — serve them first.
4. If a node is OOM crashed, restart it before doing anything else on that node.
5. Take exactly ONE action per step — choose the highest priority.

Respond ONLY with a valid JSON object:
{
  "action_type": "route_batch" | "load_model" | "evict_model" | "restart_node" | "noop",
  "target_node": "node_a" | "node_b" | "node_c" | null,
  "model_name": "<model name>" | null,
  "request_ids": ["req_0001", ...]
}

No explanation. Return only the JSON.
"""


def get_action(
    client: OpenAI,
    obs: dict,
    history: List[str],
    model_chain: List[str],
    model_state: dict,
) -> Optional[dict]:
    # Use last_action_result which now contains the full NL status report
    nl_report = obs.get("last_action_result", "")
    if not nl_report or not nl_report.strip().startswith("==="):
        # Fallback: if NL report not present, build a minimal summary from raw obs fields
        nodes_summary = json.dumps(obs.get("nodes", {}), separators=(",", ":"))
        queue_summary = json.dumps(obs.get("request_queue", [])[:PROMPT_QUEUE_LIMIT], separators=(",", ":"))
        nl_report = (
            f"Step {obs.get('step', '?')}/{obs.get('step_budget', '?')} | "
            f"served={obs.get('requests_served', 0)} violations={obs.get('sla_violations', 0)}\n"
            f"Nodes: {nodes_summary}\nQueue: {queue_summary}"
        )

    history_str = "\n".join(history[-3:]) if history else "None yet."
    user_msg = f"{nl_report}\n\nRecent actions:\n{history_str}\n\nWhat is your next action?"

    while True:
        model_name = model_chain[model_state["index"]]
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            error_str = str(e)

            if "402" in error_str or "credit" in error_str.lower() or "quota" in error_str.lower():
                debug_log("Credit exhaustion - stopping inference")
                model_state["stopped_due_credit"] = True
                return None

            if _should_try_next_model(error_str) and model_state["index"] < len(model_chain) - 1:
                previous = model_name
                model_state["index"] += 1
                next_model = model_chain[model_state["index"]]
                debug_log(f"Switching model from {previous} to {next_model} due to API error")
                continue

            debug_log(f"Model error ({model_name}): {e}")
            return {"action_type": "noop", "target_node": None, "model_name": None, "request_ids": []}


#  Main loop 

async def main():
    if not API_BASE_URL:
        print("[ERROR] API_BASE_URL is not set. Use the injected LiteLLM proxy URL.", flush=True)
        raise SystemExit(1)
    if not API_KEY:
        print("[ERROR] API_KEY is not set. Use the injected evaluator API key.", flush=True)
        raise SystemExit(1)

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llmfleet_sre.client import LLMFleetSreEnv
        from llmfleet_sre.models import LLMFleetAction
    except ImportError as e:
        print(f"[ERROR] Required packages missing: {e}. Run: pip install openenv-core", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await LLMFleetSreEnv.from_docker_image(IMAGE_NAME, provider=Port7860DockerProvider())
    model_chain = _build_model_chain(MODEL_NAME, FALLBACK_MODELS_RAW)
    model_state = {"index": 0, "stopped_due_credit": False}

    tasks_to_run = _parse_tasks_to_run(TASK_NAME)

    try:
        for task_name in tasks_to_run:
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False
            completed_episode = False

            model_state = {"index": 0, "stopped_due_credit": False}

            log_start(task=task_name, env=BENCHMARK, model=model_chain[0])

            result = await env.reset(task_name=task_name)
            obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation

            for step in range(1, MAX_STEPS + 1):
                if isinstance(obs, dict) and obs.get("done", False):
                    break

                action_dict = get_action(client, obs, history, model_chain, model_state)
                if action_dict is None:
                    break
                action_str = json.dumps(action_dict, separators=(",", ":"))

                try:
                    action_obj = LLMFleetAction(**action_dict)
                except Exception as e:
                    debug_log(f"Invalid LLM action format: {e}")
                    action_obj = LLMFleetAction(action_type="noop", target_node=None, model_name=None, request_ids=[])

                step_result = await env.step(action_obj)
                new_obs = step_result.observation.model_dump() if hasattr(step_result.observation, "model_dump") else step_result.observation

                reward = float(getattr(step_result, "reward", 0.0) or (new_obs.get("reward", 0.0) if isinstance(new_obs, dict) else 0.0))
                done = getattr(step_result, "done", False) or (new_obs.get("done", False) if isinstance(new_obs, dict) else False)
                raw_error = getattr(step_result, "error", None)
                if raw_error is None and isinstance(new_obs, dict):
                    raw_error = new_obs.get("last_action_error")
                step_error = str(raw_error) if raw_error else None

                rewards.append(reward)
                steps_taken = step
                obs = new_obs

                log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)
                history.append(f"Step {step}: {action_str}  reward {reward:+.2f}")

                if done:
                    completed_episode = True
                    break

            # FIX 2: score formula from gotin repo — round(min(max(reward, 0.01), 0.99), 2)
            if rewards:
                raw_score = sum(rewards) / len(rewards)
                score = round(min(max(raw_score, 0.01), 0.99), 2)
            else:
                score = 0.01
            success = completed_episode and (score >= SUCCESS_THRESHOLD)
            # FIX 3: pass task= to log_end
            log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        import traceback
        debug_log(f"Runtime error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if asyncio.iscoroutinefunction(env.close):
                await env.close()
            else:
                env.close()
        except Exception as e:
            debug_log(f"env.close() error: {e}")


if __name__ == "__main__":
    asyncio.run(main())