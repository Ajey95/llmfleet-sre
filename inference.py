"""
LLMFleet-SRE baseline inference script.

An LLM agent acts as an SRE for a simulated GPU inference cluster.
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN (as API key), IMAGE_NAME, TASK_NAME

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
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
if "api-inference.huggingface.co" in API_BASE_URL:
    # Keep backward compatibility with older .env values.
    API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN")
IMAGE_NAME   = os.environ.get("IMAGE_NAME",   "Ajeya95/llmfleet-sre")
TASK_NAME    = os.environ.get("TASK_NAME",    "task_easy")
FALLBACK_MODELS_RAW = os.environ.get(
    "FALLBACK_MODELS",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,Qwen/Qwen2.5-Coder-3B-Instruct,google/gemma-3n-E4B-it,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-4B-Instruct-2507",
)
DEBUG_LOGS = os.environ.get("DEBUG_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}

MAX_STEPS         = 25
MAX_TOTAL_REWARD  = 1.0
SUCCESS_THRESHOLD = 0.6
MAX_COMPLETION_TOKENS = int(os.environ.get("MAX_COMPLETION_TOKENS", "120"))
PROMPT_QUEUE_LIMIT = int(os.environ.get("PROMPT_QUEUE_LIMIT", "12"))

BENCHMARK = "llmfleet-sre"


def debug_log(message: str) -> None:
    if DEBUG_LOGS:
        print(f"[DEBUG] {message}", flush=True)


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


def _compact_observation(obs: dict, queue_limit: int) -> dict:
    nodes = {}
    for node_id, node in obs.get("nodes", {}).items():
        nodes[node_id] = {
            "status": node.get("status"),
            "vram_used_gb": node.get("vram_used_gb"),
            "vram_total_gb": node.get("vram_total_gb"),
            "loaded_models": node.get("loaded_models", []),
            "load_latency_remaining": node.get("load_latency_remaining", 0),
        }

    queue = obs.get("request_queue", [])
    sorted_queue = sorted(
        queue,
        key=lambda r: (
            0 if r.get("sla_tier") == "premium" else 1,
            -int(r.get("age_steps", 0)),
        ),
    )
    trimmed_queue = sorted_queue[:max(1, queue_limit)]
    compact_queue = [
        {
            "request_id": r.get("request_id"),
            "required_model": r.get("required_model"),
            "sla_tier": r.get("sla_tier"),
            "age_steps": r.get("age_steps", 0),
        }
        for r in trimmed_queue
    ]

    return {
        "step": obs.get("step", 0),
        "step_budget": obs.get("step_budget", 0),
        "sla_violations": obs.get("sla_violations", 0),
        "requests_served": obs.get("requests_served", 0),
        "queue_total": len(queue),
        "queue_shown": len(compact_queue),
        "nodes": nodes,
        "request_queue": compact_queue,
        "last_action_result": obs.get("last_action_result", ""),
    }


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


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(bool(success)).lower()} steps={steps} score={float(score):.2f} rewards={rewards_str}",
        flush=True,
    )


#  Agent System Prompt 

SYSTEM_PROMPT = """You are an SRE managing an LLM inference cluster.
You will receive the cluster state as JSON and must decide the next action.

Available models and their VRAM cost (GB):
- llama3-8b-chat: 18GB   (for chat tasks)
- llama3-70b-chat: 45GB  (for chat tasks, large)
- codellama-34b: 35GB    (for code tasks)
- mistral-7b-sum: 16GB   (for summarize tasks)
- code-lora-adapter: 8GB (adapter)

Node capacities:
- node_a: 80GB VRAM
- node_b: 80GB VRAM
- node_c: 40GB VRAM

Available actions:
- route_batch: Route a list of request_ids to a target_node (node must have the required model loaded and status=healthy)
- load_model: Load model_name onto target_node (CHECK: vram_used + model_vram <= vram_total to avoid OOM crash!)
- evict_model: Evict model_name from target_node to free VRAM
- restart_node: Restart a crashed node (status: oom_crashed)
- noop: Do nothing this step

IMPORTANT RULES:
1. Before load_model, always verify: node.vram_used_gb + model_vram_gb <= node.vram_total_gb
2. You can only route to nodes with status=healthy AND load_latency_remaining=0
3. Premium SLA requests must be served within 5 steps or you incur -0.3 penalty each
4. Prioritize premium requests over best_effort
5. A node with status=loading is not yet ready - do not route to it, wait or act on other nodes
6. Each step you take exactly one action - plan the highest priority action only

Respond ONLY with a valid JSON object matching this schema:
{
  "action_type": "<one of the actions above>",
  "target_node": "<node_a | node_b | node_c | null>",
  "model_name": "<model name or null>",
  "request_ids": ["<id1>", "<id2>"]
}

Do not explain. Return only JSON.
"""


def get_action(
    client: OpenAI,
    obs: dict,
    history: List[str],
    model_chain: List[str],
    model_state: dict,
) -> Optional[dict]:
    compact_obs = _compact_observation(obs, PROMPT_QUEUE_LIMIT)
    history_str = "\n".join(history[-3:]) if history else "None yet."
    user_msg = f"""Current cluster state:
{json.dumps(compact_obs, separators=(",", ":"))}

Recent history:
{history_str}

What is your next action?"""
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
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            error_str = str(e)

            # Stop immediately on quota exhaustion to avoid burning additional
            # failed requests while rotating fallbacks.
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
    if not API_KEY:
        print("[ERROR] HF_TOKEN is not set.", flush=True)
        return

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

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    completed_episode = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=model_chain[0])

    try:
        result = await env.reset(task_name=TASK_NAME)
        obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation

        for step in range(1, MAX_STEPS + 1):
            # Using dict .get() since obs is dumped to dict
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
            
            # get reward & done from step_result object, or from the new_obs dictionary gracefully
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

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = max(0.0, min(score, 1.0))
        # A run that exits early (e.g., due to credits) should not be marked successful.
        success = completed_episode and (score >= SUCCESS_THRESHOLD)

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
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

