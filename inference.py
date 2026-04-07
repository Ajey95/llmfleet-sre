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
    "Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen2.5-Coder-3B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,google/gemma-3n-E4B-it",
)

MAX_STEPS         = 25
MAX_TOTAL_REWARD  = 1.0
SUCCESS_THRESHOLD = 0.6

BENCHMARK = "llmfleet-sre"


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


#  Logging helpers (MUST match this exact format) 

def log_start(task, env, model):
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step, action, reward, done, error=None):
    print(json.dumps({
        "type": "STEP", "step": step,
        "action": action, "reward": reward,
        "done": done, "error": error
    }), flush=True)

def log_end(success, steps, score, rewards):
    print(json.dumps({
        "type": "END", "success": success,
        "steps": steps, "score": score, "rewards": rewards
    }), flush=True)


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
    history_str = "\n".join(history[-5:]) if history else "None yet."
    user_msg = f"""Current cluster state:
{json.dumps(obs, indent=2)}

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
                max_tokens=200,
                temperature=0.1,
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

            if _should_try_next_model(error_str) and model_state["index"] < len(model_chain) - 1:
                previous = model_name
                model_state["index"] += 1
                next_model = model_chain[model_state["index"]]
                print(
                    f"[DEBUG] Switching model from {previous} to {next_model} due to API error",
                    flush=True,
                )
                continue

            # Exit cleanly on credit exhaustion rather than spamming noop.
            if "402" in error_str or "credit" in error_str.lower() or "quota" in error_str.lower():
                print("[DEBUG] Credit exhaustion - stopping inference", flush=True)
                return None

            print(f"[DEBUG] Model error ({model_name}): {e}", flush=True)
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
    model_state = {"index": 0}

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

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
            action_str = json.dumps(action_dict)

            try:
                action_obj = LLMFleetAction(**action_dict)
            except Exception as e:
                print(f"[DEBUG] Invalid LLM action format: {e}", flush=True)
                action_obj = LLMFleetAction(action_type="noop", target_node=None, model_name=None, request_ids=[])

            step_result = await env.step(action_obj)
            
            new_obs = step_result.observation.model_dump() if hasattr(step_result.observation, "model_dump") else step_result.observation
            
            # get reward & done from step_result object, or from the new_obs dictionary gracefully
            reward = float(getattr(step_result, "reward", 0.0) or (new_obs.get("reward", 0.0) if isinstance(new_obs, dict) else 0.0))
            done = getattr(step_result, "done", False) or (new_obs.get("done", False) if isinstance(new_obs, dict) else False)

            rewards.append(reward)
            steps_taken = step
            obs = new_obs

            log_step(step=step, action=action_str, reward=reward, done=done)
            history.append(f"Step {step}: {action_str}  reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        import traceback
        print(f"[DEBUG] Runtime error: {e}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            if asyncio.iscoroutinefunction(env.close):
                await env.close()
            else:
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

