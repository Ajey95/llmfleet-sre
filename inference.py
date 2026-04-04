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
from typing import List

from openai import OpenAI

#  Config 
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("HF_TOKEN")
IMAGE_NAME   = os.environ.get("IMAGE_NAME",   "Ajeya95/llmfleet-sre")
TASK_NAME    = os.environ.get("TASK_NAME",    "task_easy")

MAX_STEPS         = 25
MAX_TOTAL_REWARD  = 1.0
SUCCESS_THRESHOLD = 0.6

BENCHMARK = "llmfleet-sre"


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
1. Before load_model, always check: node.vram_used_gb + model_vram_gb <= node.vram_total_gb
2. You can only route to nodes with status=healthy
3. Premium SLA requests must be served within 5 steps or you incur -0.3 penalty each
4. Prioritize premium requests over best_effort

Respond ONLY with a valid JSON object matching this schema:
{
  "action_type": "<one of the actions above>",
  "target_node": "<node_a | node_b | node_c | null>",
  "model_name": "<model name or null>",
  "request_ids": ["<id1>", "<id2>"]
}

Do not explain. Return only JSON.
"""


def get_action(client: OpenAI, obs: dict, history: List[str]) -> dict:
    history_str = "\n".join(history[-5:]) if history else "None yet."
    user_msg = f"""Current cluster state:
{json.dumps(obs, indent=2)}

Recent history:
{history_str}

What is your next action?"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
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
        print(f"[DEBUG] Model error: {e}", flush=True)
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
    env = await LLMFleetSreEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=TASK_NAME)
        obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation

        for step in range(1, MAX_STEPS + 1):
            # Using dict .get() since obs is dumped to dict
            if isinstance(obs, dict) and obs.get("done", False):
                break

            action_dict = get_action(client, obs, history)
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

