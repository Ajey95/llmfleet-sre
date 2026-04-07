from server.environment import LLMFleetEnvironment, MODEL_CATALOGUE
from server.tasks import grade
from models import LLMFleetAction


def greedy_policy(obs):
    nodes = obs.nodes
    queue = obs.request_queue

    # Priority 1: restart crashed nodes
    for node_id, node in nodes.items():
        if node.status == "oom_crashed":
            return LLMFleetAction(action_type="restart_node", target_node=node_id)

    # Priority 2: route requests where model already loaded
    for req in queue:
        for node_id, node in nodes.items():
            if node.status == "healthy" and req.required_model in node.loaded_models:
                return LLMFleetAction(
                    action_type="route_batch",
                    target_node=node_id,
                    request_ids=[req.request_id],
                )

    # Priority 3: load required model if VRAM free
    for req in queue:
        model_name = req.required_model
        vram_needed = MODEL_CATALOGUE[model_name].vram_gb
        for node_id, node in nodes.items():
            if node.status != "healthy":
                continue
            if model_name in node.loaded_models:
                continue
            if node.vram_used_gb + vram_needed <= node.vram_total_gb:
                return LLMFleetAction(
                    action_type="load_model",
                    target_node=node_id,
                    model_name=model_name,
                )

    return LLMFleetAction(action_type="noop")


def run_greedy(task_name: str, seed: int):
    env = LLMFleetEnvironment(task_name=task_name)
    obs = env.reset(seed=seed)

    for _ in range(env.step_budget):
        action = greedy_policy(obs)
        obs = env.step(action)
        if obs.done:
            break

    final_state = env.get_full_state()
    score = grade(task_name, final_state)
    return score, final_state.requests_served, final_state.sla_violations


if __name__ == "__main__":
    tasks = ["task_easy", "task_medium", "task_hard", "task_longhaul"]
    seeds = [42, 7, 13, 99, 21]

    print("\nGreedy baseline verification")
    print("=" * 60)

    for task in tasks:
        results = [run_greedy(task, s) for s in seeds]
        scores = [r[0] for r in results]
        avg = sum(scores) / len(scores)
        hi = max(scores)
        lo = min(scores)

        print(f"\n{task}")
        print(f"  scores : {[f'{s:.2f}' for s in scores]}")
        print(f"  avg={avg:.2f}  hi={hi:.2f}  lo={lo:.2f}")

        if task == "task_longhaul":
            if avg < 0.50:
                print(f"  PASS - greedy avg {avg:.2f} < 0.50 - environment requires RL")
            else:
                print(f"  FAIL - greedy avg {avg:.2f} >= 0.50 - environment still rule-based solvable")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  task_easy     - expect greedy ~1.0  (trivial by design)")
    print("  task_medium   - expect greedy ~0.5-0.7")
    print("  task_hard     - expect greedy ~0.3-0.5")
    print("  task_longhaul - expect greedy < 0.50 to confirm RL")
    print("=" * 60)
