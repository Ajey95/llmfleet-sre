from server.environment import LLMFleetEnvironment
from server.tasks import grade
from models import LLMFleetAction

import random

for task in ["task_easy", "task_medium", "task_hard", "task_longhaul"]:
    env = LLMFleetEnvironment(task_name=task)
    obs = env.reset(seed=42)

    for _ in range(env.step_budget):
        # Prefer routing when possible, otherwise recover/load opportunistically.
        route_candidates = []
        for node_id, node in obs.nodes.items():
            if node.status != "healthy":
                continue
            req_ids = [
                req.request_id
                for req in obs.request_queue
                if req.required_model in node.loaded_models
            ]
            if req_ids:
                route_candidates.append((node_id, req_ids))

        if route_candidates and random.random() < 0.8:
            node_id, req_ids = random.choice(route_candidates)
            action = LLMFleetAction(action_type="route_batch", target_node=node_id, request_ids=req_ids[:3])
        else:
            crashed_nodes = [nid for nid, n in obs.nodes.items() if n.status == "oom_crashed"]
            if crashed_nodes and random.random() < 0.5:
                action = LLMFleetAction(action_type="restart_node", target_node=random.choice(crashed_nodes))
            else:
                action = LLMFleetAction(action_type="noop")

        obs = env.step(action)
        if obs.done:
            break

    score = grade(task, env.get_full_state())
    final_state = env.get_full_state()
    print(
        f"{task}: score={score:.3f}, "
        f"served={final_state.requests_served}, "
        f"sla_violations={final_state.sla_violations}"
    )
