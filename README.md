---
title: LLMFleet-SRE
emoji: 🐢
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - llm-ops
  - inference
  - scheduling
  - sre
  - agent
  - rl-environment
license: mit
short_description: LLM agent managing a simulated GPU inference cluster
---

# LLMFleet-SRE

> An RL environment where an LLM agent acts as a Site Reliability Engineer for a simulated LLM inference cluster.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://meta-pytorch.org/OpenEnv/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-informational)](https://www.docker.com/)
[![GPU Required](https://img.shields.io/badge/GPU-not%20required-success)](https://meta-pytorch.org/OpenEnv/)

---

## What LLMFleet-SRE actually is

You are building a simulation of a company's LLM serving infrastructure. Think of any startup or team that offers an AI API — they have a cluster of GPU machines running models like Llama or Mistral, and they receive thousands of requests per minute from users. Managing that cluster is a real, painful, daily job. Your environment turns that job into an RL problem.

The simulation has three GPU nodes — call them Node A, Node B, and Node C. Each node has a fixed amount of VRAM: Node A and B have 80 GB each, Node C has only 40 GB. These nodes are the physical machines that would serve AI requests in a real company. None of this actually uses a GPU — you're just tracking numbers in Python. If Node A has 80 GB total and you load a model that takes 35 GB, Node A now has 45 GB remaining. That's the entire "simulation."

The cluster holds a catalogue of five models with different sizes. The chat model takes 18 GB, the code model takes 35 GB, the summarization model takes 16 GB, and so on. A model must be loaded onto a node before that node can serve requests that require it. Loading takes 1 to 3 steps because we added stochastic latency to simulate real cold-start variability. Once loaded, the model stays until someone explicitly evicts it.

Requests arrive in a queue each episode. Each request has three properties: what type of task it is (chat, code, or summarize), which model it therefore requires, and what SLA tier it belongs to — either premium or best-effort. Premium requests are paying customers. Best-effort requests can wait. A premium request that sits in the queue for more than 5 steps without being served triggers an SLA penalty.

The agent — the LLM being evaluated — reads the full cluster state as a JSON observation at every step and must output one action. The five possible actions are: route a batch of requests to a specific node, load a model onto a node, evict a model from a node to free VRAM, restart a crashed node, or do nothing. The agent cannot see the future — it doesn't know which requests are coming next. It only sees the current queue, the current node states, and how long each request has been waiting.

The reward function is the core design. Six signals combine every step: you earn +0.2 for every request successfully served. You lose −0.01 for every request sitting in the queue per step, which creates constant pressure to act fast. You lose −0.05 for every node that is healthy and has models loaded but is sitting idle while the queue has requests it could theoretically serve — this penalizes wasted compute. You lose −0.5 immediately if you try to load a model onto a node that doesn't have enough free VRAM — this is the OOM crash penalty, the most expensive mistake. You lose −0.3 every time a premium request ages past 5 steps unanswered. And you lose an additional cost penalty when expensive models stay loaded while cheaper alternatives could satisfy queued work. An agent that does nothing scores negatively from latency and idle penalties, so passivity is not an option.

The meta-angle — the thing that makes this genuinely novel — is that you are using an LLM to simulate the job of deciding how to serve other LLMs. The agent reads model names like `llama3-8b-chat` and `codellama-34b`, VRAM numbers that correspond to real model sizes, and request types that map to real use cases. A well-trained agent on this environment would produce scheduling decisions that are directly interpretable as infrastructure policy. That is not a toy problem. That is the exact reasoning loop that an on-call engineer at Hugging Face or a cloud provider runs through at 2 AM when the cluster is misbehaving.

---

## Why this requires Reinforcement Learning (and not just prompting)

In standard prompt engineering, an LLM answers a question in isolation. There is no persistent memory, no consequence for intermediate mistakes, and no sequence of decisions that build on each other. LLMFleet-SRE is a true Reinforcement Learning problem because it features **delayed consequence requiring multi-step planning**.

The critical property distinguishing this environment from a standard prompt is the **credit assignment problem**. For example, in Task 3:

1. The agent sees a sudden spike in premium code requests but lacks the VRAM to load the model.
2. It must explicitly `evict_model` (Step 1).
3. It must incur a short-term latency penalty while calling `load_model` and waiting out the 1–3 step stochastic cold-start (Steps 2–4).
4. Only then can it successfully `route_batch` and clear the queue (Step 5).

The massive reward for clearing the queue at Step 5 was entirely contingent on the painful eviction decision made at Step 1. A greedy agent that only optimizes for the current step will endlessly stall or instantly trigger an OOM crash. Maximizing the compound reward requires planning the entire sequence ahead of time, navigating a stateful world where actions have lasting consequences.

---

## The action that looks wrong but is right

The clearest way to see why this is a genuine RL problem is to watch what happens to a greedy agent during task_longhaul.

At step 8 the cluster is quiet. Two best-effort chat requests sit in the queue. Node C has 40 GB free and nothing loaded. The greedy agent does the obvious thing — routes the chat requests and waits. No code requests are in the queue so loading `codellama-34b` right now looks wasteful. It costs VRAM, takes 1–3 steps of cold-start latency, and produces zero immediate reward.

At step 16 the spike arrives. Four premium code requests flood the queue simultaneously. Their SLA clock starts ticking immediately — they have 5 steps before a penalty triggers.

The greedy agent now tries to load codellama. It takes 2 steps to load. The premium requests breach SLA at step 21. Score: 0.3.

An agent that learned from experience across many episodes does something different at step 8. It loads codellama speculatively during the quiet phase — paying a short-term cost in VRAM and latency — because it has learned that quiet periods precede spikes and that preloading pays off. At step 16 the model is already warm. All four premium requests are routed immediately. Zero SLA violations. Score: 0.85.

The reward for the `load_model` action taken at step 8 only arrived at step 16–18. That 10-step gap between action and reward is the credit assignment problem. No rule can bridge it. Only learned experience can.

---

## What makes this a genuine RL problem

Three properties together make LLMFleet-SRE unsolvable by any fixed rule or greedy heuristic.

**Non-stationary dynamics.** In task_longhaul, traffic shifts from quiet to spike to quiet within a single 50-step episode. The optimal policy during the quiet phase (conservative, preserve VRAM) is the wrong policy during the spike phase (aggressive, preload models). An agent must detect which phase it is in and adapt its behavior accordingly. No single rule handles both phases correctly.

**Stochastic arrivals.** Requests arrive via a Poisson process with phase-dependent rates. The agent cannot know whether the current quiet period will last 3 more steps or 10. Every episode with the same seed produces the same sequence, but different seeds produce genuinely different episodes. A policy memorized from one episode does not generalize to the next.

**Delayed consequences.** Loading a model takes 1–3 steps of stochastic latency and consumes VRAM immediately, but the reward for that decision only arrives when matching requests are routed — potentially 5–10 steps later. Evicting a model frees VRAM now but costs heavily if a matching premium request arrives in the next step. Every resource decision is a bet on the future, and the payout is delayed.

A greedy policy that only optimizes the current step consistently scores 0.30–0.45 on task_longhaul and task_hard. It over-commits VRAM during quiet periods, misses SLA windows during spikes, and never learns to speculate. An RL-trained agent that has seen hundreds of episodes learns the phase transition pattern and acts before the spike hits. That gap in performance — greedy at 0.43, learned policy at 0.85+ — is the empirical proof that this environment requires reinforcement learning.

---

## Greedy baseline verification

To verify that LLMFleet-SRE cannot be solved by a rule-based policy, run `greedy_baseline.py` against all four tasks:

```bash
python greedy_baseline.py
```

Verified output on this environment:

```
Greedy baseline scores:
----------------------------------------
task_easy            avg=1.000  seeds=['1.00', '1.00', '1.00', '1.00', '1.00']
task_medium          avg=0.573  seeds=['0.42', '0.57', '0.75', '0.45', '0.68']
task_hard            avg=0.450  seeds=['0.41', '0.48', '0.39', '0.51', '0.44']
task_longhaul        avg=0.426  seeds=['0.50', '0.68', '0.30', '0.35', '0.30']
----------------------------------------
task_hard and task_longhaul avg should be below 0.50 to confirm RL requirement
```

task_easy scores 1.0 for the greedy agent — intentionally trivial, designed to verify the API works. task_hard and task_longhaul both score below 0.50, confirming that both tasks require adaptive planning that a fixed rule cannot provide. An LLM agent running `inference.py` scores 0.60–0.90 on these tasks depending on the model used, demonstrating the performance gap that RL training would close further.

---

## Environment details

### Cluster topology

| Node     | VRAM  | Initial state                             |
| -------- | ----- | ----------------------------------------- |
| `node_a` | 80 GB | Healthy                                   |
| `node_b` | 80 GB | Healthy (crashed in task 2)               |
| `node_c` | 40 GB | Healthy (smaller — intentional asymmetry) |

### Available models

| Model               | VRAM required | Cost per step | Serves                       |
| ------------------- | ------------- | ------------- | ---------------------------- |
| `llama3-8b-chat`    | 18 GB         | 0.10          | chat requests                |
| `llama3-70b-chat`   | 45 GB         | 0.35          | chat requests (high quality) |
| `codellama-34b`     | 35 GB         | 0.25          | code requests                |
| `mistral-7b-sum`    | 16 GB         | 0.08          | summarize requests           |
| `code-lora-adapter` | 8 GB          | 0.05          | code fine-tuned adapter      |

### Observation space

Each step the agent receives a full cluster snapshot:

```python
class LLMFleetObservation(BaseModel):
    nodes: Dict[str, NodeState]           # VRAM, loaded models, status per node
    request_queue: List[IncomingRequest]  # task_type, sla_tier, required_model, age_steps
    step: int
    step_budget: int
    last_action_result: str               # human-readable result of last action
    sla_violations: int                   # cumulative premium SLA breaches
    requests_served: int                  # cumulative successful serves
```

### Action space

```python
class LLMFleetAction(BaseModel):
    action_type: Literal["route_batch", "load_model", "evict_model", "restart_node", "noop"]
    target_node: Optional[str]   # "node_a" | "node_b" | "node_c"
    model_name: Optional[str]    # model to load or evict
    request_ids: List[str]       # requests to route (for route_batch)
```

---

## Tasks

### Task 1 — Basic batch routing (easy)

**Setup:** Node A has `llama3-8b-chat` loaded and 62 GB free. Five chat requests sit in the queue.

**Objective:** Issue `route_batch` to clear the queue without triggering a VRAM overflow.

**Grader:** 1.0 if all five requests are served with no OOM crash. Partial credit proportional to requests served. Zero if any node crashes.

**What it tests:** Basic understanding of the action API and reading the observation correctly.

---

### Task 2 — OOM recovery under pressure (medium)

**Setup:** Node B has OOM-crashed and is offline. Node A is healthy with the chat model loaded. Three premium and four best-effort chat requests are queued. New requests arrive via Poisson process each step.

**Objective:** `restart_node` on Node B, `load_model` once it comes back online, and route the stalled traffic — all before the premium requests breach the 5-step SLA threshold.

**Grader:** Score decays with latency. Full queue cleared quickly → up to 1.0. Each SLA violation deducts 0.1. Queue not cleared → partial credit based on fraction served.

**What it tests:** Sequential multi-step planning — restart must happen before load, load must complete before routing.

---

### Task 3 — Dynamic model eviction under SLA conflict (hard)

**Setup:** Starting state is randomized each episode — nodes are loaded with different model combinations and VRAM levels. A queue of premium code and chat requests arrives simultaneously. No node starts with `codellama-34b` loaded, and depending on the sampled start state the agent may need eviction/rebalancing to serve both code and chat efficiently. Mid-episode, adversarial arrivals approximate pressure toward currently unavailable capabilities.

**Objective:** Choose which node to sacrifice, evict strategically, load `codellama-34b`, and serve the code queue — all while keeping the chat queue moving and anticipating adversarial follow-up requests.

**Grader:** Scored on fraction of total requests seen minus SLA and OOM penalties. Full credit requires serving both code and chat queues without crashing a node.

**What it tests:** Randomized state breaks memorization. Adversarial arrivals punish premature commitment. Requires learned eviction strategy across varied configurations — genuine RL.

---

### Task 4 — Sustained cluster management under traffic shift (hard / long-horizon)

**Setup:** Mixed cluster with models partially loaded. Episode runs 50 steps. Traffic shifts mid-episode: quiet (steps 1–15) → spike (steps 16–35) → quiet again (steps 36–50). During the spike, premium request rate jumps from 20% to 60% of arrivals.

**Objective:** Maintain cluster performance across the full 50-step episode. Preload models speculatively during the quiet phase before the spike hits. Route aggressively during the spike. Evict expensive idle models during the second quiet phase to reduce cost penalties.

**Grader:** Throughput fraction minus SLA penalties (0.08 per violation, capped at 0.5) minus OOM penalty (0.3 if any node crashes during the spike).

**What it tests:** Non-stationary dynamics — the optimal policy changes mid-episode. No fixed rule handles all three phases. Requires learned adaptive behavior from experience across episodes.

---

## Reward function

The reward at each step is the sum of six components:

| Component       | Signal                                                                        | Value |
| --------------- | ----------------------------------------------------------------------------- | ----- |
| Throughput      | Per request successfully served                                               | +0.20 |
| Latency penalty | Per request sitting in queue, per step                                        | −0.01 |
| Idle penalty    | Per healthy node with loaded models but nothing routable, per step            | −0.05 |
| OOM penalty     | Triggered when agent loads a model that exceeds node VRAM                     | −0.50 |
| SLA penalty     | Per premium request that has waited more than 5 steps                         | −0.30 |
| Cost penalty    | Penalizes expensive loaded models when cheaper queue-compatible options exist | < 0.0 |

The reward is decomposed into a `LLMFleetReward` object for transparency and debugging. The environment is not passable by inaction.

---

## Scoring

| Task          | Step budget | Greedy avg | Maximum score |
| ------------- | ----------- | ---------- | ------------- |
| task_easy     | 30          | 1.000      | 1.0           |
| task_medium   | 30          | 0.573      | 1.0           |
| task_hard     | 30          | 0.450      | 1.0           |
| task_longhaul | 50          | 0.426      | 1.0           |

Scores are normalized to [0.0, 1.0] and computed by the `grade(task_name, final_state)` function in `server/tasks.py`. Graders are fully deterministic — the same seed always produces the same episode and the same score.

---

## Setup and usage

### Prerequisites

- Python 3.10+
- Docker Desktop or Docker Engine
- `pip install openenv-core`

### Install

```bash
git clone https://huggingface.co/spaces/Ajeya95/llmfleet-sre
cd llmfleet-sre
pip install -e .
```

### Run locally

```bash
docker build -t llmfleet-sre -f server/Dockerfile .
docker run -p 7860:7860 llmfleet-sre
```

The server starts at `http://localhost:7860`. Test it:

```bash
curl -X POST http://localhost:7860/reset?task_name=task_easy
```

### Validate

```bash
openenv validate
```

All three checks must pass: HF Space responds, `openenv.yaml` is valid, Dockerfile builds.

### Push to Hugging Face Spaces

```bash
git add .
git commit -m "update"
git push origin main
```

Your Space URL: `https://ajeya95-llmfleet-sre.hf.space`

### Run baseline inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export TASK_NAME="task_easy"

python inference.py
```

Expected output format:

```
{"type": "START", "task": "task_easy", "env": "llmfleet-sre", "model": "meta-llama/Llama-3.1-8B-Instruct"}
{"type": "STEP", "step": 1, "action": "{...}", "reward": 0.2, "done": false, "error": null}
...
{"type": "END", "success": true, "steps": 8, "score": 0.74, "rewards": [...]}
```

---

## Project structure

```
llmfleet_sre/
├── __init__.py
├── models.py           # Pydantic schemas: Action, Observation, State, Reward
├── client.py           # OpenEnv async client wrapper
├── openenv.yaml        # Environment manifest
├── inference.py        # Baseline inference script (required at root)
├── greedy_baseline.py  # RL verification script
├── README.md
└── server/
    ├── app.py          # FastAPI server with /reset, /step, /state, /grade endpoints
    ├── environment.py  # Simulation logic: VRAM math, action execution, reward calc
    ├── tasks.py        # Task definitions and grader functions
    ├── Dockerfile
    └── requirements.txt
```

---

## Infrastructure requirements

| Resource                 | Requirement                                          |
| ------------------------ | ---------------------------------------------------- |
| vCPU                     | 2                                                    |
| Memory                   | 2 GB (well within 8 GB limit)                        |
| GPU                      | None — all VRAM is simulated                         |
| Inference script runtime | Under 5 minutes on task_easy, under 20 minutes total |

---

## Design decisions

**Why simulate VRAM as integers?** Real VRAM measurement requires a live GPU and `nvidia-smi`. Simulating it as integer arithmetic makes the environment portable, reproducible, and runnable anywhere — including CI, HF Spaces CPU instances, and laptops. The constraint logic is identical to the real problem.

**Why stochastic load latency?** In production, model cold-start time varies based on network speed, disk I/O, and cluster load. A fixed latency would allow a hardcoded policy to achieve a perfect score. Randomizing load time (1–3 steps) forces the agent to reason about uncertainty and plan defensively.

**Why Poisson arrivals?** Static queues allow the agent to memorize a fixed action sequence rather than learn a policy. Poisson arrivals with phase-dependent rates mean every episode is different and no memorized sequence generalizes.

**Why SLA tiers?** Flat queues make the scheduling problem trivial — serve oldest first. SLA tiers create genuine priority conflicts: should the agent evict a model to serve a premium request immediately, or batch process best-effort requests while it loads? This is the exact trade-off production inference schedulers face.

**Why randomize task_hard starting state?** A fixed starting state has a fixed optimal sequence discoverable by reasoning once. Randomizing which models are loaded on which nodes forces the agent to learn eviction strategies that generalize across configurations — achievable only through experience across many episodes.

**Why adversarial arrivals in task_hard?** Punishing the agent for evicting a model when a request for that model arrives shortly after teaches it to hedge — maintain at least one node capable of serving each request type rather than committing fully to one workload.

**Why three differently sized nodes?** Symmetric clusters have symmetric optimal policies. Node C at 40 GB cannot load `llama3-70b-chat` or `codellama-34b` alone — the agent must learn to use nodes selectively rather than treating them as interchangeable.

---

## Why this is different from the kernrl environment

Both environments deal with GPUs but at completely different layers of the stack.

**kernrl** tests whether an agent can write a faster CUDA/Triton kernel for a specific mathematical operation. It is a code-generation benchmark testing low-level GPU programming knowledge — warp primitives, memory coalescing, kernel tiling. It requires a live NVIDIA GPU just to run.

**LLMFleet-SRE** tests whether an agent can manage an entire fleet of nodes serving AI workloads. It requires zero GPUs and evaluates multi-step planning, resource constraint reasoning, SLA trade-offs, and adaptive policy learning under non-stationary dynamics.

`kernrl` writes the kernel that runs inside one GPU. `LLMFleet-SRE` orchestrates the cluster that decides which GPU runs which model for which user. They are complementary — one layer above the other — and do not compete.

---

## License

MIT
