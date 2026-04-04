---
title: LLMFleet-SRE
emoji: 🖥️
colorFrom: blue
colorTo: purple
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
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-informational)](https://www.docker.com/)
[![GPU Required](https://img.shields.io/badge/GPU-not%20required-success)](https://meta-pytorch.org/OpenEnv/)

---

## What LLMFleet-SRE actually is

You are building a simulation of a company's LLM serving infrastructure. Think of any startup or team that offers an AI API they have a cluster of GPU machines running models like Llama or Mistral, and they receive thousands of requests per minute from users. Managing that cluster is a real, painful, daily job. Your environment turns that job into an RL problem.

The simulation has three GPU nodes call them Node A, Node B, and Node C. Each node has a fixed amount of VRAM: Node A and B have 80 GB each, Node C has only 40 GB. These nodes are the physical machines that would serve AI requests in a real company. None of this actually uses a GPU you're just tracking numbers in Python. If Node A has 80 GB total and you load a model that takes 35 GB, Node A now has 45 GB remaining. That's the entire "simulation."

The cluster holds a catalogue of five models with different sizes. The chat model takes 18 GB, the code model takes 35 GB, the summarization model takes 16 GB, and so on. A model must be loaded onto a node before that node can serve requests that require it. Loading takes 1 to 3 steps because you added stochastic latency to simulate real cold-start variability. Once loaded, the model stays until someone explicitly evicts it.

Requests arrive in a queue each episode. Each request has three properties: what type of task it is (chat, code, or summarize), which model it therefore requires, and what SLA tier it belongs to either premium or best-effort. Premium requests are paying customers. Best-effort requests can wait. A premium request that sits in the queue for more than 5 steps without being served triggers an SLA penalty.

The agent the LLM you're evaluating reads the full cluster state as a JSON observation at every step and must output one action. The five possible actions are: route a batch of requests to a specific node, load a model onto a node, evict a model from a node to free VRAM, restart a crashed node, or do nothing. The agent cannot see the future it doesn't know which requests are coming next. It only sees the current queue, the current node states, and how long each request has been waiting.

The reward function is the core design. Five signals combine every step: you earn +0.2 for every request successfully served. You lose 0.01 for every request sitting in the queue per step, which creates constant pressure to act fast. You lose 0.05 for every node that is healthy and has models loaded but is sitting idle while the queue has requests it could theoretically serve this penalizes wasted compute. You lose 0.5 immediately if you try to load a model onto a node that doesn't have enough free VRAM this is the OOM crash penalty, the most expensive mistake. And you lose 0.3 every time a premium request ages past 5 steps unanswered. An agent that does nothing scores roughly 0.15 per step purely from latency and idle penalties, so passivity is not an option.

The three tasks are the same environment with different starting conditions. Task 1 is easy: Node A already has the right model loaded, five requests are in the queue, just route them. It tests whether the agent understands the basic API. Task 2 is medium: Node B has crashed with an OOM error, the queue is backing up, and new premium requests keep arriving every few steps. The agent must restart Node B, wait for it to come back, load the model, and route traffic three sequential actions where each one blocks on the previous. If the agent is slow, the SLA clock runs out on the premium requests. Task 3 is hard: every node is full of chat and summarize models, and suddenly four premium code requests land in the queue. There is no free VRAM anywhere. The agent must choose a node to sacrifice, evict one of its models, load the code model while tolerating 13 steps of load latency, and then serve the code queue all without letting the chat queue collapse on the other two nodes simultaneously.

The graders score each task on a 0.0 to 1.0 scale with deterministic logic. In task 1, did any node OOM crash? If yes, zero. Otherwise, fraction of requests served. In task 2, score decays with how many steps it took to clear the queue, minus 0.1 per SLA violation. In task 3, partial credit for each queue cleared independently, with SLA violations deducting 0.15 each up to a maximum deduction of 0.5.

The meta-angle the thing that makes this genuinely novel is that you are using an LLM to simulate the job of deciding how to serve other LLMs. The agent reads model names like `llama3-8b-chat` and `codellama-34b`, VRAM numbers that correspond to real model sizes, and request types that map to real use cases. A well-trained agent on this environment would produce scheduling decisions that are directly interpretable as infrastructure policy. That is not a toy problem. That is the exact reasoning loop that an on-call engineer at Hugging Face or a cloud provider runs through at 2 AM when the cluster is misbehaving.

---

## Why this requires Reinforcement Learning (and not just prompting)

In standard prompt engineering, an LLM answers a question in isolation. There is no persistent memory, no consequence for intermediate mistakes, and no sequence of decisions that build on each other. LLMFleet-SRE is a true Reinforcement Learning problem because it features **delayed consequence requiring multi-step planning**.

The critical property distinguishing this environment from a standard prompt is the **credit assignment problem**. For example, in Task 3:

1. The agent sees a sudden spike in premium code requests but lacks the VRAM to load the model.
2. It must explicitly `evict_model` (Step 1).
3. It must incur a short-term latency penalty while calling `load_model` and waiting out the 13 step stochastic cold-start (Steps 24).
4. Only then can it successfully `route_batch` and clear the queue (Step 5).

The massive reward for clearing the queue at Step 5 was entirely contingent on the painful eviction decision made at Step 1. A greedy, prompt-based LLM that only optimizes for the current step will endlessly stall or instantly trigger an OOM crash. Maximizing the compound reward requires planning the entire sequence ahead of time, navigating a stateful world where actions have lasting consequences.

---

## Environment details

### Cluster topology

| Node     | VRAM  | Initial state                           |
| -------- | ----- | --------------------------------------- |
| `node_a` | 80 GB | Healthy                                 |
| `node_b` | 80 GB | Healthy (crashed in task 2)             |
| `node_c` | 40 GB | Healthy (smaller intentional asymmetry) |

### Available models

| Model               | VRAM required | Serves                       |
| ------------------- | ------------- | ---------------------------- |
| `llama3-8b-chat`    | 18 GB         | chat requests                |
| `llama3-70b-chat`   | 45 GB         | chat requests (high quality) |
| `codellama-34b`     | 35 GB         | code requests                |
| `mistral-7b-sum`    | 16 GB         | summarize requests           |
| `code-lora-adapter` | 8 GB          | code fine-tuned adapter      |

### Observation space

Each step the agent receives a full cluster snapshot:

```python
class LLMFleetObservation(BaseModel):
    nodes: Dict[str, NodeState]       # VRAM, loaded models, status per node
    request_queue: List[IncomingRequest]  # task_type, sla_tier, required_model, age_steps
    step: int
    step_budget: int
    last_action_result: str           # human-readable result of last action
    sla_violations: int               # cumulative premium SLA breaches
    requests_served: int              # cumulative successful serves
```

### Action space

```python
class LLMFleetAction(BaseModel):
    action_type: Literal["route_batch", "load_model", "evict_model", "restart_node", "noop"]
    target_node: Optional[str]        # "node_a" | "node_b" | "node_c"
    model_name: Optional[str]         # model to load or evict
    request_ids: List[str]            # requests to route (for route_batch)
```

---

## Tasks

### Task 1 Basic batch routing (easy)

**Setup:** Node A has `llama3-8b-chat` loaded and 62 GB free. Five chat requests sit in the queue.

**Objective:** Issue `route_batch` to clear the queue without triggering a VRAM overflow.

**Grader:** 1.0 if all five requests are served with no OOM crash. Partial credit proportional to requests served. Zero if any node crashes.

**What it tests:** Basic understanding of the action API and reading the observation correctly.

---

### Task 2 OOM recovery under pressure (medium)

**Setup:** Node B has OOM-crashed and is offline. Node A is healthy with the chat model loaded. Three premium and four best-effort chat requests are queued. New premium requests arrive every three steps.

**Objective:** `restart_node` on Node B, `load_model` once it comes back online, and route the stalled traffic all before the premium requests breach the 5-step SLA threshold.

**Grader:** Score decays with latency. Full queue cleared quickly up to 1.0. Each SLA violation deducts 0.1. Queue not cleared partial credit based on fraction served.

**What it tests:** Sequential multi-step planning restart must happen before load, load must complete before routing.

---

### Task 3 Dynamic model eviction under SLA conflict (hard)

**Setup:** All three nodes are full of chat and summarize models. A sudden influx of four premium code requests arrives. No node has `codellama-34b` loaded and no node has free VRAM to load it without evicting something first.

**Objective:** Choose which node to free, evict a model, load the code model (accounting for stochastic load latency of 13 steps), and serve the code queue all while keeping the chat queue moving on the remaining nodes.

**Grader:** Full credit (1.0) only if both code and chat queues are cleared with two or fewer SLA violations. Partial credit for each queue served independently. SLA violations deduct 0.15 each, capped at 0.5 total deduction.

**What it tests:** Resource trade-off reasoning, multi-queue management, tolerance for stochastic latency.

---

## Reward function

The reward at each step is the sum of five components:

| Component       | Signal                                                             | Value |
| --------------- | ------------------------------------------------------------------ | ----- |
| Throughput      | Per request successfully served                                    | +0.2  |
| Latency penalty | Per request sitting in queue, per step                             | 0.01  |
| Idle penalty    | Per healthy node with loaded models but nothing routable, per step | 0.05  |
| OOM penalty     | Triggered when agent loads a model that exceeds node VRAM          | 0.5   |
| SLA penalty     | Per premium request that has waited more than 5 steps              | 0.3   |

The reward is decomposed into a `LLMFleetReward` object for transparency and debugging. A baseline agent that does nothing scores approximately 0.15 per step due to latency and idle penalties the environment is not passable by inaction.

---

## Scoring

| Task        | Maximum score | Baseline score |
| ----------- | ------------- | -------------- |
| task_easy   | 1.0           | 0.2            |
| task_medium | 1.0           | 0.1            |
| task_hard   | 1.0           | 0.0            |

Scores are normalized to [0.0, 1.0] and computed by the `grade(task_name, final_state)` function in `server/tasks.py`. Graders are fully deterministic the same sequence of actions always produces the same score.

---

## Setup and usage

### Prerequisites

- Python 3.11+
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
cd server
docker build -t llmfleet-sre .
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

1. Authenticate your terminal:

```bash
huggingface-cli login
```

2. Confirm you are in the project root (the folder containing `openenv.yaml`, `inference.py`, and `server/`).

3. Push the environment with OpenEnv:

```bash
openenv push --repo-id <your-hf-username>/llmfleet-sre
```

4. Wait for the build to finish. The CLI prints your Space URL:

```text
https://<your-hf-username>-llmfleet-sre.hf.space
```

Use that Space URL as your submission URL.

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
 __init__.py
 models.py          # Pydantic schemas: Action, Observation, State, Reward
 client.py          # OpenEnv async client wrapper
 openenv.yaml       # Environment manifest
 inference.py       # Baseline inference script (required at root)
 README.md
 server/
     app.py         # FastAPI server with /reset, /step, /state, /grade endpoints
     environment.py # Simulation logic: VRAM math, action execution, reward calc
     tasks.py       # Task definitions and grader functions
     Dockerfile
     requirements.txt
```

---

## Infrastructure requirements

| Resource                 | Requirement                                          |
| ------------------------ | ---------------------------------------------------- |
| vCPU                     | 2                                                    |
| Memory                   | 2 GB (well within 8 GB limit)                        |
| GPU                      | None all VRAM is simulated                           |
| Inference script runtime | Under 5 minutes on task_easy, under 20 minutes total |

---

## Design decisions

**Why simulate VRAM as integers?** Real VRAM measurement requires a live GPU and `nvidia-smi`. Simulating it as integer arithmetic makes the environment portable, reproducible, and runnable anywhere including CI, HF Spaces CPU instances, and laptops. The constraint logic is identical to the real problem.

**Why stochastic load latency?** In production, model cold-start time varies based on network speed, disk I/O, and cluster load. A fixed latency would allow a hardcoded policy to achieve a perfect score. Randomizing load time (13 steps) forces the agent to reason about uncertainty and plan defensively.

**Why SLA tiers?** Flat queues make the scheduling problem trivial serve oldest first. SLA tiers create genuine priority conflicts: should the agent evict a model to serve a premium request immediately, or batch process best-effort requests while it loads? This is the exact trade-off production inference schedulers face.

**Why three differently sized nodes?** Symmetric clusters have symmetric optimal policies. Node C at 40 GB cannot load `llama3-70b-chat` or `codellama-34b` alone the agent must learn to use nodes selectively rather than treating them as interchangeable.

---

## Why this is different from Kernrl Environment

If you are comparing this environment to the existing `kernrl` OpenEnv environment, keep in mind they tackle fundamentally different problems at opposite ends of the GPU stack:

**kernrl** tests whether an agent can write a faster CUDA/Triton software kernel for a specific mathematical operation. It is a code-generation benchmark testing low-level GPU programming knowledge (warp primitives, memory coalescing). It actually requires a live NVIDIA GPU to measure the compiled code's speedup.

**LLMFleet-SRE** tests whether an agent can act as the infrastructure scheduler across an entire live fleet. It requires zero GPUs and instead evaluates multi-step planning, resource constraints, queuing SLA trade-offs, and sequential decision-making.

`kernrl` writes the code layer. `LLMFleet-SRE` orchestrates the operational load-balancing layer entirely above it. They are complementary challenges, not competing ones, offering distinct, immense value to the community.

---

## License

MIT
