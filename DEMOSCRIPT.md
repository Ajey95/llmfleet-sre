# LLMFleet-SRE Demo Script (Submission)

This is a reviewer-facing, runnable script to verify the submitted environment.

## 1) Build and run the server

```powershell
Set-Location d:\projects\openenv\llmfleet_sre
docker build -t llmfleet-sre:latest .
docker run --rm -d --name llmfleet-demo -p 7860:7860 llmfleet-sre:latest
```

Expected: container starts and serves HTTP on port 7860.

## 2) Confirm task registry

```powershell
Invoke-RestMethod -Uri "http://localhost:7860/tasks" -Method GET
```

Expected tasks: `task_easy`, `task_medium`, `task_hard`, `task_longhaul`.

## 3) Verify horizon settings

```powershell
$easy = Invoke-RestMethod -Uri "http://localhost:7860/reset?task_name=task_easy" -Method POST
$long = Invoke-RestMethod -Uri "http://localhost:7860/reset?task_name=task_longhaul" -Method POST
"easy=$($easy.observation.step_budget) longhaul=$($long.observation.step_budget)"
```

Expected output:

```text
easy=30 longhaul=50
```

## 4) Run greedy baseline verification

```powershell
python greedy_baseline.py
```

Expected pattern:

- `task_easy` near 1.0
- `task_hard` below 0.5 (around 0.45)
- `task_longhaul` below 0.5 (around 0.43)

## 5) Run inference script log-format check

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "<your_hf_token>"
$env:TASK_NAME = "task_easy"
python inference.py
```

Required log shape:

- one `START` JSON line
- multiple `STEP` JSON lines
- one `END` JSON line

## 6) Validate deployed HF Space

```powershell
$res = Invoke-RestMethod -Uri "https://ajeya95-llmfleet-sre.hf.space/reset?task_name=task_longhaul" -Method POST
$res.observation.step_budget
```

Expected output:

```text
50
```

## 7) Cleanup

```powershell
docker stop llmfleet-demo
```

---

## What this demonstrates

- OpenEnv-compatible API is live
- `task_longhaul` uses the required 50-step episode budget
- baseline scripts execute in expected format
- hard tasks remain non-trivial for greedy policy
