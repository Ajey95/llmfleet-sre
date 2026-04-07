# LLMFleet-SRE Demo Script

This script is for a clean demo of the environment, including the longhaul step budget check.

## 1) Build the Docker image

```powershell
Set-Location d:\projects\openenv\llmfleet_sre
docker build -t llmfleet-sre:latest .
```

Expected: build succeeds and image `llmfleet-sre:latest` exists.

## 2) Run locally on port 7860

```powershell
docker run --rm -d --name llmfleet-demo -p 7860:7860 llmfleet-sre:latest
```

Expected: container starts without errors.

## 3) Verify task list is available

```powershell
Invoke-RestMethod -Uri "http://localhost:7860/tasks" -Method GET
```

Expected: includes `task_easy`, `task_medium`, `task_hard`, `task_longhaul`.

## 4) Verify longhaul reset uses 50-step budget

```powershell
$res = Invoke-RestMethod -Uri "http://localhost:7860/reset?task_name=task_longhaul" -Method POST
$res.observation.step_budget
```

Expected output:

```text
50
```

## 5) Quick check that easy task still resets normally

```powershell
$res = Invoke-RestMethod -Uri "http://localhost:7860/reset?task_name=task_easy" -Method POST
$res.observation.step_budget
```

Expected output:

```text
30
```

## 6) Run baseline inference logs format check

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "<your_hf_token>"
$env:TASK_NAME = "task_easy"
python inference.py
```

Expected log markers in stdout:

- JSON line with `"type":"START"`
- Multiple JSON lines with `"type":"STEP"`
- Final JSON line with `"type":"END"`

## 7) Stop local container

```powershell
docker stop llmfleet-demo
```

## 8) Verify deployed HF Space (after push/redeploy)

```powershell
$res = Invoke-RestMethod -Uri "https://ajeya95-llmfleet-sre.hf.space/reset?task_name=task_longhaul" -Method POST
$res.observation.step_budget
```

Expected output:

```text
50
```

If this still shows `30`, the Space is serving an older revision and needs another deploy cycle.
