# Running Celery Locally

## Prerequisites

1. **Redis** running on `localhost:6379`

   Easiest on Windows — run via Docker:
   ```bash
   docker run -d -p 6379:6379 --name redis redis:latest
   ```

   Or install [Memurai](https://www.memurai.com/) for a native Windows Redis-compatible server.

2. **Python dependencies** installed:
   ```bash
   pip install celery[redis] redis
   ```

---

## How It All Fits Together

```
┌────────────────────────────────────────────────────────────────┐
│                        YOUR MACHINE                             │
│                                                                │
│  ┌──────────┐     ┌───────┐     ┌─────────────────────────┐  │
│  │ Your     │────▶│ Redis │────▶│ Worker Process(es)      │  │
│  │ Script / │     │:6379  │     │                         │  │
│  │ FastAPI  │◀────│       │◀────│  - picks up tasks       │  │
│  └──────────┘     └───────┘     │  - runs your functions  │  │
│       │                         │  - stores results       │  │
│   .delay()                      └─────────────────────────┘  │
│   or .apply_async()                                           │
└────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step

### 1. Start Redis

```bash
docker start redis
```

Or if first time:
```bash
docker run -d -p 6379:6379 --name redis redis:latest
```

### 2. Start a Worker

Open a terminal in the project root and run:

```bash
# Single worker listening on ALL queues (good for dev):
celery -A worker.celery_app worker --loglevel=info -Q default,orchestration,generation,extraction,judging,evaluation
```

You'll see output like:
```
 -------------- celery@YOUR-PC v5.3.x
--- ***** -----
-- ******* ---- [config]
- *** --- * --- .> app:         transcript_pipeline
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/1
- ** ---------- .> concurrency: 8 (prefork)
- *** --- * ---
-- ******* ----
--- ***** ----- [queues]
 -------------- .> default          exchange=default(direct) key=default
                .> orchestration    exchange=orchestration(direct) key=orchestration
                .> generation       exchange=generation(direct) key=generation
                .> extraction       exchange=extraction(direct) key=extraction
                .> judging          exchange=judging(direct) key=judging
                .> evaluation       exchange=evaluation(direct) key=evaluation

[tasks]
  . worker.tasks.evaluation.run_evaluation
  . worker.tasks.extraction.run_extraction_batch
  . worker.tasks.generation.generate_prompt
  . worker.tasks.generation.generate_schema
  . worker.tasks.judging.run_judging_batch
  . worker.tasks.pipeline.run_pipeline

[2026-08-17 12:00:00] Ready.
```

### 3. Dispatch a Task

In a separate terminal (or a Python script / notebook):

```python
from worker.tasks.generation import generate_prompt

# .delay() = fire and forget (async)
result = generate_prompt.delay(
    user_input="What value addition opportunities exist in this call?",
    model="gpt-4.1-mini",
    temperature="0.4",
)

# result is an AsyncResult — the task is running in the worker process
print(result.id)        # task UUID
print(result.status)    # PENDING → STARTED → SUCCESS / FAILURE

# Block and wait for the result (useful for testing):
output = result.get(timeout=120)
print(output)
# {"status": "success", "system_prompt": "...", "user_prompt": "...", ...}
```

### 4. Run the Full Pipeline

```python
from worker.tasks.pipeline import run_pipeline
import pandas as pd

df = pd.read_csv("tests/TEST_DATA.csv")
rows = df.to_dict("records")

result = run_pipeline.delay(
    user_input="What value addition opportunities exist in this call?",
    rows=rows,
    output_dir=r"C:\Users\P3311043\Python Projects\Transcript_Analysis_Automation\tests",
    project_name="VAO",
    max_workers=5,
    n_runs=3,
)

# Check status
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE

# Wait for completion
output = result.get(timeout=600)
print(output["status"])
print(output["evaluation"])
```

---

## Useful Commands

| Command | What it does |
|---------|-------------|
| `celery -A worker.celery_app worker --loglevel=info` | Start a worker (all queues) |
| `celery -A worker.celery_app worker -Q extraction --concurrency=20` | Worker for extraction only, 20 threads |
| `celery -A worker.celery_app worker -Q generation --concurrency=4` | Worker for generation only, 4 threads |
| `celery -A worker.celery_app inspect active` | Show currently running tasks |
| `celery -A worker.celery_app inspect reserved` | Show queued tasks waiting for a worker |
| `celery -A worker.celery_app inspect stats` | Worker statistics |
| `celery -A worker.celery_app purge` | Clear all queued tasks (careful!) |
| `celery -A worker.celery_app flower` | Start web UI for monitoring (install flower first) |

---

## Production-Style (Multiple Specialized Workers)

When you want isolation between task types, run separate workers:

```bash
# Terminal 1 — generation (low concurrency, fast response)
celery -A worker.celery_app worker -Q generation --concurrency=4 -n gen@%h

# Terminal 2 — extraction (high concurrency, I/O-bound)
celery -A worker.celery_app worker -Q extraction --concurrency=30 -P gevent -n ext@%h

# Terminal 3 — judging (high concurrency, I/O-bound)
celery -A worker.celery_app worker -Q judging --concurrency=30 -P gevent -n judge@%h

# Terminal 4 — evaluation (CPU-bound, use prefork)
celery -A worker.celery_app worker -Q evaluation --concurrency=4 -P prefork -n eval@%h

# Terminal 5 — orchestration (lightweight coordinator)
celery -A worker.celery_app worker -Q orchestration --concurrency=2 -n orch@%h
```

`-n gen@%h` gives each worker a unique name (required when running multiple on one machine).
`-P gevent` uses green threads (better for I/O-bound tasks — needs `pip install gevent`).

---

## Monitoring with Flower (Optional)

```bash
pip install flower
celery -A worker.celery_app flower --port=5555
```

Open `http://localhost:5555` — gives you a real-time dashboard of tasks, workers, and queues.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` on Redis | Make sure Redis is running: `docker start redis` |
| Tasks stuck in PENDING | Worker might not be listening on that queue. Check `-Q` flag. |
| `ModuleNotFoundError` in worker | Run the worker from the project root so imports resolve. |
| Results disappear | Results expire after 24h (configurable in `celery_app.py`). |
| Windows `prefork` issues | Windows doesn't support `prefork` well. Use `--pool=solo` or `--pool=threads`. |
