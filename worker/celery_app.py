"""
worker/celery_app.py

Central Celery application instance.

This is the single entry point that:
- Connects to Redis (broker + result backend)
- Registers task modules
- Configures serialization, queues, and routing

Usage:
    from worker.celery_app import app

    # Start a worker (from project root):
    # celery -A worker.celery_app worker --loglevel=info
"""

from celery import Celery

# -------------------------------------------------------------------
# App instance
# -------------------------------------------------------------------
# broker: where tasks are queued (Redis)
# backend: where results are stored (also Redis)

app = Celery(
    "transcript_pipeline",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# -------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------
# JSON is safer and more debuggable than pickle.
# All task arguments and return values must be JSON-serializable.

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]

# -------------------------------------------------------------------
# Result settings
# -------------------------------------------------------------------
# Results expire after 24 hours (don't fill up Redis forever)
app.conf.result_expires = 86400  # seconds

# -------------------------------------------------------------------
# Task discovery
# -------------------------------------------------------------------
# Celery will import these modules and find any @app.task decorators.


app.conf.include = [
    "worker.tasks.generation",
    "worker.tasks.extraction",
    "worker.tasks.judging",
    "worker.tasks.evaluation",
    "worker.tasks.pipeline",
]

# -------------------------------------------------------------------
# Queue & Routing Configuration
# -------------------------------------------------------------------
# Queues define WHERE tasks wait. Routing decides WHICH queue a task goes to.
#
# Think of it like airport terminals:
# - "generation" terminal = small, few gates (low concurrency)
# - "extraction" terminal = big, many gates (high concurrency)
# - A task's route = its boarding pass to a specific terminal
#
# Workers subscribe to specific queues:
#   celery -A worker.celery_app worker -Q extraction --concurrency=30
#
# If no queue is specified, tasks go to "default".

from kombu import Queue

app.conf.task_queues = [
    Queue("default"),           # Fallback for unrouted tasks
    Queue("orchestration"),     # Pipeline coordination (lightweight)
    Queue("generation"),        # Prompt + schema generation (few concurrent, latency-sensitive)
    Queue("extraction"),        # Batch extraction (many concurrent, I/O-bound)
    Queue("judging"),           # Batch judging (many concurrent, I/O-bound)
    Queue("evaluation"),        # Metrics computation (CPU-bound, fast)
]

# Route tasks to queues by task name.
# Format: "module.task_name" -> {"queue": "queue_name"}
app.conf.task_routes = {
    # Generation tasks → low-concurrency queue
    "worker.tasks.generation.generate_prompt": {"queue": "generation"},
    "worker.tasks.generation.generate_schema": {"queue": "generation"},

    # Extraction → high-concurrency I/O queue
    "worker.tasks.extraction.run_extraction_batch": {"queue": "extraction"},

    # Judging → high-concurrency I/O queue
    "worker.tasks.judging.run_judging_batch": {"queue": "judging"},

    # Evaluation → CPU queue
    "worker.tasks.evaluation.run_evaluation": {"queue": "evaluation"},

    # Pipeline coordination → lightweight queue
    "worker.tasks.pipeline.run_pipeline": {"queue": "orchestration"},
}

# Default queue for anything not explicitly routed
app.conf.task_default_queue = "default"
