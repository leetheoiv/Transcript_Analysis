# Transcript Analysis Automation

Automates the end-to-end pipeline by which analysts turn a plain-language question into a validated, structured analysis of call transcripts. The system uses Jinja2 prompt templates, Pydantic structured outputs, and a Databricks/Azure OpenAI batch pipeline, with two human-approval gates and two automated judge passes before any full-batch run is triggered.

---

## Project Structure

```
Transcript_Analysis_Automation/
├── agents/
│   ├── __init__.py
│   ├── prompt_generator.py
│   ├── schema_generator.py
│   └── judge.py
├── db/
│   ├── __init__.py
│   ├── database.py
│   ├── repositories.py
│   ├── csv_repositories.py
│   ├── store.py
│   └── schema.sql
├── models/
│   ├── __init__.py
│   ├── run_spec.py
│   └── feedback.py
├── orchestrator/
│   ├── __init__.py
│   └── state_machine.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_repositories.py
│   ├── test_csv_repositories.py
│   ├── test_state_machine.py
│   └── test_notebook.ipynb
├── sandbox.ipynb
├── requirements.txt
└── README.md
```

---

## File Summaries

### `models/__init__.py`
Public re-exports for the `models` package. Import all shared data models from here (`RunSpec`, `RunStatus`, `JudgeFeedback`, `UserFeedback`, `RevisionBrief`) rather than from individual submodules, so import paths stay stable as the package grows.

### `models/run_spec.py`
Defines the core data models for a pipeline run:
- `RunStatus` — enum of every valid state in the orchestration state machine (`draft_prompt` → ... → `complete` / `flagged_for_review`).
- `DatasetParameters` — date range, arbitrary filters, and sample size used to select the transcript set for a run.
- `RunSpec` — the central record for a single analysis run. Holds the analyst's question, the generated Jinja2 prompt template, the serialised Pydantic output schema, tool definitions, current status, version number, lineage (parent version UUID), dataset parameters, a reference to the fixed validation transcript set, and aggregated judge/validation scores. Every state transition produces a new or updated `RunSpec` persisted to Postgres.

### `models/feedback.py`
Defines the three feedback and review data models:
- `JudgeFeedback` — per-row output from the `JudgeAgent`: a 0–1 score, free-text reasoning, and a list of typed issue flags. Produced during the sample-judge pass (step 6) and the full-set validation pass (step 9).
- `UserFeedback` — per-row comment entered by a human analyst during the sample review step (step 6). Tied to a specific `RunSpec` version and row ID.
- `RevisionBrief` — aggregates all `JudgeFeedback` and `UserFeedback` from a rejected sample run into a single object handed back to the `PromptGeneratorAgent` to drive the next revision cycle (step 7 → step 1).

### `db/__init__.py`
Public re-exports for the `db` package. Import `get_connection` and all repository classes from here.

### `db/schema.sql`
Postgres DDL for all four tables (`run_specs`, `judge_feedback`, `user_feedback`, `revision_briefs`). Run once against a fresh database to initialise the schema. Uses `ON CONFLICT` upsert on `run_specs` to support version updates without deletes.

### `db/database.py`
Connection helper. Loads Postgres credentials from `config/secrets.toml` under the `[postgres]` key and returns a `psycopg2` connection via `get_connection()`.

### `db/csv_repositories.py`
CSV-backed versions of all four repository classes. Stores each record as a newline-delimited JSON row under `data/` so nested fields (JSONB equivalents) round-trip cleanly. Exposes the exact same `insert()` / `get()` / `get_by_run_spec()` interface as the Postgres repositories — swap backends without touching any other code.

### `db/store.py`
Repository factory. Call `get_repositories()` to get a `Repositories` bundle (run_specs, judge_feedback, user_feedback, revision_briefs) for the active backend. Backend is selected by the `STORAGE_BACKEND` environment variable (`csv` by default, `postgres` when available).

### `tests/test_csv_repositories.py`
Unit tests for all four CSV repository classes using `tmp_path` for isolation. Covers insert, upsert, get, filtering by run_spec_id, latest-record semantics on revision briefs, and the `get_repositories` factory. No database or config required.

### `tests/test_notebook.ipynb`
Jupyter notebook version of all tests. Run cells top to bottom to execute every test and see `PASSED` / `FAILED` results inline. Each section mirrors a test file in `tests/`. No database or external services required. A final summary cell prints the total pass/fail count.

### `agents/__init__.py`
Public re-exports for the `agents` package. Import all three agent classes from here.

### `agents/prompt_generator.py`
`PromptGeneratorAgent` — takes an analyst's plain-language question, optional `metadata` dict, optional clarifying-question `answers`, and an optional `RevisionBrief`. Returns a `PromptGeneratorResult` which is either a finished Jinja2 template (`needs_clarification=False`) or a list of clarifying questions to present to the analyst (`needs_clarification=True`). On revision cycles the full judge and user feedback is injected into the prompt so the model knows exactly what to fix.

### `agents/schema_generator.py`
`SchemaGeneratorAgent` — takes the approved Jinja2 prompt and returns a `SchemaGeneratorResult` containing a JSON Schema dict, an optional tools list, and a `prompt_feedback` string. If `prompt_feedback` is non-empty the caller should route back to `PromptGeneratorAgent` before proceeding.

### `agents/judge.py`
`JudgeAgent` — scores a single extraction output row against the original question and prompt. Returns a `JudgeFeedback` instance (score 0–1, reasoning, typed flags). Used in both the sample judge pass (step 6) and the full-set validation pass (step 9).

### `tests/test_agents.py`
Unit tests for all three agent classes. All LLM calls are mocked via `unittest.mock` — no API key or network connection required. Covers correct output parsing, revision brief injection, markdown fence stripping, and empty-response error handling.

### `orchestrator/__init__.py`
Public re-exports for the `orchestrator` package. Import `RunSpecStateMachine` and `InvalidTransitionError` from here.

### `orchestrator/state_machine.py`
Explicit state machine for a `RunSpec` pipeline run. Holds the allowed-transitions map and a `RunSpecStateMachine` class that enforces valid state changes, updates `spec.status` and `spec.updated_at` in place, and persists every transition via the repository layer. Raises `InvalidTransitionError` on illegal moves and `ValueError` when a `RevisionBrief` is missing on the `sample_rejected → draft_prompt` cycle.

### `db/repositories.py`
One repository class per table. Each exposes `insert()` and `get()` / `get_by_run_spec()` methods that map directly between the Pydantic models and database rows. `RunSpecRepository.insert()` uses upsert so the same method handles both creates and status updates.

### `tests/test_state_machine.py`
Unit tests for `RunSpecStateMachine`. Covers initial status, valid and invalid transitions, `can_transition`, the full happy path to `complete`, the `sample_rejected → draft_prompt` revision cycle (with and without a `RevisionBrief`), and the `flagged_for_review` path. Uses the CSV backend with a temp directory — no database required.

### `tests/test_models.py`
Unit tests for all Pydantic models. Covers defaults, required fields, score bounds on `JudgeFeedback`, lineage on `RunSpec`, and `RevisionBrief` aggregation. No database required.

### `tests/test_repositories.py`
Integration tests for all four repository classes. Each test runs inside a transaction that is rolled back on teardown — no permanent test data is written. Requires a live Postgres database configured in `config/secrets.toml`.

### `sandbox.ipynb`
Interactive notebook for running the full agent pipeline against real transcripts without any infrastructure. Walks through: client setup → question + metadata definition → prompt generation (with clarifying question loop) → prompt approval → schema generation → schema approval → sample extraction → judge scoring → approve/reject with revision cycle. Edit `QUESTION`, `METADATA`, and `SAMPLE_TRANSCRIPTS` in cell 2 to get started.

### `spectrum_client.py`
Wrapper around the internal SpectrumGPT API (Azure OpenAI-compatible). Provides:
- `chat()` — single-turn call with optional Pydantic structured output.
- `run_agent()` — full agentic loop that executes tool calls until the model produces a plain-text response.
- `ask_transcript_questions()` — renders a Jinja2 prompt template, calls the API, and validates the response against a Pydantic model with automatic retry on validation failure.
- `process_batch()` — parallelised batch runner over a list of transcript rows; writes results to CSV incrementally.

Credentials (url, API_KEY, ASSISTANT_ID) are loaded from `config/secrets.toml`. This client is the execution backbone for the `PromptGeneratorAgent`, `SchemaGeneratorAgent`, `JudgeAgent`, and both the sample and full runners.

### `requirements.txt`
Python dependencies for the project. Currently includes Pydantic v2 (data models), LangChain (agent layer), pandas / numpy / scipy / statsmodels (data analysis), FastAPI-related and config utilities, Streamlit and reporting libraries, and existing project dependencies.

---

## Build Order

| Step | What | Status |
|------|------|--------|
| 1 | Data model layer (`RunSpec`, `JudgeFeedback`, `UserFeedback`, `RevisionBrief`) | ✅ Done |
| 2 | Postgres schema (`run_specs`, `judge_feedback`, `user_feedback`, `revision_briefs`) | ✅ Done |
| 3 | State machine / orchestrator | ✅ Done |
| 4 | Agent layer (`PromptGeneratorAgent`, `SchemaGeneratorAgent`, `JudgeAgent`) | ✅ Done |
| 5 | Celery wiring (tasks + beat) | ⬜ |
| 6 | FastAPI layer (endpoints + state transitions) | ⬜ |
| 7 | Full runner integration (Databricks/Azure OpenAI batch pipeline) | ⬜ |
| 8 | React frontend (approval screens, sample review grid, flagged-run dashboard) | ⬜ |
| 9 | Docker (containerise + docker-compose) | ⬜ |
