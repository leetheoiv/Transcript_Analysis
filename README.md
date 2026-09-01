# Transcript Analysis Automation

Turns a plain-language analyst question into a validated, structured extraction over call transcripts — automatically. You describe what you want to know; the system generates an extraction prompt and output schema, runs the extraction across transcripts, judges every field against the transcript evidence, scores quality, and (when quality is below threshold) revises the prompt and tries again.

The pipeline is built around **LLM agents** coordinated by an **orchestrator**, with human-approval gates, an automated grounding **judge**, and an automated **prompt-revision loop** that keeps the best-scoring prompt.

---

## What it does (end to end)

```
Analyst question
      │
      ▼
┌─────────────────┐   review    ┌─────────────────┐   review
│ Prompt          │────gate────▶│ Schema          │────gate────▶ Extraction
│ Generator Agent │             │ Generator Agent │             (N runs for
└─────────────────┘             └─────────────────┘              consistency)
                                                                     │
                                                                     ▼
                                                              ┌────────────┐
                                                              │ Judge      │  grounds every
                                                              │ Agent      │  field in the
                                                              └────────────┘  transcript
                                                                     │
                                                                     ▼
                                                              ┌────────────┐
                                                              │ Evaluation │  correctness /
                                                              │ + Assess   │  consistency /
                                                              └────────────┘  hallucination
                                                                     │
                                          PASS ◀───────────── threshold check ─────────────▶ FAIL
                                                                                              │
                                                                              aggregate judge feedback
                                                                              into a revision brief,
                                                                              regenerate the prompt,
                                                                              re-run — keep the best
                                                                              prompt, stop early if the
                                                                              failures are retrieval-driven
```

---

## Core concepts

- **Agents** — focused LLM workers (prompt generation, schema generation, extraction, judging). All inherit a common base and talk to the LLM through a single client.
- **Orchestrator** — drives the whole run: prompt → schema → extraction → judging → evaluation → assessment → optional revision loop. Owns the human-in-the-loop gates and the best-prompt selection.
- **Judge / grounding** — for each extracted field, the judge retrieves supporting evidence from the transcript (quote-first), then decides whether the claim is grounded. It distinguishes a genuine extraction error from a case where it simply could not retrieve evidence.
- **Evaluation** — turns judge output into correctness, consistency, and hallucination rates (per field and overall), which the orchestrator compares against thresholds to decide PASS/FAIL.
- **Category registry** — an optional controlled-vocabulary tool that keeps categorical fields consistent across transcripts, scoped per field, with fuzzy de-duplication.

---

## Project structure

```
Transcript_Analysis_Automation/
├── agents/                     # LLM agents
│   ├── BaseAgent.py            # Shared base (LLM calls, Jinja2 rendering, tool loop)
│   ├── prompt_generator_agent.py
│   ├── schema_generator_agent.py
│   ├── ExtractionAgent.py      # Structured extraction (inline + tool/search modes)
│   └── JudgeAgent.py           # Grounding judge + evidence retrieval
│
├── orchestration/
│   ├── orchestrator.py         # The pipeline coordinator (see below)
│   └── exceptions.py           # Typed pipeline errors
│
├── data_models/                # Pydantic models
│   ├── agent_models/           # Per-agent request/response models (incl. Judge)
│   ├── prompt_datamodel.py     # PromptManagement / prompt records
│   ├── prompt_eval_datamodel.py# SemanticQuality, ConsistencyQuality, EvaluationResult
│   ├── judge_aggregation.py    # JudgeAggregation + revision-brief rendering
│   └── ...
│
├── prompts/
│   ├── agent_prompts/          # System/user prompt templates per agent (judge, etc.)
│   └── general_prompts/        # Shared prompts (e.g. search-term refinement)
│
├── tools/                      # Reusable capabilities agents can call
│   ├── CATEGORY_TOOLS/         # Category registry (controlled vocabulary)
│   ├── CHUNKING_TOOLS/         # Transcript chunk extraction/formatting
│   ├── SEARCH_TOOLS/           # Search-term derivation / n-grams
│   ├── EVAL_TOOLS/             # Judge aggregation, semantic-quality helpers
│   ├── RAG_TOOLS/ · WRITE_TOOLS/ · DIRECTORY_TOOLS/
│   └── utils/                  # transcript_helpers (matching), build_tool, schema loader
│
├── utils/
│   ├── bootstrap_extraction.py # Repeated-run extraction + consistency/semantic eval
│   ├── hitl_strategy.py        # Human-in-the-loop strategies (terminal / async)
│   ├── get_template_params.py · save_file.py · human_in_the_loop.py
│
├── worker/                     # Celery async execution (see worker/README.md)
│   ├── celery_app.py
│   └── tasks/                  # generation, extraction, judging, evaluation, pipeline
│
├── api/                        # FastAPI service
│   ├── main.py                 # REST endpoints (create/monitor runs, HITL decisions)
│   ├── run_manager.py          # In-memory run store + execution
│   └── models.py               # Request/response models
│
├── db/                         # Postgres helpers + schema.sql
├── config/                     # secrets.toml (credentials) + template
├── spectrum_client.py          # LLM client (Azure OpenAI-compatible)
├── streamlit_app.py            # Streamlit UI for running the pipeline
├── tests/                      # Test data + per-project extraction definitions
└── requirements.txt
```

---

## Components in detail

### `agents/`

- **`BaseAgent.py`** — the shared foundation. Handles LLM chat calls, Jinja2 prompt rendering (`_render_user_prompt`), message assembly, tool registration, and the tool-calling loop. Every agent inherits from it.
- **`prompt_generator_agent.py`** — `PromptGeneratorAgent`. Turns the analyst's question into a system prompt, user prompt (Jinja2 template with a `{{TRANSCRIPT}}` slot), and an output-format spec. Classifies intent (generate vs. converse) and, on revision cycles, injects a structured **revision brief** so the model knows exactly what to fix.
- **`schema_generator_agent.py`** — `SchemaGeneratorAgent`. Converts the approved output format into a Pydantic model file the extractor validates against.
- **`ExtractionAgent.py`** — `TranscriptExtractionAgent`. Runs the extraction. Uses **inline mode** for short transcripts and **tool/search mode** for long ones (searches the transcript for relevant chunks instead of stuffing the whole thing into context). Emits, per field, a `*_claim_field` value plus a supporting `*_quote` and `*_reasoning`. If category tools are attached, it's instructed to assign categories through the registry.
- **`JudgeAgent.py`** — `JudgeAgent`. For each extracted field it:
  1. **Retrieves evidence** from the transcript — quote-first, then sentence fragments, short snippets, n-grams, reasoning-derived terms, and finally LLM-generated search terms. Returns the matched lines with surrounding context.
  2. Asks the LLM whether the claim is **grounded** in that evidence and how the claim relates to it (`claim_presence`: supported / contradicted / absent / no_evidence).
  3. Derives a **`retrieval_status`** (`supported`, `contradicted`, `absent_from_transcript`, `retrieval_failure`) so downstream evaluation can tell a real extraction error apart from a retrieval miss.
  4. Stores the **verbatim** matched transcript text as evidence (no line labels, no rephrasing).

### `orchestration/orchestrator.py`

The `Orchestrator` runs the full pipeline:

1. **Prompt generation** → optional **human review gate** (approve / retry-with-notes / cancel).
2. **Schema generation** → optional **human review gate**.
3. **Extraction** — runs `n_runs` times (repeated runs power the consistency metric). Can run judging concurrently with the remaining extraction runs.
4. **Judging** — grounds every field against the transcript.
5. **Evaluation** — combines consistency (across runs) with semantic quality (from the judge) into an `EvaluationResult`.
6. **Assessment** — compares correctness / consistency / hallucination against thresholds → PASS or FAIL.
7. **Automated revision loop** (on FAIL): aggregate judge feedback into a revision brief → regenerate the prompt → re-run extraction/judging/evaluation, up to `max_revision_cycles`.

Notable behaviors:

- **Best-prompt selection & rollback** — each cycle is scored; a revision that regresses quality is never left as the final output. If the last cycle isn't the best, the orchestrator restores the best-scoring prompt.
- **Retrieval-health circuit-breaker** — if most failures are the judge failing to retrieve evidence (a data problem, not a prompt problem), the loop stops early and reports `revision_stopped_reason` instead of burning cycles.
- **Correctness source** — correctness can come from the judge's grounding verdict (default) or from **gold labels** you supply (`correctness_source="gold"`), measured independently of the judge.

### `data_models/`

Pydantic models used across the pipeline. Highlights:
- `prompt_eval_datamodel.py` — `SemanticQuality` (correctness / hallucination / retrieval-failure rates + `correctness_source`), `ConsistencyQuality`, and `EvaluationResult` (with per-field breakdowns).
- `agent_models/` — the judge's `JudgeFieldResult` and `GroundingJudgeLLMResponse` (grounded, hallucinated, evidence strength, `claim_presence`, error type).
- `judge_aggregation.py` — `JudgeAggregation` and `to_revision_brief_text()`, which renders the structured feedback the prompt generator consumes.

### `tools/`

- **`CATEGORY_TOOLS/category_registry.py`** — a CSV-backed controlled vocabulary. `lookup_category` / `register_category` keep categorical fields consistent. Scoped **per field**, with fuzzy matching and de-duplication so variants like *"No Questions"* / *"No Merger Questions"* collapse to one entry.
- **`CHUNKING_TOOLS/find_transcript_chunks.py`** — extracts transcript chunks around matched lines and formats them with line numbers.
- **`utils/transcript_helpers.py`** — the matching engine (`_search_lines`): literal match → content-word overlap → **fuzzy overlap** that tolerates corrupted transcripts (e.g. dropped letters) so quote-derived terms still locate the right lines.
- **`SEARCH_TOOLS/`** — derive/refine search terms and n-grams for evidence retrieval.
- **`EVAL_TOOLS/`** — `aggregate_judge_results.py` (builds the revision brief) and semantic-quality helpers.

### `utils/bootstrap_extraction.py`

`ExtractionBootstrapEvaluator` — runs the extractor `n_runs` times, computes **consistency** across runs, and computes **semantic quality** from judge output. The correctness calculation excludes judge retrieval failures from the denominator (they're surfaced separately as `retrieval_failure_rate`).

### `spectrum_client.py`

The LLM client (Azure OpenAI-compatible). Provides single-turn chat, an agentic tool loop, Jinja2-templated structured calls with validation + retry, and a parallel batch runner. Credentials load from `config/secrets.toml`.

---

## Entry points

There are four ways to run the pipeline. All of them ultimately drive the same `Orchestrator`.

### 1. Directly (script / notebook)

```python
import pandas as pd
from orchestration.orchestrator import Orchestrator
from agents.prompt_generator_agent import PromptGeneratorAgent
from agents.schema_generator_agent import SchemaGeneratorAgent
from agents.ExtractionAgent import TranscriptExtractionAgent
from agents.JudgeAgent import JudgeAgent

df = pd.read_csv("tests/TEST_DATA.csv")  # must include AGENTRECORDINGSESSIONID + a transcript column

orchestrator = Orchestrator(
    df=df,
    prompt="What value-addition opportunities exist in this call?",
    PromptAgent=PromptGeneratorAgent(),
    SchemaAgent=SchemaGeneratorAgent(),
    TranscriptExtractionAgent=TranscriptExtractionAgent(),
    JudgeAgent=JudgeAgent(),
    output_dir=r"tests",
    project_name="VAO",
)

result = orchestrator.run(
    review_prompt=True,
    review_schema=True,
    run_judging=True,
    run_evaluation=True,
    n_runs=3,
    min_correctness_rate=0.90,
    min_consistency_rate=0.85,
    max_hallucination_rate=0.10,
    max_revision_cycles=3,
)

print(result["final_status"])            # PASS / FAIL
print(result["revision_cycles_used"])
print(result["revision_stopped_reason"]) # set if the retrieval circuit-breaker fired
```

**Measuring correctness against gold labels (independent of the judge):**

```python
gold_df = pd.DataFrame([
    {"AGENTRECORDINGSESSIONID": "sess-1",
     "existing_customer_claim_field": "Yes",
     "merger_discussion_claim_field": "No"},
    # ... one row per labeled transcript, one column per judged field
])

orchestrator = Orchestrator(
    df=df, prompt=..., ...,
    gold_labels=gold_df,
    correctness_source="gold",
)
```

### 2. Streamlit UI

```bash
streamlit run streamlit_app.py
```

Fill in the question, thresholds, and run settings; review the prompt/schema at the gates; watch the metrics.

### 3. FastAPI service

```bash
uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for interactive docs. Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `POST /runs` | Start a run (returns immediately; processes in background) |
| `GET /runs` | List all runs |
| `GET /runs/{run_id}` | Get run status/results |
| `POST /runs/{run_id}/cancel` | Cancel a pending/running run |
| `GET /runs/{run_id}/review` | Fetch the item awaiting human review |
| `POST /runs/{run_id}/decisions` | Submit Approve / Retry / Cancel; pipeline resumes |

### 4. Celery workers (async / scaled)

For long-running or parallel batch work. See **`worker/README.md`** for the full guide. In short:

```bash
# Requires Redis on localhost:6379
celery -A worker.celery_app worker --loglevel=info \
  -Q default,orchestration,generation,extraction,judging,evaluation
```

```python
from worker.tasks.pipeline import run_pipeline
result = run_pipeline.delay(user_input="...", rows=df.to_dict("records"),
                            output_dir="tests", project_name="VAO", n_runs=3)
output = result.get(timeout=600)
```

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials** — copy `config/secrets_template.toml` to `config/secrets.toml` and fill in the LLM client credentials (and Postgres, if you use it).

3. **(Optional) Redis** for Celery — see `worker/README.md`.

---

## Quality metrics

Each run produces an `EvaluationResult` with:

- **Correctness rate** — fraction of judged fields grounded in the transcript (or matched to gold labels when `correctness_source="gold"`). Judge retrieval failures are excluded from the denominator.
- **Consistency rate** — agreement of the same field across repeated extraction runs of the same transcript.
- **Hallucination rate** — fraction of fields the judge marked as hallucinated.
- **Retrieval-failure rate** — fraction of judgments where the judge could not retrieve usable evidence (a data/retrieval signal, not an extractor-quality signal).

The orchestrator compares the first three against the thresholds you pass to `run()` to decide PASS/FAIL and whether to enter the revision loop.

---

## Outputs

A run writes artifacts to `output_dir/project_name/`, including:

- `*_prompt_v{N}.md` — each prompt version
- `*_extraction_results.csv` — extracted fields, quotes, reasoning
- `*_judge_results.csv` — per-field grounding, evidence strength, `RETRIEVAL_STATUS`, verbatim evidence chunk
- `*_evaluation_summary_results.csv` — correctness / consistency / hallucination (overall + per field)
- `*_revision_brief_cycle_{N}.md` — the structured feedback fed back into prompt revision
- `*_categories.csv` — the category registry (when enabled)
