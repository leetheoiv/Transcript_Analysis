# Transcript Analysis Automation

**An AI-powered, multi-agent pipeline that transforms plain-language analyst questions into validated, structured transcript analyses — with human-in-the-loop quality gates and self-improving prompt engineering.**

---

## Overview

Contact centers generate thousands of call transcripts daily, but extracting structured insights (compliance checks, sentiment, sales behaviors) still requires manual prompt engineering, hand-built schemas, and laborious QA. This project eliminates that bottleneck.

An analyst types a question in natural language. The system automatically:
1. Engineers an optimized extraction prompt
2. Generates a validated Pydantic data schema
3. Extracts structured data from transcripts at scale
4. Judges every extracted claim against transcript evidence
5. Self-corrects by feeding failure analysis back into prompt revision

Two human approval gates and two automated judge passes ensure nothing reaches production without validation.

---

## Architecture

```
                         +-------------------+
                         |   Analyst Input   |
                         | (plain language)  |
                         +---------+---------+
                                   |
                    +--------------v--------------+
                    |   PromptGeneratorAgent      |
                    |   (RAG + tool-augmented)    |
                    +--------------+--------------+
                                   |
                         +---------v---------+
                         | Human Approval #1 |
                         +---------+---------+
                                   |
                    +--------------v--------------+
                    |   SchemaGeneratorAgent      |
                    |   (Pydantic v2 codegen)     |
                    +--------------+--------------+
                                   |
                         +---------v---------+
                         | Human Approval #2 |
                         +---------+---------+
                                   |
             +---------------------v---------------------+
             |                                           |
   +---------v-----------+                 +-------------v-----------+
   | TranscriptExtraction|                 | Consistency Runs        |
   | Agent (Run 1)       |                 | (Runs 2..N, parallel)   |
   +---------+-----------+                 +-------------+-----------+
             |                                           |
   +---------v-----------+                               |
   |    JudgeAgent       |<------------------------------+
   | (grounding eval)    |
   +---------+-----------+
             |
   +---------v-----------+
   | Evaluation &        |
   | Aggregation         |
   +---------+-----------+
             |
      +------v------+
      | Pass / Fail |
      +------+------+
             |
    (fail)   |   (pass)
      +------v------+---------> Final Results
      | Revision    |
      | Brief       |
      +------+------+
             |
             +-------> Back to PromptGeneratorAgent
                       (with full failure context)
```

---

## Key Features

### Multi-Agent Orchestration
A central `Orchestrator` class coordinates the entire pipeline — prompt generation, schema generation, extraction, judging, and evaluation — with structured error handling, concurrent execution, and human-in-the-loop review gates at critical decision points.

### Self-Improving Prompts (Continuous Learning)
When extraction quality falls below threshold, the system doesn't just fail — it learns. A 5-step judge aggregation pipeline produces a structured `RevisionBrief` containing:
- Worst-performing fields ranked by hallucination and groundedness rates
- Error type breakdown with affected fields and sample explanations
- Representative failure examples with evidence context
- Actionable prompt lessons (keep doing / stop doing / changes to make)

This brief is injected back into the `PromptGeneratorAgent`, which revises the extraction prompt to specifically address each identified weakness. Revision history accumulates across cycles — the agent always sees the full chain of what was tried and why it failed.

### Grounding Judge with Evidence Retrieval
The `JudgeAgent` doesn't just score outputs — it retrieves transcript evidence for each claim using a multi-strategy search pipeline:
1. Direct quote matching from extractor output
2. LLM-generated search term refinement
3. N-gram transcript search with configurable context windows

Each field is judged independently with evidence strength ratings (none/low/medium/high), hallucination detection, error type classification, and prompt adjustment suggestions.

### Dual-Mode Extraction
The `TranscriptExtractionAgent` automatically selects the optimal extraction strategy based on transcript length:
- **Inline mode** — short transcripts are injected directly into the prompt
- **Tool mode** — longer transcripts use a `search_transcript` tool to locate relevant sections before extraction, with retry logic and fallback to full injection

### RAG-Augmented Prompt Engineering
The `PromptGeneratorAgent` has access to a document search tool that queries reference materials (policy documents, domain definitions, terminology guides) when generating extraction prompts — ensuring prompts incorporate specialized domain knowledge.

### Human-in-the-Loop Quality Gates
Two HITL gates enforce human oversight:
- **Prompt review** — approve, retry with feedback, or cancel
- **Schema review** — approve, retry with feedback, or cancel

Reviewers can provide free-text notes that feed directly into the revision cycle.

### Structured Evaluation Metrics
The evaluation pipeline produces:
- **Semantic quality** — correctness rate, consistency rate, hallucination rate (per-field and aggregate)
- **Consistency quality** — cross-run agreement rates from repeated extraction passes
- **Per-field breakdowns** — identifying exactly which fields need improvement

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM Client | Azure OpenAI (GPT-4.1-mini, GPT-5-mini) | Inference backbone for all agents |
| Data Validation | Pydantic v2 | Structured outputs, schema generation, model validation |
| Templating | Jinja2 | Dynamic prompt rendering with variable injection |
| Data Processing | pandas, numpy, scipy, statsmodels | Data manipulation, statistical evaluation |
| Database | PostgreSQL (psycopg2) | Durable state persistence, versioned run records |
| Task Queue | Celery + Redis | Background agent execution, batch processing |
| API | FastAPI | REST endpoints with OpenAPI spec (planned) |
| Frontend | React + Vite + shadcn/ui | Approval screens, review dashboards (planned) |
| UI/Reporting | Streamlit, ReportLab, python-pptx | Interactive dashboards, PDF/PPTX generation |
| Infrastructure | Docker + docker-compose | Containerized deployment (planned) |

---

## Agent Design

All agents inherit from `BaseToolAgent`, which provides:
- Anthropic/OpenAI-style tool registration and dispatch
- Pydantic-validated tool inputs with automatic schema conversion
- Centralized logging for tool execution and chat calls
- Jinja2 template rendering for prompt construction
- Token counting and usage tracking

```python
class BaseToolAgent(SpectrumClient):
    """Base class providing tool framework for all pipeline agents."""
    
    def register_tool(self, tool: Tool): ...
    def chat_with_tools(self, messages, temperature, ...): ...
    def _execute_tool(self, tool_name, tool_input): ...
```

### Agent Responsibilities

| Agent | Input | Output | Tools |
|-------|-------|--------|-------|
| `PromptGeneratorAgent` | Plain-language question + optional revision brief | Jinja2 prompt template (system + user + output format) | `search_documents` (RAG) |
| `SchemaGeneratorAgent` | Approved prompt output format | Pydantic v2 model class (written to file + import-validated) | — |
| `TranscriptExtractionAgent` | Transcript + prompt + schema | Validated structured extraction results | `search_transcript` |
| `JudgeAgent` | Extraction results + original transcript | Per-field grounding scores + evidence + error types | — |

---

## Data Flow & State Machine

```
draft_prompt → prompt_approved → draft_schema → schema_approved
→ sample_running → sample_judged → sample_approved → full_running
→ full_judged → complete | flagged_for_review

           sample_rejected → draft_prompt (carries RevisionBrief)
```

Every state transition is persisted atomically. Revision history is append-only — old versions are never deleted. The full lineage of any run is always reconstructable through parent version UUIDs.

---

## Database Schema

Four core tables track the full pipeline lifecycle:

- **`run_specs`** — Central pipeline record (question, prompt, schema, tools, status, version, lineage, scores)
- **`judge_feedback`** — Per-row grounding scores with reasoning and typed issue flags
- **`user_feedback`** — Per-row human analyst comments tied to specific versions
- **`revision_briefs`** — Aggregated feedback objects that drive prompt revision cycles

Storage is abstracted behind a repository interface — swap between CSV (local dev) and Postgres (production) via environment variable.

---

## Project Structure

```
Transcript_Analysis_Automation/
├── agents/                          # Multi-agent layer
│   ├── BaseAgent.py                 #   Tool-enabled base class
│   ├── prompt_generator_agent.py    #   RAG-augmented prompt engineering
│   ├── schema_generator_agent.py    #   Pydantic model codegen
│   ├── ExtractionAgent.py           #   Dual-mode transcript extraction
│   └── JudgeAgent.py                #   Evidence-based grounding evaluation
├── orchestration/                   # Pipeline coordination
│   ├── orchestrator.py              #   End-to-end pipeline with HITL gates
│   └── exceptions.py                #   Typed exception hierarchy
├── data_models/                     # Pydantic v2 models
│   ├── prompt_datamodel.py          #   PromptManagement with versioning
│   ├── feedback.py                  #   JudgeFeedback, UserFeedback, RevisionBrief
│   ├── judge_aggregation.py         #   5-step aggregation pipeline model
│   ├── prompt_eval_datamodel.py     #   Semantic + consistency quality metrics
│   └── agent_models/                #   Per-agent I/O models
├── prompts/                         # LLM prompt templates
│   ├── agent_prompts/               #   Agent-specific system/user prompts
│   └── general_prompts/             #   Shared utility prompts
├── db/                              # Persistence layer
│   ├── schema.sql                   #   Postgres DDL
│   └── database.py                  #   Connection management
├── knowledge_base/                  # RAG reference documents
├── tools/                           # Agent tools (search, chunking, RAG, write)
├── utils/                           # Shared utilities (HITL, evaluation, file I/O)
├── tests/                           # Unit tests + evaluation results
├── config/                          # Secrets and configuration
├── spectrum_client.py               # Azure OpenAI API wrapper
├── plan.md                          # Architecture design document
└── requirements.txt                 # Python dependencies
```

---

## Design Principles

| Principle | How It's Applied |
|-----------|-----------------|
| **Scalability** | Agents run independently and can be parallelized; batch processing with configurable worker pools |
| **Modularity** | Each agent, storage backend, and runner is swappable without touching other components |
| **Continuous Learning** | Structured revision briefs carry full failure history across iterations |
| **Resilience** | Typed exception hierarchy, retry logic with backoff, incremental result persistence |
| **Open Standards** | OpenAI tool-call spec, JSON Schema, Pydantic validation — provider-agnostic by design |

---

## Example Usage

```python
from orchestration.orchestrator import Orchestrator
from agents.prompt_generator_agent import PromptGeneratorAgent
from agents.schema_generator_agent import SchemaGeneratorAgent
from agents.ExtractionAgent import TranscriptExtractionAgent
from agents.JudgeAgent import JudgeAgent

# Initialize agents
prompt_agent = PromptGeneratorAgent(rag_folder="knowledge_base/")
schema_agent = SchemaGeneratorAgent()
extraction_agent = TranscriptExtractionAgent()
judge_agent = JudgeAgent()

# Run the full pipeline
orchestrator = Orchestrator(
    df=transcript_dataframe,
    prompt="Did the agent offer a Value Addition Opportunity during the call?",
    PromptAgent=prompt_agent,
    SchemaAgent=schema_agent,
    TranscriptExtractionAgent=extraction_agent,
    JudgeAgent=judge_agent,
    output_dir="output/",
    project_name="VAO_Analysis"
)

# Pipeline runs through: prompt gen → approval → schema gen → approval
# → extraction → judging → evaluation (with automatic revision on failure)
orchestrator.run()
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Data model layer | Done |
| Postgres schema + repository interface | Done |
| State machine / orchestrator | Done |
| Agent layer (all 4 agents) | Done |
| Evaluation & aggregation pipeline | Done |
| Human-in-the-loop review gates | Done |
| Self-improving revision cycle | Done |
| Celery task queue wiring | Planned |
| FastAPI REST endpoints | Planned |
| React frontend (approval UI) | Planned |
| Docker containerization | Planned |

---

## What I Learned

- **Structured feedback loops are more powerful than simple retries.** Passing a typed `RevisionBrief` with field-level failure analysis gives the LLM specific, actionable context — leading to targeted fixes rather than random variation.
- **Evidence retrieval before judgment is critical.** Having the judge retrieve and cite transcript evidence (rather than judging from memory) dramatically reduces false positives in hallucination detection.
- **Tool-augmented agents need graceful fallbacks.** The dual-mode extraction strategy (search tool with retry → full transcript fallback) handles edge cases where search terms don't match transcript language.
- **Repository abstraction pays dividends immediately.** Being able to swap between CSV (for rapid iteration) and Postgres (for production) without changing any business logic accelerated development significantly.

---

## Contact

Built as an internal tool for automating call center transcript analysis pipelines. The system generalizes an existing manual process so any analyst can define new extraction questions without engineering support.
