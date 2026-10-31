# Bug Triage – Tech Stack

Versions are the ones installed in `.venv` on 2026-10-31 (`pip list`). `requirements*.txt` pins minimums only.

## Core

| Technology | Version | Where | How it is used |
|---|---|---|---|
| Python | 3.13.7 | everywhere | Type hints, dataclasses, `pathlib`, `argparse` CLIs (`python -m src.<module>`) |
| pydantic | 2.14.2 | `schemas.py`, `triage.py`, `judge.py`, `groundedness.py` | Validates the dataset and human labels on load. Strict `model_validate_json` parsing of every LLM reply (`extra="forbid"`, `Literal` label sets, score range 1–5). |
| pandas | 3.1.0 | `metrics.py`, `calibration.py`, `report.py` | Confusion-matrix frames, crosstabs, ranks for Spearman, `read_sql_query` for the SQL section, markdown tables |
| scikit-learn | 1.9.1 | `metrics.py`, `calibration.py` | `precision_recall_fscore_support`, `confusion_matrix`, `cohen_kappa_score` (plain and `weights="quadratic"`) |
| NumPy | 2.5.4 | `calibration.py`, `compare.py` | Paired bootstrap (`default_rng(42)`, 2000 resamples), residuals for partial Spearman |
| sqlite3 | stdlib | `store.py`, `queries.sql` | Tables `runs`, `items`, `predictions`, `scores`, `judge_scores` in `reports/benchmark.db` |
| matplotlib | 3.11.2 | `report.py` | 3 PNG charts (Agg backend): accuracy by variant, judge kappa vs the 0.6 threshold, severity confusion heatmap. Colorblind-validated palette. |
| python-dotenv | 1.2.4 | `llm.py` | Loads `.env` (provider, keys, model ids) |
| pytest | 9.2.0 | `tests/` | 50 unit tests with hand-computed fixtures; no network calls |
| urllib | stdlib | `llm.py` (`OpenRouterLLM`), `sources/jira_source.py` | HTTPS calls to `https://openrouter.ai/api/v1` (`POST /chat/completions`, `GET /models`) and to Jira Cloud (`GET /rest/api/3/issue/{key}`, attachment content). No extra SDK needed. |
| base64 | stdlib | `sources/jira_source.py` | Jira Cloud Basic auth header `base64(email:api_token)` |
| hashlib | stdlib | `llm.py` (`MockLLM`) | sha256-based deterministic error injection |

## Optional (`requirements-optional.txt`)

| Technology | Version | Where | How it is used |
|---|---|---|---|
| anthropic (Python SDK) | 2.3.0 | `llm.py` (`AnthropicLLM`) | `messages.create`, or `beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default")` for models that support refusal fallbacks. No `temperature` is sent: current Claude models reject it. |
| DeepEval | 4.3.2 | `deepeval_bridge.py`, `tests/test_deepeval_bridge.py` | `CitationGroundednessMetric(BaseMetric)` runs offline through `deepeval.evaluate` (caching off, sync mode). `BenchmarkLLM(DeepEvalBaseLLM)` adapts this project's LLMs. A `GEval` "Explanation Groundedness" judge runs only with a real provider (`--geval`). Telemetry is opted out via `DEEPEVAL_TELEMETRY_OPT_OUT=YES`. |
| pytest-asyncio | (pulled in by deepeval) | `pytest.ini` | Only configured, to silence a loop-scope warning |

## LLM providers

| Provider | Selected by | Model | Notes |
|---|---|---|---|
| MockLLM | `LLM_PROVIDER=mock` (default) | n/a | Offline, deterministic, scripted. Numbers show the harness, not model quality. It cannot triage tracker tickets. |
| Anthropic | `LLM_PROVIDER=anthropic` | `ANTHROPIC_MODEL` (required) | Official SDK; key from `ANTHROPIC_API_KEY` |
| OpenRouter (**current triage provider**) | `LLM_PROVIDER=openrouter` | `OPENROUTER_MODEL`, or the Auto Router `openrouter/auto` when empty (current) | On any error except 400/401, or an empty reply, falls back to up to 5 zero-priced text-only models from `/models`; a model that returned 402/403/404 is skipped for the rest of the run |
| Judge (any of the above) | `JUDGE_PROVIDER` (empty = `LLM_PROVIDER`) | `JUDGE_MODEL` | Used by `calibration.py`, `triage.py --judge` and DeepEval GEval. **Current:** Anthropic `claude-opus-5-5`. |

## Bug sources

| Source | Technology | Notes |
|---|---|---|
| `file` | JSON + pydantic | The labelled benchmark (`data/bug_reports.json`) or any export in the same format (`--path`) |
| `jira` | Jira Cloud REST API v3 over `urllib` | `GET /rest/api/3/issue/{key}`; Basic auth with email + API token (per Atlassian's basic-auth guide). Rich text is Atlassian Document Format or wiki markup. |
| Atlassian MCP | Claude Code session only (not used by the app) | Optional way to export tickets into `data/imports/*.json` for the `file` source |

## Tooling and environment

| Tool | Use |
|---|---|
| venv (`.venv`) | Isolated Python 3.13 environment |
| Git | Repository at `DeepEvalProjects/`, remote `github.com/ishankwalia398/DeepEvalProjects`. `.env`, the DB, the report, the charts and `data/imports/` are git-ignored. |
| Mermaid | Diagrams in `Architecture.md` (rendered by GitHub and VS Code) |
| Windows 11 / PowerShell | Development environment; commands in the README use PowerShell syntax |
