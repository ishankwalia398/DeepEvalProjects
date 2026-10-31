# Bug Triage – Progress

Each entry follows the project's anti-hallucination format:
Verified Facts / Missing-Unknown / Generated Output / Self-Validation Check.

## Status board

| Step | Description | Status |
|---|---|---|
| 1 | Scaffold: git, venv, schemas, .gitignore, .env/.env.example | Done |
| 2 | Dataset (50 items) + human judgements (30) + validation tests | Done |
| 3 | `llm.py` (MockLLM, Anthropic, OpenRouter), `prompts.py`, `triage.py` | Done |
| 4 | `metrics.py`, `groundedness.py` + unit tests | Done |
| 5 | `judge.py` v1/v2, `calibration.py` + kappa tests | Done |
| 6 | `store.py`, `queries.sql`, `compare.py` | Done |
| 7 | `report.py`, DeepEval bridge, `run_all.py`, docs | Done |
| 8 | End-to-end run, definition-of-done check | Done (offline / mock) |
| 9 | Bug-tracker integration (pluggable sources + Jira) and separate judge model | Done; Jira awaiting one live issue to verify end to end |

## Decisions confirmed with the user (2026-10-31)

- Eval tooling: scikit-learn for metrics/kappa; DeepEval as an optional extra layer.
- Providers: `LLM_PROVIDER=mock|anthropic|openrouter` (OpenRouter replaces the spec's `openai`).
- Models: no hard-coded defaults. `ANTHROPIC_MODEL` is set by the user in `.env`. **Update:** the user will NOT set `OPENROUTER_MODEL`; OpenRouter must pick the best model automatically and keep working when a model's limit or quota is reached.
- `.env`: the user has filled in `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` and `OPENROUTER_API_KEY`. The assistant does not read `.env` (secrets).
- **Update (Step 9):**
  - The judge is a separate model: `JUDGE_PROVIDER=anthropic`, `JUDGE_MODEL=claude-opus-5-5`, both appended to `.env`.
  - Recommended triage model: `ANTHROPIC_MODEL=claude-sonnet-5-5`. The user's current value was not read.
  - Jira credentials go in `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN`.
- Anti-hallucination rules: applied to the build process (this file) and to the v2/v3 triage prompts and judge_v2.
- Git: repo initialised at `DeepEvalProjects/` with remote `https://github.com/ishankwalia398/DeepEvalProjects`; local only, no commits or pushes without approval.
- Python 3.13.7 in `Bug_Triage/.venv`; core deps in `requirements.txt`, optional in `requirements-optional.txt`.
- Spec gaps: 15 pairwise pairs via `alt_explanation`; partial Spearman for length bias; `judge_v2_reworded` for judge prompt sensitivity; bootstrap 2000 paired resamples, seed 42, 95% CI; ship/keep/inconclusive rule on CI bounds plus no new severity misses.

## Log

### Steps 1–2 – Scaffold and dataset
- **Verified Facts:** Python 3.13.7 venv created. Installed: pydantic 2.14.2, pandas 3.1.0, scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2, pytest 9.2.0, python-dotenv 1.2.4, anthropic 2.3.0, deepeval 4.3.2. Dataset: 50 items (severity 13/14/12/11, components 7–10 each, 4 platforms 12–13 each, 15 tricky, 2 prompt-injection items, 5 `unknown` root causes). Human judgements: 30 items, 8 deliberately bad (3 hallucinated quote, 2 vague, 3 wrong cause), 15 with a pairwise alternative. `pytest`: 5 passed.
- **Missing / Unknown:** The human scores are drafts authored by the assistant; the user will review and adjust them. The remote GitHub `main` already has commit `6d65d19`; its contents were not fetched.
- **Generated Output:** `.gitignore`, `.env` (blank values), `.env.example`, `requirements*.txt`, `pytest.ini`, `src/schemas.py`, `data/bug_reports.json`, `data/human_judgements.json`, `tests/test_data.py`.
- **Self-Validation Check (Steps 1–2):** A test asserts every quote in a `grounded=true` explanation appears verbatim (whitespace-normalised) in that item's log, and that every non-`unknown` item has at least one ERROR/WARN line. All data is invented for the fictional "StreamNova" app.

### Step 3 – Research before coding (paused here on user request, 2026-10-31)
- **Verified Facts:**
  - Anthropic (from the bundled Claude API guidance): `temperature`/`top_p`/`top_k` are rejected (HTTP 400) on current Claude models, and on every model from API version `2026-10-21`. In the installed SDK, `messages.create` has no `temperature` parameter but does have `output_config`; `beta.messages.create` accepts `betas` and `fallbacks`. The guidance recommends server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`) for `claude-fable-5-1`, `claude-opus-5-5`, `claude-opus-5`, and `claude-sonnet-5-5`. Responses can contain thinking blocks, so read only `text` blocks.
  - OpenRouter (fetched docs page `openrouter.ai/docs/features/model-routing`): the Auto Router model ids are `openrouter/auto` and `openrouter/auto-beta`. It classifies the task, ranks models by real-world spend share, and routes "with fallbacks". The response `model` field shows which model was used.
  - DeepEval 4.3.2 (installed): `DeepEvalBaseLLM` abstract methods are `load_model`, `generate`, `a_generate`, `get_model_name`. `DEEPEVAL_TELEMETRY_OPT_OUT` appears in 3 source files.
- **Missing / Unknown (must verify before writing the OpenRouter adapter):**
  - The exact shape of OpenRouter's `models` fallback-list parameter, and which errors trigger a fallback (rate limit, quota/credits 402, downtime). Two guessed doc URLs returned 404 and the web search was cancelled, so this is not verified yet.
  - The OpenRouter models-list endpoint and its fields (for picking free models when credits run out). Not verified.
  - Whether the Auto Router keeps working when account credits are exhausted. Insufficient information to determine.
- **Planned design (not yet implemented):**
  - `llm.py`: `LLM` protocol with `complete(prompt, system) -> str`; `MockLLM` (deterministic simulator, see below); `AnthropicLLM` (official SDK, no temperature, text blocks only, refusal fallbacks for supported models); `OpenRouterLLM` (stdlib HTTP; default model `openrouter/auto` when `OPENROUTER_MODEL` is empty; on 429/402/5xx, retry along a verified fallback chain; record the model actually used).
  - MockLLM: prompts begin with `### prompt_id: <id>`; data fenced as `<<<LOG ... LOG>>>`. Errors are injected by sha256 hash of (item, variant, field), with per-variant error rates (v1 > v2 > v3), higher on tricky items. A description-hash term creates paraphrase sensitivity. Some citations are hallucinated, v1 has some parse errors, and v1 obeys the injection when it is the last log line (BUG-039 only).
  - Mock judge: v1 scores by length (biased) and prefers option A in pairwise; v2 checks quotes against the log, penalises vague text, and checks claimed-cause keywords against the log (expected to miss BUG-035). `judge_v2_reworded` uses slightly different penalties.
  - Determinism note for docs: mock runs are fully deterministic; real Anthropic runs cannot use temperature 0, so repeat-run variance must be reported.
- **Generated Output:** No new code in this step yet.
- **Self-Validation Check:** Nothing about OpenRouter fallbacks is assumed beyond the fetched Auto Router page.

### Steps 3–4 – LLM layer, triage, metrics, groundedness (resumed 2026-10-31)
- **Verified Facts (OpenRouter docs fetched via links found on the routing page, not guessed):**
  - `model-fallbacks.md`: "any error can trigger the use of a fallback model, including: Context length validation errors, Moderation flags for filtered models, Rate-limiting, Downtime." Insufficient credits (402) is not mentioned. Requests are billed for the model that was actually used, which the response's `model` field names.
  - Chat completions reference: `POST /chat/completions`; `temperature` 0–2; `max_tokens` is deprecated in favour of `max_completion_tokens`; `402` means insufficient credits and `429` means rate limit exceeded; `401`, `500` and `503` are also listed.
  - Models reference: `GET https://openrouter.ai/api/v1/models` (auth required), with per-model `id`, `created`, `context_length`, `pricing.prompt` and `pricing.completion` (USD per token, as strings).
  - Anthropic SDK 2.3.0: `BetaFallbacksParam = Union[Iterable[BetaFallbackParam], Literal["default"]]`, so `fallbacks="default"` is valid.
- **Implementation:**
  - **OpenRouter:** `OpenRouterLLM` defaults to `openrouter/auto`. On 402, 408, 429, 500, 502 or 503 it walks up to 3 zero-priced models from `/models`, ordered by longest context then newest, and records which models actually served each call (shown in `runs.model`).
  - **Anthropic:** `AnthropicLLM` sends no temperature, reads only text blocks, and turns on server-side refusal fallbacks for the supported model ids.
- **Inference (low confidence):** treating `pricing == 0` as "free", and longest-context-then-newest as "best free model", are my heuristics; OpenRouter does not document them as a ranking.
- **Mock smoke run (scratch DB):**

  | Variant | Severity acc | Component acc | Root-cause acc | Exact match | Parse errors | Severity misses | Invalid citations |
  |---|---|---|---|---|---|---|---|
  | v1_basic | 0.56 | 0.58 | 0.56 | 0.26 | 0.12 | BUG-002, BUG-039 | 8 |
  | v2_structured | 0.74 | 0.60 | 0.74 | 0.34 | 0.04 | BUG-036 | 4 |
  | v3_cot_light | 0.86 | 0.84 | 0.94 | 0.68 | 0.00 | BUG-023 | 3 |

  Paraphrase label changes: v1 32, v2 16, v3 4. v1 obeys the injection on BUG-039 only.
- **Generated Output:** `src/prompts.py`, `src/llm.py`, `src/groundedness.py`, `src/metrics.py`, `src/store.py` (written early because the triage CLI stores runs), `src/triage.py`, `tests/conftest.py`, `tests/test_metrics.py`, `tests/test_groundedness.py`, `tests/test_triage.py`.
- **Decision not covered by the spec:** added an `items` table (platform, gold labels, tricky) and a `cited_log_lines` column on `predictions`. Both are needed for the SQL queries and for rebuilding runs in the report.
- **Self-Validation Check:** `pytest` reports 20 passed. Real-provider adapters have not been exercised against live APIs (no network calls made). Mock numbers demonstrate the harness, not model quality.

### Steps 5–6 – Judge, calibration, SQL, compare
- **Verified Facts (mock, scratch DB):**

  | Judge | kappa (grounded) | weighted kappa (score) | length partial Spearman | position flip rate | pairwise agreement with humans |
  |---|---|---|---|---|---|
  | judge_v1 | 0.33 | 0.27 | 0.82 | 0.40 | 0.60 |
  | judge_v2 | 0.91 | 0.84 | -0.05 | 0.00 | 1.00 |
  | judge_v2_reworded | 0.91 | 0.75 | -0.12 | n/a | n/a |

  Weighted kappa is quadratic-weighted, over 30 items and 15 pairs.
- **Generated Output:** `src/judge.py`, `src/calibration.py`, `src/compare.py`, `notebooks_or_queries/queries.sql`, `tests/test_calibration.py`, `tests/test_compare.py`. `strip_code_fence` moved to `llm.py`, shared by the triage and judge parsers.
- **Bug found by a test:** `partial_spearman` returned float noise (-0.06) instead of 0 when x equals the control variable. Fixed with an `np.allclose` check.
- **Self-Validation Check:** `pytest` reports 29 passed. The kappa test uses hand-computed values (0.5 binary, -1.0 for fully swapped weighted). The mock judge_v2 is a rubric heuristic that I designed, so its high agreement is by construction; it does miss BUG-035 (a wrong cause hidden behind misleading log lines).

### Steps 7–8 – Report, DeepEval bridge, run_all, docs, definition of done
- **Verified Facts:**
  - **DeepEval 4.3.2 API (inspected in the installed package):** `LLMTestCase` has `retrieval_context` and `metadata`; `BaseMetric` has `measure`, `a_measure` and `is_successful`; `GEval(name, evaluation_params=[SingleTurnParams...], criteria, model=DeepEvalBaseLLM)`; `evaluate(..., async_config, display_config, cache_config)`.
  - **`run_all.py` (mock):** 6 triage runs, 3 judge runs, the v1-vs-v3 comparison, the DeepEval cross-check (47/50 citation-groundedness passes on v3, matching the 3 hallucinated items), the report and 3 charts. Wall time was 13.5–36.4 s across 4 runs; the first run includes the matplotlib font cache. Two consecutive runs produced identical results.
  - **v1 vs v3:** exact match +0.42, 95% CI [+0.22, +0.60]. The recommendation is "inconclusive: B is more accurate but adds severity misses ['BUG-023']". v2 vs v3 gives the same verdict.
  - `pytest`: 31 passed. `git check-ignore` confirms `.env`, `reports/benchmark.db`, `reports/report.md` and `.venv` are ignored. The secret-pattern scan's only hit is `anthropic.Anthropic(api_key=api_key)`, a variable, not a key.
- **Bugs fixed during this step:**
  - DeepEval's local test-run cache crashed on a second run (`cached_test_run` was None). Fixed with `CacheConfig(write_cache=False, use_cache=False)`, and `run_all.py` now warns and continues if the optional layer fails.
  - The citation metric failed empty-log items that correctly cite nothing; with an empty log, "no citations" now scores 1.
  - The kappa chart label collided with a bar; counts were shown as floats; SQL query 2 was re-pointed to v2 → v3, because v3 → v3+paraphrase has no regressions.
- **Missing / Unknown:**
  - The Anthropic and OpenRouter adapters and the GEval path have **not** been run against live APIs (no paid calls made).
  - The human scores still need the user's review.
- **Generated Output:** `src/report.py`, `src/deepeval_bridge.py`, `run_all.py`, `tests/test_deepeval_bridge.py`, `README.md`, `Architecture.md`, `TechStack.md`, plus generated `reports/report.md`, `reports/*.png` and `reports/deepeval_summary.json`.
- **Self-Validation Check:** Every number in the README comes from the final run's report. Claims about real providers are labelled untested, and OpenRouter free-model ranking is labelled "Inference (low confidence)".

### Docs update (user request)
- **Verified Facts:** the "Mapping to airline operations" and "How I would explain this" sections, and the opening key-message quote, moved from `README.md` to a new `interview.md`. README now holds project information only, and lists `interview.md` in the project structure.
- **Self-Validation Check:** the content was moved without changes; no numbers were altered.

### Step 9 – Bug-tracker integration and separate judge model (user request)
- **Verified Facts:**
  - **Jira auth:** Atlassian's basic-auth guide says to use an `Authorization: Basic` header over `useremail:api_token`, and that password auth is deprecated.
  - **Atlassian MCP** (`getAccessibleAtlassianResources`) lists an employer Jira site with read/write scopes. The user's `.env` instead points at a separate personal Jira Cloud site, set by the user. (Site names are left out of this public file.)
  - **First live call:** a read-only `GET /rest/api/3/issue/PROJ-123` (a made-up key) returned 404 with the expected hint. It was not a 401, so the credentials were likely accepted.
  - **Judge model ids and prices** come from the bundled Claude API guidance (cached 2026-10-06): `claude-opus-5-5` costs $4 / $20, `claude-sonnet-5-5` $2 / $10, and `claude-haiku-5-5` $0.10 / $0.50 per million tokens.
  - `pytest`: 41 passed. `run_all.py` reproduces the same offline results (kappa 0.27 → 0.84, DeepEval 47/50).
- **Missing / Unknown:**
  - The REST reference pages for "Get issue" and JQL search came back truncated, so field shapes were coded defensively (both Atlassian Document Format and wiki markup are handled) and still need verification on a real issue.
  - JQL search is not implemented; its endpoint was not verified.
  - Attachment downloads through redirects and Data Center Bearer auth are untested.
- **Inference (low confidence):** a 404 instead of a 401 suggests the credentials are valid, but Jira also returns 404 when an issue isn't visible to the account.
- **Generated Output:**
  - **New:** `src/sources/` (`base.py`, `file_source.py`, `jira_source.py`, `__init__.py`) and `tests/test_sources.py`.
  - **`schemas.py`:** generic ids, optional gold, `unknown` platform, `source` and `url` fields.
  - **`llm.py`:** `get_judge_llm`, `model` override, mock refuses tracker tickets.
  - **`triage.py`:** `--source`, `--ids`, `--path`, `--judge`, `--judge-provider`.
  - **Other code:** `store.py` and `metrics.py` handle missing gold; calibration and DeepEval GEval use the judge model; `run_all.py` gained `--judge-provider`.
  - **Config:** `.env.example`, appended `.env` variables, and `.gitignore` (`data/imports/`).
  - **Docs:** README, Architecture and TechStack updated.
- **Decision not covered by the spec:** gold labels for tracker tickets come only from explicit labels (`severity:`, `component:`, `root_cause:`, optional `platform:`), never inferred from priority.
- **Self-Validation Check:**
  - Unit tests make no network calls.
  - The `JUDGE_MODEL` leak case (a Claude id sent to OpenRouter) is covered by a test, after fixing an operator-precedence bug in `get_judge_llm`.

### Step 10 – Provider configuration (user request)
- **Verified Facts:**
  - `.env` now has `LLM_PROVIDER=openrouter` (`OPENROUTER_MODEL` empty, so the Auto Router is used), `JUDGE_PROVIDER=anthropic` and `JUDGE_MODEL=claude-opus-5-5`.
  - `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN` are each present once and non-empty. Only names and emptiness were checked; values were not read.
  - The OpenRouter adapter now marks a model that returns 402 as exhausted and skips it for the rest of the run.
  - `pytest`: 43 passed, including 2 new stubbed fallback tests.
- **Missing / Unknown:** no live OpenRouter, Anthropic or Jira-issue call has been made with the new configuration yet.
- **Generated Output:** `src/llm.py` (exhausted-model skip), `tests/test_openrouter_fallback.py`, README model-selection section, TechStack providers table.
- **Self-Validation Check:** `run_all.py` still defaults to mock for both triage and judge, so the offline benchmark makes no paid calls. Only `src.triage` and `src.calibration` without `--provider` pick up the new `.env` providers.

### Step 11 – Live runs: BUG-012 and Jira SCRUM-6 (user request)
- **Verified Facts:**
  - The OpenRouter fallback was hardened: only 400/401 stop the chain; 402/403/404 models are skipped for the rest of the run; free models are filtered to text-in/text-out; `max_completion_tokens=16000`; an empty 200 reply falls back to the next model.
  - Run 13 (BUG-012): major/player/manifest_error, correct, 3 valid citations. The Opus 5.5 judge gave score 4, grounded=True. Served by `nvidia/nemotron-3.5-super-vl:free`.
  - Run 15 (SCRUM-6): major/player/config_error, 3 valid and 1 invalid citation. Served by the same model. The run printed no judge line.
  - Re-calling the judge directly on the saved run 15 explanation returned valid JSON: score 3, grounded=False. The judge flagged the claim "same issue in 12.2.0, so not a regression" as unsupported. That sentence is in the Jira description, not in the log.
  - `pytest`: 46 passed.
- **Missing / Unknown:** why the judge call inside run 15 failed. `judge_explanation` returns None on any LLMError or ValidationError and does not record which. The retry succeeded, so the failure was not reproducible.
- **Generated Output:** `src/llm.py`, `tests/test_openrouter_fallback.py`. Temp probe scripts deleted.
- **Self-Validation Check:** the judge prompt contains only the log (as the spec defines it), so facts taken from the description are scored as unsupported. Run 15 has no `judge_score` metric rows; Step 12 re-ran the ticket as run 16.

### Step 12 – Judge diagnostics, description context, re-run, commit (user request)
- **Verified Facts:**
  - `judge.score_explanation` returns `(score, None)` or `(None, reason)`, where the reason is `llm error: …` or `invalid judge output: …` with the start of the reply. `judge_explanation` (used by calibration) is unchanged in behaviour.
  - `triage.py --judge` prints `judge_v2: no score, <reason>` under a ticket when the judge fails. `execute` now also returns the failure reasons.
  - `judge_prompt(..., description=None)` adds a fenced, untrusted DESCRIPTION block only when given. `judge_results` passes it only for tracker tickets (`source != "file"`), so the calibrated benchmark judge prompt is byte-for-byte unchanged.
  - Run 16 (SCRUM-6, live): major/api/config_error, 4 valid and 1 invalid citation. Served by `nvidia/nemotron-3.5-super-vl:free`. Opus 5.5 judge: score 4, grounded=True, and it accepted the "12.2.0, not a regression" claim as coming from the description.
  - The one invalid citation is a JSON body that the log has pretty-printed over several lines; the model rewrote it on one line. The checker stays strict (rule: copy exactly); the judge was more lenient on this.
  - `pytest`: 50 passed (4 new in `tests/test_judge.py`).
- **Missing / Unknown:** the description context is not calibrated against human labels. `run_all.py` was not re-run, because it resets `reports/benchmark.db` and would delete live runs 13–16.
- **Generated Output:** `src/judge.py`, `src/prompts.py`, `src/triage.py`, `tests/test_judge.py`; README (tracker judge note, limitations), Architecture, TechStack.
- **Self-Validation Check:** staged files were scanned for API keys and real site names before committing; `.env`, the DB and `.venv` are git-ignored. Human judgement scores were not changed: they need the user's own judgement.

### Definition of done (spec §16)
| Criterion | Status |
|---|---|
| Runs offline with one command in < 2 min | Yes: `python run_all.py`, 13–36 s |
| Prompt variants differ measurably | Yes: exact match 0.26 / 0.34 / 0.68 |
| Severity-miss cases listed | Yes: BUG-002, BUG-039 (v1), BUG-036 (v2), BUG-023 (v3) |
| Hallucinated citations caught | Yes: 8 / 4 / 3 invalid citations |
| Judge kappa improves v1 → v2 | Yes: weighted 0.27 → 0.84, grounded 0.33 → 0.91 |
| `pytest` passes | Yes: 31 passed |
| No secrets, no real company data | Yes: `.env` ignored; all data invented for "StreamNova" |

## Resume here (optional next steps, each needs the user's go-ahead)
1. User reviews and adjusts `data/human_judgements.json` scores (their own judgement), then re-runs `python run_all.py` and refreshes the README numbers. Note: `run_all.py` resets the DB, deleting live runs.
2. Optional paid run: Anthropic as the triage provider, e.g. `python -m src.triage --prompt v3_cot_light --provider anthropic --ids BUG-012`.
3. Commit and push later changes on `main` (only `Bug_Triage/`, not `DRM_Copilot/`).
