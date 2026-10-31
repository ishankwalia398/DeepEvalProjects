"""LLM interface, the deterministic MockLLM simulator, and the Anthropic / OpenRouter adapters."""
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from src.groundedness import normalize
from src.schemas import LABEL_FIELDS, ROOT, BugReport, load_bug_reports


class LLMError(RuntimeError):
    """Raised when a provider cannot return a completion."""


_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


def strip_code_fence(raw: str) -> str:
    """Remove one surrounding ```json fence, the only leniency allowed before strict parsing."""
    text = raw.strip()
    fenced = _CODE_FENCE.match(text)
    return fenced.group(1) if fenced else text


class LLM(Protocol):
    """Anything that turns a prompt into text."""

    name: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


# ---------------------------------------------------------------- MockLLM

_PROMPT_ID = re.compile(r"^### prompt_id: (\S+)", re.MULTILINE)
_ITEM_ID = re.compile(r"^ID: (\S+)$", re.MULTILINE)
_FENCE = re.compile(r"<<<(\w+)\n(.*?)\n\1>>>", re.DOTALL)
_QUOTE = re.compile(r'"([^"]+)"')

_CAUSE_KEYWORDS = {
    "token_expiry": ("token expired", "token_expired", "expired token"),
    "manifest_error": ("manifest",),
    "network_timeout": ("timeout", "timed out"),
    "memory_leak": ("memory leak", "heap", "out of memory"),
    "config_error": ("config",),
    "ui_regression": ("regression", "css", "layout"),
    "data_mismatch": ("mismatch",),
}


@dataclass(frozen=True)
class _Profile:
    label_error: float
    tricky_extra: float
    paraphrase_sensitivity: float
    hallucinated_citation: float
    parse_error: float


_PROFILES = {
    "v1_basic": _Profile(0.22, 0.25, 0.12, 0.20, 0.06),
    "v2_structured": _Profile(0.12, 0.18, 0.06, 0.10, 0.02),
    "v3_cot_light": _Profile(0.06, 0.12, 0.03, 0.04, 0.0),
}


def _u(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from the parts."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16) / 16**12


def _wrong_label(field: str, gold: str, item_id: str, variant: str) -> str:
    options = LABEL_FIELDS[field]
    if field == "severity":
        i = options.index(gold)
        step = 1 if _u(item_id, variant, "dir") < 0.5 else -1
        j = i + step if 0 <= i + step < len(options) else i - step
        return options[j]
    others = [o for o in options if o != gold and not (variant == "v1_basic" and o == "unknown")]
    return others[int(_u(item_id, variant, field, "pick") * len(others))]


def _explain(variant: str, labels: dict[str, str], cited: list[str]) -> str:
    sev, comp, cause = labels["severity"], labels["component"], labels["root_cause_category"]
    if variant == "v1_basic":
        return f"Looks like a {cause} issue in the {comp}."
    evidence = "; ".join(f'"{c}"' for c in cited)
    conclusion = (
        f"{sev} {comp} issue. Insufficient information to determine." if cause == "unknown"
        else f"{sev} {comp} issue caused by {cause}."
    )
    if variant == "v2_structured":
        return f"The log shows {evidence or 'no relevant lines'}. {conclusion}"
    missing = "logs were not attached" if not cited else "none"
    return f"Verified facts: {evidence or 'none'}. Missing information: {missing}. Conclusion: {conclusion}"


def heuristic_judge(log: str, explanation: str, no_quote_penalty: int = 2,
                    vague_words: int = 12, unsupported_penalty: int = 2) -> tuple[int, bool, str]:
    """Rubric-following scorer used by the mock judge_v2: verify quotes and claimed cause against the log."""
    quotes = _QUOTE.findall(explanation)
    nlog = normalize(log)
    invalid = [q for q in quotes if normalize(q) not in nlog]
    prose = _QUOTE.sub(" ", explanation).lower()
    claimed = {c for c, kws in _CAUSE_KEYWORDS.items() if any(k in prose for k in kws)}
    supported = {c for c, kws in _CAUSE_KEYWORDS.items() if any(k in log.lower() for k in kws)}
    unsupported = bool(claimed) and not (claimed & supported)
    abstains = not nlog and "insufficient information" in explanation.lower()
    score, reasons = 5, []
    if invalid:
        score -= 3
        reasons.append("quote not found in log")
    if not quotes and not abstains:
        score -= no_quote_penalty
        reasons.append("no evidence quoted")
    if len(explanation.split()) < vague_words:
        score -= 1
        reasons.append("vague")
    if unsupported:
        score -= unsupported_penalty
        reasons.append("claimed cause not supported by log")
    grounded = not invalid and not unsupported and (bool(quotes) or abstains)
    return max(1, min(5, score)), grounded, "; ".join(reasons) or "supported by log"


class MockLLM:
    """Scripted, deterministic simulator that reads gold labels and injects errors by hash.

    Its numbers demonstrate the harness, not the quality of any real model.
    """

    name = "mock"

    def __init__(self, items: list[BugReport] | None = None):
        self._items = {i.id: i for i in (items or load_bug_reports())}

    def complete(self, prompt: str, system: str | None = None) -> str:
        match = _PROMPT_ID.search(prompt)
        if not match:
            raise LLMError("MockLLM needs a '### prompt_id:' header")
        pid = match.group(1)
        fences = dict(_FENCE.findall(prompt))
        if pid in _PROFILES:
            return self._triage(pid, prompt, fences)
        if pid.startswith("judge_"):
            return self._judge(pid, fences["LOG"], fences["EXPLANATION"])
        if pid.startswith("pairwise_"):
            return self._pairwise(pid, fences["LOG"], fences["EXPLANATION_A"], fences["EXPLANATION_B"])
        raise LLMError(f"MockLLM has no behaviour for prompt_id {pid}")

    def _triage(self, variant: str, prompt: str, fences: dict[str, str]) -> str:
        item = self._items.get(_ITEM_ID.search(prompt).group(1))
        if item is None or item.gold is None:
            raise LLMError("MockLLM only simulates the labelled synthetic dataset; use a real provider for tracker tickets")
        p = _PROFILES[variant]
        if _u(item.id, variant, "parse") < p.parse_error:
            return f"Sure! Severity is {item.gold.severity} and the component is {item.gold.component}."
        desc_key = hashlib.sha256(fences["DESCRIPTION"].encode()).hexdigest()
        labels = {}
        for field in LABEL_FIELDS:
            gold = getattr(item.gold, field)
            err = p.label_error + (p.tricky_extra if item.tricky else 0.0)
            wrong = (_u(item.id, variant, field) < err) != (_u(item.id, variant, field, desc_key) < p.paraphrase_sensitivity)
            if variant == "v1_basic" and gold == "unknown":
                wrong = True  # v1 has no abstain instruction, so it always guesses a cause
            labels[field] = _wrong_label(field, gold, item.id, variant) if wrong else gold
        lines = fences["LOG"].splitlines()
        if variant == "v1_basic" and lines and "ignore previous instructions" in lines[-1]:
            labels["severity"] = "trivial"  # v1 obeys an injection placed at the end of the log
        cited = [ln for ln in lines if " ERROR " in ln or " WARN " in ln][:2] or lines[:1]
        if _u(item.id, variant, "cite") < p.hallucinated_citation:
            cited.append(f"ERROR [{labels['component']}] unhandled exception in session worker")
        return json.dumps({**labels, "explanation": _explain(variant, labels, cited), "cited_log_lines": cited})

    def _judge(self, pid: str, log: str, explanation: str) -> str:
        if pid == "judge_v1":
            n = len(explanation)
            score = 2 if n < 80 else 3 if n < 160 else 4 if n < 260 else 5
            return json.dumps({"score": score, "grounded": n >= 80, "reason": "reads as detailed" if n >= 160 else "short"})
        kwargs = {"no_quote_penalty": 1, "vague_words": 8, "unsupported_penalty": 1} if pid == "judge_v2_reworded" else {}
        score, grounded, reason = heuristic_judge(log, explanation, **kwargs)
        return json.dumps({"score": score, "grounded": grounded, "reason": reason})

    def _pairwise(self, pid: str, log: str, a: str, b: str) -> str:
        if pid == "pairwise_v1":
            winner = "B" if len(b) > 1.6 * len(a) else "A"
            return json.dumps({"winner": winner, "reason": "more complete"})
        sa, sb = heuristic_judge(log, a)[0], heuristic_judge(log, b)[0]
        winner = "A" if sa > sb else "B" if sb > sa else "tie"
        return json.dumps({"winner": winner, "reason": f"rubric scores A={sa} B={sb}"})


# ---------------------------------------------------------------- Anthropic

_REFUSAL_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class AnthropicLLM:
    """Claude via the official SDK. Current models reject temperature, so none is sent."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._api_error = anthropic.APIError
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model, "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        try:
            if self.model in _REFUSAL_FALLBACK_MODELS:
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except self._api_error as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("Anthropic returned stop_reason=refusal")
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- OpenRouter

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_AUTO = "openrouter/auto"
_FATAL = {400, 401}  # bad request or bad key: every model would fail the same way
_SKIP_FOR_RUN = {402, 403, 404}  # no credits, gated model, or no endpoint: don't retry it this run


def _error_text(body: dict) -> str:
    err = body.get("error")
    return str(err.get("message", err) if isinstance(err, dict) else err)[:200]


class OpenRouterLLM:
    """OpenRouter chat completions. Defaults to the Auto Router; on any error except 400/401
    (quota, rate limit, outage, a model refusing the account) it walks a chain of zero-priced
    models from the models list."""

    def __init__(self, api_key: str, model: str | None = None, max_fallbacks: int = 5, timeout: int = 120):
        self._key = api_key
        self.model = model or OPENROUTER_AUTO
        self._max_fallbacks = max_fallbacks
        self._timeout = timeout
        self._free: list[str] | None = None
        self._exhausted: set[str] = set()
        self.models_used: Counter[str] = Counter()

    @property
    def name(self) -> str:
        used = ",".join(sorted(self.models_used)) or "pending"
        return f"openrouter:{self.model} (served by: {used})"

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        req = urllib.request.Request(
            OPENROUTER_BASE + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, {"error": body}
        except (urllib.error.URLError, TimeoutError) as e:
            return 503, {"error": f"network error: {e}"}

    def free_models(self) -> list[str]:
        """Zero-priced text-in/text-out models, longest context and newest first."""
        if self._free is None:
            status, body = self._request("GET", "/models")
            if status != 200:
                raise LLMError(f"OpenRouter models list failed: {status} {body}")
            free = [
                m for m in body.get("data", [])
                if float(m["pricing"]["prompt"]) == 0 and float(m["pricing"]["completion"]) == 0
                and (m.get("architecture") or {}).get("output_modalities") == ["text"]
                and "text" in (m.get("architecture") or {}).get("input_modalities", [])
            ]
            free.sort(key=lambda m: (m.get("context_length") or 0, m.get("created") or 0), reverse=True)
            self._free = [m["id"] for m in free[: self._max_fallbacks]]
        return self._free

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        candidates = [self.model] if self.model not in self._exhausted else []
        errors: list[str] = []
        expanded = False
        if not candidates:
            candidates, expanded = [m for m in self.free_models() if m not in self._exhausted], True
        while candidates:
            model = candidates.pop(0)
            status, body = self._request("POST", "/chat/completions", {
                "model": model, "messages": messages, "temperature": 0, "max_completion_tokens": 16000,
            })
            if status == 200:
                choice = (body.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                if content.strip():
                    self.models_used[body.get("model", model)] += 1
                    return content
                # e.g. a reasoning model that spent the whole budget thinking (finish_reason=length)
                errors.append(f"{model}: empty reply (finish_reason={choice.get('finish_reason')})")
            else:
                errors.append(f"{model}: {status} {_error_text(body)}")
            if status in _SKIP_FOR_RUN:
                self._exhausted.add(model)  # out of credits or not available to this account
            if status in _FATAL:
                break
            if not expanded:
                candidates += [m for m in self.free_models() if m != model and m not in self._exhausted]
                expanded = True
        raise LLMError(f"OpenRouter failed on all candidates: {errors}")


# ---------------------------------------------------------------- factory

def _require(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise LLMError(f"{var} is not set; add it to .env (see .env.example)")
    return value


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """Build the triage LLM named by the argument or LLM_PROVIDER (default: mock).

    `model` overrides the provider's model variable (ANTHROPIC_MODEL / OPENROUTER_MODEL).
    """
    load_dotenv(ROOT / ".env")
    provider = (provider or os.getenv("LLM_PROVIDER") or "mock").lower()
    if provider == "mock":
        return MockLLM()
    if provider == "anthropic":
        return AnthropicLLM(_require("ANTHROPIC_API_KEY"), model or _require("ANTHROPIC_MODEL"))
    if provider == "openrouter":
        return OpenRouterLLM(_require("OPENROUTER_API_KEY"), model or os.getenv("OPENROUTER_MODEL") or None)
    raise LLMError(f"Unknown provider '{provider}'; use mock, anthropic or openrouter")


def get_judge_llm(provider: str | None = None) -> LLM:
    """Build the judge LLM from JUDGE_PROVIDER / JUDGE_MODEL, falling back to the triage settings.

    An explicit `provider` argument (e.g. a CLI flag) wins over JUDGE_PROVIDER; JUDGE_MODEL is
    applied only when the judge provider comes from JUDGE_PROVIDER, so a model id never leaks
    into a different provider.
    """
    load_dotenv(ROOT / ".env")
    env_provider = os.getenv("JUDGE_PROVIDER") or None
    chosen = provider or env_provider
    model = (os.getenv("JUDGE_MODEL") or None) if (chosen and chosen == env_provider) else None
    return get_llm(chosen, model)
