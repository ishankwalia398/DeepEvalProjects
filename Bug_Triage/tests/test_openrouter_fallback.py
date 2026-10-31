"""OpenRouter fallback chain with a stubbed HTTP layer (no network)."""
import pytest

from src.llm import LLMError, OpenRouterLLM


class StubRouter(OpenRouterLLM):
    def __init__(self, replies: dict[str, int]):
        super().__init__("test-key")
        self.replies, self.calls = replies, []

    def _request(self, method, path, payload=None):
        if path == "/models":
            text = {"input_modalities": ["text"], "output_modalities": ["text"]}
            return 200, {"data": [
                {"id": "small:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 8000,
                 "created": 2, "architecture": text},
                {"id": "big:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 128000,
                 "created": 1, "architecture": text},
                {"id": "music:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 999999,
                 "architecture": {"input_modalities": ["text"], "output_modalities": ["text", "audio"]}},
                {"id": "paid", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "context_length": 200000,
                 "architecture": text},
            ]}
        model = payload["model"]
        self.calls.append(model)
        status = self.replies[model]
        if status == 200:
            return 200, {"model": model, "choices": [{"message": {"content": "ok"}}]}
        if status == "empty":
            return 200, {"model": model, "choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        return status, {"error": "x"}


def test_quota_falls_back_to_free_models_and_skips_exhausted_primary():
    llm = StubRouter({"openrouter/auto": 402, "big:free": 429, "small:free": 200})
    assert llm.complete("hi") == "ok"
    assert llm.calls == ["openrouter/auto", "big:free", "small:free"]
    llm.calls.clear()
    assert llm.complete("again") == "ok"
    assert llm.calls == ["big:free", "small:free"]
    assert llm.models_used == {"small:free": 2}
    assert llm.name == "openrouter:openrouter/auto (served by: small:free)"


def test_forbidden_free_model_moves_on_and_is_skipped_later():
    llm = StubRouter({"openrouter/auto": 402, "big:free": 403, "small:free": 200})
    assert llm.complete("hi") == "ok"
    assert llm.calls == ["openrouter/auto", "big:free", "small:free"]
    llm.calls.clear()
    assert llm.complete("again") == "ok"
    assert llm.calls == ["small:free"]


def test_empty_reply_falls_back_to_next_model():
    llm = StubRouter({"openrouter/auto": "empty", "big:free": 200})
    assert llm.complete("hi") == "ok"
    assert llm.calls == ["openrouter/auto", "big:free"]


def test_free_models_are_text_only():
    assert StubRouter({}).free_models() == ["big:free", "small:free"]


def test_non_retryable_error_stops_immediately():
    llm = StubRouter({"openrouter/auto": 400})
    with pytest.raises(LLMError, match="openrouter/auto: 400"):
        llm.complete("hi")
    assert llm.calls == ["openrouter/auto"]
