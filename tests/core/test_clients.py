import json

import httpx
import pytest
from openai import OpenAI

from llm_eval.api_testing.metrics import request_metrics
from llm_eval.core.clients import OpenAICompatibleClient


@pytest.fixture
def client_factory(fake_config, monkeypatch):
    clients = []

    def create(handler):
        monkeypatch.setattr(
            "llm_eval.core.clients.OpenAI",
            lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
        )
        client = OpenAICompatibleClient(fake_config)
        monkeypatch.setattr(client.cancel_event, "wait", lambda _: False)
        clients.append(client)
        return client

    yield create
    for client in clients:
        client.client.close()


def sse(content="answer", usage=True, finish_reason="stop"):
    chunks = [{"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}]
    if finish_reason is not None:
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
    if usage:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}})
    return b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks)


@pytest.mark.parametrize(
    ("status", "attempts", "fatal"),
    [
        (429, 3, False),
        (500, 3, False),
        (408, 3, False),
        (409, 3, False),
        (401, 1, True),
        (403, 1, True),
        (400, 1, False),
        (404, 1, False),
        (422, 1, False),
    ],
)
def test_http_retry_count_and_fatal_status(client_factory, status, attempts, fatal):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "localized error", "type": "test_error"}})

    result = client_factory(handler).generate([{"role": "user", "content": "test"}])
    assert len(calls) == result.attempts == attempts
    assert result.http_status_code == status
    assert result.fatal is fatal
    assert result.error


def test_empty_completions_keep_usage_from_all_attempts(client_factory):
    client = client_factory(lambda request: httpx.Response(200, content=sse("")))
    result = client.generate([])
    assert result.error
    assert result.attempts == 3
    assert result.usage == {"prompt_tokens": 15, "completion_tokens": 21, "total_tokens": 36}
    assert result.usage_complete


@pytest.mark.parametrize("usage", [True, False])
@pytest.mark.parametrize("recover", [True, False])
def test_interrupted_stream_accounts_for_usage_and_closes(client_factory, usage, recover):
    streams = []

    class BrokenStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield sse("partial", usage=usage)
            raise httpx.ReadError("stream disconnected")

        def close(self):
            self.closed = True

    def handler(request):
        if recover and streams:
            return httpx.Response(200, content=sse("recovered"))
        stream = BrokenStream()
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    result = client_factory(handler).generate([])
    assert result.content == ("recovered" if recover else "partial")
    assert result.attempt_records[0]["content"] == "partial"
    assert bool(result.error) is not recover
    assert result.attempts == (2 if recover else 3)
    assert result.usage["total_tokens"] == (12 * len(streams) if usage else 0) + (12 if recover else 0)
    assert result.usage_complete is usage
    assert all(stream.closed for stream in streams)


def test_success_without_usage_is_marked_incomplete(client_factory):
    result = client_factory(lambda request: httpx.Response(200, content=sse(usage=False))).generate([])
    assert result.content == "answer"
    assert not result.error
    assert not result.usage_complete
    assert result.usage["total_tokens"] == 0


def test_fatal_error_keeps_usage_from_previous_attempt(client_factory):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, content=sse(""))
        return httpx.Response(403, json={"error": {"message": "access denied"}})

    result = client_factory(handler).generate([])
    assert result.fatal
    assert result.attempts == len(calls) == 2
    assert result.usage["total_tokens"] == 12


def test_connection_errors_have_one_retry_budget(client_factory):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection failed", request=request)

    result = client_factory(handler).generate([])
    assert result.attempts == len(calls) == 3
    assert result.error
    assert not result.fatal
    assert not result.usage_complete


def test_cancelled_generation_does_not_send_a_request(client_factory):
    client = client_factory(lambda request: pytest.fail("Unexpected API request"))
    client.cancel_event.set()
    result = client.generate([])
    assert result.cancelled
    assert result.attempts == 0


def test_cancelled_stream_retains_known_usage(client_factory):
    class CancelStream(httpx.SyncByteStream):
        def __iter__(self):
            yield sse("", usage=True)
            client.cancel_event.set()
            yield sse("partial", usage=False)

    client = client_factory(lambda request: httpx.Response(200, stream=CancelStream()))
    result = client.generate([])
    assert result.cancelled
    assert result.attempts == 1
    assert result.usage["total_tokens"] == 12


@pytest.mark.parametrize("recover", [False, True])
def test_clean_eof_without_finish_reason_is_not_a_complete_answer(client_factory, recover):
    calls = []

    def handler(request):
        calls.append(request)
        recovered = recover and len(calls) > 1
        body = sse("#### 4", finish_reason="stop" if recovered else None)
        return httpx.Response(200, content=body + b"data: [DONE]\n\n")

    result = client_factory(handler).generate([])
    assert bool(result.error) is not recover
    assert result.attempts == (2 if recover else 3)
    assert result.usage["total_tokens"] == 12 * result.attempts
    assert result.attempt_records[0]["content"] == "#### 4"
    assert "without a finish_reason" in result.attempt_records[0]["error"]


def test_stream_does_not_concatenate_multiple_choices(client_factory):
    extra = {"choices": [{"index": 1, "delta": {"content": "wrong alternative"}, "finish_reason": "length"}]}
    body = sse("answer") + ("data: " + json.dumps(extra) + "\n\n").encode()
    result = client_factory(lambda request: httpx.Response(200, content=body)).generate([])
    assert result.content == "answer"
    assert result.attempt_records[0]["finish_reason"] == "stop"


@pytest.mark.parametrize("content", ["answer", "", " \n"])
def test_length_finish_reason_is_a_completed_model_outcome_without_retry(client_factory, content):
    result = client_factory(lambda request: httpx.Response(200, content=sse(content, finish_reason="length"))).generate(
        []
    )
    assert bool(result.error) is (not bool(content.strip()))
    assert result.attempts == 1


def test_cancel_at_stream_eof_does_not_accept_partial_answer(client_factory):
    class CancelAtEof(httpx.SyncByteStream):
        def __iter__(self):
            yield sse("partial")
            client.cancel_event.set()

    client = client_factory(lambda request: httpx.Response(200, stream=CancelAtEof()))
    result = client.generate([])
    assert result.cancelled
    assert result.error
    assert result.attempts == 1


@pytest.mark.parametrize("failure", [429, 500, 408, 409, "timeout", "connect", "disconnect", "empty", "eof"])
def test_timing_requests_never_retry_failures(client_factory, failure):
    calls = []
    streams = []

    class BrokenStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield sse("partial", usage=False, finish_reason=None)
            raise httpx.ReadError("connection broke")

        def close(self):
            self.closed = True

    def handler(request):
        calls.append(request)
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "failed"}})
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure == "connect":
            raise httpx.ConnectError("unreachable", request=request)
        if failure == "disconnect":
            stream = BrokenStream()
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        return httpx.Response(
            200,
            content=sse("" if failure == "empty" else "partial", finish_reason=None if failure == "eof" else "stop"),
        )

    client = client_factory(handler)
    result = client.generate([], max_attempts=1, capture_timing=True)
    assert result.error and len(calls) == result.attempts == len(result.attempt_records) == 1
    attempt = result.attempt_records[0]
    metrics = request_metrics(attempt, thinking=False, stall_threshold=1, slo={})
    assert not metrics["success"]
    assert metrics["end_to_end_seconds"] >= 0
    if failure in ("disconnect", "eof"):
        assert result.content == "partial"
        assert attempt["timing"]["events"][0]["content"] == "partial"
    assert all(stream.closed for stream in streams)
