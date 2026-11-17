from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from httpx import TransportError
from openai import APIConnectionError, OpenAI

from llm_eval.core.settings import FrameworkConfig
from llm_eval.core.timing import StreamTimeline

MAX_ATTEMPTS = 3
FATAL_ERROR_MARKERS = ("invalid api key", "authentication", "unauthorized", "incorrect api key")


class GenerationCancelled(Exception):
    pass


class IncompleteStream(Exception):
    pass


@dataclass(frozen=True)
class GenerationResult:
    content: str
    usage: dict[str, int]
    error: str | None = None
    fatal: bool = False
    http_status_code: int | None = None
    attempts: int = 1
    usage_complete: bool = True
    cancelled: bool = False
    attempt_records: list[dict[str, Any]] = field(default_factory=list)


def request_parameters(config: FrameworkConfig) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "model": config.model.model_name,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if config.run.reasoning_effort == "none" or (config.run.thinking_enabled and config.run.reasoning_effort):
        # A gateway maps the standard field to its provider-specific controls.
        # Do not send a second, potentially conflicting vendor thinking flag.
        parameters["reasoning_effort"] = config.run.reasoning_effort
    elif config.run.thinking_enabled:
        parameters["extra_body"] = {"thinking": {"type": "enabled"}}
    if config.run.max_output_tokens is not None:
        parameters[config.run.output_token_parameter] = config.run.max_output_tokens
    if config.run.temperature is not None:
        parameters["temperature"] = config.run.temperature
    if config.run.seed is not None:
        parameters["seed"] = config.run.seed
    return parameters


class OpenAICompatibleClient:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.cancel_event = threading.Event()
        self._streams_lock = threading.Lock()
        self._active_streams: list[Any] = []
        self.client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url,
            timeout=config.model.timeout_seconds,
            max_retries=0,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        parameters: dict[str, Any] | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        capture_timing: bool = False,
    ) -> GenerationResult:
        if not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS}")
        last_error = "Unknown generation error"
        last_status_code: int | None = None
        # Tokens consumed by failed attempts still count against the run.
        total_usage = _empty_usage()
        usage_complete = True
        fatal = False
        attempts = 0
        attempt_records: list[dict[str, Any]] = []
        partial_content = ""
        for attempt in range(max_attempts):
            if self.cancel_event.is_set():
                last_error = "Generation cancelled"
                break
            attempts += 1
            attempt_usage: dict[str, int] = {}
            stream_started = False
            content = ""
            last_status_code = None
            retryable = True
            parts: list[str] = []
            reasoning: list[str] = []
            response_metadata: dict[str, Any] = {}
            attempt_error = None
            attempt_started = time.monotonic()
            timeline = StreamTimeline(time.perf_counter()) if capture_timing else None
            try:
                kwargs = {**request_parameters(self.config), **(parameters or {}), "messages": messages}

                raw_response = self.client.chat.completions.with_raw_response.create(**kwargs)
                last_status_code = raw_response.status_code
                response_metadata["request_id"] = raw_response.headers.get("x-request-id")
                stream_started = True
                with raw_response.parse() as stream:
                    with self._streams_lock:
                        self._active_streams.append(stream)
                    try:
                        if self.cancel_event.is_set():
                            raise GenerationCancelled("Generation cancelled")
                        content = self._consume_stream(
                            stream, attempt_usage, parts, reasoning, response_metadata, timeline
                        )
                    finally:
                        with self._streams_lock:
                            self._active_streams.remove(stream)
                if not content.strip():
                    last_error = "Model returned an empty completion"
                    attempt_error = last_error
                    # Thinking can exhaust the output budget before any body text.
                    # A length limit is a terminal outcome even when the body is empty.
                    retryable = response_metadata.get("finish_reason") != "length"
            except Exception as exc:
                last_error = str(exc).replace(self.config.model.api_key, "[REDACTED]").replace("\n", " ")
                attempt_error = last_error
                response_metadata["error_type"] = type(exc).__name__
                error_status = getattr(exc, "status_code", None)
                last_status_code = error_status if error_status is not None else last_status_code
                fatal = error_status in (401, 403) or (error_status is None and self._is_fatal_error(last_error))
                transport_error = isinstance(exc, (APIConnectionError, TransportError, IncompleteStream))
                retryable = (
                    transport_error
                    or error_status in (408, 409, 429)
                    or (isinstance(error_status, int) and error_status >= 500)
                )
                if transport_error and not attempt_usage:
                    usage_complete = False
                content = ""
            finally:
                # Keep usage already received even when stream consumption raises.
                _merge_usage(total_usage, attempt_usage)
                if stream_started and not attempt_usage:
                    usage_complete = False
                partial_content = "".join(parts)
                if timeline is not None:
                    response_metadata["timing"] = timeline.finish(time.perf_counter())
                    response_metadata["request_concurrency"] = self.config.run.workers
                attempt_records.append(
                    {
                        "attempt": attempts,
                        "content": partial_content,
                        "reasoning_content": "".join(reasoning),
                        "usage": dict(attempt_usage),
                        "http_status_code": last_status_code,
                        "error": attempt_error,
                        "duration_seconds": time.monotonic() - attempt_started,
                        **response_metadata,
                    }
                )
            if content.strip():
                return GenerationResult(
                    content=content,
                    usage=total_usage,
                    http_status_code=last_status_code,
                    attempts=attempts,
                    usage_complete=usage_complete,
                    attempt_records=attempt_records,
                )
            if fatal or not retryable:
                break
            if attempt < max_attempts - 1:
                self.cancel_event.wait(1 + attempt)

        return GenerationResult(
            content=partial_content,
            usage=total_usage,
            error=last_error,
            fatal=fatal,
            http_status_code=last_status_code,
            attempts=attempts,
            usage_complete=usage_complete,
            cancelled=self.cancel_event.is_set(),
            attempt_records=attempt_records,
        )

    def _consume_stream(
        self,
        stream: Any,
        usage: dict[str, int],
        parts: list[str],
        reasoning: list[str],
        metadata: dict[str, Any],
        timeline: StreamTimeline | None = None,
    ) -> str:
        for chunk in stream:
            if getattr(chunk, "model", None):
                metadata["response_model"] = chunk.model
            if getattr(chunk, "usage", None):
                usage.update(
                    {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }
                )
                details = getattr(chunk.usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                if type(reasoning_tokens) is int:
                    metadata["reasoning_tokens"] = reasoning_tokens
                prompt_details = getattr(chunk.usage, "prompt_tokens_details", None)
                cached = getattr(prompt_details, "cached_tokens", None)
                if cached is None:
                    cached = getattr(chunk.usage, "prompt_cache_hit_tokens", None)
                if type(cached) is int:
                    metadata["cached_prompt_tokens"] = cached
            if self.cancel_event.is_set():
                raise GenerationCancelled("Generation cancelled")
            for choice in getattr(chunk, "choices", None) or []:
                # Only the requested first completion is gradable. Concatenating
                # choices can turn competing answers into a fabricated answer.
                if getattr(choice, "index", None) != 0:
                    continue
                if getattr(choice, "finish_reason", None):
                    metadata["finish_reason"] = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                    parts.append(delta.content)
                if delta and getattr(delta, "reasoning_content", None):
                    reasoning.append(delta.reasoning_content)
                if timeline is not None and delta:
                    timeline.observe(
                        time.perf_counter(),
                        getattr(delta, "content", None) or "",
                        getattr(delta, "reasoning_content", None) or "",
                    )
        if self.cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if not metadata.get("finish_reason"):
            raise IncompleteStream("Model stream ended without a finish_reason; answer may be truncated")
        return "".join(parts)

    def _is_fatal_error(self, message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in FATAL_ERROR_MARKERS)

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._streams_lock:
            streams = list(self._active_streams)
        for stream in streams:
            # The consumer may have closed it concurrently. Its finally block
            # still accounts for usage and releases the response.
            with suppress(Exception):
                stream.close()


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(total: dict[str, int], delta: dict[str, int]) -> None:
    for key in total:
        total[key] += delta.get(key, 0)
