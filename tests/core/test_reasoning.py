import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from llm_eval.core.clients import OpenAICompatibleClient, request_parameters
from llm_eval.core.reasoning import reasoning_presets, validate_reasoning


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_gateway_receives_exact_effort_and_rejections_do_not_downgrade(fake_config, monkeypatch, effort):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "unsupported reasoning effort"}})

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    config = replace(fake_config, run=replace(fake_config.run, thinking_enabled=True, reasoning_effort=effort))
    client = OpenAICompatibleClient(config)
    try:
        result = client.generate([{"role": "user", "content": "hello"}])
    finally:
        client.client.close()
    assert result.attempts == len(bodies) == 1
    assert result.http_status_code == 400 and result.error
    assert bodies[0]["reasoning_effort"] == effort
    assert "thinking" not in bodies[0] and "enable_thinking" not in bodies[0]
    assert config.run.thinking_enabled is (effort != "none")
    assert config.run.reasoning_display == effort


def test_presets_validate_model_specific_levels_and_custom_models():
    presets = reasoning_presets()
    assert len(presets["models"]) == 7
    for model in presets["models"]:
        for effort in presets["efforts"]:
            if effort in model["efforts"]:
                assert validate_reasoning("gateway/" + model["id"].upper(), effort, False) is (effort != "none")
            else:
                with pytest.raises(ValueError):
                    validate_reasoning(model["id"], effort, True)
    assert validate_reasoning("custom", "xhigh", False)
    assert not validate_reasoning("custom", None, False)


def test_legacy_default_thinking_request_remains_compatible(fake_config):
    config = replace(fake_config, run=replace(fake_config.run, thinking_enabled=True))
    assert request_parameters(config)["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in request_parameters(config)
