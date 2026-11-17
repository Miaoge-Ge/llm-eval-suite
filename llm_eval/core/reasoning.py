"""User-supplied gateway model presets; these are not provider capability probes."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def reasoning_presets() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("reasoning_presets.json").read_text(encoding="utf-8"))


def validate_reasoning(model: str, effort: str | None, thinking_enabled: bool) -> bool:
    if effort is None:
        return thinking_enabled
    presets = reasoning_presets()
    supported = presets["efforts"]
    name = model.strip().rsplit("/", 1)[-1].lower()
    for preset in presets["models"]:
        if preset["id"].lower() == name:
            supported = preset["efforts"]
            break
    if effort not in supported:
        raise ValueError("此模型的思考强度无效，请重新选择。")
    return effort != "none"
