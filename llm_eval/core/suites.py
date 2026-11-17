"""Validation of versioned, fixed-size evaluation manifests."""

import json
from pathlib import Path
from typing import Any


def load_suite(path: Path, version: str, count: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != version or len(manifest.get("cases", [])) != count:
        raise ValueError(f"Expected {version} with exactly {count} cases")
    ids = [case["id"] for case in manifest["cases"]]
    if len(set(ids)) != count:
        raise ValueError("Suite contains duplicate case IDs")
    return manifest
