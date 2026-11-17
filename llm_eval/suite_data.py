"""Versioned, immutable selections from the repository's original datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .utils import load_jsonl, repo_root


def row_hash(row: dict[str, Any]) -> str:
    original = {key: value for key, value in row.items() if key != "_row_index"}
    return hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_suite(path: Path, version: str, count: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != version or len(manifest.get("cases", [])) != count:
        raise ValueError(f"Expected {version} with exactly {count} cases")
    ids = [case["id"] for case in manifest["cases"]]
    if len(set(ids)) != count:
        raise ValueError("Suite contains duplicate case IDs")
    return manifest


def selected_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sources: dict[str, list[dict[str, Any]]] = {}
    selected = []
    for item in manifest["cases"]:
        name = item["source_file"]
        path = (repo_root() / name).resolve()
        if repo_root().resolve() not in path.parents:
            raise ValueError("Suite source must stay inside the repository")
        if name not in sources:
            sources[name] = load_jsonl(path)
        row = sources[name][item["row_index"]]
        if row_hash(row) != item["sha256"]:
            raise ValueError(f"Original dataset row changed: {item['id']}; create a new suite version")
        selected.append({**item, "source_case": row, "suite_version": manifest["version"]})
    return selected
