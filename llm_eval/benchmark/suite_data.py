"""Versioned, immutable selections from the repository's original datasets."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_eval.core.utils import load_jsonl, repo_root


def row_hash(row: dict[str, Any]) -> str:
    original = {key: value for key, value in row.items() if key != "_row_index"}
    return hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


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
