"""Download benchmark datasets from Hugging Face and normalize them to the
JSONL layout expected by llm_eval tasks.

Run with a throwaway dependency set so the runtime project stays lean:

    uv run --with datasets python scripts/benchmark/fetch_datasets.py aime
    uv run --with datasets python scripts/benchmark/fetch_datasets.py all

Each builder writes a single file under datasets/benchmark/ with the field names the
matching task class reads.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from llm_eval.benchmark.livecodebench import LCB_REPO, LCB_REVISION, normalize_problem
from llm_eval.core.artifacts import write_atomic

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "benchmark"


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    # LiveCodeBench's raw JSONL importer does not need the datasets package.
    from datasets import load_dataset as hf_load_dataset

    return hf_load_dataset(*args, **kwargs)


def _write_jsonl(filename: str, rows: Iterable[dict[str, Any]]) -> None:
    path = DATASETS_DIR / filename
    records = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    write_atomic(path, "".join(records))
    count = len(records)
    print(f"wrote {count} rows -> {path.relative_to(DATASETS_DIR.parent)}")


def build_aime2025() -> None:
    rows: list[dict[str, Any]] = []
    for config in ("AIME2025-I", "AIME2025-II"):
        dataset = load_dataset("opencompass/AIME2025", config, split="test")
        for index, record in enumerate(dataset):
            rows.append(
                {
                    "id": f"{config}/{index}",
                    "problem": record["question"],
                    "answer": str(record["answer"]).strip(),
                }
            )
    _write_jsonl("AIME2025.jsonl", rows)


def _infer_entry_point(code: str, test: str) -> str:
    defs = re.findall(r"def (\w+)\(", code)
    used = [name for name in defs if name in test]
    candidates = used or defs
    return candidates[-1] if candidates else "solution"


def build_mbppplus() -> None:
    dataset = load_dataset("evalplus/mbppplus", split="test")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        rows.append(
            {
                "task_id": f"Mbpp/{record['task_id']}",
                "prompt": record["prompt"],
                "entry_point": _infer_entry_point(record["code"], record["test"]),
                "test_imports": list(record["test_imports"]),
                "test": record["test"],
            }
        )
    _write_jsonl("MBPPPlus.jsonl", rows)


def build_ifeval() -> None:
    dataset = load_dataset("google/IFEval", split="train")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        rows.append(
            {
                "key": record["key"],
                "prompt": record["prompt"],
                "instruction_id_list": list(record["instruction_id_list"]),
                "kwargs": [dict(item) for item in record["kwargs"]],
            }
        )
    _write_jsonl("IFEval.jsonl", rows)


def build_livecodebench(source: Path | None = None, revision: str = LCB_REVISION) -> None:
    if source is None:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(LCB_REPO, "test.jsonl", repo_type="dataset", revision=revision))
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized = normalize_problem(record, DATASETS_DIR, revision)
            if normalized is not None:
                rows.append(normalized)
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate LiveCodeBench problem IDs")
    _write_jsonl("LiveCodeBench.jsonl", rows)


def build_aime2026() -> None:
    dataset = load_dataset("math-ai/aime26", split="test")
    rows = [
        {
            "id": f"AIME2026/{record['id']}",
            "problem": record["problem"],
            "answer": str(record["answer"]).strip(),
        }
        for record in dataset
    ]
    _write_jsonl("AIME2026.jsonl", rows)


def build_mmlu_pro(per_category: int = 30) -> None:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset:
        by_category[record["category"]].append(record)

    rng = random.Random(2026)
    rows: list[dict[str, Any]] = []
    for category in sorted(by_category):
        items = by_category[category]
        picked = items if len(items) <= per_category else rng.sample(items, per_category)
        for record in sorted(picked, key=lambda item: item["question_id"]):
            rows.append(
                {
                    "id": f"mmlu_pro/{record['question_id']}",
                    "question": record["question"],
                    "options": list(record["options"]),
                    "answer": str(record["answer"]).strip().upper(),
                    "category": record["category"],
                }
            )
    _write_jsonl("MMLUPro.jsonl", rows)


BUILDERS: dict[str, Callable[[], None]] = {
    "aime": build_aime2025,
    "aime2026": build_aime2026,
    "mbppplus": build_mbppplus,
    "ifeval": build_ifeval,
    "livecodebench": build_livecodebench,
    "mmlu_pro": build_mmlu_pro,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=[*BUILDERS, "all"], help="dataset to fetch")
    parser.add_argument("--source", type=Path, help="Local upstream LiveCodeBench test.jsonl")
    parser.add_argument("--revision", default=LCB_REVISION, help="LiveCodeBench dataset revision")
    args = parser.parse_args()

    if args.target == "livecodebench":
        build_livecodebench(args.source, args.revision)
        return
    if args.source:
        parser.error("--source is only supported for livecodebench")

    targets = BUILDERS.values() if args.target == "all" else [BUILDERS[args.target]]
    for builder in targets:
        builder()


if __name__ == "__main__":
    main()
