"""Maintainer-only: uv run --with tiktoken python scripts/build_lightweight_suites.py.

Review selections before publishing a new version. Runtime never samples or rebuilds.
"""

import json

import tiktoken

from llm_eval.settings import DEFAULT_DATASET_PATHS
from llm_eval.suite_data import row_hash
from llm_eval.tasks import TASK_REGISTRY
from llm_eval.utils import load_jsonl, repo_root

SELECTION = {
    "humanevalplus": (
        "code",
        [
            (1, "nested parenthesis parsing"),
            (10, "shortest palindrome construction"),
            (21, "floating-point normalization"),
            (40, "three-sum distinct indices"),
            (69, "frequency counting and boundary conditions"),
        ],
    ),
    "mbppplus": (
        "code",
        [
            (0, "set intersection"),
            (5, "list transformation"),
            (15, "digit reversal"),
            (30, "number theory"),
            (50, "nested collection aggregation"),
        ],
    ),
    "gsm": (
        "math",
        [
            (7, "restart and rate arithmetic"),
            (26, "pairs and money"),
            (57, "capacity rounding"),
            (103, "distance and rate"),
            (209, "dozens and mixed prices"),
            (401, "annual income units"),
        ],
    ),
    "aime2025": (
        "math",
        [
            (1, "area ratios and reflections"),
            (5, "tangency and trapezoid algebra"),
            (18, "logarithm telescoping product"),
            (24, "constrained combinatorics"),
        ],
    ),
    "mmlu_pro": (
        "knowledge",
        [
            (0, "biology"),
            (60, "chemistry"),
            (90, "computer science"),
            (120, "economics"),
            (240, "law"),
            (330, "philosophy"),
        ],
    ),
    "gpqa": (
        "knowledge",
        [
            (0, "physics: lifetime and energy"),
            (10, "molecular biology"),
            (30, "organic chemistry and symmetry"),
            (150, "physics: quantum measurement"),
        ],
    ),
    "ifeval": (
        "instructions",
        [
            (1, "no commas"),
            (2, "placeholder count"),
            (3, "verbatim repetition and title"),
            (4, "lowercase"),
            (5, "exact bullet count"),
            (11, "JSON format"),
            (12, "paragraph count"),
            (16, "two distinct responses"),
            (21, "exact ending"),
            (28, "forbidden words"),
        ],
    ),
}


def main():
    destination = repo_root() / "llm_eval" / "suites"
    destination.mkdir(exist_ok=True)
    cases = []
    for task, (dimension, selections) in SELECTION.items():
        source = DEFAULT_DATASET_PATHS[task]
        rows = load_jsonl(repo_root() / source)
        for index, reason in selections:
            row = rows[index]
            source_id = next((str(row[key]) for key in TASK_REGISTRY[task].id_fields if key in row), str(index))
            cases.append(
                {
                    "id": f"{task}/{source_id}",
                    "source_task": task,
                    "source_id": source_id,
                    "source_file": source.as_posix(),
                    "row_index": index,
                    "sha256": row_hash(row),
                    "dimension": dimension,
                    "reason": reason,
                }
            )
    save(
        destination / "quick40-v1.json",
        {
            "version": "quick40-v1",
            "selection_policy": (
                "Manually curated topic coverage, not statistically calibrated difficulty; original rows unchanged."
            ),
            "cases": cases,
        },
    )

    encoding = tiktoken.get_encoding("cl100k_base")
    topics = [
        "planning a weekly grocery list",
        "organizing a home workspace",
        "preparing for a train journey",
        "taking useful meeting notes",
        "backing up personal documents",
        "learning a new programming language",
        "planning a small community event",
        "building a reading habit",
        "writing a clear bug report",
        "preparing a project handover",
    ]
    cases = []
    for index, topic in enumerate(topics):
        for scenario, target, cap in [
            ("short_answer", 128, 256),
            ("long_answer", 512, 768),
            ("long_context", 128, 256),
        ]:
            system = "You are a helpful assistant. Follow the user's requested length and write in English."
            prompt = (
                f"Session example {index + 1}: Give practical advice about {topic}. "
                f"Write approximately {90 if target == 128 else 360} words in complete sentences."
            )
            if scenario == "long_context":
                prompt = (
                    f"Session example {index + 1}: Read these project notes and summarize the main actions "
                    "in approximately 90 words.\n\n"
                )
                note = 0
                while len(encoding.encode(system + prompt)) < 4000:
                    note += 1
                    prompt += (
                        f"Note {note}: The team discussed {topic}. Record the owner, check dependencies, "
                        "confirm the schedule, and explain the next action clearly. "
                        "Keep a copy of decisions and review unfinished work at the next meeting.\n"
                    )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            cases.append(
                {
                    "id": f"{scenario}/{index + 1:02d}",
                    "scenario": scenario,
                    "messages": messages,
                    "target_output_tokens": target,
                    "max_output_tokens": cap,
                    "reference_input_tokens": sum(len(encoding.encode(m["content"])) for m in messages),
                }
            )
    save(
        destination / "api-experience-v1.json",
        {
            "version": "api-experience-v1",
            "reference_tokenizer": "cl100k_base (content only; excludes chat framing)",
            "order": "interleaved scenarios, ten rounds; no warmup",
            "cases": cases,
        },
    )


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
