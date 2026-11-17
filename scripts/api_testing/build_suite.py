"""Maintainer-only: uv run --with tiktoken python scripts/api_testing/build_suite.py."""

import json

import tiktoken

from llm_eval.core.utils import repo_root


def main():
    destination = repo_root() / "llm_eval" / "api_testing" / "suites"
    destination.mkdir(exist_ok=True)
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
