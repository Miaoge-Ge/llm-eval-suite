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
            ("structured_output", 128, 256),
            ("multi_turn", 128, 256),
            ("multilingual", 128, 256),
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
            check = None
            if scenario == "structured_output":
                expected = {"case": index + 1, "topic": topic, "ready": True}
                messages = [
                    {"role": "system", "content": "Return a valid JSON object, without Markdown or commentary."},
                    {"role": "user", "content": "Return exactly this JSON object: " + json.dumps(expected)},
                ]
                check = {"kind": "json_object", "expected": expected}
            elif scenario == "multi_turn":
                marker = f"PROJECT-{index + 1:02d}-FINAL"
                messages = [
                    {
                        "role": "system",
                        "content": "Follow the user's latest instruction. Return only the requested value.",
                    },
                    {"role": "user", "content": f"For {topic}, the project code is PROJECT-{index + 1:02d}-OLD."},
                    {"role": "assistant", "content": "Understood. I will remember that project code."},
                    {"role": "user", "content": f"Correction: replace the project code with {marker}."},
                    {"role": "assistant", "content": "I have updated the project code."},
                    {"role": "user", "content": "What is the updated project code? Return only that code."},
                ]
                check = {"kind": "exact_text", "expected": marker}
            elif scenario == "multilingual":
                marker = f"测试{index + 1:02d}：你好，世界！Hello world — café 🚀 / {topic}"
                messages = [
                    {"role": "system", "content": "Repeat the user's text exactly. Do not add quotes or commentary."},
                    {"role": "user", "content": marker},
                ]
                check = {"kind": "exact_text", "expected": marker}
            cases.append(
                {
                    "id": f"{scenario}/{index + 1:02d}",
                    "scenario": scenario,
                    "messages": messages,
                    "target_output_tokens": target,
                    "max_output_tokens": cap,
                    "reference_input_tokens": sum(len(encoding.encode(m["content"])) for m in messages),
                    "output_check": check,
                }
            )
    save(
        destination / "api-experience-v2.json",
        {
            "version": "api-experience-v2",
            "reference_tokenizer": "cl100k_base (content only; excludes chat framing)",
            "order": "interleaved scenarios, ten rounds; no warmup",
            "cases": cases,
        },
    )


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
