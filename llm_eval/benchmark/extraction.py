"""Parsing helpers that pull gradable answers out of raw model output."""

from __future__ import annotations

import re
import textwrap

PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python|py)?[ \t]*(?:\r?\n)?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    match = PYTHON_BLOCK_PATTERN.search(text)
    extracted = match.group(1) if match else text
    return dedent_code(extracted)


def dedent_code(code: str) -> str:
    # Dedent before stripping: stripping the first line first loses its common
    # indent and changes the relative indentation of nested function bodies.
    return textwrap.dedent(code).strip()


def indent_block(code: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else "" for line in code.splitlines())


def last_numeric_token(text: str) -> str | None:
    if not text:
        return None
    number = r"-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
    if "####" in text:
        # An invalid final marker must not revive an earlier answer.
        answer = text.rsplit("####", 1)[1].strip().splitlines()
        if not answer:
            return None
        time_unit = r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
        conversion = rf"(?:\s*\((?:or\s+)?(?:{number}\s+{time_unit}\s*)+\))?"
        unit = rf"(?:\s+{time_unit}{conversion}|\s+(?:dollars?|cents?|euros?|pounds?)|\s*%)?"
        rate_unit = rf"(?:\s+per\s+(?:{time_unit}|cups?|items?|pieces?|persons?|people|students?|tickets?))?"
        # Explicit answer markers may include a currency or a known unit. Do not
        # accept arithmetic, prose alternatives, or numbers buried in reasoning.
        explicit = re.fullmatch(rf"[$£€]?\s*({number}){unit}{rate_unit}[.]?", answer[0].strip(), re.IGNORECASE)
        return explicit.group(1).replace(",", "") if explicit else None
    lines = text.strip().splitlines()
    if not lines:
        return None
    # A fragment of a fraction/expression, or a number mentioned in reasoning,
    # is not a final answer. Accept only a whole numeric final line/declaration.
    match = re.fullmatch(
        rf"(?:(?:the\s+)?(?:final\s+)?(?:answer|total)\s*(?:is\s+|:\s*))?({number})[.]?",
        lines[-1].strip(),
        re.IGNORECASE,
    )
    return match.group(1).replace(",", "") if match else None


def extract_last_boxed(text: str) -> str | None:
    if not text:
        return None
    needle = "\\boxed"
    start = text.rfind(needle)
    if start == -1:
        return None
    index = start + len(needle)
    while index < len(text) and text[index] != "{":
        if text[index] == " ":
            index += 1
            continue
        return None
    if index >= len(text):
        return None
    depth = 0
    content_start = index + 1
    for position in range(index, len(text)):
        char = text[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:position]
    return None


def extract_choice_letter(text: str, last_letter: str = "D") -> str | None:
    if not text:
        return None
    char_class = f"[A-{last_letter}]"
    boxed = extract_last_boxed(text)
    choice = rf"(?:\(({char_class})\)|({char_class}))"
    if "\\boxed" in text:
        # Boxes are authoritative, including malformed or ambiguous ones.
        match = re.fullmatch(
            rf"\s*(?:answer\s*(?:is\s+|:\s*))?{choice}\s*",
            boxed or "",
            re.IGNORECASE,
        )
        return next((group.upper() for group in match.groups() if group), None) if match else None
    stripped = text.strip()
    if not stripped:
        return None
    markers = list(re.finditer(r"\banswer\s*(?:is\b\s*|:\s*|\s+)", stripped, re.IGNORECASE))
    if markers:
        answer = stripped[markers[-1].end() :].strip()
        match = re.fullmatch(rf"{choice}[.!]?", answer, re.IGNORECASE)
        return next((group.upper() for group in match.groups() if group), None) if match else None
    # Bare letters must stay case-sensitive: with IGNORECASE the article "a"
    # at the end of a line would be read as answer A.
    match = re.fullmatch(rf"{choice}[.!]?", stripped.splitlines()[-1])
    return next((group for group in match.groups() if group), None) if match else None
