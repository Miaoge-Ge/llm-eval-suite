"""Corrections for the bundled EvalPlus exported assertion programs.

Keep the input/output corpus unchanged. Enforce the upstream evaluator's
type/length guards before tolerant comparisons, and repair the known broken
find_zero oracle in the exported HumanEval/32 test program.
"""

import re


def prepare_test_code(source: str, entry_point: str) -> str:
    if entry_point == "find_zero":
        source = source.replace(
            "assert _poly(*candidate(*inp), inp) <= 0.0001",
            "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001",
        )

    def guarded(match: re.Match[str]) -> str:
        indent, assertion = match.groups()
        return (
            f"{indent}assert type(out) is type(exp)\n"
            f"{indent}if isinstance(exp, (list, tuple)):\n"
            f"{indent}    assert len(out) == len(exp)\n"
            f"{indent}{assertion}"
        )

    return re.sub(
        r"(?m)^([ \t]*)(assert np\.allclose\(out, exp, rtol=1e-07, atol=atol\))[ \t]*$",
        guarded,
        source,
    )
