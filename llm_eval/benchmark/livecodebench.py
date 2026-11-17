"""LiveCodeBench test import, portable test assets, and stdin output comparison."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import pickle
import shutil
import uuid
import zlib
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
MAX_PRIVATE_BYTES = 512 * 1024 * 1024


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise ValueError("Executable pickle globals are forbidden in test data")

    def persistent_load(self, pid: Any) -> Any:
        raise ValueError("Persistent pickle references are forbidden in test data")


def decode_private_tests(value: str) -> list[dict[str, str]]:
    """Upstream stores either JSON or base64(zlib(pickle(JSON string)))."""
    if not isinstance(value, str) or not value:
        raise ValueError("Missing private test payload")
    try:
        tests = json.loads(value)
    except json.JSONDecodeError:
        compressed = base64.b64decode(value, validate=True)
        decoder = zlib.decompressobj()
        data = decoder.decompress(compressed, MAX_PRIVATE_BYTES + 1)
        if len(data) > MAX_PRIVATE_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("Private test payload is too large or has an invalid compressed envelope") from None
        decoded = _DataOnlyUnpickler(io.BytesIO(data)).load()
        if not isinstance(decoded, str):
            raise ValueError("Private test pickle must contain only a JSON string") from None
        tests = json.loads(decoded)
    validate_tests(tests)
    return tests


def validate_tests(tests: Any) -> None:
    if not isinstance(tests, list):
        raise ValueError("Tests must be a list")
    for test in tests:
        if not isinstance(test, dict) or not all(isinstance(test.get(key), str) for key in ("input", "output")):
            raise ValueError("Each test needs string input and output")
        if test.get("testtype", "stdin") not in ("stdin", "functional"):
            raise ValueError("Unknown LiveCodeBench test type")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(root: Path, descriptor: dict[str, Any]) -> Path:
    relative = Path(descriptor["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Private test asset must stay within its dataset/run directory")
    # Hash-named assets inside nested regrade directories can exceed MAX_PATH.
    if os.name == "nt":
        text = str(path)
        if not text.startswith("\\\\?\\"):
            path = Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
    return path


def prepare_assets(cases: list[Any], source: Path, destination: Path) -> None:
    """Copy verified compressed test files into each run, including offline runs."""
    for case in cases:
        descriptor = case.payload.get("private_tests")
        if descriptor is None:
            continue
        origin = asset_path(source, descriptor)
        if not origin.is_file():
            raise ValueError(f"Private test asset missing: {origin}; regenerate the LiveCodeBench dataset")
        target = asset_path(destination, descriptor)
        if origin == target:
            if file_digest(origin) != descriptor["sha256"]:
                raise ValueError(f"Private test asset checksum mismatch: {origin}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(origin, temporary)
            if file_digest(temporary) != descriptor["sha256"]:
                raise ValueError(f"Private test asset checksum mismatch: {origin}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def iter_private_tests(root: Path, descriptor: dict[str, Any]) -> Iterator[dict[str, str]]:
    count = 0
    with gzip.open(asset_path(root, descriptor), "rt", encoding="utf-8") as handle:
        for line in handle:
            test = json.loads(line)
            validate_tests([test])
            if test.get("testtype", "stdin") != "stdin":
                raise ValueError("Functional tests cannot be graded as stdin/stdout")
            count += 1
            yield test
    if count != descriptor["count"]:
        raise ValueError("Private test count does not match the dataset manifest")


def write_private_tests(root: Path, tests: list[dict[str, str]]) -> dict[str, Any]:
    folder = root / "LiveCodeBench.private"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            for test in tests:
                compressed.write((json.dumps(test, ensure_ascii=False) + "\n").encode())
        checksum = file_digest(temporary)
        target = folder / f"{checksum}.jsonl.gz"
        temporary.replace(target)
        return {"path": target.relative_to(root).as_posix(), "sha256": checksum, "count": len(tests)}
    finally:
        temporary.unlink(missing_ok=True)


def normalize_problem(record: dict[str, Any], root: Path, revision: str) -> dict[str, Any] | None:
    public = json.loads(record["public_test_cases"])
    validate_tests(public)
    # This task deliberately supports only the stdin subset. Do not silently
    # retain just the stdin part of a problem with mixed test types.
    if not public or any(test.get("testtype") != "stdin" for test in public):
        return None
    private = decode_private_tests(record["private_test_cases"])
    if any(test.get("testtype") != "stdin" for test in private):
        raise ValueError(f"Mixed public/private test types for {record['question_id']}")
    if not private:
        raise ValueError(f"No private tests for {record['question_id']}; refusing a public-only import")
    return {
        "question_id": record["question_id"],
        "title": record["question_title"],
        "platform": record["platform"],
        "difficulty": record["difficulty"],
        "problem": record["question_content"],
        "tests": [{"input": test["input"], "output": test["output"]} for test in public],
        "private_tests": write_private_tests(root, private),
        "test_scope": "stdin_public_and_private",
        "source": {"repo": LCB_REPO, "revision": revision, "file": "test.jsonl"},
    }


def outputs_match(actual: str, expected: str) -> bool:
    """Match upstream's stripped lines and exact decimal-token fallback.

    No float tolerance, token reordering, or guessed case-insensitivity.
    """
    actual_lines = [line.strip() for line in actual.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for left, right in zip(actual_lines, expected_lines, strict=True):
        if left == right:
            continue
        try:
            left_numbers = [Decimal(token) for token in left.split()]
            right_numbers = [Decimal(token) for token in right.split()]
            if not all(number.is_finite() for number in (*left_numbers, *right_numbers)):
                return False
            if left_numbers != right_numbers:
                return False
        except InvalidOperation:
            return False
    return True
