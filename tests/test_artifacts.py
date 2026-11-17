from pathlib import Path

import pytest

from llm_eval.artifacts import write_atomic


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_write_recovers_from_temporary_windows_lock(tmp_path, monkeypatch, winerror):
    target = tmp_path / "status.json"
    target.write_text("previous", encoding="utf-8")
    original_replace = Path.replace
    attempts = []
    delays = []

    def locked_replace(source, destination):
        attempts.append(source)
        assert source.read_text(encoding="utf-8") == "new"
        assert target.read_text(encoding="utf-8") == "previous"
        if len(attempts) <= 2:
            error = PermissionError("file temporarily locked")
            error.winerror = winerror
            raise error
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("llm_eval.artifacts.time.sleep", delays.append)
    write_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert delays == [0.02, 0.04]
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("winerror,attempt_count", [(5, 5), (None, 1)])
def test_atomic_write_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch, winerror, attempt_count):
    target = tmp_path / "status.json"
    target.write_text("previous", encoding="utf-8")
    attempts = []

    def denied_replace(source, destination):
        attempts.append(source)
        error = PermissionError("denied")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, "replace", denied_replace)
    monkeypatch.setattr("llm_eval.artifacts.time.sleep", lambda delay: None)
    with pytest.raises(PermissionError, match="denied"):
        write_atomic(target, "new")
    assert len(attempts) == attempt_count
    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [target]
