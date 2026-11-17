import json
from dataclasses import replace

import pytest

from llm_eval.api_testing.task import ApiExperienceTask
from llm_eval.core.clients import GenerationResult
from llm_eval.core.settings import DEFAULT_DATASET_PATHS, DatasetConfig
from llm_eval.core.utils import repo_root


@pytest.fixture
def task(fake_config):
    return ApiExperienceTask(
        replace(
            fake_config,
            dataset=DatasetConfig("api_experience", repo_root() / DEFAULT_DATASET_PATHS["api_experience"]),
            run=replace(fake_config.run, task="api_experience"),
        )
    )


def generation(answer, finish="stop"):
    return GenerationResult(
        answer,
        {},
        attempt_records=[
            {"finish_reason": finish, "timing": {"events": [{"at_seconds": 1, "content": answer, "reasoning": ""}]}}
        ],
    )


@pytest.mark.parametrize("scenario", ["structured_output", "multi_turn", "multilingual"])
def test_output_checks_do_not_change_transport_success(task, scenario):
    case = next(c for c in task.load_cases() if c.payload["scenario"] == scenario)
    expected = case.payload["output_check"]["expected"]
    answer = json.dumps(expected, ensure_ascii=False) if isinstance(expected, dict) else expected
    good = task.grade_case(case, generation(answer))
    wrong = task.grade_case(case, generation("incorrect output"))
    truncated = task.grade_case(case, generation(answer, "length"))
    for row, passed in ((good, True), (wrong, False), (truncated, None)):
        assert row.status == "PASSED"
        assert row.metadata["experience"]["success"]
        assert row.metadata["experience"]["output_check"]["passed"] is passed
        row.metadata.update(task.case_metadata(case))
    overall = task.summarize([r.to_dict() for r in (good, wrong, truncated)])["overall"]
    assert overall["successful"] == 3 and overall["truncated"] == 1
    assert overall["output_checks"] == {"checked": 2, "passed": 1, "pass_rate": 0.5}


def test_json_requires_exact_object_and_real_boolean(task):
    case = next(c for c in task.load_cases() if c.payload["scenario"] == "structured_output")
    answer = {**case.payload["output_check"]["expected"], "ready": 1}
    result = task.grade_case(case, generation(json.dumps(answer)))
    assert result.metadata["experience"]["output_check"]["passed"] is False
    assert task.generation_options(case)["parameters"]["response_format"] == {"type": "json_object"}


def test_previous_api_suite_can_still_be_loaded_for_reproducibility(task):
    old = ApiExperienceTask(
        replace(
            task.config,
            dataset=DatasetConfig("api_experience", repo_root() / "llm_eval/api_testing/suites/api-experience-v1.json"),
        )
    )
    cases = old.load_cases()
    assert len(cases) == 30
    assert old.case_metadata(cases[0])["suite_version"] == "api-experience-v1"
