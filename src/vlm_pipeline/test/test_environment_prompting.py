from dataclasses import dataclass

from vlm_pipeline.environment_prompting import EnvironmentPromptBuilder


@dataclass
class Detection:
    detection_id: int = 3
    detector_class: str = "bottle"
    confidence: float = 0.9
    x_min: int = 10
    y_min: int = 20
    x_max: int = 30
    y_max: int = 40


def test_environment_prompt_keeps_detector_identity(tmp_path):
    (tmp_path / "system.txt").write_text("semantic only", encoding="utf-8")
    (tmp_path / "environment.txt").write_text(
        "payload=__OBSERVATION_JSON__", encoding="utf-8"
    )
    prompt = EnvironmentPromptBuilder(str(tmp_path)).build("obs-1", [Detection()])

    assert prompt.system_prompt == "semantic only"
    assert '"observation_id":"obs-1"' in prompt.user_prompt
    assert '"detection_id":3' in prompt.user_prompt
    assert '"bbox":[10,20,30,40]' in prompt.user_prompt


def test_repair_prompt_quotes_invalid_response_and_schema_error(tmp_path):
    (tmp_path / "system.txt").write_text("semantic only", encoding="utf-8")
    (tmp_path / "environment.txt").write_text(
        "payload=__OBSERVATION_JSON__", encoding="utf-8"
    )
    (tmp_path / "repair.txt").write_text(
        "request=__OBSERVATION_JSON__ error=__SCHEMA_ERROR_JSON__ "
        "invalid=__INVALID_RESPONSE_JSON__",
        encoding="utf-8",
    )
    prompt = EnvironmentPromptBuilder(str(tmp_path)).build_repair(
        "obs-1", [Detection()], '{"x": 1}', "unsupported field x"
    )

    assert '"observation_id":"obs-1"' in prompt.user_prompt
    assert '"unsupported field x"' in prompt.user_prompt
    assert '"{\\"x\\": 1}"' in prompt.user_prompt
