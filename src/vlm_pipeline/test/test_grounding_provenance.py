"""Regression tests for metadata assigned before grounding inference."""

from pathlib import Path
from types import SimpleNamespace

from robot_interfaces.action import GroundObjects

from vlm_pipeline.grounding_prompting import GroundingPromptBuilder
from vlm_pipeline.grounding_schema import PROMPT_VERSION
from vlm_pipeline.vlm_node import VlmNode


def test_grounding_provenance_accepts_v5_prompt_builder() -> None:
    prompt_root = Path(__file__).parents[1] / "prompts" / PROMPT_VERSION
    node = SimpleNamespace(
        _backend=SimpleNamespace(
            config=SimpleNamespace(model_id="Qwen/Qwen3-VL-2B-Instruct"),
            model_revision="main",
        ),
        _grounding_prompt_builder=GroundingPromptBuilder(str(prompt_root)),
    )
    result = GroundObjects.Result()

    VlmNode._set_grounding_provenance(node, result, "observation-123")

    assert result.observation_id == "observation-123"
    assert result.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert result.model_revision == "main"
    assert result.prompt_version == PROMPT_VERSION
