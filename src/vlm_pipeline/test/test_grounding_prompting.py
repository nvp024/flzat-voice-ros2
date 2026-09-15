from pathlib import Path

from vlm_pipeline.grounding_prompting import GroundingPromptBuilder
from vlm_pipeline.grounding_schema import PROMPT_VERSION


def test_grounding_prompt_snapshot_and_open_vocabulary_contract() -> None:
    prompt_root = Path(__file__).parents[1] / "prompts" / PROMPT_VERSION
    prompt = GroundingPromptBuilder(str(prompt_root)).build()

    assert prompt.version == "environment_grounding_qwen3_v5"
    assert "preferred" not in prompt.user_prompt.lower()
    assert "outside this list" not in prompt.user_prompt.lower()
    assert "Do not return confidence, image dimensions" in prompt.user_prompt
    assert '"objects": []' in prompt.user_prompt
    assert "__" not in prompt.user_prompt
