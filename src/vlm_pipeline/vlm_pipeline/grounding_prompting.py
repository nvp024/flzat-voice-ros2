"""Render the production Qwen3-VL discovery-grounding prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.grounding_schema import PROMPT_VERSION


class GroundingPromptError(ValueError):
    """The prompt template or request violates the grounding contract."""


@dataclass(frozen=True)
class GroundingPromptBundle:
    """Separated system and user prompts for one discovery request."""

    system_prompt: str
    user_prompt: str
    version: str = PROMPT_VERSION


class GroundingPromptBuilder:
    """Load and render the immutable production grounding prompt profile."""

    def __init__(self, prompt_directory: str = "") -> None:
        if prompt_directory.strip():
            root = Path(prompt_directory).expanduser()
        else:
            from ament_index_python.packages import get_package_share_directory

            root = (
                Path(get_package_share_directory("vlm_pipeline"))
                / "prompts"
                / PROMPT_VERSION
            )
        if not root.is_dir():
            raise FileNotFoundError(f"Grounding prompt directory not found: {root}")
        self.root = root
        self._system_prompt = self._read("system.txt")
        self._template = self._read("discovery.txt")

    def build(self) -> GroundingPromptBundle:
        """Build one unrestricted open-vocabulary discovery request."""
        rendered = self._template
        unresolved = re.findall(r"__[A-Z0-9_]+__", rendered)
        if unresolved:
            raise GroundingPromptError(
                f"prompt has unresolved placeholders: {sorted(set(unresolved))}"
            )
        return GroundingPromptBundle(
            system_prompt=self._system_prompt,
            user_prompt=rendered.strip(),
        )

    def _read(self, filename: str) -> str:
        path = self.root / filename
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Could not read grounding prompt {path}: {exc}") from exc
        if not value:
            raise ValueError(f"Grounding prompt file is empty: {path}")
        return value
