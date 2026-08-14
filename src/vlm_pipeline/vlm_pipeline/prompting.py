from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_EVENT_TEMPLATES = {
    "voice": "voice.txt",
    "voice_motion": "voice_motion.txt",
    "motion": "motion.txt",
    "standalone_test": "standalone_test.txt",
}


@dataclass(frozen=True)
class PromptBundle:
    """Separated instructions and event data passed to a VLM backend."""

    system_prompt: str
    user_prompt: str


class PromptBuilder:
    """Load a prompt profile once and safely wrap live event data."""

    def __init__(self, profile: str, prompt_directory: str = "") -> None:
        if not profile.strip():
            raise ValueError("prompt_profile cannot be empty")
        if prompt_directory.strip():
            root = Path(prompt_directory).expanduser()
        else:
            from ament_index_python.packages import get_package_share_directory

            root = (
                Path(get_package_share_directory("vlm_pipeline"))
                / "prompts"
                / profile
            )
        if not root.is_dir():
            raise FileNotFoundError(f"VLM prompt directory not found: {root}")

        self.profile = profile
        self.root = root
        self._system_prompt = self._read("system.txt")
        self._templates: Mapping[str, str] = {
            event_type: self._read(filename)
            for event_type, filename in _EVENT_TEMPLATES.items()
        }

    def build(
        self,
        event_type: str,
        transcript: str,
        trigger_reason: str,
        default_command: str = "",
    ) -> PromptBundle:
        normalized_type = event_type.strip() or "standalone_test"
        if normalized_type not in self._templates:
            raise ValueError(
                f"Unsupported event_type '{normalized_type}'. Expected one of: "
                f"{', '.join(sorted(self._templates))}"
            )

        human_command = transcript.strip()
        if normalized_type == "standalone_test" and not human_command:
            human_command = default_command.strip()
        if normalized_type in {"voice", "voice_motion"} and not human_command:
            raise ValueError(f"{normalized_type} event requires a transcript")

        replacements = {
            "__EVENT_TYPE_JSON__": json.dumps(normalized_type, ensure_ascii=False),
            "__TRIGGER_REASON_JSON__": json.dumps(
                trigger_reason.strip() or "unspecified",
                ensure_ascii=False,
            ),
            "__HUMAN_COMMAND_JSON__": (
                "null"
                if normalized_type == "motion"
                else json.dumps(human_command, ensure_ascii=False)
            ),
        }
        user_prompt = self._templates[normalized_type]
        for token, value in replacements.items():
            user_prompt = user_prompt.replace(token, value)
        return PromptBundle(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt.strip(),
        )

    def _read(self, filename: str) -> str:
        path = self.root / filename
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Could not read VLM prompt file {path}: {exc}") from exc
        if not value:
            raise ValueError(f"VLM prompt file is empty: {path}")
        return value
