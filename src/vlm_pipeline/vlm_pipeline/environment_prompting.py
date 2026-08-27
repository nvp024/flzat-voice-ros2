from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EnvironmentPromptBundle:
    system_prompt: str
    user_prompt: str


class EnvironmentPromptBuilder:
    """Build the detector-linked environment prompt without parsing its output."""

    def __init__(self, prompt_directory: str = "") -> None:
        if prompt_directory.strip():
            root = Path(prompt_directory).expanduser()
        else:
            from ament_index_python.packages import get_package_share_directory

            root = (
                Path(get_package_share_directory("vlm_pipeline"))
                / "prompts"
                / "environment_memory_v1"
            )
        if not root.is_dir():
            raise FileNotFoundError(f"Environment prompt directory not found: {root}")
        self.root = root
        self._system_prompt = self._read("system.txt")
        self._environment_template = self._read("environment.txt")

    def build(self, observation_id: str, detections: Iterable[object]):
        payload = self._payload(observation_id, detections)
        user_prompt = self._environment_template.replace(
            "__OBSERVATION_JSON__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return EnvironmentPromptBundle(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt.strip(),
        )

    def build_repair(
        self,
        observation_id: str,
        detections: Iterable[object],
        invalid_response: str,
        schema_error: str,
    ) -> EnvironmentPromptBundle:
        template = self._read("repair.txt")
        replacements = {
            "__OBSERVATION_JSON__": json.dumps(
                self._payload(observation_id, detections),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "__INVALID_RESPONSE_JSON__": json.dumps(
                invalid_response, ensure_ascii=False
            ),
            "__SCHEMA_ERROR_JSON__": json.dumps(schema_error, ensure_ascii=False),
        }
        for token, value in replacements.items():
            template = template.replace(token, value)
        return EnvironmentPromptBundle(
            system_prompt=self._system_prompt,
            user_prompt=template.strip(),
        )

    @staticmethod
    def _payload(observation_id: str, detections: Iterable[object]) -> dict:
        detection_values = []
        for detection in detections:
            detection_values.append(
                {
                    "detection_id": int(detection.detection_id),
                    "detector_class": str(detection.detector_class),
                    "detector_confidence": float(detection.confidence),
                    "bbox": [
                        int(detection.x_min),
                        int(detection.y_min),
                        int(detection.x_max),
                        int(detection.y_max),
                    ],
                }
            )
        return {
            "observation_id": observation_id,
            "detections": detection_values,
        }

    def _read(self, filename: str) -> str:
        path = self.root / filename
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Could not read environment prompt {path}: {exc}"
            ) from exc
        if not value:
            raise ValueError(f"Environment prompt file is empty: {path}")
        return value
