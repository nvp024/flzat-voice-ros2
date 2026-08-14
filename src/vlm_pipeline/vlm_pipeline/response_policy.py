from __future__ import annotations

import json
from dataclasses import asdict, dataclass


_DECISIONS = {
    "respond",
    "ask_confirmation",
    "observe",
    "alert",
    "locate_object",
    "pick_object",
    "place_object",
}
_EMOTIONS = {"uncertain", "neutral", "positive", "negative", "distressed"}
_CONFIDENCE = {"low", "medium", "high"}


@dataclass(frozen=True)
class VlmDecision:
    decision: str
    target: str
    observation: str
    possible_emotion: str
    confidence: str
    should_speak: bool
    speech: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def normalize_vlm_response(raw_response: str, event_type: str) -> VlmDecision:
    """Normalize every non-empty model response into speech for TTS."""

    text = _strip_code_fence(raw_response)
    if not text:
        raise ValueError("VLM returned an empty response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text

    is_motion_only = event_type == "motion"
    default_decision = "observe" if is_motion_only else "respond"
    if not isinstance(value, dict):
        speech = value.strip() if isinstance(value, str) else text
        if not speech:
            raise ValueError("VLM returned an empty response")
        return VlmDecision(
            decision=default_decision,
            target="",
            observation="",
            possible_emotion="uncertain",
            confidence="low",
            should_speak=True,
            speech=speech,
        )

    data = value
    decision = _string(data, "decision", default_decision).lower()
    if decision not in _DECISIONS:
        decision = default_decision

    target = _string(data, "target")
    observation = _string(data, "observation")
    emotion = _string(data, "possible_emotion", "uncertain").lower()
    if emotion not in _EMOTIONS:
        emotion = "uncertain"
    confidence = _string(data, "confidence", "low").lower()
    if confidence not in _CONFIDENCE:
        confidence = "low"

    speech = _string(data, "speech")
    if not speech:
        speech = observation or text

    # Motion remains passive perception, but its description is now spoken.
    if is_motion_only:
        if decision not in {"observe", "alert"}:
            decision = "observe"

    return VlmDecision(
        decision=decision,
        target=target,
        observation=observation,
        possible_emotion=emotion,
        confidence=confidence,
        should_speak=True,
        speech=speech,
    )


def _strip_code_fence(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _string(data: dict[str, object], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"VLM field '{key}' must be a string")
    return value.strip()
