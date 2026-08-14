from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechDecision:
    decision: str
    should_speak: bool
    speech: str


def speech_from_vlm_response(response_text: str, event_type: str) -> SpeechDecision:
    """Route every non-empty VLM response to TTS."""

    response = response_text.strip()
    if not response:
        raise ValueError("VLM action returned an empty response")

    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        # Small VLMs are more reliable with short natural-language answers.
        return SpeechDecision("respond", True, response)

    if not isinstance(value, dict):
        speech = value.strip() if isinstance(value, str) else response
        decision = "observe" if event_type == "motion" else "respond"
        return SpeechDecision(decision, True, speech)

    decision = value.get("decision", "observe" if event_type == "motion" else "respond")
    speech = value.get("speech", "")
    if not isinstance(decision, str):
        raise ValueError("VLM decision must be a string")
    if not isinstance(speech, str):
        raise ValueError("VLM speech must be a string")

    speech = speech.strip()
    if not speech:
        observation = value.get("observation", "")
        speech = observation.strip() if isinstance(observation, str) else response
    if not speech:
        speech = response
    return SpeechDecision(decision.strip(), True, speech)
