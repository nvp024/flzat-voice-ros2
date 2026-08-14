import json

from vlm_pipeline.response_policy import normalize_vlm_response


def test_voice_response_keeps_short_speech() -> None:
    raw = json.dumps({
        "decision": "ask_confirmation",
        "target": "apple",
        "observation": "Two apples are visible.",
        "possible_emotion": "neutral",
        "confidence": "medium",
        "should_speak": True,
        "speech": "Do you mean the red apple?",
    })
    result = normalize_vlm_response(raw, "voice_motion")
    assert result.should_speak is True
    assert result.speech == "Do you mean the red apple?"


def test_motion_response_is_forced_passive_and_spoken() -> None:
    raw = """```json
    {"decision":"pick_object","should_speak":true,"speech":"I will pick it up."}
    ```"""
    result = normalize_vlm_response(raw, "motion")
    assert result.decision == "observe"
    assert result.should_speak is True
    assert result.speech == "I will pick it up."


def test_plain_text_response_is_spoken() -> None:
    result = normalize_vlm_response("I see a person.", "motion")
    assert result.should_speak is True
    assert result.speech == "I see a person."


def test_json_encoded_string_is_spoken() -> None:
    result = normalize_vlm_response('"Hello!"', "voice")
    assert result.should_speak is True
    assert result.speech == "Hello!"
