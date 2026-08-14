import json

from trigger_engine.output_policy import speech_from_vlm_response


def test_voice_speech_is_allowed() -> None:
    result = speech_from_vlm_response(
        json.dumps({
            "decision": "respond",
            "should_speak": True,
            "speech": "Hello there.",
        }),
        "voice",
    )
    assert result.should_speak is True
    assert result.speech == "Hello there."


def test_motion_reaches_tts() -> None:
    result = speech_from_vlm_response(
        json.dumps({
            "decision": "alert",
            "should_speak": True,
            "speech": "Something moved.",
        }),
        "motion",
    )
    assert result.should_speak is True
    assert result.speech == "Something moved."


def test_plain_text_voice_response_is_spoken() -> None:
    result = speech_from_vlm_response("Hello! How can I help?", "voice")
    assert result.decision == "respond"
    assert result.should_speak is True
    assert result.speech == "Hello! How can I help?"


def test_json_encoded_string_is_spoken() -> None:
    result = speech_from_vlm_response('"Hello from the robot."', "voice")
    assert result.should_speak is True
    assert result.speech == "Hello from the robot."


def test_plain_text_motion_response_is_spoken() -> None:
    result = speech_from_vlm_response("A person entered the room.", "motion")
    assert result.should_speak is True
    assert result.speech == "A person entered the room."


def test_should_speak_false_is_overridden() -> None:
    result = speech_from_vlm_response(
        json.dumps({"should_speak": False, "speech": "I can see you."}),
        "voice",
    )
    assert result.should_speak is True
    assert result.speech == "I can see you."
