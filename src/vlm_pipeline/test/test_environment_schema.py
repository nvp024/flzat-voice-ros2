import json

import pytest

from vlm_pipeline.environment_schema import (
    EnvironmentSchemaError,
    parse_environment_response,
)


def response(**overrides):
    payload = {
        "schema_version": "environment_memory.v1",
        "scene": "Hotel Lobby",
        "objects": [
            {
                "detection_id": 3,
                "label": "Blue Suitcase",
                "description": "A blue suitcase beside the luggage cart.",
                "attributes": {"Color": "blue", "type": "hard shell"},
                "relationships": ["beside the luggage cart"],
                "useful": True,
                "confidence": 0.88,
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_response_normalizes_labels_and_keeps_detector_identity():
    analysis = parse_environment_response(response(), [3, 4])

    assert analysis.scene == "hotel_lobby"
    assert len(analysis.objects) == 1
    assert analysis.objects[0].detection_id == 3
    assert analysis.objects[0].label == "blue_suitcase"
    assert dict(analysis.objects[0].attributes) == {
        "color": "blue",
        "type": "hard shell",
    }


def test_json_fence_is_allowed_but_extra_fields_and_wrong_ids_are_rejected():
    fenced = f"```json\n{response()}\n```"
    assert parse_environment_response(fenced, [3]).objects[0].detection_id == 3

    payload = json.loads(response())
    payload["objects"][0]["x"] = 5.0
    with pytest.raises(EnvironmentSchemaError, match="unsupported"):
        parse_environment_response(json.dumps(payload), [3])

    payload = json.loads(response())
    payload["objects"][0]["detection_id"] = 99
    with pytest.raises(EnvironmentSchemaError, match="not supplied"):
        parse_environment_response(json.dumps(payload), [3])


def test_duplicates_limits_confidence_and_unsafe_content_are_rejected():
    payload = json.loads(response())
    payload["objects"].append(dict(payload["objects"][0]))
    with pytest.raises(EnvironmentSchemaError, match="at most once"):
        parse_environment_response(json.dumps(payload), [3])

    payload = json.loads(response())
    payload["objects"][0]["confidence"] = 1.1
    with pytest.raises(EnvironmentSchemaError, match="confidence"):
        parse_environment_response(json.dumps(payload), [3])

    payload = json.loads(response())
    payload["objects"][0]["description"] = "Navigate to x=4 and publish /cmd_vel"
    with pytest.raises(EnvironmentSchemaError, match="coordinates"):
        parse_environment_response(json.dumps(payload), [3])
