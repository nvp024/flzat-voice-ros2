import json

import pytest

from vlm_pipeline.grounding_schema import (
    GroundingSchemaError,
    parse_grounding_response,
)


def test_minimal_response_is_canonicalized_and_converted_to_pixels() -> None:
    raw = json.dumps(
        {"objects": [{"label": "Floor Fan", "bbox_2d": [200, 100, 600, 800]}]}
    )

    parsed = parse_grounding_response(raw, 640, 480)

    assert parsed.objects[0].label == "fan"
    assert parsed.objects[0].confidence == 0.5
    assert parsed.objects[0].pixel_bbox == (128, 48, 384, 384)


def test_empty_objects_is_valid() -> None:
    assert parse_grounding_response('{"objects": []}', 640, 480).objects == ()


@pytest.mark.parametrize(
    "payload",
    (
        {"objects": [{"label": "fan", "bbox_2d": [1, 2, 3, 4], "x": 1}]},
        {"objects": [{"label": "fan", "bbox_2d": [1, 2, 1, 4]}]},
        {"objects": [{"label": "fan", "bbox_2d": [-1, 2, 3, 4]}]},
        {"objects": [{"label": "", "bbox_2d": [1, 2, 3, 4]}]},
    ),
)
def test_malformed_grounding_is_rejected(payload) -> None:
    with pytest.raises(GroundingSchemaError):
        parse_grounding_response(json.dumps(payload), 640, 480)


def test_same_label_duplicate_is_removed_but_other_class_is_retained() -> None:
    raw = json.dumps(
        {
            "objects": [
                {"label": "sofa", "bbox_2d": [100, 100, 700, 700]},
                {"label": "couch", "bbox_2d": [110, 110, 710, 710]},
                {"label": "table", "bbox_2d": [110, 110, 710, 710]},
            ]
        }
    )

    parsed = parse_grounding_response(raw, 640, 480)

    assert [item.label for item in parsed.objects] == ["sofa", "table"]
