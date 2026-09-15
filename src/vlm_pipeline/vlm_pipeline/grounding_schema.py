"""Validate minimal VLM grounding JSON and convert normalized boxes to pixels."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping


PROMPT_VERSION = "environment_grounding_qwen3_v5"
MAX_OBJECTS = 20
MAX_RAW_RESPONSE_CHARS = 32_768
UNCALIBRATED_MANAGER_CONFIDENCE = 0.50
DEFAULT_NMS_IOU_THRESHOLD = 0.50
DEFAULT_LABEL_ALIASES = {
    "floor_fan": "fan",
    "standing_fan": "fan",
    "storage_chest": "chest",
    "woven_basket": "basket",
    "potted_plant": "plant",
    "floor_plant": "plant",
    "television_set": "television",
    "tv": "television",
    "couch": "sofa",
}


class GroundingSchemaError(ValueError):
    """A VLM response cannot safely enter depth/TF post-processing."""


@dataclass(frozen=True)
class GroundedObject:
    """One manager-validated object in normalized and pixel coordinates."""

    label: str
    confidence: float
    normalized_bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class GroundingResponse:
    """Validated detections for the exact source image dimensions."""

    image_width: int
    image_height: int
    objects: tuple[GroundedObject, ...]


def parse_grounding_response(
    raw_response: str,
    image_width: int,
    image_height: int,
    *,
    label_aliases: Mapping[str, str] | None = None,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
) -> GroundingResponse:
    """Parse the v4 minimal schema without silently repairing model output."""
    width = _positive_int(image_width, "image_width")
    height = _positive_int(image_height, "image_height")
    threshold = _unit_interval(nms_iou_threshold, "nms_iou_threshold")
    text = raw_response.strip()
    if not text or len(text) > MAX_RAW_RESPONSE_CHARS:
        raise GroundingSchemaError(
            f"response must contain 1 to {MAX_RAW_RESPONSE_CHARS} characters"
        )
    text = _unwrap_json_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroundingSchemaError(
            f"response is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise GroundingSchemaError("response root must be an object")
    _exact_fields(payload, {"objects"}, "response")
    values = payload["objects"]
    if not isinstance(values, list) or len(values) > MAX_OBJECTS:
        raise GroundingSchemaError(
            f"objects must be a list with at most {MAX_OBJECTS} entries"
        )

    aliases = _normalized_aliases(label_aliases)
    parsed = [
        _parse_object(value, index, width, height, aliases)
        for index, value in enumerate(values)
    ]
    return GroundingResponse(
        image_width=width,
        image_height=height,
        objects=_suppress_duplicates(parsed, threshold),
    )


def canonicalize_label(
    value: object,
    label_aliases: Mapping[str, str] | None = None,
) -> str:
    """Normalize an open-vocabulary label and apply stable aliases."""
    normalized = _snake_case(value, "label")
    return _normalized_aliases(label_aliases).get(normalized, normalized)


def _parse_object(
    value: object,
    index: int,
    width: int,
    height: int,
    aliases: Mapping[str, str],
) -> GroundedObject:
    name = f"objects[{index}]"
    if not isinstance(value, dict):
        raise GroundingSchemaError(f"{name} must be an object")
    _exact_fields(value, {"label", "bbox_2d"}, name)
    raw_label = _snake_case(value["label"], f"{name}.label")
    label = aliases.get(raw_label, raw_label)
    raw_bbox = value["bbox_2d"]
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise GroundingSchemaError(f"{name}.bbox_2d must contain four numbers")
    normalized = tuple(
        _finite_number(item, f"{name}.bbox_2d") for item in raw_bbox
    )
    if not all(0.0 <= item <= 1000.0 for item in normalized):
        raise GroundingSchemaError(
            f"{name}.bbox_2d must be normalized to [0, 1000]"
        )
    x_min, y_min, x_max, y_max = normalized
    if x_max <= x_min or y_max <= y_min:
        raise GroundingSchemaError(f"{name}.bbox_2d has no positive area")
    pixels = (
        int(math.floor(x_min * width / 1000.0)),
        int(math.floor(y_min * height / 1000.0)),
        int(math.ceil(x_max * width / 1000.0)),
        int(math.ceil(y_max * height / 1000.0)),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise GroundingSchemaError(
            f"{name}.bbox_2d collapses after pixel conversion"
        )
    return GroundedObject(
        label=label,
        confidence=UNCALIBRATED_MANAGER_CONFIDENCE,
        normalized_bbox=normalized,
        pixel_bbox=pixels,
    )


def _suppress_duplicates(
    objects: list[GroundedObject], threshold: float
) -> tuple[GroundedObject, ...]:
    kept: list[GroundedObject] = []
    for candidate in objects:
        duplicate = any(
            candidate.label == existing.label
            and _bbox_iou(candidate.pixel_bbox, existing.pixel_bbox) >= threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return tuple(kept)


def _bbox_iou(left, right) -> float:
    intersection_width = max(
        0.0, min(left[2], right[2]) - max(left[0], right[0])
    )
    intersection_height = max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _normalized_aliases(
    label_aliases: Mapping[str, str] | None,
) -> dict[str, str]:
    aliases = dict(DEFAULT_LABEL_ALIASES)
    if label_aliases is not None:
        for source, target in label_aliases.items():
            aliases[_snake_case(source, "alias source")] = _snake_case(
                target, "alias target"
            )
    return aliases


def _exact_fields(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GroundingSchemaError(
            f"{name} fields mismatch; missing={sorted(expected - actual)}, "
            f"unsupported={sorted(actual - expected)}"
        )


def _unwrap_json_fence(value: str) -> str:
    match = re.fullmatch(
        r"\x60\x60\x60(?:json)?\s*(.*?)\s*\x60\x60\x60",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return value if match is None else match.group(1).strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GroundingSchemaError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundingSchemaError(f"{name} must contain numbers")
    result = float(value)
    if not math.isfinite(result):
        raise GroundingSchemaError(f"{name} must contain finite numbers")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise GroundingSchemaError(f"{name} must be in [0, 1]")
    return result


def _snake_case(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise GroundingSchemaError(f"{name} must be a string")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized or len(normalized) > 64:
        raise GroundingSchemaError(
            f"{name} must normalize to 1-64 snake_case characters"
        )
    return normalized
