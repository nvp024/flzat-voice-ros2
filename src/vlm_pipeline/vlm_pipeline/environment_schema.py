from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Iterable


SCHEMA_VERSION = "environment_memory.v1"
MAX_RAW_RESPONSE_CHARS = 32_768


class EnvironmentSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticObjectData:
    detection_id: int
    label: str
    description: str
    attributes: tuple[tuple[str, str], ...]
    relationships: tuple[str, ...]
    useful: bool
    confidence: float


@dataclass(frozen=True)
class EnvironmentAnalysis:
    scene: str
    objects: tuple[SemanticObjectData, ...]


def parse_environment_response(
    raw_response: str,
    supplied_detection_ids: Iterable[int],
) -> EnvironmentAnalysis:
    allowed_ids = {int(value) for value in supplied_detection_ids}
    if not allowed_ids:
        raise EnvironmentSchemaError("at least one supplied detection ID is required")
    if len(allowed_ids) > 8:
        raise EnvironmentSchemaError("at most eight supplied detections are allowed")
    text = raw_response.strip()
    if not text or len(text) > MAX_RAW_RESPONSE_CHARS:
        raise EnvironmentSchemaError("response must contain 1 to 32768 characters")
    text = _unwrap_json_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvironmentSchemaError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise EnvironmentSchemaError("response root must be an object")
    _exact_fields(payload, {"schema_version", "scene", "objects"}, "response")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise EnvironmentSchemaError("schema_version is unsupported")
    scene = _snake_case(payload["scene"], "scene", 64)
    objects = payload["objects"]
    if not isinstance(objects, list) or len(objects) > 8:
        raise EnvironmentSchemaError("objects must be a list with at most eight items")

    parsed = []
    seen_ids = set()
    for index, value in enumerate(objects):
        parsed_object = _parse_object(value, index, allowed_ids)
        if parsed_object.detection_id in seen_ids:
            raise EnvironmentSchemaError("each detection_id may appear at most once")
        seen_ids.add(parsed_object.detection_id)
        parsed.append(parsed_object)
    return EnvironmentAnalysis(scene=scene, objects=tuple(parsed))


def _parse_object(
    value: object, index: int, allowed_ids: set[int]
) -> SemanticObjectData:
    if not isinstance(value, dict):
        raise EnvironmentSchemaError(f"objects[{index}] must be an object")
    _exact_fields(
        value,
        {
            "detection_id",
            "label",
            "description",
            "attributes",
            "relationships",
            "useful",
            "confidence",
        },
        f"objects[{index}]",
    )
    detection_id = value["detection_id"]
    if isinstance(detection_id, bool) or not isinstance(detection_id, int):
        raise EnvironmentSchemaError(f"objects[{index}].detection_id must be an integer")
    if detection_id not in allowed_ids:
        raise EnvironmentSchemaError(
            f"objects[{index}].detection_id was not supplied by the detector"
        )
    label = _snake_case(value["label"], f"objects[{index}].label", 64)
    description = _short_text(
        value["description"], f"objects[{index}].description", 240
    )
    attributes_value = value["attributes"]
    if not isinstance(attributes_value, dict) or len(attributes_value) > 8:
        raise EnvironmentSchemaError(
            f"objects[{index}].attributes must contain at most eight strings"
        )
    attributes = []
    normalized_keys = set()
    for key, item in attributes_value.items():
        normalized_key = _snake_case(key, "attribute key", 64)
        if normalized_key in {
            "x",
            "y",
            "z",
            "position",
            "pose",
            "coordinate",
            "coordinates",
            "latitude",
            "longitude",
        }:
            raise EnvironmentSchemaError("coordinate attributes are not allowed")
        if normalized_key in normalized_keys:
            raise EnvironmentSchemaError("normalized attribute keys must be unique")
        normalized_keys.add(normalized_key)
        attributes.append(
            (normalized_key, _short_text(item, f"attribute {key}", 120))
        )
    relationships_value = value["relationships"]
    if not isinstance(relationships_value, list) or len(relationships_value) > 5:
        raise EnvironmentSchemaError(
            f"objects[{index}].relationships must contain at most five strings"
        )
    relationships = tuple(
        _short_text(item, f"objects[{index}].relationship", 120)
        for item in relationships_value
    )
    useful = value["useful"]
    if not isinstance(useful, bool):
        raise EnvironmentSchemaError(f"objects[{index}].useful must be boolean")
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise EnvironmentSchemaError(
            f"objects[{index}].confidence must be finite and in [0, 1]"
        )
    for semantic_text in (description, *relationships, *(item for _, item in attributes)):
        _reject_unsafe_semantics(semantic_text)
    return SemanticObjectData(
        detection_id=detection_id,
        label=label,
        description=description,
        attributes=tuple(attributes),
        relationships=relationships,
        useful=useful,
        confidence=float(confidence),
    )


def _exact_fields(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unsupported = sorted(actual - expected)
        raise EnvironmentSchemaError(
            f"{name} fields mismatch; missing={missing}, unsupported={unsupported}"
        )


def _unwrap_json_fence(value: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.I)
    return value if match is None else match.group(1).strip()


def _snake_case(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise EnvironmentSchemaError(f"{name} must be a string")
    if "/" in value:
        raise EnvironmentSchemaError(f"{name} must not contain ROS-style names")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized or len(normalized) > maximum:
        raise EnvironmentSchemaError(
            f"{name} must normalize to 1 to {maximum} lowercase snake_case characters"
        )
    return normalized


def _short_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise EnvironmentSchemaError(f"{name} must be a string")
    normalized = re.sub(r"\s+", " ", value.strip())
    if not normalized or len(normalized) > maximum:
        raise EnvironmentSchemaError(f"{name} must contain 1 to {maximum} characters")
    return normalized


def _reject_unsafe_semantics(value: str) -> None:
    lowered = value.casefold()
    forbidden = (
        r"(?<!\w)/[a-z][a-z0-9_/]*\b",
        r"\b(?:cmd_vel|navigate_to_pose|compute_path_to_pose)\b",
        r"\b[xyz]\s*[:=]\s*-?\d",
        r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)?\s*\)",
        r"<(?:script|iframe|object)\b",
        r"```",
        r"^(?:go|move|navigate|drive|turn|execute|run|publish|call)\b",
        r"\b(?:move the robot|turn the robot|run command|execute command)\b",
    )
    if any(re.search(pattern, lowered) for pattern in forbidden):
        raise EnvironmentSchemaError(
            "semantic text contains coordinates, commands, ROS names, or executable content"
        )
