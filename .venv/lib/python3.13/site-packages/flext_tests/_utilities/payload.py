"""Shared payload conversion helpers for flext_tests.

Low-level module with no dependency on flext_tests.utilities,
importable by both utilities.py and matchers.py without cycles.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, tzinfo
from enum import Enum
from pathlib import Path

from flext_infra import u
from flext_tests import c, m, p, t


class FlextTestsPayloadUtilities:
    """Namespace class for shared payload conversion helpers in flext_tests."""

    @staticmethod
    def _stable_sort_key(value: object) -> tuple[str, str]:
        """Return a total deterministic key for heterogeneous payload values."""
        return type(value).__name__, str(value)

    @staticmethod
    def to_payload(value: p.AttributeProbe) -> t.Tests.TestobjectSerializable:
        """Recursively flatten any runtime value to ``TestobjectSerializable``."""
        to_p = FlextTestsPayloadUtilities.to_payload
        match value:
            case m.RootModel():
                result = to_p(value.root)
            case Enum():
                result = to_p(value.value)
            case None:
                result = None
            case (
                str()
                | int()
                | float()
                | bool()
                | bytes()
                | datetime()
                | Path()
                | m.BaseModel()
            ):
                result = value
            case Mapping():
                normalized_map = {str(k): to_p(v) for k, v in value.items()}
                try:
                    validated_map = t.Tests.TESTOBJECT_MAPPING_ADAPTER.validate_python(
                        normalized_map
                    )
                except c.ValidationError:
                    result = normalized_map
                else:
                    result = {k: to_p(v) for k, v in validated_map.items()}
            case list() | tuple() | set() | frozenset():
                normalized_seq = [to_p(item) for item in value]
                if isinstance(value, (set, frozenset)):
                    normalized_seq = sorted(
                        normalized_seq, key=FlextTestsPayloadUtilities._stable_sort_key
                    )
                try:
                    validated_seq = t.Tests.TESTOBJECT_SEQUENCE_ADAPTER.validate_python(
                        normalized_seq
                    )
                except c.ValidationError:
                    result = normalized_seq
                else:
                    result = [to_p(item) for item in validated_seq]
            case _:
                result = str(value)
        return result

    @staticmethod
    def to_normalized_value(value: t.Tests.NormalizationInput) -> t.JsonValue:
        """Flatten to pure Container via canonical runtime helper."""
        to_n = FlextTestsPayloadUtilities.to_normalized_value
        match value:
            case m.RootModel():
                result = to_n(FlextTestsPayloadUtilities.to_payload(value.root))
            case m.BaseModel():
                result = str(value)
            case bytes():
                result = value.decode(errors="ignore")
            case type() | tzinfo():
                result = str(value)
            case bool() | datetime() | Path() | None | str() | int() | float():
                result = u.normalize_to_metadata(value)
            case Mapping():
                result = u.normalize_to_metadata({
                    key: to_n(item) for key, item in value.items()
                })
            case list() | tuple():
                normalized_seq = [to_n(item) for item in value]
                result = u.normalize_to_metadata(normalized_seq)
            case _:
                result = str(value)
        return result

    @staticmethod
    def to_config_map(
        value: (
            m.BaseModel
            | t.MappingKV[str, t.Tests.TestobjectSerializable]
            | t.MappingKV[str, t.JsonPayload]
            | t.JsonMapping
        ),
    ) -> m.ConfigMap:
        """Convert a model or payload mapping to the canonical ConfigMap shape."""
        source = (
            value.model_dump(mode="python") if isinstance(value, m.BaseModel) else value
        )
        return m.ConfigMap.model_validate({
            key: (
                payload
                if isinstance(
                    payload := FlextTestsPayloadUtilities.to_payload(item), m.BaseModel
                )
                else FlextTestsPayloadUtilities.to_normalized_value(payload)
            )
            for key, item in source.items()
        })

    @staticmethod
    def deep_match(
        obj: m.BaseModel | t.MappingKV[str, t.Tests.TestobjectSerializable],
        spec: t.Tests.DeepSpec,
        *,
        path_sep: str = ".",
    ) -> m.Tests.DeepMatchResult:
        """Match t.JsonValue against deep specification.

        Uses u.extract() for path extraction.
        Supports unlimited nesting depth via dot notation paths.

        Args:
            obj: Object to match against (dict or Pydantic model)
            spec: DeepSpec mapping of path -> expected value or predicate
            path_sep: Path separator (default: ".")

        Returns:
            DeepMatchResult with match status and details

        """
        source_obj = FlextTestsPayloadUtilities.to_config_map(obj)
        to_payload = FlextTestsPayloadUtilities.to_payload
        object_payload = to_payload(obj)
        for path, expected in spec.items():
            result = u.extract(source_obj, path, separator=path_sep)
            if result.failure:
                return m.Tests.DeepMatchResult(
                    path=path,
                    expected=expected,
                    actual=None,
                    matched=False,
                    reason=f"Path not found: {path}",
                )
            actual = result.value
            actual_payload = to_payload(actual)
            if callable(expected):
                if not expected(actual_payload):
                    return m.Tests.DeepMatchResult(
                        path=path,
                        expected="<predicate>",
                        actual=actual_payload,
                        matched=False,
                        reason="Predicate failed",
                    )
            elif actual != expected:
                return m.Tests.DeepMatchResult(
                    path=path,
                    expected=expected,
                    actual=actual_payload,
                    matched=False,
                    reason="Value mismatch",
                )
        return m.Tests.DeepMatchResult(
            path="",
            expected=object_payload,
            actual=object_payload,
            matched=True,
            reason="",
        )
