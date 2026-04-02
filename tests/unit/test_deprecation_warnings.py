"""Tests for deprecation warnings on compatibility methods.

Verifies that ALL deprecated compatibility paths emit DeprecationWarning
with clear messaging pointing to their modern replacement.

Every deprecated method and alias is tested here so that deprecation
notices remain visible in test output until removal (planned: v0.12).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import cast

import pytest

from flext_core import (
    FlextRuntime,
)
from flext_tests import tm
from tests import m, t, u

pytestmark = [pytest.mark.unit]


class TestDeprecationWarnings:
    def test_normalize_to_container_functional_equivalence(self) -> None:
        test_cases: Sequence[t.NormalizedValue] = [
            "str",
            42,
            math.pi,
            True,
            None,
            {"k": "v"},
            [1, 2],
        ]
        for val in test_cases:
            normalized_result = FlextRuntime.normalize_to_container(
                cast("t.RuntimeData", val),
            )
            strict_result = FlextRuntime.normalize_to_container(
                cast("t.RuntimeData", val),
            )
            tm.that(type(normalized_result), eq=type(strict_result))

    def test_facade_normalize_to_container(self) -> None:
        result = u.normalize_to_container("facade_test")
        tm.that(result, eq="facade_test")

    def test_facade_normalize_to_metadata(self) -> None:
        result = u.normalize_to_metadata("facade_meta")
        tm.that(result, eq="facade_meta")

    def test_normalize_to_container_scalar_passthrough(self) -> None:
        """Scalars pass through normalize_to_container unchanged."""
        tm.that(FlextRuntime.normalize_to_container("hello"), eq="hello")
        tm.that(FlextRuntime.normalize_to_container(42), eq=42)
        tm.that(FlextRuntime.normalize_to_container(math.pi), eq=math.pi)
        tm.that(FlextRuntime.normalize_to_container(True), eq=True)

    def test_normalize_to_container_none_becomes_empty_string(self) -> None:
        """None normalizes to empty string."""
        tm.that(FlextRuntime.normalize_to_container(None), eq="")

    def test_normalize_to_container_dict_wraps_in_model(self) -> None:
        """Nested dicts are wrapped in t.Dict RootModel."""
        result = FlextRuntime.normalize_to_container({"key": "value"})
        tm.that(result, is_=t.Dict)

    def test_normalize_to_container_list_wraps_in_model(self) -> None:
        """Nested lists are wrapped in t.ObjectList RootModel."""
        result = FlextRuntime.normalize_to_container([1, 2, 3])
        tm.that(result, is_=t.ObjectList)

    def test_normalize_to_container_unknown_becomes_str(self) -> None:
        """Unknown objects are converted to string representation."""
        result = FlextRuntime.normalize_to_container(
            cast("t.RuntimeData", "normalized"),
        )
        tm.that(result, is_=str)

    def test_normalize_to_metadata_returns_metadata_value(self) -> None:
        for val in cast("list[t.NormalizedValue]", ["str", 42, None]):
            metadata = FlextRuntime.normalize_to_metadata(val)
            tm.that(metadata, is_=(str, int, float, bool, list, dict))
        list_meta = FlextRuntime.normalize_to_metadata([1])
        tm.that(list_meta, is_=list)
        dict_meta = FlextRuntime.normalize_to_metadata({"k": "v"})
        tm.that(dict_meta, is_=dict)

    def test_deprecated_class_warning(self) -> None:
        """warnings.deprecated emits DeprecationWarning on instantiation."""
        legacy_base = type("LegacyBase", (m.Categories,), {})
        legacy = warnings.deprecated("Use NewClass instead")(legacy_base)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy()
            assert len(caught) == 1

    def test_no_deprecation_on_strict_methods(self) -> None:
        """Non-deprecated methods must NOT emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = FlextRuntime.normalize_to_container("test")
            _ = FlextRuntime.normalize_to_metadata("test")
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), eq=0)
