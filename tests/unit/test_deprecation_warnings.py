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
from typing import cast

import pytest
from flext_tests import t as test_t, tm

from flext_core import FlextRuntime, FlextUtilities
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core.runtime import RuntimeData
from tests.models import m

pytestmark = [pytest.mark.unit]


class TestRuntimeDeprecatedNormalizeMethods:
    """Deprecated runtime normalization methods must emit DeprecationWarning."""

    def test_normalize_to_general_value_emits_deprecation(self) -> None:
        """FlextRuntime.normalize_to_general_value -> normalize_to_container."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextRuntime.normalize_to_general_value("hello")
            tm.that(result, eq="hello")
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)
            tm.that(str(deprecation_warnings[0].message), has="normalize_to_container")

    def test_normalize_to_metadata_value_emits_deprecation(self) -> None:
        """FlextRuntime.normalize_to_metadata_value -> normalize_to_metadata."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextRuntime.normalize_to_metadata_value(42)
            tm.that(result, eq=42)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)
            tm.that(str(deprecation_warnings[0].message), has="normalize_to_metadata")

    def test_normalize_to_general_value_functional_equivalence(self) -> None:
        """Deprecated path must return same result as non-deprecated path."""
        test_cases: list[test_t.Tests.object] = [
            "str",
            42,
            math.pi,
            True,
            None,
            {"k": "v"},
            [1, 2],
        ]
        for val in test_cases:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                deprecated_result = FlextRuntime.normalize_to_general_value(
                    cast("RuntimeData", val),
                )
            strict_result = FlextRuntime.normalize_to_container(
                cast("RuntimeData", val),
            )
            tm.that(type(deprecated_result), eq=type(strict_result))


class TestGuardsDeprecatedMethods:
    """Deprecated guard methods must emit DeprecationWarning."""

    def test_is_general_value_type_emits_deprecation(self) -> None:
        """FlextUtilitiesGuards.is_general_value_type -> is_container."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilitiesGuards.is_general_value_type("test")
            tm.that(result, eq=True)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)
            tm.that(str(deprecation_warnings[0].message), has="is_container")


class TestMapperDeprecatedMethods:
    """Deprecated mapper methods must emit DeprecationWarning."""

    def test_narrow_to_general_value_type_emits_deprecation(self) -> None:
        """FlextUtilitiesMapper.narrow_to_general_value_type -> narrow_to_container."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilitiesMapper.narrow_to_general_value_type("hello")
            tm.that(result, eq="hello")
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)
            tm.that(str(deprecation_warnings[0].message), has="narrow_to_container")

    def test_to_general_value_from_object_emits_deprecation(self) -> None:
        """FlextUtilitiesMapper._to_general_value_from_object -> narrow_to_container."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilitiesMapper._to_general_value_from_object(99)
            tm.that(result, eq=99)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)
            tm.that(str(deprecation_warnings[0].message), has="narrow_to_container")


class TestFacadeDeprecatedAliases:
    """Deprecated facade aliases (u.*) must emit DeprecationWarning via runtime."""

    def test_facade_normalize_to_general_value_emits_deprecation(self) -> None:
        """FlextUtilities.normalize_to_general_value -> normalize_to_container."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilities.normalize_to_general_value("facade_test")
            tm.that(result, eq="facade_test")
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)

    def test_facade_normalize_to_metadata_value_emits_deprecation(self) -> None:
        """FlextUtilities.normalize_to_metadata_value -> normalize_to_metadata."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilities.normalize_to_metadata_value("facade_meta")
            tm.that(result, eq="facade_meta")
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            tm.that(len(deprecation_warnings), gt=0)


class TestStrictContainerNormalization:
    """Non-deprecated normalization methods work correctly with strict types."""

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
        """Nested dicts are wrapped in m.Dict RootModel."""
        result = FlextRuntime.normalize_to_container({"key": "value"})
        tm.that(isinstance(result, m.Dict), eq=True)

    def test_normalize_to_container_list_wraps_in_model(self) -> None:
        """Nested lists are wrapped in m.ObjectList RootModel."""
        result = FlextRuntime.normalize_to_container([1, 2, 3])
        tm.that(isinstance(result, m.ObjectList), eq=True)

    def test_normalize_to_container_unknown_becomes_str(self) -> None:
        """Unknown objects are converted to string representation."""
        result = FlextRuntime.normalize_to_container(
            cast("RuntimeData", object()),
        )
        tm.that(isinstance(result, str), eq=True)

    def test_normalize_to_metadata_returns_metadata_value(self) -> None:
        for val in ["str", 42, None]:
            metadata = FlextRuntime.normalize_to_metadata(val)
            tm.that(isinstance(metadata, (str, int, float, bool, list, dict)), eq=True)
        list_meta = FlextRuntime.normalize_to_metadata([1])
        tm.that(isinstance(list_meta, list), eq=True)
        dict_meta = FlextRuntime.normalize_to_metadata({"k": "v"})
        tm.that(isinstance(dict_meta, dict), eq=True)

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
