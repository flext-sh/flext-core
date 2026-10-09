"""Behavioral contract tests for the public ``FlextRuntime`` facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import flext_core
from flext_core.runtime import FlextRuntime
from tests import m

if TYPE_CHECKING:
    from tests import t


class TestsFlextCoreRuntime:
    """Assert the observable behavior callers depend on from ``FlextRuntime``."""

    @staticmethod
    def test_facade_exposes_stable_public_identity() -> None:
        """Test facade exposes stable public identity."""
        assert flext_core.FlextRuntime is FlextRuntime

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2025, 1, 1, tzinfo=UTC), "2025-01-01T00:00:00+00:00"),
            (Path("/a/b"), "/a/b"),
            (None, ""),
            (42, 42),
            ("text", "text"),
        ],
    )
    @staticmethod
    def test_normalize_to_metadata_converts_scalars_to_json_native(
        value: t.JsonPayload,
        expected: t.JsonValue,
    ) -> None:
        """Test normalize to metadata converts scalars to json native."""
        assert FlextRuntime.normalize_to_metadata(value) == expected

    @staticmethod
    def test_normalize_to_metadata_flattens_sequence_members() -> None:
        """Test normalize to metadata flattens sequence members."""
        normalized = FlextRuntime.normalize_to_metadata([1, Path("/x"), None])

        assert normalized == [1, "/x", None]

    @staticmethod
    def test_normalize_to_json_value_keeps_none_as_json_null() -> None:
        """``None`` is a JsonValue in its own right, never coerced to a string."""
        assert FlextRuntime.normalize_to_json_value(None) is None

    @staticmethod
    def test_normalize_to_json_mapping_normalizes_each_value() -> None:
        """Test normalize to json mapping normalizes each value."""
        normalized = FlextRuntime.normalize_to_json_mapping({"a": 1, "b": Path("/z")})

        assert normalized == {"a": 1, "b": "/z"}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, ""), (42, 42), ([1, 2], [1, 2])],
    )
    @staticmethod
    def test_normalize_to_container_returns_runtime_data(
        value: t.JsonPayload,
        expected: t.JsonValue,
    ) -> None:
        """Test normalize to container returns runtime data."""
        assert FlextRuntime.normalize_to_container(value) == expected

    @staticmethod
    def test_normalize_to_container_unwraps_config_map_model() -> None:
        """Test normalize to container unwraps config map model."""
        normalized = FlextRuntime.normalize_to_container(m.ConfigMap(root={"k": 1}))

        assert normalized == {"k": 1}

    @staticmethod
    def test_normalize_model_input_mapping_preserves_nested_mapping() -> None:
        """Test normalize model input mapping preserves nested mapping."""
        assert FlextRuntime.normalize_model_input_mapping({"x": {"y": 1}}) == {
            "x": {"y": 1},
        }

    @staticmethod
    def test_normalize_model_input_mapping_accepts_root_model() -> None:
        """Test normalize model input mapping accepts root model."""
        normalized = FlextRuntime.normalize_model_input_mapping(
            m.Dict(root={"a": 1, "b": {"c": 2}}),
        )

        assert normalized == {"a": 1, "b": {"c": 2}}

    @staticmethod
    def test_normalize_model_input_mapping_returns_none_for_none() -> None:
        """Test normalize model input mapping returns none for none."""
        assert FlextRuntime.normalize_model_input_mapping(None) is None

    @staticmethod
    def test_normalize_metadata_input_mapping_preserves_explicit_none() -> None:
        """Test normalize metadata input mapping preserves explicit none."""
        normalized = FlextRuntime.normalize_metadata_input_mapping({
            "alpha": None,
            "beta": 2,
        })

        assert normalized == {"alpha": None, "beta": 2}

    @staticmethod
    def test_normalize_metadata_input_mapping_reads_model_dump_carrier() -> None:
        """Test normalize metadata input mapping reads model dump carrier."""
        normalized = FlextRuntime.normalize_metadata_input_mapping(
            m.Dict(root={"a": 1, "b": None}),
        )

        assert normalized == {"a": 1, "b": None}

    @staticmethod
    def test_normalize_metadata_input_mapping_returns_none_for_none() -> None:
        """Test normalize metadata input mapping returns none for none."""
        assert FlextRuntime.normalize_metadata_input_mapping(None) is None

    @staticmethod
    def test_normalize_metadata_input_mapping_rejects_non_dict_like_input() -> None:
        """Test normalize metadata input mapping rejects non dict like input."""
        with pytest.raises(TypeError, match="dict-like"):
            FlextRuntime.normalize_metadata_input_mapping("not-a-mapping")

    @staticmethod
    def test_validate_metadata_attributes_drops_none_values() -> None:
        """Test validate metadata attributes drops none values."""
        assert FlextRuntime.validate_metadata_attributes({"a": 1, "b": None}) == {
            "a": 1,
        }

    @staticmethod
    def test_validate_metadata_attributes_rejects_reserved_underscore_keys() -> None:
        """Test validate metadata attributes rejects reserved underscore keys."""
        with pytest.raises(ValueError, match="_x"):
            FlextRuntime.validate_metadata_attributes({"_x": 1})

    @staticmethod
    def test_validate_metadata_model_input_binds_attributes_into_model() -> None:
        """Test validate metadata model input binds attributes into model."""
        model = FlextRuntime.validate_metadata_model_input({"a": 1}, m.Metadata)

        assert isinstance(model, m.Metadata)
        assert model.attributes == {"a": 1}

    @staticmethod
    def test_validate_metadata_model_input_returns_existing_model_unchanged() -> None:
        """Test validate metadata model input returns existing model unchanged."""
        existing = FlextRuntime.validate_metadata_model_input({"a": 1}, m.Metadata)

        assert (
            FlextRuntime.validate_metadata_model_input(existing, m.Metadata) is existing
        )

    @staticmethod
    def test_validate_metadata_model_input_yields_empty_attributes_for_none() -> None:
        """Test validate metadata model input yields empty attributes for none."""
        model = FlextRuntime.validate_metadata_model_input(None, m.Metadata)

        assert model.attributes == {}

    @staticmethod
    def test_validate_callable_input_returns_the_callable() -> None:
        """Test validate callable input returns the callable."""

        def factory() -> int:
            return 1

        assert FlextRuntime.validate_callable_input(factory, "factory") is factory

    @staticmethod
    def test_validate_callable_input_rejects_non_callable() -> None:
        """Test validate callable input rejects non callable."""
        with pytest.raises(TypeError, match="must be callable"):
            FlextRuntime.validate_callable_input(5, "factory")

    @staticmethod
    def test_normalize_registerable_service_passes_scalars_through() -> None:
        """Test normalize registerable service passes scalars through."""
        assert FlextRuntime.normalize_registerable_service("hi") == "hi"

    @staticmethod
    def test_normalize_registerable_service_rejects_unregisterable_value() -> None:
        """Test normalize registerable service rejects unregisterable value."""
        with pytest.raises(ValueError, match="RegisterableService"):
            FlextRuntime.normalize_registerable_service(bytearray(b"unsupported"))
