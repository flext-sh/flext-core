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
from tests.models import m

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextCoreRuntime:
    """Assert the observable behavior callers depend on from ``FlextRuntime``."""

    def test_facade_exposes_stable_public_identity(self) -> None:
        assert flext_core.FlextRuntime is FlextRuntime
        assert FlextRuntime.__module__ == "flext_core.runtime"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2025, 1, 1, tzinfo=UTC), "2025-01-01T00:00:00+00:00"),
            (Path("/a/b"), "/a/b"),
            (None, None),
            (42, 42),
            ("text", "text"),
        ],
    )
    def test_normalize_to_metadata_converts_scalars_to_json_native(
        self, value: t.JsonPayload, expected: t.JsonValue
    ) -> None:
        assert FlextRuntime.normalize_to_metadata(value) == expected

    def test_normalize_to_metadata_flattens_sequence_members(self) -> None:
        normalized = FlextRuntime.normalize_to_metadata([1, Path("/x"), None])

        assert normalized == [1, "/x", None]

    def test_normalize_to_json_value_preserves_none(self) -> None:
        assert FlextRuntime.normalize_to_json_value(None) is None

    def test_normalize_to_json_mapping_normalizes_each_value(self) -> None:
        normalized = FlextRuntime.normalize_to_json_mapping({"a": 1, "b": Path("/z")})

        assert normalized == {"a": 1, "b": "/z"}

    @pytest.mark.parametrize(
        ("value", "expected"), [(None, ""), (42, 42), ([1, 2], [1, 2])]
    )
    def test_normalize_to_container_returns_runtime_data(
        self, value: t.JsonPayload, expected: t.JsonValue
    ) -> None:
        assert FlextRuntime.normalize_to_container(value) == expected

    def test_normalize_to_container_unwraps_config_map_model(self) -> None:
        normalized = FlextRuntime.normalize_to_container(m.ConfigMap(root={"k": 1}))

        assert normalized == {"k": 1}

    def test_normalize_model_input_mapping_preserves_nested_mapping(self) -> None:
        assert FlextRuntime.normalize_model_input_mapping({"x": {"y": 1}}) == {
            "x": {"y": 1}
        }

    def test_normalize_model_input_mapping_accepts_root_model(self) -> None:
        normalized = FlextRuntime.normalize_model_input_mapping(
            m.Dict(root={"a": 1, "b": {"c": 2}})
        )

        assert normalized == {"a": 1, "b": {"c": 2}}

    def test_normalize_model_input_mapping_returns_none_for_none(self) -> None:
        assert FlextRuntime.normalize_model_input_mapping(None) is None

    def test_normalize_metadata_input_mapping_preserves_explicit_none(self) -> None:
        normalized = FlextRuntime.normalize_metadata_input_mapping({
            "alpha": None,
            "beta": 2,
        })

        assert normalized == {"alpha": None, "beta": 2}

    def test_normalize_metadata_input_mapping_reads_model_dump_carrier(self) -> None:
        normalized = FlextRuntime.normalize_metadata_input_mapping(
            m.Dict(root={"a": 1, "b": None})
        )

        assert normalized == {"a": 1, "b": None}

    def test_normalize_metadata_input_mapping_returns_none_for_none(self) -> None:
        assert FlextRuntime.normalize_metadata_input_mapping(None) is None

    def test_normalize_metadata_input_mapping_rejects_non_dict_like_input(self) -> None:
        with pytest.raises(TypeError, match="dict-like"):
            FlextRuntime.normalize_metadata_input_mapping("not-a-mapping")

    def test_validate_metadata_attributes_drops_none_values(self) -> None:
        assert FlextRuntime.validate_metadata_attributes({"a": 1, "b": None}) == {
            "a": 1
        }

    def test_validate_metadata_attributes_rejects_reserved_underscore_keys(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="_x"):
            FlextRuntime.validate_metadata_attributes({"_x": 1})

    def test_validate_metadata_model_input_binds_attributes_into_model(self) -> None:
        model = FlextRuntime.validate_metadata_model_input({"a": 1}, m.Metadata)

        assert isinstance(model, m.Metadata)
        assert model.attributes == {"a": 1}

    def test_validate_metadata_model_input_returns_existing_model_unchanged(
        self,
    ) -> None:
        existing = FlextRuntime.validate_metadata_model_input({"a": 1}, m.Metadata)

        assert (
            FlextRuntime.validate_metadata_model_input(existing, m.Metadata) is existing
        )

    def test_validate_metadata_model_input_yields_empty_attributes_for_none(
        self,
    ) -> None:
        model = FlextRuntime.validate_metadata_model_input(None, m.Metadata)

        assert model.attributes == {}

    def test_validate_callable_input_returns_the_callable(self) -> None:
        def factory() -> int:
            return 1

        assert FlextRuntime.validate_callable_input(factory, "factory") is factory

    def test_validate_callable_input_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            FlextRuntime.validate_callable_input(5, "factory")

    def test_normalize_registerable_service_passes_scalars_through(self) -> None:
        assert FlextRuntime.normalize_registerable_service("hi") == "hi"

    def test_normalize_registerable_service_rejects_unregisterable_value(self) -> None:
        with pytest.raises(ValueError, match="RegisterableService"):
            FlextRuntime.normalize_registerable_service(bytearray(b"unsupported"))

    def test_create_container_exposes_registered_object_provider(self) -> None:
        container = FlextRuntime.DependencyIntegration.create_container(
            services={"alpha": "beta"}
        )

        assert container.alpha() == "beta"

    def test_register_factory_with_cache_yields_singleton_instances(self) -> None:
        container = FlextRuntime.DependencyIntegration.create_container()
        _ = FlextRuntime.DependencyIntegration.register_factory(
            container, "svc", object, cache=True
        )

        assert container.svc() is container.svc()

    def test_register_factory_without_cache_yields_distinct_instances(self) -> None:
        container = FlextRuntime.DependencyIntegration.create_container()
        _ = FlextRuntime.DependencyIntegration.register_factory(
            container, "svc", object, cache=False
        )

        assert container.svc() is not container.svc()

    def test_register_object_rejects_duplicate_provider_name(self) -> None:
        container = FlextRuntime.DependencyIntegration.create_container(
            services={"alpha": "beta"}
        )

        with pytest.raises(ValueError, match="already registered"):
            _ = FlextRuntime.DependencyIntegration.register_object(
                container, "alpha", "other"
            )
