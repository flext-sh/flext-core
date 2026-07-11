"""Integration tests for FLEXT Core foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r

from flext_core import FlextContainer
from flext_core.__version__ import __version__
from tests.protocols import p
from tests.utilities import u

if TYPE_CHECKING:
    from tests.typings import t

pytestmark = [pytest.mark.integration]


class TestsFlextCoreIntegration:
    """Behavioral integration tests for flext-core public contracts.

    Exercises the composed public surface (r result type, FlextContainer
    dependency injection, identifier generation, version metadata) through
    observable behavior only - return values, r[T] outcomes, and public
    model state. No private attributes, internals, or collaborator spying.
    """

    @pytest.mark.core
    def test_result_and_container_compose_on_public_surface(
        self,
        clean_container: p.Container,
        sample_data: t.JsonMapping,
    ) -> None:
        """Bound value round-trips through container.resolve as a success r[T]."""
        # Arrange
        test_value = str(sample_data["string"])

        # Act
        bind_return = clean_container.bind("test_service", test_value)
        resolved = clean_container.resolve("test_service")

        # Assert - bind is fluent, resolve yields the exact stored value
        assert bind_return is clean_container
        assert resolved.success is True
        assert resolved.value == test_value

    @pytest.mark.core
    def test_freshly_constructed_container_satisfies_container_protocol(self) -> None:
        """A default FlextContainer honors the public Container protocol."""
        # Act
        container = FlextContainer()

        # Assert
        assert isinstance(container, p.Container)

    def test_result_ok_exposes_wrapped_value_as_success(self) -> None:
        """r.ok wraps a value and reports success with that value."""
        # Arrange / Act
        result = r[str].ok("payload")

        # Assert
        assert result.success is True
        assert u.Tests.assert_success(result) == "payload"

    def test_result_map_transforms_only_success_value(self) -> None:
        """Mapping applies the function to the success value, preserving success."""
        # Arrange
        result = r[int].ok(21)

        # Act
        mapped = result.map(lambda value: value * 2)

        # Assert
        assert mapped.success is True
        assert mapped.value == 42

    def test_result_flat_map_chains_fallible_operations(self) -> None:
        """Chaining sequences a second r-returning step on success via flat_map."""
        # Arrange
        result = r[str].ok("id-1")

        # Act
        chained = result.flat_map(lambda value: r[str].ok(f"resolved_{value}"))

        # Assert
        assert chained.success is True
        assert chained.value == "resolved_id-1"

    def test_result_fail_short_circuits_map_and_flat_map(self) -> None:
        """A failed r propagates its error through map/flat_map untouched."""
        # Arrange
        failure = r[int].fail("boom")

        # Act
        mapped = failure.map(lambda value: value + 1).flat_map(
            lambda value: r[int].ok(value),
        )

        # Assert
        assert mapped.success is False
        assert mapped.error == "boom"

    def test_result_unwrap_or_returns_default_on_failure(self) -> None:
        """Unwrapping yields the value on success and the default on failure."""
        # Assert - success keeps its value, failure falls back to the default
        assert r[int].ok(7).unwrap_or(0) == 7
        assert r[int].fail("missing").unwrap_or(0) == 0

    def test_container_factory_resolves_computed_value(
        self,
        clean_container: p.Container,
        mock_external_service: u.Tests.FunctionalExternalService,
    ) -> None:
        """A registered factory resolves to the value produced by its callable."""
        # Arrange
        input_data = "container_result"
        expected = f"processed_{input_data}"

        def create_result() -> str:
            processed: p.Result[str] = mock_external_service.process(input_data)
            if not processed.success:
                raise AssertionError(processed.error)
            value: str = processed.value
            return value

        # Act
        factory_return = clean_container.factory("result_factory", create_result)
        resolved = clean_container.resolve("result_factory")

        # Assert - factory is fluent and delivers the computed payload
        assert factory_return is clean_container
        assert resolved.success is True
        assert resolved.value == expected

    def test_container_resolve_unknown_name_fails_with_error(
        self,
        clean_container: p.Container,
    ) -> None:
        """Resolving an unregistered name yields a failure carrying the name."""
        # Act
        resolved = clean_container.resolve("does_not_exist")

        # Assert
        assert resolved.success is False
        assert resolved.error is not None
        assert "does_not_exist" in resolved.error

    def test_container_bind_is_idempotent_for_existing_name(
        self,
        clean_container: p.Container,
    ) -> None:
        """Re-binding an existing name preserves the first value (no overwrite)."""
        # Arrange
        _ = clean_container.bind("svc", "first")

        # Act
        _ = clean_container.bind("svc", "second")
        resolved = clean_container.resolve("svc")

        # Assert
        assert resolved.success is True
        assert resolved.value == "first"

    def test_generate_produces_unique_uuid_shaped_identifiers(self) -> None:
        """u.generate returns distinct 36-char, 4-dash UUID strings."""
        # Act
        first = u.generate()
        second = u.generate()

        # Assert - shape contract and uniqueness
        for identifier in (first, second):
            assert isinstance(identifier, str)
            assert len(identifier) == 36
            assert identifier.count("-") == 4
        assert first != second

    def test_generated_identifier_round_trips_through_result(self) -> None:
        """A generated identifier is preserved verbatim when wrapped in r.ok."""
        # Arrange
        entity_id = u.generate()

        # Act
        result = r[str].ok(entity_id)

        # Assert
        assert u.Tests.assert_success(result, expected_value=entity_id) == entity_id

    def test_version_metadata_is_dotted_nonempty_string(self) -> None:
        """The exported package version is a non-empty dotted string."""
        # Assert
        assert __version__
        assert "." in __version__
