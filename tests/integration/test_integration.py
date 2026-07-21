"""Integration tests for FLEXT Core foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer
from flext_core.__version__ import __version__
from flext_tests import r, tm
from tests import p, t, u

pytestmark = [pytest.mark.integration]

_DOUBLED_RESULT = 42
_UNWRAP_VALUE = 7
_UUID_DASH_COUNT = 4
_UUID_TEXT_LENGTH = 36


class TestsFlextCoreIntegration:
    """Behavioral integration tests for flext-core public contracts.

    Exercises the composed public surface (r result type, FlextContainer
    dependency injection, identifier generation, version metadata) through
    observable behavior only - return values, r[T] outcomes, and public
    model state. No private attributes, internals, or collaborator spying.
    """

    @pytest.mark.core
    def test_result_and_container_compose_on_public_surface(
        self, clean_container: p.Container, sample_data: t.JsonMapping
    ) -> None:
        """Bound value round-trips through container.resolve as a success r[T]."""
        # Arrange
        test_value = str(sample_data["string"])

        # Act
        bind_return = clean_container.bind("test_service", test_value)
        resolved = clean_container.resolve("test_service")

        # Assert - bind is fluent, resolve yields the exact stored value
        tm.that(bind_return is clean_container, eq=True)
        tm.that(resolved.success, eq=True)
        tm.that(resolved.value, eq=test_value)

    @pytest.mark.core
    def test_freshly_constructed_container_satisfies_container_protocol(self) -> None:
        """A default FlextContainer honors the public Container protocol."""
        # Act
        container = FlextContainer()

        # Assert
        tm.that(container, is_=p.Container)

    def test_result_ok_exposes_wrapped_value_as_success(self) -> None:
        """r.ok wraps a value and reports success with that value."""
        # Arrange / Act
        result = r[str].ok("payload")

        # Assert
        tm.that(result.success, eq=True)
        tm.that(u.Tests.assert_success(result), eq="payload")

    def test_result_map_transforms_only_success_value(self) -> None:
        """Mapping applies the function to the success value, preserving success."""
        # Arrange
        result = r[int].ok(21)

        # Act
        mapped = result.map(lambda value: value * 2)

        # Assert
        tm.that(mapped.success, eq=True)
        tm.that(mapped.value, eq=_DOUBLED_RESULT)

    def test_result_flat_map_chains_fallible_operations(self) -> None:
        """Chaining sequences a second r-returning step on success via flat_map."""
        # Arrange
        result = r[str].ok("id-1")

        # Act
        chained = result.flat_map(lambda value: r[str].ok(f"resolved_{value}"))

        # Assert
        tm.that(chained.success, eq=True)
        tm.that(chained.value, eq="resolved_id-1")

    def test_result_fail_short_circuits_map_and_flat_map(self) -> None:
        """A failed r propagates its error through map/flat_map untouched."""
        # Arrange
        failure = r[int].fail("boom")

        # Act
        mapped = failure.map(lambda value: value + 1).flat_map(
            lambda value: r[int].ok(value)
        )

        # Assert
        tm.that(mapped.success, eq=False)
        tm.that(mapped.error, eq="boom")

    def test_result_unwrap_or_returns_default_on_failure(self) -> None:
        """Unwrapping yields the value on success and the default on failure."""
        # Assert - success keeps its value, failure falls back to the default
        tm.that(r[int].ok(_UNWRAP_VALUE).unwrap_or(0), eq=_UNWRAP_VALUE)
        tm.that(r[int].fail("missing").unwrap_or(0), eq=0)

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
        tm.that(factory_return is clean_container, eq=True)
        tm.that(resolved.success, eq=True)
        tm.that(resolved.value, eq=expected)

    def test_container_resolve_unknown_name_fails_with_error(
        self, clean_container: p.Container
    ) -> None:
        """Resolving an unregistered name yields a failure carrying the name."""
        # Act
        resolved = clean_container.resolve("does_not_exist")

        # Assert
        tm.that(resolved.success, eq=False)
        tm.that(resolved.error, none=False)
        tm.that(tm.not_none(resolved.error), has="does_not_exist")

    def test_container_bind_is_idempotent_for_existing_name(
        self, clean_container: p.Container
    ) -> None:
        """Re-binding an existing name preserves the first value (no overwrite)."""
        # Arrange
        _ = clean_container.bind("svc", "first")

        # Act
        _ = clean_container.bind("svc", "second")
        resolved = clean_container.resolve("svc")

        # Assert
        tm.that(resolved.success, eq=True)
        tm.that(resolved.value, eq="first")

    def test_generate_produces_unique_uuid_shaped_identifiers(self) -> None:
        """u.generate returns distinct 36-char, 4-dash UUID strings."""
        # Act
        first = u.generate()
        second = u.generate()

        # Assert - shape contract and uniqueness
        for identifier in (first, second):
            tm.that(identifier, is_=str)
            tm.that(len(identifier), eq=_UUID_TEXT_LENGTH)
            tm.that(identifier.count("-"), eq=_UUID_DASH_COUNT)
        tm.that(first, ne=second)

    def test_generated_identifier_round_trips_through_result(self) -> None:
        """A generated identifier is preserved verbatim when wrapped in r.ok."""
        # Arrange
        entity_id = u.generate()

        # Act
        result = r[str].ok(entity_id)

        # Assert
        tm.that(u.Tests.assert_success(result, expected_value=entity_id), eq=entity_id)

    def test_version_metadata_is_dotted_nonempty_string(self) -> None:
        """The exported package version is a non-empty dotted string."""
        # Assert
        tm.that(__version__, empty=False, has=".")
