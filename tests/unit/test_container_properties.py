"""Behavioral tests for the FlextContainer public contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import assume, given, settings, strategies as st

from flext_core import FlextContainer
from tests import c, tm


class TestsFlextCoreContainerProperties:
    """Assert the observable public behavior of ``FlextContainer``.

    Every test drives the container through its published API
    (``factory``/``bind``/``resolve``/``has``/``names``/``drop``/``clear``/
    ``snapshot``/``shared``) and asserts return values and ``r[T]`` outcomes,
    never internal registries or DI wiring.
    """

    @pytest.fixture
    def container(self) -> Iterator[FlextContainer]:
        """Yield the shared container and reset the singleton afterwards.

        Yields:
            Each ``FlextContainer``.

        """
        instance = FlextContainer.shared()
        instance.clear()
        yield instance
        FlextContainer.reset_for_testing()

    # -- enumeration -------------------------------------------------------

    @staticmethod
    def test_names_lists_registered_and_hides_internal(
        container: FlextContainer,
    ) -> None:
        """``names`` exposes user registrations only, not core services."""
        _ = container.bind("svc_a", 1)
        _ = container.factory("svc_b", lambda: 2)

        names = set(container.names())

        tm.that({"svc_a", "svc_b"} <= names, truth=True)
        tm.that("logger" not in names, truth=True)
        tm.that("command_bus" not in names, truth=True)

    # -- invariants --------------------------------------------------------

    @staticmethod
    def test_shared_returns_singleton_instance(container: FlextContainer) -> None:
        """``shared`` returns the one canonical instance."""
        tm.that(FlextContainer.shared() is container, truth=True)

    # -- property-based roundtrip ------------------------------------------

    @staticmethod
    @given(
        name=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        ),
    )
    @settings(max_examples=50)
    def test_register_get_roundtrip_property(name: str) -> None:
        """For any valid name, factory registration then resolution roundtrips."""
        container = FlextContainer.shared()
        sanitized = "".join(ch for ch in name if ch.isalnum()) or "svc"
        assume(sanitized not in c.CONTAINER_RESERVED_NAMES)

        def produce() -> str:
            return sanitized

        _ = container.factory(sanitized, produce)

        tm.ok(container.resolve(sanitized, type_cls=str), eq=sanitized)
        FlextContainer.reset_for_testing()
