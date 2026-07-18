"""Behavioral tests for the FlextContainer public contract."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from flext_tests import tm
from hypothesis import assume, given, settings, strategies as st

from flext_core.container import FlextContainer


class TestsFlextCoreContainerProperties:
    """Assert the observable public behavior of ``FlextContainer``.

    Every test drives the container through its published API
    (``factory``/``bind``/``resolve``/``has``/``names``/``drop``/``clear``/
    ``snapshot``/``shared``) and asserts return values and ``r[T]`` outcomes,
    never internal registries or DI wiring.
    """

    _RESERVED_CONTAINER_ATTRS: frozenset[str] = frozenset({
        "override",
        "overridden",
        "providers",
        "reset_override",
        "reset_last_overriding",
        "set_providers",
        "declarative_parent",
        "settings",
    })

    @pytest.fixture
    def container(self) -> Iterator[FlextContainer]:
        """Yield the shared container and reset the singleton afterwards."""
        instance = FlextContainer.shared()
        instance.clear()
        yield instance
        FlextContainer.reset_for_testing()

    # -- registration + resolution roundtrips ------------------------------

    def test_bind_then_resolve_returns_bound_value(
        self, container: FlextContainer
    ) -> None:
        """A value bound under a name resolves back to that exact value."""
        _ = container.bind("answer", 42)

        tm.ok(container.resolve("answer", type_cls=int), eq=42)

    def test_factory_then_resolve_invokes_factory(
        self, container: FlextContainer
    ) -> None:
        """A registered factory is invoked and its product is resolved."""
        _ = container.factory("greeting", lambda: "hello")

        tm.ok(container.resolve("greeting", type_cls=str), eq="hello")

    def test_resolve_without_type_returns_service(
        self, container: FlextContainer
    ) -> None:
        """Resolving without a type constraint still returns the value."""
        _ = container.bind("plain", "value")

        tm.ok(container.resolve("plain"), eq="value")

    # -- membership + enumeration ------------------------------------------

    def test_has_reflects_registration_state(self, container: FlextContainer) -> None:
        """``has`` is False before registration and True afterwards."""
        tm.that(container.has("late"), eq=False)

        _ = container.bind("late", 1)

        tm.that(container.has("late"), eq=True)

    def test_names_lists_registered_and_hides_internal(
        self, container: FlextContainer
    ) -> None:
        """``names`` exposes user registrations only, not core services."""
        _ = container.bind("svc_a", 1)
        _ = container.factory("svc_b", lambda: 2)

        names = set(container.names())

        assert {"svc_a", "svc_b"} <= names
        tm.that(names, lacks="logger")
        tm.that(names, lacks="command_bus")

    # -- error paths -------------------------------------------------------

    def test_resolve_unknown_name_fails_not_found(
        self, container: FlextContainer
    ) -> None:
        """Resolving an unregistered name yields a not-found failure."""
        tm.fail(container.resolve("ghost"), has="ghost")

    def test_resolve_type_mismatch_fails(self, container: FlextContainer) -> None:
        """Resolving with an incompatible type constraint fails."""
        _ = container.bind("text", "not-an-int")

        tm.fail(container.resolve("text", type_cls=int), has="int")

    def test_drop_unknown_name_fails_not_found(self, container: FlextContainer) -> None:
        """Dropping an unregistered name reports a failure."""
        tm.fail(container.drop("absent"), has="absent")

    # -- removal semantics -------------------------------------------------

    def test_drop_removes_registration(self, container: FlextContainer) -> None:
        """After a successful drop the name is gone and no longer resolves."""
        _ = container.bind("temp", 7)

        tm.ok(container.drop("temp"), eq=True)

        tm.that(container.has("temp"), eq=False)
        tm.fail(container.resolve("temp"), has="temp")

    def test_clear_removes_all_user_registrations(
        self, container: FlextContainer
    ) -> None:
        """``clear`` empties user registrations reported by ``names``."""
        _ = container.bind("a", 1)
        _ = container.factory("b", lambda: 2)

        container.clear()

        tm.that(list(container.names()), eq=[])
        tm.that(container.has("a"), eq=False)

    # -- invariants --------------------------------------------------------

    def test_bind_is_idempotent_first_write_wins(
        self, container: FlextContainer
    ) -> None:
        """Re-binding an existing name keeps the original value."""
        _ = container.bind("dup", 1)
        _ = container.bind("dup", 2)

        tm.ok(container.resolve("dup", type_cls=int), eq=1)

    def test_bind_returns_same_container_for_chaining(
        self, container: FlextContainer
    ) -> None:
        """Mutating operations return the container to support chaining."""
        assert container.bind("x", 1) is container
        assert container.factory("y", lambda: 2) is container

    def test_shared_returns_singleton_instance(self, container: FlextContainer) -> None:
        """``shared`` returns the one canonical instance."""
        assert FlextContainer.shared() is container

    def test_snapshot_exposes_merged_settings_mapping(
        self, container: FlextContainer
    ) -> None:
        """``snapshot`` returns a mapping serializable via its public API."""
        snapshot = container.snapshot()

        tm.that(snapshot.model_dump(), is_=dict)

    # -- property-based roundtrip ------------------------------------------

    @given(
        name=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        )
    )
    @settings(max_examples=50)
    def test_register_get_roundtrip_property(self, name: str) -> None:
        """For any valid name, factory registration then resolution roundtrips."""
        container = FlextContainer.shared()
        sanitized = "".join(ch for ch in name if ch.isalnum()) or "svc"
        assume(sanitized not in self._RESERVED_CONTAINER_ATTRS)

        _ = container.factory(sanitized, lambda value=sanitized: value)

        tm.ok(container.resolve(sanitized, type_cls=str), eq=sanitized)
        FlextContainer.reset_for_testing()
