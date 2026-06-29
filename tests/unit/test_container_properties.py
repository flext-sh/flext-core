"""Container property-based tests."""

from __future__ import annotations

from flext_tests import tm
from hypothesis import assume, given, settings, strategies as st

from flext_core.container import FlextContainer


class TestsFlextContainerProperties:
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

    @given(
        name=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        ),
    )
    @settings(max_examples=50)
    def test_register_get_roundtrip_property(self, name: str) -> None:
        """Property: register then get roundtrips for any valid name."""
        container = FlextContainer.shared()
        sanitized = "".join(ch for ch in name if ch.isalnum()) or "svc"
        assume(sanitized not in self._RESERVED_CONTAINER_ATTRS)

        def dynamic_factory() -> str:
            return sanitized

        _ = container.factory(sanitized, dynamic_factory)
        tm.ok(container.resolve(sanitized, type_cls=str), eq=sanitized)
        FlextContainer.reset_for_testing()
