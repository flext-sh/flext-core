"""Real API tests for flext_core.registry using flext_tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from flext_tests import tb, tm
from hypothesis import given, settings, strategies as st

from flext_core import FlextRegistry, r
from tests.models import m


class _RegistryHandlerCallable:
    message_type = "tests.registry.handler"

    def __call__(self, _message: m.Command) -> m.ConfigMap:
        return m.ConfigMap(root={"handled": "yes"})


_registry_handler = _RegistryHandlerCallable()


class TestAutomatedFlextRegistry:
    def test_create_registry_and_execute(self) -> None:
        registry = FlextRegistry.create()
        tm.that(registry, is_=FlextRegistry)
        tm.ok(registry.execute(), eq=True)

    def test_register_service_with_metadata(self) -> None:
        registry = FlextRegistry.create()
        service = "service_impl"
        metadata = m.ConfigMap(root={"owner": "tests"})
        result = registry.register("sample_service", service, metadata=metadata)
        tm.ok(result, eq=True)

    def test_register_handler(self) -> None:
        registry = FlextRegistry.create()
        details = tm.ok(registry.register_handler(_registry_handler))
        tm.that(details.registration_id, none=False)

    @pytest.mark.parametrize(
        ("category", "name"),
        tb.Tests.Batch.scenarios(("validators", "email"), ("validators", "phone")),
        ids=lambda case: f"{case[0]}-{case[1]}",
    )
    def test_register_get_list_unregister_plugin(
        self, category: str, name: str
    ) -> None:
        registry = FlextRegistry.create()
        plugin = "plugin_impl"
        tm.ok(registry.register_plugin(category, name, plugin), eq=True)
        tm.ok(registry.get_plugin(category, name), none=False)
        listed = tm.ok(registry.list_plugins(category))
        tm.that(listed, has=name)
        tm.ok(registry.unregister_plugin(category, name), eq=True)
        tm.fail(registry.get_plugin(category, name), has="not found")

    @given(
        name=st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=40)
    def test_hypothesis_plugin_roundtrip(self, name: str) -> None:
        registry = FlextRegistry.create()
        category = "validators"
        plugin = "plugin_impl"
        tm.ok(registry.register_plugin(category, name, plugin), eq=True)
        tm.ok(registry.get_plugin(category, name), none=False)

    @pytest.mark.performance
    def test_registry_plugin_benchmark(self, benchmark: Callable[..., object]) -> None:
        def register_and_read() -> r[bool]:
            registry = FlextRegistry.create()
            plugin = "plugin_impl"
            _ = registry.register_plugin("bench", "p1", plugin)
            fetched = registry.get_plugin("bench", "p1")
            if fetched.is_failure:
                return r[bool].fail(fetched.error or "failed")
            return r[bool].ok(value=True)

        tm.ok(register_and_read(), eq=True)
        _ = benchmark(register_and_read)
