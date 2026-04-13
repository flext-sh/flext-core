"""Behavior tests for the registry DSL public surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

from hypothesis import given, settings, strategies as st

from tests import c, p, r, t, u


class TestRegistryDsl:
    class NamedHandler:
        """Simple public handler with explicit route metadata."""

        def __init__(self, route_name: str) -> None:
            self.message_type = route_name

        def handle(self, message: p.Routable) -> p.Result[str]:
            route = (
                message.command_type or message.query_type or message.event_type or ""
            )
            return r[str].ok(f"{self.message_type}:{route}")

    class RejectingDispatcher(p.Dispatcher):
        """Dispatcher seam that lets tests observe registry error propagation."""

        error_message: str

        def __init__(self, error_message: str) -> None:
            self.error_message = error_message

        @override
        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> p.Result[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> p.Result[bool]:
            _ = handler
            _ = is_event
            return r[bool].fail(self.error_message)

        @override
        def dispatch(self, message: p.Routable) -> p.Result[t.RuntimeAtomic]:
            _ = message
            return r[t.RuntimeAtomic].fail(self.error_message)

    class ReadyDispatcher(p.Dispatcher):
        """Dispatcher seam that accepts every registration."""

        @override
        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> p.Result[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> p.Result[bool]:
            _ = handler
            _ = is_event
            return r[bool].ok(True)

        @override
        def dispatch(self, message: p.Routable) -> p.Result[t.RuntimeAtomic]:
            _ = message
            return r[t.RuntimeAtomic].ok(True)

    def test_registry_execute_uses_canonical_runtime_dispatcher(self) -> None:
        registry = u.build_registry()
        assert registry.execute().success

    def test_registry_register_handler_exposes_public_registration_details(
        self,
    ) -> None:
        registry = u.build_registry(dispatcher=self.ReadyDispatcher())
        result = registry.register_handler(self.NamedHandler("alpha"))
        assert result.success
        assert result.value is not None
        assert result.value.registration_id != ""
        assert result.value.handler_mode == c.HandlerType.COMMAND
        assert result.value.status == c.CommonStatus.ACTIVE

    def test_registry_register_handler_surfaces_dispatcher_failures(self) -> None:
        registry = u.build_registry(
            dispatcher=self.RejectingDispatcher("dispatcher-fail"),
        )
        result = registry.register_handler(self.NamedHandler("alpha"))
        assert result.failure
        assert "dispatcher-fail" in (result.error or "")

    def test_registry_batch_registration_returns_public_summary(self) -> None:
        registry = u.build_registry(dispatcher=self.ReadyDispatcher())
        result = registry.register_handlers([
            self.NamedHandler("alpha"),
            self.NamedHandler("beta"),
        ])
        assert result.success
        assert len(result.value.registered) == 2
        assert result.value.errors == []

    def test_registry_binding_registration_returns_public_summary(self) -> None:
        registry = u.build_registry(dispatcher=self.ReadyDispatcher())
        result = registry.register_bindings({
            str: self.NamedHandler("alpha"),
            int: self.NamedHandler("beta"),
        })
        assert result.success
        assert len(result.value.registered) == 2
        assert result.value.errors == []

    def test_registry_plugin_roundtrip_uses_only_public_methods(self) -> None:
        registry = u.build_registry(dispatcher=self.ReadyDispatcher())
        assert registry.register_plugin("validators", "local", "plugin").success
        assert registry.fetch_plugin("validators", "local").value == "plugin"
        assert registry.list_plugins("validators").value == ["local"]
        assert registry.unregister_plugin("validators", "local").success
        assert registry.fetch_plugin("validators", "local").failure

    def test_registry_class_scope_is_shared_across_instances(self) -> None:
        registry_a = u.build_registry(dispatcher=self.ReadyDispatcher())
        registry_b = u.build_registry(dispatcher=self.ReadyDispatcher())
        assert registry_a.register_plugin(
            "validators",
            "shared",
            "plugin",
            scope=c.RegistrationScope.CLASS,
        ).success
        assert (
            registry_b.fetch_plugin(
                "validators",
                "shared",
                scope=c.RegistrationScope.CLASS,
            ).value
            == "plugin"
        )
        assert registry_b.unregister_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).success
        assert registry_a.fetch_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).failure

    @given(
        name=st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=40)
    def test_registry_plugin_roundtrip_property(self, name: str) -> None:
        registry = u.build_registry(dispatcher=self.ReadyDispatcher())
        assert registry.register_plugin("validators", name, "plugin").success
        assert registry.fetch_plugin("validators", name).value == "plugin"
