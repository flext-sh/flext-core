"""Tests for the current slim service bootstrap surface."""

from __future__ import annotations

from typing import Annotated, override

from flext_tests.base import s

from tests import m, p, r, t, u


class TestsFlextServiceBootstrap:
    """Test service initialization and execution patterns."""

    class RuntimeBootstrapSource(m.ArbitraryTypesModel):
        """Minimal service-like source for runtime bootstrap resolution tests."""

        __test__ = False

        subproject: Annotated[
            str | None,
            m.Field(description="Runtime subproject value exposed by the source."),
        ] = None
        wire_packages: Annotated[
            t.StrSequence | None,
            m.Field(description="Runtime package wiring hints exposed by the source."),
        ] = None
        runtime_dispatcher: Annotated[
            p.Dispatcher | None,
            m.Field(description="Runtime dispatcher override exposed by the source."),
        ] = None

        def _runtime_bootstrap_options(self) -> m.RuntimeBootstrapOptions:
            return m.RuntimeBootstrapOptions(
                subproject="source-options",
                wire_packages=("source-options",),
            )

    class ConcreteTestService(s[bool]):
        """Concrete service for constructor/execute tests."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    def test_service_constructor_accepts_runtime_data(self) -> None:
        service = self.ConcreteTestService()
        assert isinstance(service, self.ConcreteTestService)

    def test_service_execute_returns_ok_result(self) -> None:
        result = self.ConcreteTestService().execute()
        assert result.success
        assert result.value is True

    def test_resolve_runtime_options_sanitizes_invalid_wire_packages(self) -> None:
        resolved = u.resolve_runtime_options(
            {
                "subproject": "demo",
                "wire_packages": ["valid", 3],
            },
        )

        assert resolved.subproject == "demo"
        assert resolved.wire_packages is None

    def test_resolve_runtime_options_applies_source_and_override_precedence(
        self,
    ) -> None:
        source = self.RuntimeBootstrapSource(
            subproject="source-attrs",
            wire_packages=("source-attrs",),
            runtime_dispatcher=u.Tests.OkDispatcher(),
        )

        resolved = u.resolve_runtime_options(
            source,
            subproject="override",
            wire_packages=("override",),
        )

        assert resolved.subproject == "override"
        assert list(resolved.wire_packages or ()) == ["override"]
        assert resolved.dispatcher is source.runtime_dispatcher
