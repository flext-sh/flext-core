"""Behavioral tests for the slim service bootstrap surface.

Asserts observable public contract only: the ``execute`` result of a concrete
service and the resolved :class:`RuntimeBootstrapOptions` returned by
``resolve_runtime_options`` for every supported source shape. No private
attribute access, no collaborator spying, no internal patching.
"""

from __future__ import annotations

from typing import Annotated, override

import pytest
from flext_tests import r

from flext_core import FlextSettings
from tests.base import s
from tests import m
from tests import p
from tests import t
from tests import u


class TestsFlextCoreServiceBootstrap:
    """Public-contract tests for service execution and option resolution."""

    class RuntimeBootstrapSource(m.ArbitraryTypesModel):
        """Service-like source exposing the runtime bootstrap resolution hook."""

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

        def _runtime_bootstrap_options(self) -> p.RuntimeBootstrapOptions:
            return m.RuntimeBootstrapOptions(
                subproject="source-options", wire_packages=("source-options",)
            )

    class ConcreteTestService(s[bool]):
        """Concrete service whose execute contract yields a successful result."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    # --- Service execution contract ------------------------------------

    def test_execute_returns_successful_result_with_payload(self) -> None:
        result = self.ConcreteTestService().execute()

        assert result.success
        assert result.value is True

    def test_execute_result_unwraps_to_payload(self) -> None:
        result = self.ConcreteTestService().execute()

        assert result.unwrap() is True

    # --- resolve_runtime_options: empty / passthrough ------------------

    def test_resolve_with_no_source_yields_empty_options(self) -> None:
        resolved = u.resolve_runtime_options()

        assert resolved.model_dump(exclude_none=True) == {}

    def test_resolve_returns_supplied_model_unchanged(self) -> None:
        options = m.RuntimeBootstrapOptions(subproject="keep")

        resolved = u.resolve_runtime_options(options)

        assert resolved.subproject == "keep"

    def test_runtime_options_accepts_settings_class_contract(self) -> None:
        """Settings class validation uses its method-only class protocol."""
        options = m.RuntimeBootstrapOptions(settings_type=FlextSettings)

        assert options.settings_type is FlextSettings

    def test_resolve_is_idempotent_for_resolved_model(self) -> None:
        once = u.resolve_runtime_options({"subproject": "demo"})

        twice = u.resolve_runtime_options(once)

        assert twice.model_dump() == once.model_dump()

    # --- resolve_runtime_options: mapping validation -------------------

    @pytest.mark.parametrize(
        ("wire_packages", "expected"),
        [
            (["valid", 3], None),
            (["only-int", 7, 9], None),
            (["a", "b"], ("a", "b")),
            (["x"], ("x",)),
            ([], ()),
        ],
    )
    def test_resolve_mapping_sanitizes_wire_packages_by_element_type(
        self, wire_packages: list[str | int], expected: tuple[str, ...] | None
    ) -> None:
        resolved = u.resolve_runtime_options({
            "subproject": "demo",
            "wire_packages": wire_packages,
        })

        assert resolved.subproject == "demo"
        assert resolved.wire_packages == expected

    # --- resolve_runtime_options: service-like source ------------------

    def test_resolve_uses_source_bootstrap_hook_when_attrs_absent(self) -> None:
        source = self.RuntimeBootstrapSource()

        resolved = u.resolve_runtime_options(source)

        assert resolved.subproject == "source-options"
        assert resolved.wire_packages == ("source-options",)

    def test_resolve_source_attrs_override_bootstrap_hook(self) -> None:
        dispatcher = u.Tests.OkDispatcher()
        source = self.RuntimeBootstrapSource(
            subproject="source-attrs",
            wire_packages=("source-attrs",),
            runtime_dispatcher=dispatcher,
        )

        resolved = u.resolve_runtime_options(source)

        assert resolved.subproject == "source-attrs"
        assert resolved.wire_packages == ("source-attrs",)
        assert resolved.dispatcher is dispatcher

    def test_resolve_keyword_overrides_take_precedence_over_source(self) -> None:
        dispatcher = u.Tests.OkDispatcher()
        source = self.RuntimeBootstrapSource(
            subproject="source-attrs",
            wire_packages=("source-attrs",),
            runtime_dispatcher=dispatcher,
        )

        resolved = u.resolve_runtime_options(
            source, subproject="override", wire_packages=("override",)
        )

        assert resolved.subproject == "override"
        assert list(resolved.wire_packages or ()) == ["override"]
        assert resolved.dispatcher is dispatcher
