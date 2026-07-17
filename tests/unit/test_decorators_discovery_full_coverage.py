"""Behavior contract for u.scan_module — factory discovery helper.

Exercises the PUBLIC contract end-to-end: the ``d.factory`` decorator is the
real producer of factory metadata and ``u.scan_module`` is its consumer. Tests
assert only observable output (the returned ``(name, config)`` pairs and the
public config model state), never how the metadata is stored on the function.
"""

from __future__ import annotations

import types

import pytest
from flext_tests import d, r

from tests import p
from tests import u


class TestsFlextDecoratorsDiscovery:
    """Behavior contract for u.scan_module — used by FlextContainer to register factories."""

    @staticmethod
    def _module(name: str) -> types.ModuleType:
        return types.ModuleType(name)

    def test_scan_module_with_no_factories_returns_empty_list(self) -> None:
        mod = self._module("empty_mod")

        assert u.scan_module(mod) == []

    def test_scan_module_ignores_undecorated_functions(self) -> None:
        mod = self._module("plain_mod")

        def plain() -> p.Result[int]:
            return r[int].ok(1)

        mod.__dict__["plain"] = plain

        assert u.scan_module(mod) == []

    def test_scan_module_ignores_non_callable_attributes(self) -> None:
        mod = self._module("noncallable_mod")
        mod.__dict__["some_string"] = "not callable"
        mod.__dict__["some_number"] = 42

        assert u.scan_module(mod) == []

    def test_scan_module_discovers_decorated_public_function(self) -> None:
        mod = self._module("factory_mod")

        @d.factory("my_service")
        def build_service() -> p.Result[int]:
            return r[int].ok(7)

        mod.__dict__["build_service"] = build_service

        result = u.scan_module(mod)

        assert len(result) == 1
        attr_name, config = result[0]
        assert attr_name == "build_service"
        assert config.name == "my_service"

    def test_scan_module_skips_private_names(self) -> None:
        mod = self._module("private_mod")

        @d.factory("hidden")
        def _private_factory() -> p.Result[int]:
            return r[int].ok(0)

        mod.__dict__["_private_factory"] = _private_factory

        assert u.scan_module(mod) == []

    @pytest.mark.parametrize(
        ("singleton", "lazy"),
        [
            (False, True),
            (True, False),
            (True, True),
            (False, False),
        ],
    )
    def test_scan_module_preserves_config_metadata(
        self,
        *,
        singleton: bool,
        lazy: bool,
    ) -> None:
        mod = self._module("metadata_mod")

        @d.factory("configured", singleton=singleton, lazy=lazy)
        def build() -> p.Result[int]:
            return r[int].ok(1)

        mod.__dict__["build"] = build

        ((_, config),) = u.scan_module(mod)

        assert config.model_dump() == {
            "name": "configured",
            "singleton": singleton,
            "lazy": lazy,
        }

    def test_scan_module_returns_results_sorted_by_attribute_name(self) -> None:
        mod = self._module("multi_mod")

        @d.factory("z")
        def zebra() -> p.Result[int]:
            return r[int].ok(1)

        @d.factory("a")
        def alpha() -> p.Result[int]:
            return r[int].ok(2)

        mod.__dict__["zebra"] = zebra
        mod.__dict__["alpha"] = alpha

        result = u.scan_module(mod)

        assert [name for name, _ in result] == ["alpha", "zebra"]

    def test_scan_module_returns_only_decorated_functions_from_mixed_module(
        self,
    ) -> None:
        mod = self._module("mixed_mod")

        @d.factory("kept")
        def decorated() -> p.Result[int]:
            return r[int].ok(1)

        def undecorated() -> p.Result[int]:
            return r[int].ok(2)

        mod.__dict__["decorated"] = decorated
        mod.__dict__["undecorated"] = undecorated
        mod.__dict__["constant"] = "value"

        result = u.scan_module(mod)

        assert [name for name, _ in result] == ["decorated"]

    def test_scan_module_is_idempotent(self) -> None:
        mod = self._module("idempotent_mod")

        @d.factory("svc")
        def build() -> p.Result[int]:
            return r[int].ok(1)

        mod.__dict__["build"] = build

        first = u.scan_module(mod)
        second = u.scan_module(mod)

        assert first == second
        assert [name for name, _ in first] == ["build"]
