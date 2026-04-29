"""Behavior contract for u.scan_module — factory discovery helper."""

from __future__ import annotations

import types

from tests import c, m, u


class TestsFlextDecoratorsDiscovery:
    """Behavior contract for u.scan_module — used by FlextContainer to register factories."""

    def test_scan_module_with_no_factories_returns_empty_list(self) -> None:
        mod = types.ModuleType("empty_mod")
        assert u.scan_module(mod) == []

    def test_scan_module_ignores_functions_without_factory_attribute(self) -> None:
        mod = types.ModuleType("plain_mod")
        mod.__dict__["my_func"] = lambda: None
        assert u.scan_module(mod) == []

    def test_scan_module_skips_private_names_even_with_factory_attribute(self) -> None:
        mod = types.ModuleType("private_mod")

        def _private_factory() -> None:
            msg = "intentionally unused factory body"
            raise NotImplementedError(msg)

        setattr(
            _private_factory,
            c.FACTORY_ATTR,
            m.FactoryDecoratorConfig(name="private"),
        )
        mod.__dict__["_private_factory"] = _private_factory
        assert u.scan_module(mod) == []

    def test_scan_module_returns_factory_decorated_public_functions(self) -> None:
        mod = types.ModuleType("factory_mod")

        def my_factory() -> None:
            msg = "intentionally unused factory body"
            raise NotImplementedError(msg)

        setattr(
            my_factory,
            c.FACTORY_ATTR,
            m.FactoryDecoratorConfig(name="my_factory"),
        )
        mod.__dict__["my_factory"] = my_factory
        result = u.scan_module(mod)
        assert len(result) == 1
        assert result[0][0] == "my_factory"
        assert result[0][1].name == "my_factory"

    def test_scan_module_returns_results_sorted_alphabetically(self) -> None:
        mod = types.ModuleType("multi_mod")

        def zebra_factory() -> None:
            msg = "intentionally unused factory body"
            raise NotImplementedError(msg)

        def alpha_factory() -> None:
            msg = "intentionally unused factory body"
            raise NotImplementedError(msg)

        setattr(zebra_factory, c.FACTORY_ATTR, m.FactoryDecoratorConfig(name="zebra"))
        setattr(alpha_factory, c.FACTORY_ATTR, m.FactoryDecoratorConfig(name="alpha"))
        mod.__dict__["zebra_factory"] = zebra_factory
        mod.__dict__["alpha_factory"] = alpha_factory
        result = u.scan_module(mod)
        assert [entry[0] for entry in result] == ["alpha_factory", "zebra_factory"]

    def test_scan_module_ignores_non_callable_attributes(self) -> None:
        mod = types.ModuleType("noncallable_mod")
        mod.__dict__["some_string"] = "not callable"
        assert u.scan_module(mod) == []
