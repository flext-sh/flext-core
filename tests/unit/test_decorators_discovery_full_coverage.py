"""Tests for FactoryDecoratorsDiscovery - scan_module and has_factories.

Module: flext_core._decorators.discovery
Coverage target: lines 50-63, 85 (scan_module body, has_factories body)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types

from flext_core import c, m
from flext_core._decorators.discovery import FactoryDecoratorsDiscovery


class TestFactoryDecoratorsDiscoveryScanModule:
    """Tests for FactoryDecoratorsDiscovery.scan_module()."""

    def test_scan_empty_module_returns_empty_list(self) -> None:
        """Scanning a module with no factory-decorated functions returns []."""
        mod = types.ModuleType("empty_mod")
        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert result == []

    def test_scan_module_with_non_factory_functions(self) -> None:
        """Functions without factory config attribute are ignored."""
        mod = types.ModuleType("plain_mod")
        mod.__dict__["my_func"] = lambda: None
        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert result == []

    def test_scan_module_skips_private_names(self) -> None:
        """Names starting with _ are skipped even if they have factory config."""
        mod = types.ModuleType("private_mod")

        def _private_factory() -> None:
            pass

        config = m.Container.FactoryDecoratorConfig(name="private")
        setattr(_private_factory, c.Discovery.FACTORY_ATTR, config)
        mod.__dict__["_private_factory"] = _private_factory
        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert result == []

    def test_scan_module_finds_factory_decorated_function(self) -> None:
        """Finds functions that have the factory config attribute."""
        mod = types.ModuleType("factory_mod")

        def my_factory() -> None:
            pass

        config = m.Container.FactoryDecoratorConfig(name="my_factory")
        setattr(my_factory, c.Discovery.FACTORY_ATTR, config)
        mod.__dict__["my_factory"] = my_factory

        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert len(result) == 1
        assert result[0][0] == "my_factory"
        assert result[0][1].name == "my_factory"

    def test_scan_module_returns_sorted_by_name(self) -> None:
        """Results are sorted alphabetically by function name."""
        mod = types.ModuleType("multi_mod")

        def zebra_factory() -> None:
            pass

        def alpha_factory() -> None:
            pass

        config_z = m.Container.FactoryDecoratorConfig(name="zebra")
        config_a = m.Container.FactoryDecoratorConfig(name="alpha")
        setattr(zebra_factory, c.Discovery.FACTORY_ATTR, config_z)
        setattr(alpha_factory, c.Discovery.FACTORY_ATTR, config_a)
        mod.__dict__["zebra_factory"] = zebra_factory
        mod.__dict__["alpha_factory"] = alpha_factory

        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert len(result) == 2
        assert result[0][0] == "alpha_factory"
        assert result[1][0] == "zebra_factory"

    def test_scan_module_ignores_non_callable_attributes(self) -> None:
        """Non-callable attributes are skipped even if they have factory attr."""
        mod = types.ModuleType("noncallable_mod")
        mod.__dict__["some_string"] = "not callable"
        result = FactoryDecoratorsDiscovery.scan_module(mod)
        assert result == []


class TestFactoryDecoratorsDiscoveryHasFactories:
    """Tests for FactoryDecoratorsDiscovery.has_factories()."""

    def test_has_factories_returns_false_for_empty_module(self) -> None:
        """Empty module has no factories."""
        mod = types.ModuleType("empty_mod")
        assert FactoryDecoratorsDiscovery.has_factories(mod) is False

    def test_has_factories_returns_false_for_plain_functions(self) -> None:
        """Module with plain functions (no factory attr) returns False."""
        mod = types.ModuleType("plain_mod")
        mod.__dict__["plain"] = lambda: None
        assert FactoryDecoratorsDiscovery.has_factories(mod) is False

    def test_has_factories_returns_true_when_factory_exists(self) -> None:
        """Module with at least one factory-decorated function returns True."""
        mod = types.ModuleType("factory_mod")

        def my_factory() -> None:
            pass

        config = m.Container.FactoryDecoratorConfig(name="my_factory")
        setattr(my_factory, c.Discovery.FACTORY_ATTR, config)
        mod.__dict__["my_factory"] = my_factory
        assert FactoryDecoratorsDiscovery.has_factories(mod) is True

    def test_has_factories_skips_private_names(self) -> None:
        """Private names are skipped by has_factories."""
        mod = types.ModuleType("private_mod")

        def _hidden() -> None:
            pass

        config = m.Container.FactoryDecoratorConfig(name="hidden")
        setattr(_hidden, c.Discovery.FACTORY_ATTR, config)
        mod.__dict__["_hidden"] = _hidden
        assert FactoryDecoratorsDiscovery.has_factories(mod) is False
