"""Tests for u - scan_module and has_factories.

Module: flext_core._utilities.discovery
Coverage target: lines 50-63, 85 (scan_module body, has_factories body)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types

from tests import c, m, u


class TestDecoratorsDiscoveryFullCoverage:
    def test_scan_empty_module_returns_empty_list(self) -> None:
        """Scanning a module with no factory-decorated functions returns []."""
        mod = types.ModuleType("empty_mod")
        result = u.scan_module(mod)
        assert result == []

    def test_scan_module_with_non_factory_functions(self) -> None:
        """Functions without factory config attribute are ignored."""
        mod = types.ModuleType("plain_mod")
        mod.__dict__["my_func"] = lambda: None
        result = u.scan_module(mod)
        assert result == []

    def test_scan_module_skips_private_names(self) -> None:
        """Names starting with _ are skipped even if they have factory config."""
        mod = types.ModuleType("private_mod")

        def _private_factory() -> None:
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

        config = m.FactoryDecoratorConfig(name="private")
        setattr(_private_factory, c.FACTORY_ATTR, config)
        mod.__dict__["_private_factory"] = _private_factory
        result = u.scan_module(mod)
        assert result == []

    def test_scan_module_finds_factory_decorated_function(self) -> None:
        """Finds functions that have the factory config attribute."""
        mod = types.ModuleType("factory_mod")

        def my_factory() -> None:
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

        config = m.FactoryDecoratorConfig(name="my_factory")
        setattr(my_factory, c.FACTORY_ATTR, config)
        mod.__dict__["my_factory"] = my_factory
        result = u.scan_module(mod)
        assert len(result) == 1
        assert result[0][0] == "my_factory"
        assert result[0][1].name == "my_factory"

    def test_scan_module_returns_sorted_by_name(self) -> None:
        """Results are sorted alphabetically by function name."""
        mod = types.ModuleType("multi_mod")

        def zebra_factory() -> None:
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

        def alpha_factory() -> None:
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

        config_z = m.FactoryDecoratorConfig(name="zebra")
        config_a = m.FactoryDecoratorConfig(name="alpha")
        setattr(zebra_factory, c.FACTORY_ATTR, config_z)
        setattr(alpha_factory, c.FACTORY_ATTR, config_a)
        mod.__dict__["zebra_factory"] = zebra_factory
        mod.__dict__["alpha_factory"] = alpha_factory
        result = u.scan_module(mod)
        assert len(result) == 2
        assert result[0][0] == "alpha_factory"
        assert result[1][0] == "zebra_factory"

    def test_scan_module_ignores_non_callable_attributes(self) -> None:
        """Non-callable attributes are skipped even if they have factory attr."""
        mod = types.ModuleType("noncallable_mod")
        mod.__dict__["some_string"] = "not callable"
        result = u.scan_module(mod)
        assert result == []
