"""Behavior contract for flext_core.lazy — module-level lazy export installer."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from flext_core.lazy import install_lazy_exports, lazy
from flext_core.typings import t


class TestsFlextLazy:
    """Behavior contract for install_lazy_exports, merge_lazy_imports, lazy runtime."""

    @pytest.mark.parametrize(
        ("module_name", "facade_name", "alias_name"),
        [
            ("flext_core.constants", "FlextConstants", "c"),
            ("flext_core.exceptions", "FlextExceptions", "e"),
            ("flext_core.models", "FlextModels", "m"),
            ("flext_core.protocols", "FlextProtocols", "p"),
            ("flext_core.typings", "FlextTypes", "t"),
            ("flext_core.utilities", "FlextUtilities", "u"),
        ],
    )
    def test_thin_facade_modules_export_facade_and_alias(
        self,
        module_name: str,
        facade_name: str,
        alias_name: str,
    ) -> None:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)

        facade = getattr(module, facade_name)
        alias = getattr(module, alias_name)

        assert alias is facade
        assert facade_name in module.__all__
        assert alias_name in module.__all__

    def test_root_package_lazy_exports_resolve_primary_facades(self) -> None:
        package = importlib.import_module("flext_core")
        package = importlib.reload(package)

        assert package.c is package.FlextConstants
        assert package.e is package.FlextExceptions
        assert package.m is package.FlextModels
        assert package.p is package.FlextProtocols
        assert package.t is package.FlextTypes
        assert package.u is package.FlextUtilities
        assert "FlextConstants" in package.__all__
        assert "FlextUtilities" in package.__all__
        assert "u" in package.__all__

    def test_install_without_publish_all_omits_dunder_all_attribute(self) -> None:
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_pkg.transformers",
            module_globals,
            {"Alpha": ("test_pkg.transformers.alpha", "Alpha")},
            publish_all=False,
        )
        assert "__all__" not in module_globals
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    def test_install_with_publish_all_populates_dunder_all(self) -> None:
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_pkg",
            module_globals,
            {"Alpha": ("test_pkg.alpha", "Alpha")},
        )
        assert module_globals["__all__"] == ("Alpha",)
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    def test_install_resolves_relative_lazy_targets_against_installing_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        package_name = "test_lazy_pkg_relative"
        alpha_module_name = f"{package_name}.alpha"
        alpha_module = ModuleType(alpha_module_name)

        class Alpha:
            pass

        setattr(alpha_module, "Alpha", Alpha)
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))
        monkeypatch.setitem(sys.modules, alpha_module_name, alpha_module)

        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (".alpha", "Alpha")},
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("Alpha") is Alpha

    def test_string_alias_does_not_probe_child_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target_module_name = "test_lazy_pkg_alias_target"
        probed_child_name = "test_lazy_pkg_alias.alias"
        target_module = ModuleType(target_module_name)
        setattr(target_module, "alias", "resolved")

        def import_module(module_name: str) -> ModuleType:
            assert module_name != probed_child_name
            if module_name == target_module_name:
                return target_module
            raise ModuleNotFoundError(module_name)

        lazy.reset()
        monkeypatch.setattr(lazy, "_import_module", import_module)

        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_lazy_pkg_alias",
            module_globals,
            {"alias": target_module_name},
            publish_all=False,
        )

        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("alias") == "resolved"

    def test_install_with_identical_inputs_reuses_cached_wiring(self) -> None:
        module_globals: t.ModuleGlobals = {}
        lazy.reset()

        lazy_map = {"Alpha": ("test_pkg.alpha", "Alpha")}
        lazy.install("test_pkg", module_globals, lazy_map, publish_all=False)
        first_getattr = module_globals["__getattr__"]
        first_dir = module_globals["__dir__"]

        lazy.install("test_pkg", module_globals, lazy_map, publish_all=False)
        assert module_globals["__getattr__"] is first_getattr
        assert module_globals["__dir__"] is first_dir
        assert lazy.cache_stats["install_cache"] >= 1
