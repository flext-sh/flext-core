"""Behavior contract for flext_core.lazy — module-level lazy export installer."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from flext_core import lazy
from flext_core.lazy import install_lazy_exports, merge_lazy_imports


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
        module_globals: dict[str, object] = {}
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
        module_globals: dict[str, object] = {}
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

        module_globals: dict[str, object] = {}
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (".alpha", "Alpha")},
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("Alpha") is Alpha

    def test_install_with_identical_inputs_reuses_cached_wiring(self) -> None:
        module_globals: dict[str, object] = {}
        lazy.reset()

        lazy_map = {"Alpha": ("test_pkg.alpha", "Alpha")}
        lazy.install("test_pkg", module_globals, lazy_map, publish_all=False)
        first_getattr = module_globals["__getattr__"]
        first_dir = module_globals["__dir__"]

        lazy.install("test_pkg", module_globals, lazy_map, publish_all=False)
        assert module_globals["__getattr__"] is first_getattr
        assert module_globals["__dir__"] is first_dir
        assert lazy.cache_stats["install_cache"] >= 1

    def test_merge_normalizes_child_relative_targets_to_absolute_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        child_package_name = "test_lazy_pkg_merge.child"
        alpha_module_name = f"{child_package_name}.alpha"

        child_package = ModuleType(child_package_name)
        setattr(child_package, "_LAZY_IMPORTS", {"Alpha": (".alpha", "Alpha")})
        alpha_module = ModuleType(alpha_module_name)

        class Alpha:
            pass

        setattr(alpha_module, "Alpha", Alpha)
        monkeypatch.setitem(sys.modules, child_package_name, child_package)
        monkeypatch.setitem(sys.modules, alpha_module_name, alpha_module)

        merged = merge_lazy_imports((child_package_name,), {})
        assert merged["Alpha"] == (alpha_module_name, "Alpha")

    def test_merge_normalizes_relative_child_package_paths_against_parent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parent_package_name = "test_lazy_pkg_parent"
        child_package_name = f"{parent_package_name}.child"
        alpha_module_name = f"{child_package_name}.alpha"

        child_package = ModuleType(child_package_name)
        setattr(child_package, "_LAZY_IMPORTS", {"Alpha": (".alpha", "Alpha")})
        alpha_module = ModuleType(alpha_module_name)

        class Alpha:
            pass

        setattr(alpha_module, "Alpha", Alpha)
        monkeypatch.setitem(sys.modules, child_package_name, child_package)
        monkeypatch.setitem(sys.modules, alpha_module_name, alpha_module)

        merged = merge_lazy_imports(
            (".child",),
            {},
            module_name=parent_package_name,
        )
        assert merged["Alpha"] == (alpha_module_name, "Alpha")

    def test_cache_stats_increment_on_usage_and_reset_to_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        package_name = "test_lazy_pkg_state"
        module_name = f"{package_name}.module"
        child_module = ModuleType(module_name)

        lazy.reset()
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))
        monkeypatch.setitem(sys.modules, module_name, child_module)

        module_globals: dict[str, object] = {}
        install_lazy_exports(
            package_name,
            module_globals,
            {"module": module_name},
            publish_all=False,
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("module") is child_module
        assert lazy.cache_stats["module_cache"] >= 1

        lazy.reset()
        assert lazy.cache_stats == {
            "module_cache": 0,
            "child_lazy_cache": 0,
            "child_merge_cache": 0,
            "normalized_map_cache": 0,
            "install_cache": 0,
        }

    def test_build_map_returns_keys_in_alphabetic_order_by_default(self) -> None:
        mapping = lazy.build_map({"pkg.mod": ("zeta", "alpha")})
        assert list(mapping) == ["alpha", "zeta"]

    def test_build_map_preserves_insertion_order_when_sort_keys_disabled(self) -> None:
        mapping = lazy.build_map(
            {"pkg.mod": ("zeta", "alpha")},
            alias_groups={"pkg.alias": (("beta", "Thing"),)},
            sort_keys=False,
        )
        assert list(mapping) == ["zeta", "alpha", "beta"]
        assert mapping["beta"] == ("pkg.alias", "Thing")
