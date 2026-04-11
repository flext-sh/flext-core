"""Tests for module-level lazy export installation behavior."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from flext_core import install_lazy_exports, merge_lazy_imports
from flext_core.lazy import lazy


class TestInstallLazyExports:
    """Verify __all__ publication stays root-only when requested."""

    def test_publish_all_disabled_omits_all(self) -> None:
        """Subpackages keep __dir__ but do not publish __all__."""
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

    def test_publish_all_enabled_keeps_all(self) -> None:
        """Root packages still publish __all__ by default."""
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

    def test_relative_lazy_targets_resolve(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Relative lazy targets are normalized against the installing module."""
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

    def test_repeated_install_reuses_cached_wiring(self) -> None:
        """Second install with same inputs reuses previously installed handlers."""
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


class TestMergeLazyImports:
    """Verify merged child maps normalize relative lazy targets."""

    def test_child_relative_targets_are_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Merged child lazy maps expand relative targets to absolute module paths."""
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

    def test_relative_child_package_paths_are_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Relative child package paths expand against the parent module name."""
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

        merged = merge_lazy_imports((".child",), {}, module_name=parent_package_name)
        assert merged["Alpha"] == (alpha_module_name, "Alpha")


class TestLazyRuntimeState:
    """Verify cache diagnostics and reset behavior."""

    def test_cache_stats_and_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache counters increase on usage and return to zero after reset."""
        package_name = "test_lazy_pkg_state"
        module_name = f"{package_name}.module"
        module_globals: dict[str, object] = {}
        child_module = ModuleType(module_name)

        lazy.reset()
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))
        monkeypatch.setitem(sys.modules, module_name, child_module)

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


class TestBuildLazyImportMap:
    """Verify build_map behavior stays deterministic and explicit."""

    def test_build_map_sorts_keys_by_default(self) -> None:
        """Default behavior returns a stable alphabetic key order."""
        mapping = lazy.build_map({"pkg.mod": ("zeta", "alpha")})
        assert list(mapping) == ["alpha", "zeta"]

    def test_build_map_keeps_insertion_order_when_disabled(self) -> None:
        """Disabling sorting preserves insertion order from inputs."""
        mapping = lazy.build_map(
            {"pkg.mod": ("zeta", "alpha")},
            alias_groups={"pkg.alias": (("beta", "Thing"),)},
            sort_keys=False,
        )
        assert list(mapping) == ["zeta", "alpha", "beta"]
        assert mapping["beta"] == ("pkg.alias", "Thing")
