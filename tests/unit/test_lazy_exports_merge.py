"""Behavior contract for flext_core.lazy merge/cache helpers."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from flext_core import t
from flext_core.lazy import install_lazy_exports, lazy, merge_lazy_imports


class TestsFlextLazyMerge:
    """Behavior contract for merge/cache/build-map helpers."""

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

    def test_merge_keeps_local_map_when_child_has_no_lazy_imports(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-lazy child packages contribute no lazy entries."""
        parent_package_name = "test_lazy_pkg_without_child_map"
        child_package_name = f"{parent_package_name}.child"
        child_package = ModuleType(child_package_name)
        monkeypatch.setitem(sys.modules, child_package_name, child_package)

        merged = merge_lazy_imports(
            (".child",),
            {"Local": (f"{parent_package_name}.local", "Local")},
            module_name=parent_package_name,
        )

        assert merged == {"Local": (f"{parent_package_name}.local", "Local")}

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

        module_globals: t.ModuleGlobals = {}
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
