"""Behavior contract for flext_core.lazy merge/cache/build-map helpers.

All assertions target the OBSERVABLE PUBLIC surface of the lazy helpers:
- ``merge_lazy_imports`` / ``normalize_lazy_imports`` return values.
- The PEP 562 ``__getattr__`` that ``install_lazy_exports`` publishes into
  module globals (the caller-visible attribute-resolution contract).
- ``lazy.build_map`` ordering and alias contract.
- ``lazy.cache_stats`` public ``computed_field`` and ``lazy.reset`` invariant.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from flext_core.lazy import (
    install_lazy_exports,
    lazy,
    merge_lazy_imports,
    normalize_lazy_imports,
)

if TYPE_CHECKING:
    from flext_core import t


class TestsFlextCoreLazyExportsMerge:
    """Behavior contract for merge/cache/normalize/build-map helpers."""

    def test_merge_normalizes_child_relative_targets_to_absolute_paths(
        self, monkeypatch: pytest.MonkeyPatch
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
        self, monkeypatch: pytest.MonkeyPatch
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

        merged = merge_lazy_imports((".child",), {}, module_name=parent_package_name)

        assert merged["Alpha"] == (alpha_module_name, "Alpha")

    def test_merge_keeps_local_map_when_child_has_no_lazy_imports(
        self, monkeypatch: pytest.MonkeyPatch
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

    def test_merge_relative_child_path_without_module_name_raises_value_error(
        self,
    ) -> None:
        """A relative child path with no parent module is an invalid request."""
        with pytest.raises(ValueError, match="relative lazy-import paths"):
            merge_lazy_imports((".child",), {})

    @pytest.mark.parametrize(
        ("module_path", "raw", "expected"),
        [
            pytest.param(
                "pkg.parent",
                {"Alpha": (".alpha", "Alpha")},
                {"Alpha": ("pkg.parent.alpha", "Alpha")},
                id="relative-pair-resolves-against-module",
            ),
            pytest.param(
                "pkg.parent",
                {"Alpha": ("other.pkg.mod", "Alpha")},
                {"Alpha": ("other.pkg.mod", "Alpha")},
                id="absolute-pair-unchanged",
            ),
            pytest.param(
                "pkg.parent",
                {"Alpha": ".alpha"},
                {"Alpha": "pkg.parent.alpha"},
                id="relative-string-resolves-against-module",
            ),
            pytest.param(
                "pkg.parent",
                {"Alpha": "other.pkg.mod"},
                {"Alpha": "other.pkg.mod"},
                id="absolute-string-unchanged",
            ),
        ],
    )
    def test_normalize_map_resolves_relative_and_preserves_absolute(
        self,
        module_path: str,
        raw: dict[str, str | tuple[str, str]],
        expected: dict[str, str | tuple[str, str]],
    ) -> None:
        """Relative targets bind to the module; absolute targets pass through."""
        assert normalize_lazy_imports(module_path, raw) == expected

    def test_installed_getattr_resolves_submodule_and_caches_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The published ``__getattr__`` returns the live submodule object."""
        package_name = "test_lazy_pkg_state"
        module_name = f"{package_name}.module"
        child_module = ModuleType(module_name)

        lazy.reset()
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))
        monkeypatch.setitem(sys.modules, module_name, child_module)

        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            package_name, module_globals, {"module": module_name}, publish_all=False
        )

        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("module") is child_module
        assert lazy.cache_stats["module_cache"] >= 1

    def test_reset_clears_all_cache_stats_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a resolution + reset, every diagnostic counter returns to zero."""
        package_name = "test_lazy_pkg_reset"
        module_name = f"{package_name}.module"

        lazy.reset()
        monkeypatch.setitem(sys.modules, package_name, ModuleType(package_name))
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            package_name, module_globals, {"module": module_name}, publish_all=False
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        getattr_fn("module")
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

    def test_build_map_merges_module_groups_and_alias_groups(self) -> None:
        """Module groups map names to the module; alias groups map to (module, attr)."""
        mapping = lazy.build_map(
            {"pkg.mod": ("alpha",)}, alias_groups={"pkg.alias": (("beta", "Thing"),)}
        )

        assert mapping == {"alpha": "pkg.mod", "beta": ("pkg.alias", "Thing")}
