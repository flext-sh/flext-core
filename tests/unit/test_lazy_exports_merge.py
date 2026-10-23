"""Behavior contract for flext_core.lazy merge/cache/build-map helpers.

All assertions target the OBSERVABLE PUBLIC surface of the lazy helpers:
- ``merge_lazy_imports`` / ``normalize_lazy_imports`` return values.
- The PEP 562 ``__getattr__`` that ``install_lazy_exports`` publishes into
  module globals (the caller-visible attribute-resolution contract).
- ``lazy.build_map`` ordering and alias contract.
- ``lazy.cache_stats`` public ``computed_field`` and ``lazy.reset`` invariant.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

from flext_core.lazy import (
    install_lazy_exports,
    lazy,
    merge_lazy_imports,
    normalize_lazy_imports,
)
from tests import tm, u

if TYPE_CHECKING:
    from pathlib import Path

    from tests import t


def _register_alpha_child(child_package_name: str, alpha_module_name: str) -> None:
    """Register a child package and its alpha module carrying a relative lazy map."""
    child_package = ModuleType(child_package_name)
    child_package.__dict__["_LAZY_IMPORTS"] = {"Alpha": (".alpha", "Alpha")}
    alpha_module = ModuleType(alpha_module_name)

    class Alpha:
        pass

    alpha_module.__dict__["Alpha"] = Alpha
    sys.modules[child_package_name] = child_package
    sys.modules[alpha_module_name] = alpha_module


class TestsFlextCoreLazyExportsMerge:
    """Behavior contract for merge/cache/normalize/build-map helpers."""

    @staticmethod
    @pytest.mark.parametrize(
        ("raw_literal", "accepted"),
        [
            ("{'Alpha': 'pkg.alpha'}", True),
            ("{'Alpha': ('pkg.alpha', 'Alpha')}", True),
            ("{'Alpha': ['pkg.alpha', 'Alpha']}", True),
            ("None", False),
            ("{'Alpha': ('pkg.alpha',)}", False),
            ("{'Alpha': ('pkg.alpha', 'Alpha', 'extra')}", False),
            ("{'Alpha': 123}", False),
            ("{123: 'pkg.alpha'}", False),
        ],
    )
    def test_fresh_normalizer_validates_the_public_lazy_entry_shape(
        tmp_path: Path,
        raw_literal: str,
        *,
        accepted: bool,
    ) -> None:
        """The entry contract accepts strings/pairs and rejects malformed maps."""
        script = (
            "from flext_core.lazy import normalize_lazy_imports\n"
            "try:\n"
            f"    normalized = normalize_lazy_imports('pkg', {raw_literal})\n"
            "except TypeError as exc:\n"
            "    print('rejected', exc.__cause__ is not None)\n"
            "else:\n"
            "    print('accepted', normalized['Alpha'] in "
            "('pkg.alpha', ('pkg.alpha', 'Alpha')))\n"
        )
        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=tmp_path)
        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.outcome.raw_return_code, eq=0, msg=result.value.stderr)
        tm.that(
            result.value.stdout.strip(),
            eq="accepted True" if accepted else "rejected True",
        )
        tm.that(result.value.stderr, eq="")

    @staticmethod
    def test_merge_normalizes_child_relative_targets_to_absolute_paths() -> None:
        """Test merge normalizes child relative targets to absolute paths."""
        child_package_name = "test_lazy_pkg_merge.child"
        alpha_module_name = f"{child_package_name}.alpha"

        _register_alpha_child(child_package_name, alpha_module_name)
        try:
            merged = merge_lazy_imports((child_package_name,), {})

            tm.that(merged["Alpha"], eq=(alpha_module_name, "Alpha"))
        finally:
            sys.modules.pop(child_package_name, None)
            sys.modules.pop(alpha_module_name, None)

    @staticmethod
    def test_merge_normalizes_relative_child_package_paths_against_parent() -> None:
        """Test merge normalizes relative child package paths against parent."""
        parent_package_name = "test_lazy_pkg_parent"
        child_package_name = f"{parent_package_name}.child"
        alpha_module_name = f"{child_package_name}.alpha"

        _register_alpha_child(child_package_name, alpha_module_name)
        try:
            merged = merge_lazy_imports(
                (".child",),
                {},
                module_name=parent_package_name,
            )

            tm.that(merged["Alpha"], eq=(alpha_module_name, "Alpha"))
        finally:
            sys.modules.pop(child_package_name, None)
            sys.modules.pop(alpha_module_name, None)

    @staticmethod
    def test_merge_keeps_local_map_when_child_has_no_lazy_imports() -> None:
        """Non-lazy child packages contribute no lazy entries."""
        parent_package_name = "test_lazy_pkg_without_child_map"
        child_package_name = f"{parent_package_name}.child"
        child_package = ModuleType(child_package_name)
        sys.modules[child_package_name] = child_package
        try:
            merged = merge_lazy_imports(
                (".child",),
                {"Local": (f"{parent_package_name}.local", "Local")},
                module_name=parent_package_name,
            )

            tm.that(merged, eq={"Local": (f"{parent_package_name}.local", "Local")})
        finally:
            sys.modules.pop(child_package_name, None)

    @staticmethod
    def test_merge_relative_child_path_without_module_name_raises_value_error() -> None:
        """A relative child path with no parent module is an invalid request."""
        with pytest.raises(ValueError, match="relative lazy-import paths"):
            merge_lazy_imports((".child",), {})

    @staticmethod
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
        module_path: str,
        raw: t.LazyImportMap,
        expected: t.LazyImportDict,
    ) -> None:
        """Relative targets bind to the module; absolute targets pass through."""
        tm.that(normalize_lazy_imports(module_path, raw), eq=expected)

    @staticmethod
    def test_installed_getattr_resolves_submodule_and_caches_it() -> None:
        """The published ``__getattr__`` returns the live submodule object."""
        package_name = "test_lazy_pkg_state"
        module_name = f"{package_name}.module"
        child_module = ModuleType(module_name)

        lazy.reset()
        sys.modules[package_name] = ModuleType(package_name)
        sys.modules[module_name] = child_module
        try:
            module_globals: t.ModuleGlobals = {}
            install_lazy_exports(
                package_name,
                module_globals,
                {"module": module_name},
                publish_all=False,
            )

            getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
            tm.that(callable(getattr_fn), truth=True)
            tm.that(getattr_fn("module"), where=lambda value: value is child_module)
            tm.that(lazy.cache_stats["module_cache"], gte=1)
        finally:
            sys.modules.pop(package_name, None)
            sys.modules.pop(module_name, None)

    @staticmethod
    def test_reset_clears_all_cache_stats_to_zero() -> None:
        """After a resolution + reset, every diagnostic counter returns to zero."""
        package_name = "test_lazy_pkg_reset"
        module_name = f"{package_name}.module"

        lazy.reset()
        sys.modules[package_name] = ModuleType(package_name)
        sys.modules[module_name] = ModuleType(module_name)
        try:
            module_globals: t.ModuleGlobals = {}
            # second install pass for merge verification
            install_lazy_exports(
                package_name,
                module_globals,
                {"module": module_name},
                publish_all=False,
            )
            getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
            tm.that(callable(getattr_fn), truth=True)
            getattr_fn("module")
            tm.that(lazy.cache_stats["module_cache"], gte=1)

            lazy.reset()

            tm.that(
                lazy.cache_stats,
                eq={
                    "module_cache": 0,
                    "child_lazy_cache": 0,
                    "child_merge_cache": 0,
                    "normalized_map_cache": 0,
                    "install_cache": 0,
                },
            )
        finally:
            sys.modules.pop(package_name, None)
            sys.modules.pop(module_name, None)

    @staticmethod
    def test_build_map_returns_keys_in_alphabetic_order_by_default() -> None:
        """Test build map returns keys in alphabetic order by default."""
        mapping = lazy.build_map({"pkg.mod": ("zeta", "alpha")})

        tm.that(list(mapping), eq=["alpha", "zeta"])

    @staticmethod
    def test_build_map_preserves_insertion_order_when_sort_keys_disabled() -> None:
        """Test build map preserves insertion order when sort keys disabled."""
        mapping = lazy.build_map(
            {"pkg.mod": ("zeta", "alpha")},
            alias_groups={"pkg.alias": (("beta", "Thing"),)},
            sort_keys=False,
        )

        tm.that(list(mapping), eq=["zeta", "alpha", "beta"])
        tm.that(mapping["beta"], eq=("pkg.alias", "Thing"))

    @staticmethod
    def test_build_map_merges_module_groups_and_alias_groups() -> None:
        """Module groups map names to the module; alias groups map to (module, attr)."""
        mapping = lazy.build_map(
            {"pkg.mod": ("alpha",)},
            alias_groups={"pkg.alias": (("beta", "Thing"),)},
        )

        tm.that(mapping, eq={"alpha": "pkg.mod", "beta": ("pkg.alias", "Thing")})
