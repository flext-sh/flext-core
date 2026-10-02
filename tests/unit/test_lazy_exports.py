"""Behavioral contract for flext_core.lazy — public PEP 562 lazy export surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    lazy,
    lazy_attribute,
)
from tests import u

if TYPE_CHECKING:
    from flext_core import t


class TestsFlextCoreLazyExports:
    """Behavioral contract: what the lazy export surface promises callers."""

    @staticmethod
    @pytest.fixture
    def registered_alpha_module() -> Iterator[tuple[str, type]]:
        """Register a real child module exposing ``Alpha`` and return its name.

        Yields:
            Each ``tuple[str, type]``.

        """
        lazy.reset()
        module_name = "test_lazy_pkg.alpha"
        child = ModuleType(module_name)

        class Alpha:
            pass

        child.__dict__["Alpha"] = Alpha
        sys.modules["test_lazy_pkg"] = ModuleType("test_lazy_pkg")
        sys.modules[module_name] = child
        try:
            yield module_name, Alpha
        finally:
            sys.modules.pop("test_lazy_pkg", None)
            sys.modules.pop(module_name, None)

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
    @staticmethod
    def test_thin_facade_modules_export_facade_and_short_alias(
        module_name: str,
        facade_name: str,
        alias_name: str,
    ) -> None:
        # Arrange / Act
        """Test thin facade modules export facade and short alias."""
        module = importlib.import_module(module_name)

        # Assert — public contract: facade + alias resolve to the same object
        facade = getattr(module, facade_name)
        alias = getattr(module, alias_name)
        assert alias is facade

    @staticmethod
    def test_root_package_resolves_primary_facades_via_aliases() -> None:
        # Arrange / Act
        """Test root package resolves primary facades via aliases."""
        package = importlib.import_module("flext_core")

        # Assert — each short alias resolves to its full facade
        assert package.c is package.FlextConstants
        assert package.e is package.FlextExceptions
        assert package.m is package.FlextModels
        assert package.p is package.FlextProtocols
        assert package.t is package.FlextTypes
        assert package.u is package.FlextUtilities
        assert {"FlextConstants", "FlextUtilities", "u"} <= set(package.__all__)

    @staticmethod
    def test_model_facade_does_not_import_web_runtime() -> None:
        """Loading model declarations must not import the web framework stack."""
        script = (
            "import sys\n"
            "from flext_core import m\n"
            "assert m.StrictModel\n"
            "for name in sorted(sys.modules):\n"
            "    if name == 'fastapi' or name.startswith('fastapi.'):\n"
            "        print(name)\n"
        )

        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=Path.cwd())

        assert result.success, result.error
        assert not result.value.stdout

    @staticmethod
    def test_install_without_publish_all_omits_dunder_all(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install without publish all omits dunder all."""
        module_name, _ = registered_alpha_module
        package_name = module_name.rpartition(".")[0]
        module_globals: t.ModuleGlobals = vars(sys.modules[package_name])

        # Act
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (module_name, "Alpha")},
            publish_all=False,
        )

        # Assert — no __all__ published, but __dir__ still lists the public name
        assert "__all__" not in module_globals
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    @staticmethod
    def test_install_with_publish_all_publishes_dunder_all(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install with publish all publishes dunder all."""
        module_name, _ = registered_alpha_module
        package_name = module_name.rpartition(".")[0]
        module_globals: t.ModuleGlobals = vars(sys.modules[package_name])

        # Act
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (module_name, "Alpha")},
        )

        # Assert
        assert module_globals["__all__"] == ("Alpha",)
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    @staticmethod
    def test_install_with_public_exports_filters_dunder_all(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install with public exports filters dunder all."""
        module_name, alpha_cls = registered_alpha_module
        package_name = module_name.rpartition(".")[0]
        module_globals: t.ModuleGlobals = vars(sys.modules[package_name])

        # Act — private symbol wired but excluded from the published surface
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (module_name, "Alpha"), "InternalAlpha": (module_name, "Alpha")},
            public_exports=("Alpha",),
        )

        # Assert
        assert module_globals["__all__"] == ("Alpha",)
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

        assert sys.modules[package_name].Alpha is alpha_cls

    @staticmethod
    def test_installed_getattr_resolves_absolute_target(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test installed getattr resolves absolute target."""
        module_name, alpha_cls = registered_alpha_module
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_lazy_pkg",
            module_globals,
            {"Alpha": (module_name, "Alpha")},
        )

        # Act
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        resolved = getattr_fn("Alpha")

        # Assert — lazy symbol resolves to the real class and is cached in globals
        assert resolved is alpha_cls
        assert module_globals["Alpha"] is alpha_cls

    @staticmethod
    def test_installed_getattr_resolves_relative_target(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange — relative path resolved against the installing package
        """Test installed getattr resolves relative target."""
        _, alpha_cls = registered_alpha_module
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_lazy_pkg",
            module_globals,
            {"Alpha": (".alpha", "Alpha")},
        )

        # Act
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)

        # Assert
        assert getattr_fn("Alpha") is alpha_cls

    @staticmethod
    def test_installed_getattr_resolves_bare_string_module_entry() -> None:
        # Arrange — a bare-string entry names a module whose same-named attr is used;
        # resolution must succeed without any '<pkg>.alias' child module existing.
        """Test installed getattr resolves bare string module entry."""
        lazy.reset()
        target_name = "test_lazy_alias_target"
        target = ModuleType(target_name)
        target.__dict__["alias"] = "resolved"
        package_name = "test_lazy_alias_pkg"
        package = ModuleType(package_name)
        sys.modules[target_name] = target
        sys.modules[package_name] = package
        try:
            module_globals: t.ModuleGlobals = vars(package)
            install_lazy_exports(
                package_name,
                module_globals,
                {"alias": target_name},
                publish_all=False,
            )

            # Act
            getattr_fn = module_globals["__getattr__"]
            assert callable(getattr_fn)

            # Assert — resolves the attribute, and never required a probed child submodule
            assert getattr_fn("alias") == "resolved"
            assert "test_lazy_alias_pkg.alias" not in sys.modules
        finally:
            sys.modules.pop(target_name, None)
            sys.modules.pop(package_name, None)

    @staticmethod
    def test_installed_getattr_raises_attribute_error_for_unknown_name(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test installed getattr raises attribute error for unknown name."""
        module_name, _ = registered_alpha_module
        package_name = module_name.rpartition(".")[0]
        module_globals: t.ModuleGlobals = vars(sys.modules[package_name])
        install_lazy_exports(
            package_name,
            module_globals,
            {"Alpha": (module_name, "Alpha")},
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)

        # Act / Assert
        with pytest.raises(AttributeError, match="Missing"):
            getattr_fn("Missing")

    @staticmethod
    def test_install_is_idempotent_and_keeps_getattr_working(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install is idempotent and keeps getattr working."""
        module_name, alpha_cls = registered_alpha_module
        module_globals: t.ModuleGlobals = {}
        lazy_map = {"Alpha": (module_name, "Alpha")}

        # Act — installing twice with identical inputs must not break resolution
        install_lazy_exports("test_lazy_pkg", module_globals, lazy_map)
        install_lazy_exports("test_lazy_pkg", module_globals, lazy_map)

        # Assert
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        assert getattr_fn("Alpha") is alpha_cls
        assert module_globals["__all__"] == ("Alpha",)

    @staticmethod
    def test_get_resolves_symbol_and_caches_into_module_globals(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test get resolves symbol and caches into module globals."""
        module_name, alpha_cls = registered_alpha_module
        module_globals: t.ModuleGlobals = {}

        # Act
        resolved = lazy.get(
            "Alpha",
            {"Alpha": (module_name, "Alpha")},
            module_globals,
            "test_lazy_pkg",
        )

        # Assert
        assert resolved is alpha_cls
        assert module_globals["Alpha"] is alpha_cls

    @staticmethod
    @pytest.fixture
    def rebinding_package(tmp_path: Path) -> Iterator[str]:
        """Write a real lazy package whose child rebinds ``u`` mid-body.

        ``pkg/__init__.py`` lazily exports ``u`` from ``pkg.child``; the child
        binds a first class, resolves ``pkg.u`` while its own body is still
        executing (same-thread circular import), then rebinds ``u``.

        Yields:
            Each ``str``.

        """
        lazy.reset()
        package_name = "flext_lazy_rebinding_pkg"
        package_dir = tmp_path / package_name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            "from flext_core.lazy import install_lazy_exports\n"
            "install_lazy_exports(__name__, globals(), {'u': '.child'})\n",
            encoding="utf-8",
        )
        (package_dir / "child.py").write_text(
            "class FirstAlias: ...\n"
            "u = FirstAlias\n"
            f"import {package_name}\n"
            f"RESOLVED_DURING_INIT = {package_name}.u\n"
            f"CACHED_DURING_INIT = 'u' in vars({package_name})\n"
            "class FinalAlias: ...\n"
            "u = FinalAlias\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))
        try:
            yield package_name
        finally:
            sys.path.remove(str(tmp_path))
            for name in [
                name
                for name in sys.modules
                if name == package_name or name.startswith(f"{package_name}.")
            ]:
                sys.modules.pop(name)
            lazy.reset()

    @staticmethod
    def test_get_serves_but_never_caches_symbol_of_initializing_module(
        rebinding_package: str,
    ) -> None:
        # Act — first access imports the child, whose body resolves ``u`` again
        """Test get serves but never caches symbol of initializing module."""
        package = importlib.import_module(rebinding_package)
        first_access = package.u
        child = importlib.import_module(f"{rebinding_package}.child")

        # Assert — the mid-body resolution saw the current binding, uncached;
        # the completed module's final binding is what gets cached.
        assert child.RESOLVED_DURING_INIT is child.FirstAlias
        assert child.CACHED_DURING_INIT is False
        assert first_access is child.FinalAlias
        assert vars(package)["u"] is child.FinalAlias
        assert package.u is child.FinalAlias

    @staticmethod
    def test_get_caches_symbol_of_fully_imported_module(
        rebinding_package: str,
    ) -> None:
        # Arrange — import the child completely before any lazy resolution
        """Test get caches symbol of fully imported module."""
        child = importlib.import_module(f"{rebinding_package}.child")
        package = importlib.import_module(rebinding_package)

        # Act
        resolved = package.u

        # Assert — resolution from a finished module is cached in module globals
        assert resolved is child.FinalAlias
        assert vars(package)["u"] is child.FinalAlias

    @staticmethod
    def test_attribute_resolves_class_namespace_symbol_and_caches_global(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test attribute resolves class namespace symbol and caches global."""
        module_name, alpha_cls = registered_alpha_module
        module_globals: t.ModuleGlobals = {}

        class Namespace:
            Alpha = lazy_attribute(
                "Alpha",
                {"Alpha": (module_name, "Alpha")},
                module_globals,
                "test_lazy_pkg",
                resolved_type=type,
            )

        # Act / Assert
        assert Namespace.Alpha is alpha_cls
        assert module_globals["Alpha"] is alpha_cls

    @staticmethod
    def test_get_raises_attribute_error_for_name_absent_from_map() -> None:
        # Arrange
        """Test get raises attribute error for name absent from map."""
        module_globals: t.ModuleGlobals = {}

        # Act / Assert
        with pytest.raises(AttributeError, match="Missing"):
            lazy.get("Missing", {}, module_globals, "test_pkg")

    @staticmethod
    def test_build_map_produces_flat_sorted_import_map() -> None:
        # Act
        """Test build map produces flat sorted import map."""
        result = build_lazy_import_map(
            {"pkg.mod": ("Beta", "Alpha")},
            alias_groups={"pkg.aliases": (("Zeta", "ZetaImpl"),)},
        )

        # Assert — module-group names map to the module; alias-groups keep target+attr
        assert list(result) == ["Alpha", "Beta", "Zeta"]
        assert result["Alpha"] == "pkg.mod"
        assert result["Zeta"] == ("pkg.aliases", "ZetaImpl")

    @staticmethod
    def test_normalize_map_resolves_relative_paths_against_module() -> None:
        # Act
        """Test normalize map resolves relative paths against module."""
        normalized = lazy.normalize_map(
            "pkg.sub",
            {"Rel": (".child", "Rel"), "Abs": "other.mod"},
        )

        # Assert
        assert normalized["Rel"] == ("pkg.sub.child", "Rel")
        assert normalized["Abs"] == "other.mod"

    @staticmethod
    def test_merge_combines_child_and_local_with_local_precedence() -> None:
        # Arrange — a child package exposing its own _LAZY_IMPORTS
        """Test merge combines child and local with local precedence."""
        lazy.reset()
        child_name = "test_merge_child"
        child = ModuleType(child_name)
        child.__dict__["_LAZY_IMPORTS"] = {
            "ChildOnly": (f"{child_name}.a", "ChildOnly"),
            "Shared": (f"{child_name}.a", "SharedChild"),
        }
        sys.modules[child_name] = child
        try:
            local = {"Shared": ("local.mod", "SharedLocal")}

            # Act
            merged = lazy.merge([child_name], local)
        finally:
            sys.modules.pop(child_name, None)
            lazy.reset()

        # Assert — child entry preserved, local wins on collision
        assert "ChildOnly" in merged
        assert merged["Shared"] == ("local.mod", "SharedLocal")

    @staticmethod
    def test_reset_clears_caches_observed_via_cache_stats(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange — perform an install to populate caches
        """Test reset clears caches observed via cache stats."""
        module_name, _ = registered_alpha_module
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_lazy_pkg",
            module_globals,
            {"Alpha": (module_name, "Alpha")},
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)
        getattr_fn("Alpha")
        assert any(size > 0 for size in lazy.cache_stats.values())

        # Act
        lazy.reset()

        # Assert — all caches empty after reset
        assert all(size == 0 for size in lazy.cache_stats.values())
