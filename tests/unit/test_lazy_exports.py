"""Behavioral contract for flext_core.lazy — public PEP 562 lazy export surface."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from flext_core.lazy import build_lazy_import_map, install_lazy_exports, lazy

if TYPE_CHECKING:
    from flext_core import t


class TestsFlextCoreLazyExports:
    """Behavioral contract: what the lazy export surface promises callers."""

    @pytest.fixture
    def registered_alpha_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[str, type]:
        """Register a real child module exposing ``Alpha`` and return its name."""
        lazy.reset()
        module_name = "test_lazy_pkg.alpha"
        child = ModuleType(module_name)

        class Alpha:
            pass

        setattr(child, "Alpha", Alpha)
        monkeypatch.setitem(sys.modules, "test_lazy_pkg", ModuleType("test_lazy_pkg"))
        monkeypatch.setitem(sys.modules, module_name, child)
        return module_name, Alpha

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
    def test_thin_facade_modules_export_facade_and_short_alias(
        self,
        module_name: str,
        facade_name: str,
        alias_name: str,
    ) -> None:
        # Arrange / Act
        module = importlib.import_module(module_name)

        # Assert — public contract: facade + alias resolve to the same object
        facade = getattr(module, facade_name)
        alias = getattr(module, alias_name)
        assert alias is facade
        assert facade_name in module.__all__
        assert alias_name in module.__all__

    def test_root_package_resolves_primary_facades_via_aliases(self) -> None:
        # Arrange / Act
        package = importlib.import_module("flext_core")

        # Assert — each short alias resolves to its full facade
        assert package.c is package.FlextConstants
        assert package.e is package.FlextExceptions
        assert package.m is package.FlextModels
        assert package.p is package.FlextProtocols
        assert package.t is package.FlextTypes
        assert package.u is package.FlextUtilities
        assert {"FlextConstants", "FlextUtilities", "u"} <= set(package.__all__)

    def test_install_without_publish_all_omits_dunder_all(self) -> None:
        # Arrange
        module_globals: t.ModuleGlobals = {}

        # Act
        install_lazy_exports(
            "test_pkg.transformers",
            module_globals,
            {"Alpha": ("test_pkg.transformers.alpha", "Alpha")},
            publish_all=False,
        )

        # Assert — no __all__ published, but __dir__ still lists the public name
        assert "__all__" not in module_globals
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    def test_install_with_publish_all_publishes_dunder_all(self) -> None:
        # Arrange
        module_globals: t.ModuleGlobals = {}

        # Act
        install_lazy_exports(
            "test_pkg",
            module_globals,
            {"Alpha": ("test_pkg.alpha", "Alpha")},
        )

        # Assert
        assert module_globals["__all__"] == ("Alpha",)
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    def test_install_with_public_exports_filters_dunder_all(self) -> None:
        # Arrange
        module_globals: t.ModuleGlobals = {}

        # Act — private symbol wired but excluded from the published surface
        install_lazy_exports(
            "test_pkg",
            module_globals,
            {
                "Alpha": ("test_pkg.alpha", "Alpha"),
                "InternalAlpha": ("test_pkg._alpha", "InternalAlpha"),
            },
            public_exports=("Alpha",),
        )

        # Assert
        assert module_globals["__all__"] == ("Alpha",)
        dir_fn = module_globals["__dir__"]
        assert callable(dir_fn)
        assert dir_fn() == ["Alpha"]

    def test_installed_getattr_resolves_absolute_target(
        self,
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
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

    def test_installed_getattr_resolves_relative_target(
        self,
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange — relative path resolved against the installing package
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

    def test_installed_getattr_resolves_bare_string_module_entry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — a bare-string entry names a module whose same-named attr is used;
        # resolution must succeed without any '<pkg>.alias' child module existing.
        lazy.reset()
        target_name = "test_lazy_alias_target"
        target = ModuleType(target_name)
        setattr(target, "alias", "resolved")
        monkeypatch.setitem(sys.modules, target_name, target)

        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_lazy_alias_pkg",
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

    def test_installed_getattr_raises_attribute_error_for_unknown_name(self) -> None:
        # Arrange
        module_globals: t.ModuleGlobals = {}
        install_lazy_exports(
            "test_pkg",
            module_globals,
            {"Alpha": ("test_pkg.alpha", "Alpha")},
        )
        getattr_fn = module_globals["__getattr__"]
        assert callable(getattr_fn)

        # Act / Assert
        with pytest.raises(AttributeError, match="Missing"):
            getattr_fn("Missing")

    def test_install_is_idempotent_and_keeps_getattr_working(
        self,
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
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

    def test_get_resolves_symbol_and_caches_into_module_globals(
        self,
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
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

    def test_get_raises_attribute_error_for_name_absent_from_map(self) -> None:
        # Arrange
        module_globals: t.ModuleGlobals = {}

        # Act / Assert
        with pytest.raises(AttributeError, match="Missing"):
            lazy.get("Missing", {}, module_globals, "test_pkg")

    def test_build_map_produces_flat_sorted_import_map(self) -> None:
        # Act
        result = build_lazy_import_map(
            {"pkg.mod": ("Beta", "Alpha")},
            alias_groups={"pkg.aliases": (("Zeta", "ZetaImpl"),)},
        )

        # Assert — module-group names map to the module; alias-groups keep target+attr
        assert list(result) == ["Alpha", "Beta", "Zeta"]
        assert result["Alpha"] == "pkg.mod"
        assert result["Zeta"] == ("pkg.aliases", "ZetaImpl")

    def test_normalize_map_resolves_relative_paths_against_module(self) -> None:
        # Act
        normalized = lazy.normalize_map(
            "pkg.sub",
            {"Rel": (".child", "Rel"), "Abs": "other.mod"},
        )

        # Assert
        assert normalized["Rel"] == ("pkg.sub.child", "Rel")
        assert normalized["Abs"] == "other.mod"

    def test_merge_combines_child_and_local_with_local_precedence(self) -> None:
        # Arrange — a child package exposing its own _LAZY_IMPORTS
        lazy.reset()
        child_name = "test_merge_child"
        child = ModuleType(child_name)
        setattr(
            child,
            "_LAZY_IMPORTS",
            {
                "ChildOnly": (f"{child_name}.a", "ChildOnly"),
                "Shared": (f"{child_name}.a", "SharedChild"),
            },
        )
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

    def test_reset_clears_caches_observed_via_cache_stats(
        self,
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange — perform an install to populate caches
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
