"""Behavioral contract for flext_core.lazy — public PEP 562 lazy export surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

from flext_core.lazy import build_lazy_import_map, install_lazy_exports, lazy
from tests import tm, u

if TYPE_CHECKING:
    from tests import t


def _install_alpha(
    registered_alpha_module: tuple[str, type],
    *,
    publish_all: bool = True,
) -> t.ModuleGlobals:
    """Install the registered alpha lazy export and return package globals.

    Returns:
        The package module globals after installation.

    """
    module_name, _ = registered_alpha_module
    package_name = module_name.rpartition(".")[0]
    module_globals: t.ModuleGlobals = vars(sys.modules[package_name])
    install_lazy_exports(
        package_name,
        module_globals,
        {"Alpha": (module_name, "Alpha")},
        publish_all=publish_all,
    )
    return module_globals


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


class TestsFlextCoreLazyExports:
    """Behavioral contract: what the lazy export surface promises callers."""

    @staticmethod
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
        tm.ok(alias is facade)

    @staticmethod
    def test_root_package_resolves_primary_facades_via_aliases() -> None:
        # Arrange / Act
        """Test root package resolves primary facades via aliases."""
        package = importlib.import_module("flext_core")

        # Assert — each short alias resolves to its full facade
        tm.ok(package.c is package.FlextConstants)
        tm.ok(package.e is package.FlextExceptions)
        tm.ok(package.m is package.FlextModels)
        tm.ok(package.p is package.FlextProtocols)
        tm.ok(package.t is package.FlextTypes)
        tm.ok(package.u is package.FlextUtilities)
        tm.that(
            {"FlextConstants", "FlextUtilities", "u"} <= set(package.__all__), eq=True
        )

    @staticmethod
    def test_cold_loader_does_not_execute_generated_child_initializer(
        tmp_path: Path,
    ) -> None:
        """The bootstrap loader works before child projections are recoverable."""
        source = Path(__file__).resolve().parents[2] / "src" / "flext_core"
        candidate = tmp_path / "flext_core"
        shutil.copytree(source, candidate, ignore=shutil.ignore_patterns("__pycache__"))
        child = candidate / "_lazy_parts"
        child.mkdir(exist_ok=True)
        (child / "__init__.py").write_text(
            "raise RuntimeError('generated child initializer is not ready')\n",
            encoding="utf-8",
        )
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "from flext_core.lazy import install_lazy_exports\n"
            "namespace = {}\n"
            "install_lazy_exports('bootstrap_probe', namespace, {}, "
            "public_exports=())\n"
            "print(namespace['__all__'] == (), namespace['__dir__']() == [], "
            "'flext_core._lazy_parts' not in sys.modules)\n"
        )
        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=tmp_path)
        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.outcome.raw_return_code, eq=0, msg=result.value.stderr)
        tm.that(result.value.stderr, eq="")
        tm.that(result.value.stdout, eq="True True True\n")

    @staticmethod
    def test_fresh_root_supports_bootstrap_public_facade_dependency(
        tmp_path: Path,
    ) -> None:
        """Fresh public imports survive a facade-dependent bootstrap cutover."""
        source = Path(__file__).resolve().parents[2] / "src" / "flext_core"
        candidate = tmp_path / "flext_core"
        shutil.copytree(source, candidate, ignore=shutil.ignore_patterns("__pycache__"))
        bootstrap = candidate / "_beartype_bootstrap.py"
        bootstrap.write_text(
            bootstrap.read_text(encoding="utf-8") + "\nfrom flext_core import c\n",
            encoding="utf-8",
        )
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "import flext_core\n"
            f"print(flext_core.__file__ == {str(candidate / '__init__.py')!r})\n"
            "from flext_core import c, t, m, u\n"
            "print(c is flext_core.FlextConstants, t is flext_core.FlextTypes, "
            "m is flext_core.FlextModels, u is flext_core.FlextUtilities)\n"
            "from flext_cli import FlextCli\n"
            "print(bool(FlextCli))\n"
        )
        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=tmp_path)
        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.outcome.raw_return_code, eq=0, msg=result.value.stderr)
        tm.that(result.value.stderr, eq="")
        tm.that(result.value.stdout, eq="True\nTrue True True True\nTrue\n")

    @staticmethod
    @pytest.mark.parametrize(
        "first_import",
        [
            "from flext_core import t, p, m",
            "from flext_cli import t, p, m",
            "from flext_infra import t, p, m",
            "from flext_infra import u, t, p, m",
        ],
    )
    def test_cold_public_declarations_resolve_from_live_sources(
        first_import: str,
    ) -> None:
        """Public consumers resolve declarations without warming constants first."""
        workspace = Path(__file__).resolve().parents[3]
        script = (
            f"{first_import}\n"
            "from pathlib import Path\n"
            "from typing import get_args\n"
            "import flext_core, flext_cli, flext_infra\n"
            "from flext_core import FlextSettings, c, m, p, t\n"
            f"workspace = Path({str(workspace)!r})\n"
            "for package in (flext_core, flext_cli, flext_infra):\n"
            "    source = workspace / package.__name__.replace('_', '-') / 'src'\n"
            "    print(Path(package.__file__).resolve().is_relative_to(source), "
            "issubclass(package.t, t), issubclass(package.p, p), "
            "issubclass(package.m, m))\n"
            "print(isinstance(FlextSettings, p.SettingsType))\n"
            "for operation in ('register_plugin', 'unregister_plugin', "
            "'fetch_plugin', 'list_plugins'):\n"
            "    method = getattr(p.Registry, operation)\n"
            "    print(method.__kwdefaults__['scope'] is "
            "c.RegistrationScope.INSTANCE)\n"
            "options = m.RuntimeBootstrapOptions(settings_type=FlextSettings)\n"
            "print(options.settings_type is FlextSettings)\n"
            "print(options.model_dump() == {'settings_overrides': None})\n"
        )
        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=workspace)
        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.outcome.raw_return_code, eq=0, msg=result.value.stderr)
        tm.that(result.value.stderr, eq="")
        tm.that(
            result.value.stdout,
            eq="True True True True\n" * 3 + "True\n" * 7,
        )

    @staticmethod
    def test_model_facade_does_not_import_web_runtime() -> None:
        """Loading model declarations must not import the web framework stack."""
        script = (
            "import sys\n"
            "from flext_core import m\n"
            "print(bool(m.StrictModel))\n"
            "for name in sorted(sys.modules):\n"
            "    if name == 'fastapi' or name.startswith('fastapi.'):\n"
            "        print(name)\n"
        )

        result = u.Cli.run_raw([sys.executable, "-c", script], cwd=Path.cwd())

        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.stdout, eq="True\n")

    @staticmethod
    def test_install_without_publish_all_omits_dunder_all(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install without publish all omits dunder all."""
        module_globals = _install_alpha(registered_alpha_module, publish_all=False)

        # Assert — no __all__ published, but __dir__ still lists the public name
        tm.that(module_globals, where=lambda namespace: "__all__" not in namespace)
        dir_fn = cast("Callable[[], list[str]]", module_globals["__dir__"])
        tm.that(callable(dir_fn), truth=True)
        tm.that(dir_fn(), eq=["Alpha"])

    @staticmethod
    def test_install_with_publish_all_publishes_dunder_all(
        registered_alpha_module: tuple[str, type],
    ) -> None:
        # Arrange
        """Test install with publish all publishes dunder all."""
        module_globals = _install_alpha(registered_alpha_module)

        # Assert
        tm.that(module_globals["__all__"], eq=("Alpha",))
        dir_fn = cast("Callable[[], list[str]]", module_globals["__dir__"])
        tm.that(callable(dir_fn), truth=True)
        tm.that(dir_fn(), eq=["Alpha"])

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
        tm.that(module_globals["__all__"], eq=("Alpha",))
        dir_fn = cast("Callable[[], list[str]]", module_globals["__dir__"])
        tm.that(callable(dir_fn), truth=True)
        tm.that(dir_fn(), eq=["Alpha"])

        tm.that(sys.modules[package_name].Alpha, where=lambda value: value is alpha_cls)

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
        getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
        tm.that(callable(getattr_fn), truth=True)
        resolved = getattr_fn("Alpha")

        # Assert — lazy symbol resolves to the real class and is cached in globals
        tm.that(resolved, where=lambda value: value is alpha_cls)
        tm.that(module_globals["Alpha"], where=lambda value: value is alpha_cls)

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
        getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
        tm.that(callable(getattr_fn), truth=True)

        # Assert
        tm.that(getattr_fn("Alpha"), where=lambda value: value is alpha_cls)

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
            getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
            tm.that(callable(getattr_fn), truth=True)

            # Assert — resolves the attribute, and never required a
            # probed child submodule
            tm.that(getattr_fn("alias"), eq="resolved")
            tm.that(
                sys.modules,
                where=lambda modules: "test_lazy_alias_pkg.alias" not in modules,
            )
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
        getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
        tm.that(callable(getattr_fn), truth=True)

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
        getattr_hook, dir_hook = install_lazy_exports(
            "test_lazy_pkg", module_globals, lazy_map
        )
        repeated_getattr, repeated_dir = install_lazy_exports(
            "test_lazy_pkg", module_globals, lazy_map
        )

        # Assert
        getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
        tm.that(callable(getattr_fn), truth=True)
        tm.that(getattr_fn("Alpha"), where=lambda value: value is alpha_cls)
        tm.that(module_globals["__all__"], eq=("Alpha",))
        tm.ok(getattr_hook is repeated_getattr is getattr_fn)
        tm.ok(dir_hook is repeated_dir is module_globals["__dir__"])
        tm.that(dir_hook(), eq=["Alpha"])

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
        tm.that(resolved, where=lambda value: value is alpha_cls)
        tm.that(module_globals["Alpha"], where=lambda value: value is alpha_cls)


class TestsFlextCoreLazyExportCaching:
    """In-progress import rebinding and completed export caching."""

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
        tm.that(
            child.RESOLVED_DURING_INIT, where=lambda value: value is child.FirstAlias
        )
        tm.that(child.CACHED_DURING_INIT, is_=bool, eq=False)
        tm.that(first_access, where=lambda value: value is child.FinalAlias)
        tm.that(vars(package)["u"], where=lambda value: value is child.FinalAlias)
        tm.that(package.u, where=lambda value: value is child.FinalAlias)

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
        tm.that(resolved, where=lambda value: value is child.FinalAlias)
        tm.that(vars(package)["u"], where=lambda value: value is child.FinalAlias)

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
        tm.that(list(result), eq=["Alpha", "Beta", "Zeta"])
        tm.that(result["Alpha"], eq="pkg.mod")
        tm.that(result["Zeta"], eq=("pkg.aliases", "ZetaImpl"))

    @staticmethod
    def test_normalize_map_resolves_relative_paths_against_module() -> None:
        # Act
        """Test normalize map resolves relative paths against module."""
        normalized = lazy.normalize_map(
            "pkg.sub",
            {"Rel": (".child", "Rel"), "Abs": "other.mod"},
        )

        # Assert
        tm.that(normalized["Rel"], eq=("pkg.sub.child", "Rel"))
        tm.that(normalized["Abs"], eq="other.mod")

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
        tm.that(merged, has="ChildOnly")
        tm.that(merged["Shared"], eq=("local.mod", "SharedLocal"))

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
        getattr_fn = cast("Callable[[str], object]", module_globals["__getattr__"])
        tm.that(callable(getattr_fn), truth=True)
        getattr_fn("Alpha")
        tm.that(any(size > 0 for size in lazy.cache_stats.values()), truth=True)

        # Act
        lazy.reset()

        # Assert — all caches empty after reset
        tm.that(all(size == 0 for size in lazy.cache_stats.values()), truth=True)
