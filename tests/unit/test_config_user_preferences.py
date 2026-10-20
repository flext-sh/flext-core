"""Packaged config defaults + namespace-scoped user preference overlay.

Every FLEXT package ships immutable defaults under its own import namespace
(``<pkg>/config/*.yaml``, force-included into the wheel). An operator may
override declared values from the platform config root scoped by the package
namespace (``$XDG_CONFIG_HOME/<pkg-namespace>/*.yaml``). Neither layer depends
on a source checkout or on the process CWD, so a ``pip``/``uv``/``uvx``/
``pipx`` install is fully functional with no repository present.

These tests build a synthetic installed package on ``sys.path`` — no fixture
mirrors flext-core's own config, so the contract is proven generically rather
than for one project.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from flext_core.config_sources import FlextStrictYamlConfigSource
from tests import tm, u

_SUBCLASS_SOURCE = '''
from __future__ import annotations

from flext_core import FlextConfig


class SynthConfig(FlextConfig):
    """Synthetic consumer of the packaged-config contract."""

    greeting: str = "unset"
    level: int = 0
'''


def _install_package(root: Path, package: str, defaults: str) -> ModuleType:
    """Create and import a synthetic installed package rooted at ``root``.

    Returns:
        The resulting ``ModuleType``.

    """
    package_dir = root / package
    (package_dir / "config").mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "_config.py").write_text(_SUBCLASS_SOURCE, encoding="utf-8")
    (package_dir / "config" / "defaults.yaml").write_text(defaults, encoding="utf-8")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"{package}._config")


@pytest.fixture(autouse=True)
def isolate_import_state() -> Iterator[None]:
    """Drop synthetic packages from the import cache between tests."""
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    yield
    sys.path[:] = original_path
    for name in set(sys.modules) - original_modules:
        del sys.modules[name]


class TestPackagedConfigWithUserPreferences:
    """Packaged defaults plus optional user overlay, keyed by package namespace."""

    @staticmethod
    def _install_packaged(
        tmp_path: Path,
        package: str,
        packaged_yaml: str,
        xdg_root: Path,
    ) -> ModuleType:
        """Install one synthetic package under an isolated XDG root.

        Returns:
            The synthetic package module carrying ``SynthConfig``.

        """
        with u.Tests.env_vars_context(env_vars={"XDG_CONFIG_HOME": str(xdg_root)}):
            return _install_package(tmp_path / "site-packages", package, packaged_yaml)

    @staticmethod
    def _seed_user_preferences(
        xdg_root: Path,
        package_namespace: str,
        preferences_yaml: str,
    ) -> None:
        """Write one user preferences overlay under the XDG package namespace."""
        user_dir = xdg_root / package_namespace
        user_dir.mkdir(parents=True)
        (user_dir / "preferences.yaml").write_text(
            preferences_yaml,
            encoding="utf-8",
        )

    @staticmethod
    def test_packaged_defaults_load_without_a_source_checkout(
        tmp_path: Path,
    ) -> None:
        """Given an installed package, When loaded, Then packaged YAML applies."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        ):
            module = TestPackagedConfigWithUserPreferences._install_packaged(
                tmp_path,
                "synthpkg_alpha",
                "greeting: packaged\nlevel: 1\n",
                tmp_path / "xdg",
            )

            config = module.SynthConfig.fetch_global()

            tm.that(config.greeting, eq="packaged")
            tm.that(config.level, eq=1)

    @staticmethod
    def test_user_preferences_override_only_declared_keys(tmp_path: Path) -> None:
        """Given a user YAML, When loaded, Then it wins and leaves the rest."""
        xdg = tmp_path / "xdg"
        with u.Tests.env_vars_context(env_vars={"XDG_CONFIG_HOME": str(xdg)}):
            TestPackagedConfigWithUserPreferences._seed_user_preferences(
                xdg,
                "synthpkg-beta",
                "greeting: operator\n",
            )
            module = _install_package(
                tmp_path / "site-packages",
                "synthpkg_beta",
                "greeting: packaged\nlevel: 7\n",
            )

            config = module.SynthConfig.fetch_global()

            tm.that(config.greeting, eq="operator")
            tm.that(config.level, eq=7)

    @staticmethod
    def test_absent_user_config_keeps_packaged_defaults(tmp_path: Path) -> None:
        """Given no user directory, When loaded, Then defaults still apply."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "empty-xdg")},
        ):
            module = TestPackagedConfigWithUserPreferences._install_packaged(
                tmp_path,
                "synthpkg_gamma",
                "greeting: packaged\nlevel: 3\n",
                tmp_path / "empty-xdg",
            )

            config = module.SynthConfig.fetch_global()

            tm.that(config.greeting, eq="packaged")
            tm.that(config.level, eq=3)

    @staticmethod
    def test_namespaces_stay_isolated(tmp_path: Path) -> None:
        """Given two packages, When both load, Then neither reads the other."""
        xdg = tmp_path / "xdg"
        with u.Tests.env_vars_context(env_vars={"XDG_CONFIG_HOME": str(xdg)}):
            TestPackagedConfigWithUserPreferences._seed_user_preferences(
                xdg,
                "synthpkg-delta",
                "greeting: delta\n",
            )
            site = tmp_path / "site-packages"
            delta = _install_package(site, "synthpkg_delta", "greeting: packaged\n")
            epsilon = _install_package(site, "synthpkg_epsilon", "greeting: packaged\n")

            tm.that(delta.SynthConfig.fetch_global().greeting, eq="delta")
            tm.that(epsilon.SynthConfig.fetch_global().greeting, eq="packaged")

    @staticmethod
    @pytest.mark.parametrize("layout", ["site-packages", "src", "member/src"])
    def test_nested_mro_declaration_uses_its_package_resource(
        tmp_path: Path,
        layout: str,
    ) -> None:
        """Declaration depth and source layout cannot select repository defaults."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        ):
            root = tmp_path / layout
            parent = _install_package(root, "synthpkg_parent", "greeting: parent\n")
            child = _install_package(root, "synthpkg_nested", "greeting: packaged\n")
            nested = root / "synthpkg_nested" / "declarations"
            nested.mkdir()
            (nested / "__init__.py").write_text("", encoding="utf-8")
            (nested / "owner.py").write_text(
                "from synthpkg_parent._config import SynthConfig\n"
                "class NestedConfig(SynthConfig):\n"
                "    pass\n",
                encoding="utf-8",
            )
            for directory in (nested / "config", tmp_path / "config"):
                directory.mkdir()
                (directory / "defaults.yaml").write_text(
                    "greeting: foreign\n", encoding="utf-8"
                )
            module = importlib.import_module("synthpkg_nested.declarations.owner")
            config = module.NestedConfig.fetch_global()
            tm.that(config.greeting, eq=child.SynthConfig.fetch_global().greeting)
            tm.that(config.greeting, ne=parent.SynthConfig.fetch_global().greeting)
            tm.that(module.NestedConfig.fetch_global() is config, truth=True)
            tm.that(config is not parent.SynthConfig.fetch_global(), truth=True)

    @staticmethod
    @pytest.mark.parametrize("layout", ["site-packages", "src", "member/src"])
    def test_inherited_yaml_and_child_overrides_share_one_source_chain(
        tmp_path: Path,
        layout: str,
    ) -> None:
        """Required inherited values and child overrides come from their own YAML."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        ):
            root = tmp_path / layout
            _install_package(
                root,
                "synthpkg_base_chain",
                "greeting: base\nlevel: 7\nparent_value: inherited\nentries: [base]\n",
            )
            _install_package(
                root,
                "synthpkg_child_chain",
                "greeting: child\nchild_value: declared\nentries: [child]\n",
            )
            package = root / "synthpkg_child_chain"
            (package / "owner.py").write_text(
                "from synthpkg_base_chain._config import SynthConfig\n"
                "class ChildConfig(SynthConfig):\n"
                "    parent_value: str\n"
                "    child_value: str\n"
                "    entries: list[str]\n"
                "class GrandchildConfig(ChildConfig):\n"
                "    pass\n",
                encoding="utf-8",
            )
            module = importlib.import_module("synthpkg_child_chain.owner")
            inherited = module.GrandchildConfig.fetch_global()
            expected = FlextStrictYamlConfigSource(
                module.GrandchildConfig,
                yaml_file=(
                    root / "synthpkg_base_chain" / "config" / "defaults.yaml",
                    package / "config" / "defaults.yaml",
                ),
                deep_merge=True,
            )()

            tm.that(inherited.model_dump(), eq=expected)
            tm.that(module.GrandchildConfig.fetch_global() is inherited, eq=True)

    @staticmethod
    def test_child_declared_file_order_overrides_its_parent(tmp_path: Path) -> None:
        """A child's explicit order wins when the same sources appear in its MRO."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        ):
            root = tmp_path / "site-packages"
            _install_package(root, "synthpkg_ordered_chain", "greeting: defaults\n")
            package = root / "synthpkg_ordered_chain"
            first = package / "config" / "first.yaml"
            second = package / "config" / "second.yaml"
            first.write_text("greeting: first\nlevel: 1\n", encoding="utf-8")
            second.write_text("greeting: second\nlevel: 2\n", encoding="utf-8")
            (package / "owner.py").write_text(
                "from typing import ClassVar\n"
                "from synthpkg_ordered_chain._config import SynthConfig\n"
                "class ParentConfig(SynthConfig):\n"
                "    CONFIG_FILENAMES: ClassVar[tuple[str, ...]] = (\n"
                "        'first.yaml', 'second.yaml')\n"
                "class ChildConfig(ParentConfig):\n"
                "    CONFIG_FILENAMES: ClassVar[tuple[str, ...]] = (\n"
                "        'second.yaml', 'first.yaml')\n",
                encoding="utf-8",
            )
            module = importlib.import_module("synthpkg_ordered_chain.owner")
            expected = FlextStrictYamlConfigSource(
                module.ChildConfig,
                yaml_file=(second, first),
                deep_merge=True,
            )()

            tm.that(module.ChildConfig.fetch_global().model_dump(), eq=expected)

    @staticmethod
    def test_missing_declared_resource_never_borrows_repository_config(
        tmp_path: Path,
    ) -> None:
        """A missing packaged declaration fails even with a repository lookalike."""
        with u.Tests.env_vars_context(
            env_vars={"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        ):
            root = tmp_path / "src"
            _install_package(root, "synthpkg_missing", "greeting: packaged\n")
            package = root / "synthpkg_missing"
            (package / "config" / "defaults.yaml").unlink()
            (package / "config").rmdir()
            (tmp_path / "config").mkdir()
            (tmp_path / "config" / "defaults.yaml").write_text(
                "greeting: foreign\n", encoding="utf-8"
            )
            (package / "declared.py").write_text(
                "from typing import ClassVar\n"
                "from synthpkg_missing._config import SynthConfig\n"
                "class DeclaredConfig(SynthConfig):\n"
                "    CONFIG_FILENAMES: ClassVar[tuple[str, ...]] = (\n"
                "        'defaults.yaml',)\n",
                encoding="utf-8",
            )
            module = importlib.import_module("synthpkg_missing.declared")
            with pytest.raises(FileNotFoundError, match="declared config directory"):
                module.DeclaredConfig.fetch_global()

    @staticmethod
    def test_explicit_config_dir_env_override_replaces_the_root(
        tmp_path: Path,
    ) -> None:
        """Given ``<PKG>_CONFIG_DIR``, When loaded, Then that directory is used."""
        explicit = tmp_path / "explicit-config"
        with u.Tests.env_vars_context(
            env_vars={
                "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
                "SYNTHPKG_ZETA_CONFIG_DIR": str(explicit),
            },
        ):
            explicit.mkdir()
            (explicit / "defaults.yaml").write_text(
                "greeting: explicit\n",
                encoding="utf-8",
            )
            module = _install_package(
                tmp_path / "site-packages",
                "synthpkg_zeta",
                "greeting: packaged\n",
            )

            tm.that(module.SynthConfig.fetch_global().greeting, eq="explicit")
