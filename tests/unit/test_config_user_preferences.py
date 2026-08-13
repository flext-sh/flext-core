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
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_SUBCLASS_SOURCE = '''
from __future__ import annotations

from flext_core import FlextConfig


class SynthConfig(FlextConfig):
    """Synthetic consumer of the packaged-config contract."""

    greeting: str = "unset"
    level: int = 0
'''


def _install_package(root: Path, package: str, defaults: str) -> ModuleType:
    """Create and import a synthetic installed package rooted at ``root``."""
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

    def test_packaged_defaults_load_without_a_source_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an installed package, When loaded, Then packaged YAML applies."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        module = _install_package(
            tmp_path / "site-packages",
            "synthpkg_alpha",
            "greeting: packaged\nlevel: 1\n",
        )

        config = module.SynthConfig.fetch_global()

        assert config.greeting == "packaged"
        assert config.level == 1

    def test_user_preferences_override_only_declared_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a user YAML, When loaded, Then it wins and leaves the rest."""
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        user_dir = xdg / "synthpkg-beta"
        user_dir.mkdir(parents=True)
        (user_dir / "preferences.yaml").write_text(
            "greeting: operator\n", encoding="utf-8"
        )
        module = _install_package(
            tmp_path / "site-packages",
            "synthpkg_beta",
            "greeting: packaged\nlevel: 7\n",
        )

        config = module.SynthConfig.fetch_global()

        assert config.greeting == "operator"
        assert config.level == 7

    def test_absent_user_config_keeps_packaged_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given no user directory, When loaded, Then defaults still apply."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
        module = _install_package(
            tmp_path / "site-packages",
            "synthpkg_gamma",
            "greeting: packaged\nlevel: 3\n",
        )

        config = module.SynthConfig.fetch_global()

        assert config.greeting == "packaged"
        assert config.level == 3

    def test_namespaces_stay_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given two packages, When both load, Then neither reads the other."""
        xdg = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        foreign = xdg / "synthpkg-delta"
        foreign.mkdir(parents=True)
        (foreign / "preferences.yaml").write_text("greeting: delta\n", encoding="utf-8")
        site = tmp_path / "site-packages"
        delta = _install_package(site, "synthpkg_delta", "greeting: packaged\n")
        epsilon = _install_package(site, "synthpkg_epsilon", "greeting: packaged\n")

        assert delta.SynthConfig.fetch_global().greeting == "delta"
        assert epsilon.SynthConfig.fetch_global().greeting == "packaged"

    def test_explicit_config_dir_env_override_replaces_the_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given ``<PKG>_CONFIG_DIR``, When loaded, Then that directory is used."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        explicit = tmp_path / "explicit-config"
        explicit.mkdir()
        (explicit / "defaults.yaml").write_text(
            "greeting: explicit\n", encoding="utf-8"
        )
        monkeypatch.setenv("SYNTHPKG_ZETA_CONFIG_DIR", str(explicit))
        module = _install_package(
            tmp_path / "site-packages", "synthpkg_zeta", "greeting: packaged\n"
        )

        assert module.SynthConfig.fetch_global().greeting == "explicit"
