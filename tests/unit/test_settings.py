"""Behavior contract for flext_core.FlextSettings — public API only.

Tests the minimal canonical surface: ``fetch_global``/``update_global``/
``clone``/``reset_for_testing`` and the universal scalar fields
(``debug``/``trace``/``log_level``/``timezone``/``async_logging``). Namespaced
project fields are plain nested Pydantic-2 model Fields; there is no namespace
registry, ``app_name`` field, ``validate_overrides`` or ``clone_for_injection``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from flext_core import FlextSettings


class TestsFlextCoreSettings:
    """Public-contract behaviour for the exported settings facade."""

    def test_fetch_global_returns_stable_singleton(self) -> None:
        """fetch_global returns the same instance across calls."""
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        assert first is second

    def test_universal_fields_present_with_defaults(self) -> None:
        """The five universal runtime fields exist with scalar defaults."""
        s = FlextSettings.fetch_global()
        assert isinstance(s.debug, bool)
        assert isinstance(s.trace, bool)
        assert isinstance(s.log_level, str)
        assert isinstance(s.timezone, str)
        assert isinstance(s.async_logging, bool)

    def test_update_global_replaces_singleton_and_propagates(self) -> None:
        """update_global installs a new singleton reflected by fetch_global."""
        FlextSettings.reset_for_testing()
        updated = FlextSettings.update_global(log_level="ERROR")
        assert updated.log_level == "ERROR"
        assert FlextSettings.fetch_global().log_level == "ERROR"
        FlextSettings.reset_for_testing()

    def test_update_global_rejects_unknown_field(self) -> None:
        """update_global raises for keys that are not declared fields."""
        with pytest.raises(ValueError, match="Unknown settings override"):
            FlextSettings.update_global(typo_field="x")

    def test_clone_produces_isolated_snapshot(self) -> None:
        """Clone returns an isolated copy without mutating the singleton."""
        FlextSettings.reset_for_testing()
        original = FlextSettings.fetch_global()
        original_level = original.log_level
        snapshot = original.clone(log_level="CRITICAL")
        assert snapshot.log_level == "CRITICAL"
        assert original.log_level == original_level
        assert FlextSettings.fetch_global().log_level == original_level
        FlextSettings.reset_for_testing()

    def test_fetch_global_overrides_returns_isolated_clone(self) -> None:
        """fetch_global(overrides=...) yields a clone, not the singleton."""
        FlextSettings.reset_for_testing()
        singleton = FlextSettings.fetch_global()
        singleton_level = singleton.log_level
        derived = FlextSettings.fetch_global(overrides={"log_level": "WARNING"})
        assert derived.log_level == "WARNING"
        assert FlextSettings.fetch_global().log_level == singleton_level
        FlextSettings.reset_for_testing()

    def test_trace_requires_debug_invariant(self) -> None:
        """trace=True without debug raises the documented validation error."""
        with pytest.raises(ValidationError):
            FlextSettings.update_global(trace=True, debug=False)
        FlextSettings.reset_for_testing()

    def test_reset_for_testing_drops_singleton(self) -> None:
        """reset_for_testing forces a fresh instance on next fetch_global."""
        first = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        second = FlextSettings.fetch_global()
        assert first is not second


class TestsFlextCoreSettingsWorkDir:
    """Platform-aware per-namespace ``work_dir`` universal field contract."""

    def test_base_default_derives_flext_namespace(self) -> None:
        """Base FlextSettings work_dir uses the 'flext' namespace name."""
        FlextSettings.reset_for_testing()
        s = FlextSettings.fetch_global()
        assert isinstance(s.work_dir, Path)
        assert s.work_dir.name == "flext"
        FlextSettings.reset_for_testing()

    def test_default_root_is_platform_cache_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default root follows the platform cache convention."""
        FlextSettings.reset_for_testing()
        if sys.platform == "darwin":
            expected = Path.home() / "Library" / "Caches" / "flext"
        elif sys.platform == "win32":
            monkeypatch.setenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
            expected = Path.home() / "AppData" / "Local" / "flext"
        else:
            monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
            expected = Path.home() / ".cache" / "flext"
        assert FlextSettings.fetch_global().work_dir == expected
        FlextSettings.reset_for_testing()

    def test_linux_honours_xdg_cache_home(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Linux XDG_CACHE_HOME redirects the work_dir root."""
        if sys.platform not in {"linux", "linux2"}:
            pytest.skip("XDG_CACHE_HOME is Linux-specific")
        FlextSettings.reset_for_testing()
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")
        assert FlextSettings.fetch_global().work_dir == Path("/xdg/cache/flext")
        FlextSettings.reset_for_testing()

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FLEXT_WORK_DIR overrides the derived default."""
        FlextSettings.reset_for_testing()
        monkeypatch.setenv("FLEXT_WORK_DIR", "/tmp/flext-custom-wd")
        assert FlextSettings.fetch_global().work_dir == Path("/tmp/flext-custom-wd")
        FlextSettings.reset_for_testing()

    def test_subclass_namespace_names_the_dir(self) -> None:
        """A subclass derives its dir name from its env_prefix namespace."""

        class _AcmeSettings(FlextSettings):
            model_config = FlextSettings.model_config | {"env_prefix": "ACME_TOOL_"}

        _AcmeSettings.reset_for_testing()
        assert _AcmeSettings.fetch_global().work_dir.name == "acme-tool"
        _AcmeSettings.reset_for_testing()

    def test_linux_namespaced_xdg_roots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux XDG roots are independently namespaced."""
        if sys.platform not in {"linux", "linux2"}:
            pytest.skip("XDG directories are Linux-specific")
        roots = {
            "XDG_CACHE_HOME": "/xdg/cache",
            "XDG_DATA_HOME": "/xdg/data",
            "XDG_STATE_HOME": "/xdg/state",
            "XDG_CONFIG_HOME": "/xdg/config",
            "XDG_RUNTIME_DIR": "/xdg/runtime",
        }
        for name, value in roots.items():
            monkeypatch.setenv(name, value)

        class _AcmeSettings(FlextSettings):
            model_config = FlextSettings.model_config | {"env_prefix": "ACME_TOOL_"}

        _AcmeSettings.reset_for_testing()
        settings = _AcmeSettings.fetch_global()
        assert settings.work_dir == Path("/xdg/cache/acme-tool")
        assert settings.data_dir == Path("/xdg/data/acme-tool")
        assert settings.state_dir == Path("/xdg/state/acme-tool")
        assert settings.config_dir == Path("/xdg/config/acme-tool")
        assert settings.runtime_dir == Path("/xdg/runtime/acme-tool")
        _AcmeSettings.reset_for_testing()

    def test_runtime_dir_falls_back_under_work_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The runtime directory falls back below an explicit work directory."""
        work_dir = tmp_path / "work"
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setenv("FLEXT_WORK_DIR", str(work_dir))
        FlextSettings.reset_for_testing()
        assert FlextSettings.fetch_global().runtime_dir == work_dir / "run"
        FlextSettings.reset_for_testing()

    def test_explicit_runtime_directory_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The project runtime override wins over XDG_RUNTIME_DIR."""
        runtime_dir = tmp_path / "runtime"
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/xdg/runtime")
        monkeypatch.setenv("FLEXT_RUNTIME_DIR", str(runtime_dir))
        FlextSettings.reset_for_testing()
        assert FlextSettings.fetch_global().runtime_dir == runtime_dir
        FlextSettings.reset_for_testing()

    def test_runtime_directories_reject_relative_overrides(self) -> None:
        """Relative runtime directory overrides fail validation."""
        FlextSettings.reset_for_testing()
        with pytest.raises(ValidationError, match="must be absolute"):
            FlextSettings.fetch_global(overrides={"state_dir": "relative-state"})
        FlextSettings.reset_for_testing()
