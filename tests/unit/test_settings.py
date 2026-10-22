"""Behavior contract for flext_core.FlextSettings — public API only.

Tests the minimal canonical surface: ``fetch_global``/``update_global``/
``clone``/``reset_for_testing`` and the universal scalar fields
(``debug``/``trace``/``log_level``/``timezone``/``async_logging``). Namespaced
project fields are plain nested Pydantic-2 model Fields; there is no namespace
registry, ``app_name`` field, ``validate_overrides`` or ``clone_for_injection``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Self

import pytest

from flext_core import FlextSettings, settings
from tests import m, t, tm, u


class TestsFlextCoreSettings:
    """Public-contract behaviour for the exported settings facade."""

    class TestsFlextCoreSettingsWorkDir:
        """Platform-aware consuming-application directory contract."""

        @staticmethod
        def setup_method() -> None:
            """Provide ``setup_method``."""
            FlextSettings.reset_app_namespace()

        @staticmethod
        def teardown_method() -> None:
            """Provide ``teardown_method``."""
            FlextSettings.reset_app_namespace()

        @staticmethod
        def test_base_default_derives_flext_namespace() -> None:
            """Base FlextSettings work_dir uses the 'flext' namespace name."""
            FlextSettings.reset_for_testing()
            s = FlextSettings.fetch_global()
            tm.that(s.work_dir, is_=Path)
            tm.that(s.work_dir.name, eq="flext")
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_default_root_is_platform_cache_home() -> None:
            """Default root follows the platform cache convention."""
            FlextSettings.reset_for_testing()
            if sys.platform == "darwin":
                env_context = u.Tests.env_vars_context()
                expected = Path.home() / "Library" / "Caches" / "flext"
            elif sys.platform == "win32":
                env_context = u.Tests.env_vars_context(
                    env_vars={"LOCALAPPDATA": str(Path.home() / "AppData" / "Local")},
                )
                expected = Path.home() / "AppData" / "Local" / "flext"
            else:
                env_context = u.Tests.env_vars_context(
                    vars_to_clear=("XDG_CACHE_HOME",),
                )
                expected = Path.home() / ".cache" / "flext"
            with env_context:
                tm.that(FlextSettings.fetch_global().work_dir, eq=expected)
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_linux_honours_xdg_cache_home() -> None:
            """On Linux XDG_CACHE_HOME redirects the work_dir root."""
            if sys.platform not in {"linux", "linux2"}:
                pytest.skip("XDG_CACHE_HOME is Linux-specific")
            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context(
                env_vars={"XDG_CACHE_HOME": "/xdg/cache"},
            ):
                tm.that(
                    FlextSettings.fetch_global().work_dir, eq=Path("/xdg/cache/flext")
                )
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_env_override_wins(tmp_path: Path) -> None:
            """FLEXT_WORK_DIR overrides the derived default."""
            FlextSettings.reset_for_testing()
            custom_dir = tmp_path / "flext-custom-wd"
            custom_dir.mkdir()
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_WORK_DIR": str(custom_dir)},
            ):
                tm.that(FlextSettings.fetch_global().work_dir, eq=custom_dir)
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_libraries_share_consuming_application_namespace() -> None:
            """Imported library settings resolve via the outer identity."""
            if sys.platform not in {"linux", "linux2"}:
                pytest.skip("XDG directories are Linux-specific")
            with u.Tests.env_vars_context(
                env_vars={
                    "XDG_CACHE_HOME": "/xdg/cache",
                    "XDG_DATA_HOME": "/xdg/data",
                    "XDG_STATE_HOME": "/xdg/state",
                    "XDG_CONFIG_HOME": "/xdg/config",
                    "XDG_RUNTIME_DIR": "/xdg/runtime",
                },
            ):

                class _CliSettings(FlextSettings):
                    model_config = FlextSettings.model_config | {
                        "env_prefix": "CLI_LIB_",
                    }

                class _MeltanoSettings(FlextSettings):
                    model_config = FlextSettings.model_config | {
                        "env_prefix": "MELTANO_LIB_",
                    }

                cli_settings = _CliSettings.fetch_global()
                meltano_settings = _MeltanoSettings.fetch_global()
                FlextSettings.apply_app_namespace("flext-tap-oracle")

                expected = (
                    Path("/xdg/cache/flext-tap-oracle"),
                    Path("/xdg/data/flext-tap-oracle"),
                    Path("/xdg/state/flext-tap-oracle"),
                    Path("/xdg/config/flext-tap-oracle"),
                    Path("/xdg/runtime/flext-tap-oracle"),
                )
                for library_settings in (cli_settings, meltano_settings):
                    tm.that(
                        (
                            library_settings.work_dir,
                            library_settings.data_dir,
                            library_settings.state_dir,
                            library_settings.config_dir,
                            library_settings.runtime_dir,
                        ),
                        eq=expected,
                    )

        @staticmethod
        def test_application_namespace_is_first_wins() -> None:
            """A later library bootstrap cannot replace the outer identity."""
            FlextSettings.apply_app_namespace("flext-tap-oracle")
            FlextSettings.apply_app_namespace("flext-cli")
            tm.that(FlextSettings.fetch_global().work_dir.name, eq="flext-tap-oracle")

        @staticmethod
        def test_unregistered_owner_namespace_prevails_by_default() -> None:
            """Without registration, the owning project's own namespace is the default.

            Registration is never mandatory: a standalone project (e.g. ``ai-hub``)
            transparently owns ``<root>/ai-hub`` directories, not the ``flext``
            fallback, purely from its ``env_prefix``.
            """
            if sys.platform not in {"linux", "linux2"}:
                pytest.skip("XDG directories are Linux-specific")
            with u.Tests.env_vars_context(
                env_vars={"XDG_CACHE_HOME": "/xdg/cache", "XDG_DATA_HOME": "/xdg/data"},
                vars_to_clear=("FLEXT_APP_NAMESPACE",),
            ):

                class _AiHubSettings(FlextSettings):
                    model_config = FlextSettings.model_config | {
                        "env_prefix": "AI_HUB_",
                    }

                _AiHubSettings.reset_for_testing()
                s = _AiHubSettings.fetch_global()
                tm.that(s.cache_dir, eq=Path("/xdg/cache/ai-hub"))
                tm.that(s.data_dir, eq=Path("/xdg/data/ai-hub"))
                _AiHubSettings.reset_for_testing()

        @staticmethod
        def test_computed_directories_serialize_with_current_application() -> None:
            """Pydantic serialization exposes current application dirs."""
            settings = FlextSettings.fetch_global()
            FlextSettings.apply_app_namespace("flext-tap-oracle")
            dumped = settings.model_dump()
            tm.that(Path(str(dumped["cache_dir"])).name, eq="flext-tap-oracle")
            tm.that(Path(str(dumped["config_dir"])).name, eq="flext-tap-oracle")

        @staticmethod
        def test_environment_supplies_application_namespace() -> None:
            """FLEXT_APP_NAMESPACE identifies the app when bootstrap is unavailable."""
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_APP_NAMESPACE": "flext-target-ldap"},
            ):
                tm.that(
                    FlextSettings.fetch_global().data_dir.name, eq="flext-target-ldap"
                )

        @staticmethod
        def test_environment_application_namespace_rejects_paths() -> None:
            """Environment application identity cannot escape its XDG root."""
            with (
                u.Tests.env_vars_context(
                    env_vars={"FLEXT_APP_NAMESPACE": "../outside"},
                ),
                pytest.raises(ValueError, match="one non-empty path segment"),
            ):
                _ = FlextSettings.fetch_global().data_dir

        @staticmethod
        def test_application_scoped_directory_override_wins(
            tmp_path: Path,
        ) -> None:
            """The consuming application's directory override wins over XDG roots."""
            override = tmp_path / "application-work"
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_TAP_ORACLE_WORK_DIR": str(override)},
            ):
                FlextSettings.apply_app_namespace("flext-tap-oracle")
                tm.that(FlextSettings.fetch_global().work_dir, eq=override)

        @staticmethod
        def test_runtime_dir_falls_back_under_work_dir(tmp_path: Path) -> None:
            """The runtime directory falls back below an explicit work directory."""
            work_dir = tmp_path / "work"
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_WORK_DIR": str(work_dir)},
                vars_to_clear=("XDG_RUNTIME_DIR",),
            ):
                FlextSettings.apply_app_namespace("flext")
                FlextSettings.reset_for_testing()
                tm.that(FlextSettings.fetch_global().runtime_dir, eq=work_dir / "run")
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_explicit_runtime_directory_override_wins(tmp_path: Path) -> None:
            """The project runtime override wins over XDG_RUNTIME_DIR."""
            runtime_dir = tmp_path / "runtime"
            with u.Tests.env_vars_context(
                env_vars={
                    "XDG_RUNTIME_DIR": "/xdg/runtime",
                    "FLEXT_RUNTIME_DIR": str(runtime_dir),
                },
            ):
                FlextSettings.apply_app_namespace("flext")
                FlextSettings.reset_for_testing()
                tm.that(FlextSettings.fetch_global().runtime_dir, eq=runtime_dir)
            FlextSettings.reset_for_testing()

        @staticmethod
        def test_runtime_directories_reject_relative_overrides() -> None:
            """Relative runtime directory overrides fail validation."""
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_TAP_ORACLE_STATE_DIR": "relative-state"},
            ):
                FlextSettings.apply_app_namespace("flext-tap-oracle")
                with pytest.raises(ValueError, match="must be absolute"):
                    _ = FlextSettings.fetch_global().state_dir

    @staticmethod
    def test_fetch_global_returns_stable_singleton() -> None:
        """fetch_global returns the same instance across calls."""
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        tm.that(first is second, truth=True)

    @staticmethod
    def test_universal_fields_present_with_defaults() -> None:
        """The five universal runtime fields exist with scalar defaults."""
        s = FlextSettings.fetch_global()
        tm.that(s.debug, is_=bool)
        tm.that(s.trace, is_=bool)
        tm.that(s.log_level, is_=str)
        tm.that(s.timezone, is_=str)
        tm.that(s.async_logging, is_=bool)

    @staticmethod
    def test_update_global_preserves_imported_singleton_and_propagates() -> None:
        """An already imported root sees validated updates without rebinding."""
        FlextSettings.reset_for_testing()
        selected_level = "ERROR"
        try:
            updated = FlextSettings.update_global(log_level=selected_level)
            tm.ok(updated is settings is FlextSettings.fetch_global())
            tm.that(settings.log_level, eq=selected_level)
        finally:
            FlextSettings.reset_for_testing()

    @staticmethod
    def test_update_global_rejects_unknown_field() -> None:
        """update_global raises for keys that are not declared fields."""
        with pytest.raises(ValueError, match="Unknown settings override"):
            FlextSettings.update_global(typo_field="x")

    @staticmethod
    def test_clone_produces_isolated_snapshot() -> None:
        """Clone returns an isolated copy without mutating the singleton."""
        FlextSettings.reset_for_testing()
        original = FlextSettings.fetch_global()
        original_level = original.log_level
        snapshot = original.clone(log_level="CRITICAL")
        tm.that(snapshot.log_level, eq="CRITICAL")
        tm.that(original.log_level, eq=original_level)
        tm.that(FlextSettings.fetch_global().log_level, eq=original_level)
        FlextSettings.reset_for_testing()

    @staticmethod
    def test_fetch_global_overrides_returns_isolated_clone() -> None:
        """fetch_global(overrides=...) yields a clone, not the singleton."""
        FlextSettings.reset_for_testing()
        singleton = FlextSettings.fetch_global()
        singleton_level = singleton.log_level
        derived = FlextSettings.fetch_global(overrides={"log_level": "WARNING"})
        tm.that(derived.log_level, eq="WARNING")
        tm.that(FlextSettings.fetch_global().log_level, eq=singleton_level)
        FlextSettings.reset_for_testing()

    @staticmethod
    def test_trace_detail_is_independent_of_debug_and_severity() -> None:
        """Trace-only settings preserve the captured severity."""
        original = FlextSettings.fetch_global()
        selected = original.clone(trace=True, debug=False)
        tm.that(selected.trace, truth=True)
        tm.that(selected.debug, truth=False)
        tm.that(selected.log_level, eq=original.log_level)

    @staticmethod
    def test_reset_for_testing_reloads_and_restores_imported_handle() -> None:
        """A real environment repoint and restoration keep the published root."""
        original = settings.model_dump()
        selected = not settings.debug
        try:
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_DEBUG": str(selected)},
            ):
                FlextSettings.reset_for_testing()
                tm.ok(settings is FlextSettings.fetch_global())
                tm.that(settings.debug, eq=selected)
        finally:
            FlextSettings.reset_for_testing()
        tm.ok(settings is FlextSettings.fetch_global())
        tm.that(settings.model_dump(), eq=original)

    @staticmethod
    def test_invalid_reset_and_direct_construction_leave_root_unchanged() -> None:
        """Failed reloads and isolated constructors cannot poison a bound root."""
        original = settings.model_dump()
        with u.Tests.env_vars_context(env_vars={"FLEXT_LOG_LEVEL": "invalid-level"}):
            with pytest.raises(m.ValidationError, match="Invalid log level"):
                FlextSettings.reset_for_testing()
            with pytest.raises(m.ValidationError, match="Invalid log level"):
                FlextSettings()
        tm.ok(settings is FlextSettings.fetch_global())
        tm.that(settings.model_dump(), eq=original)
        isolated = FlextSettings(log_level="CRITICAL")
        tm.ok(isolated is not settings)
        tm.that(settings.model_dump(), eq=original)

    @staticmethod
    @pytest.mark.parametrize(
        "overrides",
        [
            {"debug": "not-a-bool"},
            {"unknown_field": True},
            {"log_level": "invalid-level"},
            {"window": {"lower": 10}},
            {"window": {"upper": "not-an-int"}},
        ],
    )
    def test_invalid_updates_preserve_root_and_nested_state(
        overrides: t.JsonMapping,
    ) -> None:
        """Scalar, unknown-key and nested-validator failures are transactional."""

        class Window(m.BaseModel):
            lower: int = 1
            upper: int = 5

            @m.model_validator(mode="after")
            def validate_order(self) -> Self:
                if self.lower >= self.upper:
                    msg = "lower must be below upper"
                    raise ValueError(msg)
                return self

        class NestedSettings(FlextSettings):
            window: Window = m.Field(default_factory=Window)

        published = NestedSettings.fetch_global()
        nested = published.window
        original = published.model_dump()
        with pytest.raises(ValueError, match=r"."):
            NestedSettings.update_global(**overrides)
        tm.ok(published is NestedSettings.fetch_global())
        tm.ok(published.window is nested)
        tm.that(published.model_dump(), eq=original)
        with pytest.raises(ValueError, match=r"."):
            published.clone(**overrides)
        tm.that(published.model_dump(), eq=original)

    @staticmethod
    def test_subclasses_and_nested_clones_are_isolated() -> None:
        """Each concrete class owns its root; clone graphs never share mutation."""

        class Labels(m.BaseModel):
            names: list[str] = m.Field(default_factory=list)

        class ParentSettings(FlextSettings):
            labels: Labels = m.Field(default_factory=Labels)

        parent = ParentSettings.fetch_global()

        class ChildSettings(ParentSettings):
            pass

        class SiblingSettings(ParentSettings):
            pass

        child = ChildSettings.fetch_global()
        sibling = SiblingSettings.fetch_global()
        snapshot = child.clone(labels={"names": ["snapshot"]})
        plain_snapshot = child.clone()
        derived = ChildSettings.fetch_global(overrides={})
        ChildSettings.update_global(labels={"names": ["global"]})
        tm.ok(child is ChildSettings.fetch_global())
        tm.that(snapshot.labels.names, eq=["snapshot"])
        for isolated in (parent, sibling, plain_snapshot, derived):
            tm.that(isolated.labels.names, eq=[])
            tm.ok(isolated is not child)
        ChildSettings.reset_for_testing()
        tm.ok(child is ChildSettings.fetch_global())
        tm.ok(parent is ParentSettings.fetch_global())
        tm.ok(sibling is SiblingSettings.fetch_global())

    @staticmethod
    def test_unused_classes_stay_lazy_after_reset_and_failed_publication() -> None:
        """Reset, snapshots and invalid first writes never publish a candidate."""

        class LazySettings(FlextSettings):
            marker: str = "unselected"

        with u.Tests.env_vars_context(env_vars={"FLEXT_MARKER": "first"}):
            LazySettings.reset_for_testing()
            snapshot = LazySettings.fetch_global(overrides={})
            tm.that(snapshot.marker, eq="first")
            with pytest.raises(m.ValidationError, match="Invalid log level"):
                LazySettings.update_global(log_level="invalid-level")
            with pytest.raises(m.ValidationError, match="Invalid log level"):
                LazySettings(log_level="invalid-level")
        with u.Tests.env_vars_context(env_vars={"FLEXT_MARKER": "second"}):
            published = LazySettings.fetch_global()
        tm.that(published.marker, eq="second")
        tm.ok(published is not snapshot)

    @staticmethod
    def test_dotenv_repoint_reloads_sources_without_rebinding(tmp_path: Path) -> None:
        """FLEXT_ENV_FILE is resolved at reload, not frozen at module import."""
        env_file = tmp_path / "selected.env"
        selected_timezone = "Etc/GMT+3"
        env_file.write_text(f"FLEXT_TIMEZONE={selected_timezone}\n", encoding="utf-8")
        original = settings.model_dump()
        try:
            with u.Tests.env_vars_context(
                env_vars={"FLEXT_ENV_FILE": str(env_file)},
                vars_to_clear=("FLEXT_TIMEZONE",),
            ):
                FlextSettings.reset_for_testing()
                tm.ok(settings is FlextSettings.fetch_global())
                tm.that(settings.timezone, eq=selected_timezone)
        finally:
            FlextSettings.reset_for_testing()
        tm.that(settings.model_dump(), eq=original)


TestsFlextCoreSettingsWorkDir = TestsFlextCoreSettings.TestsFlextCoreSettingsWorkDir
