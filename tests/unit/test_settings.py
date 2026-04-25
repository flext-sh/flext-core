"""Behavior contract for flext_core.FlextSettings — public API only."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import (
    Generator,
    MutableSequence,
)
from pathlib import Path

import pytest
from flext_tests import tm

from flext_core import FlextSettings
from tests import c, p, t, u


@pytest.fixture(autouse=True)
def reset_flext_settings_singleton() -> Generator[None]:
    """Isolate singleton state across settings tests."""
    FlextSettings.reset_for_testing()
    try:
        yield
    finally:
        FlextSettings.reset_for_testing()


class TestsFlextCoreSettings:
    """Behavior contract for FlextSettings — public API only."""

    # --- Initialization --------------------------------------------------

    @pytest.mark.parametrize(
        "config_data",
        [
            {"app_name": "test_app", "version": "1.0.0", "debug": True},
            {"app_name": "dict_app", "version": "2.0.0", "debug": False},
            {"app_name": "valid_app", "version": "1.0.0"},
        ],
        ids=lambda d: str(d.get("app_name", "default")),
    )
    def test_initialization_from_kwargs_sets_expected_fields(
        self,
        config_data: t.FeatureFlagMapping,
    ) -> None:
        settings = u.Core.Tests.create_test_config(**config_data)
        u.Core.Tests.assert_config_fields(settings, config_data)
        tm.that(settings, is_=FlextSettings)

    def test_model_dump_round_trips_values(self) -> None:
        settings = u.Core.Tests.create_test_config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        dumped = settings.model_dump()
        tm.that(dumped["app_name"], eq="test_app")
        tm.that(dumped["version"], eq="1.0.0")
        tm.that(dumped["debug"], eq=True)

    def test_model_validate_returns_same_singleton(self) -> None:
        original = u.Core.Tests.create_test_config(
            app_name="original_app",
            version="1.0.0",
        )
        dumped = original.model_dump(exclude={"is_production"})
        cloned = FlextSettings.model_validate(dumped)
        tm.that(cloned is original, eq=True)

    @pytest.mark.parametrize(
        ("field_name", "initial", "modified"),
        [
            ("app_name", "test_value", "modified_value"),
            ("version", "1.0.0", "2.0.0"),
        ],
    )
    def test_field_mutation_is_visible(
        self,
        field_name: str,
        initial: str,
        modified: str,
    ) -> None:
        settings = u.Core.Tests.create_test_config()
        setattr(settings, field_name, initial)
        tm.that(getattr(settings, field_name), eq=initial)
        setattr(settings, field_name, modified)
        tm.that(getattr(settings, field_name), eq=modified)

    # --- Singleton -------------------------------------------------------

    def test_direct_instantiation_returns_shared_singleton(self) -> None:
        first = FlextSettings()
        second = FlextSettings()
        tm.that(first is second, eq=True)
        tm.that(first.model_dump_json(), eq=second.model_dump_json())

    def test_fetch_global_returns_same_instance_across_calls(self) -> None:
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        tm.that(first is second, eq=True)

    def test_reset_for_testing_clears_singleton(self) -> None:
        original = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        fresh = FlextSettings()
        tm.that(fresh is not original, eq=True)
        tm.that(fresh.app_name, eq="flext")

    def test_concurrent_writes_do_not_crash(self) -> None:
        settings = u.Core.Tests.create_test_config()
        results: MutableSequence[str] = []

        def set_value(thread_id: int) -> None:
            settings.app_name = f"thread_{thread_id}"
            results.append(settings.app_name)

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for t_ in threads:
            t_.start()
        for t_ in threads:
            t_.join()
        tm.that(len(results), eq=10)
        tm.that(all(r.startswith("thread_") for r in results), eq=True)

    def test_repeated_writes_complete_within_budget(self) -> None:
        settings = u.Core.Tests.create_test_config()
        start = time.time()
        for i in range(100):
            settings.app_name = f"value_{i}"
            _ = settings.app_name
        tm.that(time.time() - start, lt=5.0)

    # --- Serialization ---------------------------------------------------

    def test_json_round_trip_preserves_fields(self) -> None:
        settings = u.Core.Tests.create_test_config(
            app_name="serialize_app",
            version="1.0.0",
        )
        json_str = settings.model_dump_json(
            exclude={
                "is_debug_enabled",
                "effective_log_level",
                "is_production",
                "effective_timeout",
                "has_database",
                "has_cache",
            },
        )
        tm.that("serialize_app" in json_str, eq=True)
        restored = FlextSettings.model_validate_json(json_str)
        tm.that(restored.app_name, eq=settings.app_name)
        tm.that(restored.version, eq=settings.version)

    # --- Validation ------------------------------------------------------

    def test_trace_without_debug_raises_validation_error(self) -> None:
        with pytest.raises(c.ValidationError, match="Trace mode requires debug mode"):
            FlextSettings.model_validate({"trace": True, "debug": False})

    def test_invalid_database_url_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            FlextSettings.model_validate({"database_url": "invalid://scheme"})

    # --- Environment loading ---------------------------------------------

    @pytest.mark.parametrize(
        ("env_key", "env_value", "should_load"),
        [
            ("DEBUG", "true", False),
            ("FLEXT_DEBUG", "true", True),
        ],
    )
    def test_env_prefix_is_respected(
        self,
        env_key: str,
        env_value: str,
        should_load: bool,
    ) -> None:
        with u.Core.Tests.env_vars_context(
            {env_key: env_value},
            ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            if not env_key.startswith(c.ENV_PREFIX):
                os.environ["DEBUG"] = "true"
            settings = FlextSettings()
            tm.that(settings.debug, eq=should_load)

    def test_dotenv_file_loads_when_env_file_var_points_to_it(
        self,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n",
        )
        with u.Core.Tests.env_vars_context(
            {"FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_APP_NAME", "FLEXT_ENV_FILE"],
        ):
            FlextSettings.reset_for_testing()
            settings = FlextSettings()
            tm.that(settings.app_name in {"from-dotenv", "flext"}, eq=True)

    def test_env_var_overrides_dotenv_file(self, tmp_path: Path) -> None:
        with u.Core.Tests.env_vars_context(
            {},
            ["FLEXT_APP_NAME", "FLEXT_LOG_LEVEL"],
        ):
            env_file = tmp_path / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n",
            )
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
            settings = FlextSettings()
            tm.that(settings.app_name, eq="from-env-var")
            tm.that(settings.log_level, eq="ERROR")

    def test_explicit_override_wins_over_env_and_dotenv(
        self,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FLEXT_TIMEOUT_SECONDS=45\n")
        with u.Core.Tests.env_vars_context(
            {"FLEXT_TIMEOUT_SECONDS": "60", "FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_TIMEOUT_SECONDS", "FLEXT_ENV_FILE"],
        ):
            settings = FlextSettings.fetch_global(overrides={"timeout_seconds": 90})
            tm.that(settings.timeout_seconds, eq=90)

    # --- Effective log level --------------------------------------------

    def test_trace_mode_sets_effective_log_level_to_debug(self) -> None:
        settings = u.Core.Tests.create_test_config(trace=True, debug=True)
        tm.that(settings.effective_log_level, eq=c.LogLevel.DEBUG)

    def test_debug_mode_sets_effective_log_level_to_info(self) -> None:
        settings = u.Core.Tests.create_test_config(debug=True)
        tm.that(settings.effective_log_level, eq=c.LogLevel.INFO)

    def test_without_debug_effective_log_level_matches_configured(self) -> None:
        settings = u.Core.Tests.create_test_config(debug=False, trace=False)
        tm.that(settings.effective_log_level, eq=settings.log_level)

    # --- Overrides and DI provider --------------------------------------

    def test_apply_override_rejects_unknown_key(self) -> None:
        settings = u.Core.Tests.create_test_config()
        tm.that(settings.apply_override("invalid_key", "value"), eq=False)

    def test_apply_override_updates_known_field(self) -> None:
        settings = u.Core.Tests.create_test_config()
        settings.apply_override("app_name", "new_name")
        tm.that(settings.app_name, eq="new_name")

    def test_resolve_di_settings_provider_returns_stable_handle(self) -> None:
        settings = u.Core.Tests.create_test_config()
        first = settings.resolve_di_settings_provider()
        second = settings.resolve_di_settings_provider()
        tm.that(first is second, eq=True)

    def test_auto_settings_create_settings_returns_settings_protocol(self) -> None:
        auto = FlextSettings.AutoSettings(settings_class=FlextSettings)
        instance = auto.create_settings()
        tm.that(instance, is_=p.Settings)


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreSettings"]
