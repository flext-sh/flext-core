"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import override

from flext_core import FlextSettings, c, u

from .shared import ExamplesFlextCoreShared


class Ex02FlextSettings(ExamplesFlextCoreShared):
    """Golden-file tests for ``FlextSettings`` public API."""

    class _TestConfig(FlextSettings):
        """Subclass with extra fields used across namespace/context exercises."""

        service_name: str = "test-service"
        feature_enabled: bool = True

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        self._exercise_singleton_and_global()
        self._exercise_settings_fields_and_validation()
        self._exercise_effective_log_level_and_override()
        self._exercise_fetch_global_and_provider()
        self._exercise_resolve_env_file_and_auto_settings()
        self._exercise_namespace_system()
        self._exercise_context_system()
        FlextSettings.reset_for_testing()
        self._TestConfig.reset_for_testing()

    @staticmethod
    def _set_env(key: str, value: str | None) -> str | None:
        """Set an env var, returning its previous value."""
        previous = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        return previous

    @staticmethod
    def _restore_env(key: str, previous: str | None) -> None:
        """Restore an env var to its previous value."""
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous

    def _exercise_singleton_and_global(self) -> None:
        """Exercise singleton pattern and global instance management."""
        self.section("singleton_and_global")
        FlextSettings.reset_for_testing()
        first = FlextSettings()
        second = FlextSettings()
        self.check("constructor.singleton_identity", first is second)
        global_instance = FlextSettings.fetch_global()
        self.check("fetch_global.identity", global_instance is first)
        FlextSettings.reset_for_testing()
        third = FlextSettings()
        self.check("reset_for_testing.recreates_singleton", third is not first)
        FlextSettings.reset_for_testing()
        fourth = FlextSettings.fetch_global()
        self.check("reset_for_testing.recreates_global", fourth is not third)
        self._TestConfig.reset_for_testing()
        test_a = self._TestConfig()
        test_b = self._TestConfig()
        self.check("subclass.singleton_identity", test_a is test_b)

    def _exercise_settings_fields_and_validation(self) -> None:
        """Exercise all settings fields and validation."""
        self.section("settings_fields_and_validation")
        FlextSettings.reset_for_testing()
        settings = FlextSettings(
            app_name="demo-app",
            version="9.8.7",
            debug=True,
            trace=False,
            log_level=c.LogLevel.INFO,
            async_logging=False,
            enable_caching=False,
            cache_ttl=321,
            database_url="sqlite:///tmp/demo.db",
            database_pool_size=11,
            circuit_breaker_threshold=4,
            rate_limit_max_requests=77,
            rate_limit_window_seconds=42,
            retry_delay=5,
            max_retry_attempts=6,
            enable_timeout_executor=False,
            dispatcher_enable_logging=False,
            dispatcher_auto_context=False,
            dispatcher_timeout_seconds=2.5,
            dispatcher_enable_metrics=False,
            executor_workers=3,
            timeout_seconds=13.0,
            max_workers=7,
            max_batch_size=29,
            api_key="demo-key",
            exception_failure_level=c.FAILURE_LEVEL_DEFAULT,
        )
        self.check("field.app_name", settings.app_name)
        self.check("field.version", settings.version)
        self.check("field.debug", settings.debug)
        self.check("field.trace", settings.trace)
        self.check("field.log_level", settings.log_level)
        self.check("field.async_logging", settings.async_logging)
        self.check("field.enable_caching", settings.enable_caching)
        self.check("field.cache_ttl", settings.cache_ttl)
        self.check("field.database_url", settings.database_url)
        self.check("field.database_pool_size", settings.database_pool_size)
        self.check(
            "field.circuit_breaker_threshold", settings.circuit_breaker_threshold
        )
        self.check("field.rate_limit_max_requests", settings.rate_limit_max_requests)
        self.check(
            "field.rate_limit_window_seconds", settings.rate_limit_window_seconds
        )
        self.check("field.retry_delay", settings.retry_delay)
        self.check("field.max_retry_attempts", settings.max_retry_attempts)
        self.check("field.enable_timeout_executor", settings.enable_timeout_executor)
        self.check(
            "field.dispatcher_enable_logging", settings.dispatcher_enable_logging
        )
        self.check("field.dispatcher_auto_context", settings.dispatcher_auto_context)
        self.check(
            "field.dispatcher_timeout_seconds",
            settings.dispatcher_timeout_seconds,
        )
        self.check(
            "field.dispatcher_enable_metrics", settings.dispatcher_enable_metrics
        )
        self.check("field.executor_workers", settings.executor_workers)
        self.check("field.timeout_seconds", settings.timeout_seconds)
        self.check("field.max_workers", settings.max_workers)
        self.check("field.max_batch_size", settings.max_batch_size)
        self.check("field.api_key", settings.api_key)
        self.check("field.exception_failure_level", settings.exception_failure_level)
        validated = FlextSettings.model_validate(settings.model_dump())
        self.check(
            "validate_configuration.indirect_via_model_validate",
            validated.app_name,
        )

    def _exercise_effective_log_level_and_override(self) -> None:
        """Exercise effective_log_level property and apply_override."""
        self.section("effective_log_level_and_override")
        FlextSettings.reset_for_testing()
        settings = FlextSettings(debug=False, trace=False, log_level=c.LogLevel.ERROR)
        self.check("effective_log_level.base", settings.effective_log_level)
        valid_override = settings.apply_override("debug", True)
        self.check("apply_override.valid_return", valid_override)
        self.check("apply_override.valid_applied", settings.debug)
        self.check("effective_log_level.debug", settings.effective_log_level)
        settings.apply_override("trace", True)
        self.check("effective_log_level.trace", settings.effective_log_level)
        invalid_override = settings.apply_override("does_not_exist", "x")
        self.check("apply_override.invalid_return", invalid_override)

    def _exercise_fetch_global_and_provider(self) -> None:
        """Exercise fetch_global and DI settings provider."""
        self.section("fetch_global_and_provider")
        FlextSettings.reset_for_testing()
        base = FlextSettings.fetch_global()
        cloned = FlextSettings.fetch_global()
        self.check("fetch_global.clone_same_values", cloned.app_name == base.app_name)
        self.check("fetch_global.clone_new_object", cloned is base)
        overridden = FlextSettings.fetch_global(
            overrides={"app_name": "materialized", "timeout_seconds": 55.0},
        )
        self.check("fetch_global.override.app_name", overridden.app_name)
        self.check("fetch_global.override.timeout_seconds", overridden.timeout_seconds)
        provider = overridden.resolve_di_settings_provider()
        self.check("resolve_di_settings_provider.type", type(provider).__name__)

    def _exercise_resolve_env_file_and_auto_settings(self) -> None:
        """Exercise resolve_env_file and AutoSettings."""
        self.section("resolve_env_file_and_auto_settings")
        FlextSettings.reset_for_testing()
        env_path = Path(__file__).with_name("flext_settings_example.env")
        env_path.write_text("FLEXT_APP_NAME=from_env_file\n", encoding="utf-8")
        previous = self._set_env(c.ENV_FILE_ENV_VAR, str(env_path))
        try:
            resolved = u.resolve_env_file()
            self.check("resolve_env_file.custom_path", resolved)
            auto = FlextSettings.AutoSettings(
                settings_class=self._TestConfig,
                env_prefix=c.ENV_PREFIX,
                env_file=resolved,
            )
            created = auto.create_settings()
            self.check("AutoSettings.create_settings.type", type(created).__name__)
            self.check(
                "AutoSettings.create_settings.service_name",
                created.model_dump().get("service_name"),
            )
        finally:
            self._restore_env(c.ENV_FILE_ENV_VAR, previous)
            if env_path.exists():
                env_path.unlink()

    def _exercise_namespace_system(self) -> None:
        """Exercise namespace registration and retrieval."""
        self.section("namespace_system")
        FlextSettings.reset_for_testing()
        tc = Ex02FlextSettings._TestConfig

        class _DecoratedNamespace(Ex02FlextSettings._TestConfig):
            service_name: str = "decorated"

        class _RegisteredNamespace(Ex02FlextSettings._TestConfig):
            service_name: str = "registered"

        FlextSettings.register_namespace("decorated_ns", _DecoratedNamespace)
        FlextSettings.register_namespace("registered_ns", _RegisteredNamespace)
        base = FlextSettings()
        decorated_typed = base.fetch_namespace("decorated_ns", tc)
        registered_typed = base.fetch_namespace("registered_ns", tc)
        self.check(
            "fetch_namespace.decorated.service_name",
            decorated_typed.service_name,
        )
        self.check(
            "fetch_namespace.registered.service_name",
            registered_typed.service_name,
        )

    def _exercise_context_system(self) -> None:
        """Exercise context override system."""
        self.section("context_system")
        self._TestConfig.reset_for_testing()
        self._TestConfig.register_context_overrides(
            "worker-a",
            service_name="worker-service",
            timeout_seconds=41.0,
            max_workers=2,
        )
        from_registered = self._TestConfig.for_context("worker-a")
        from_runtime = self._TestConfig.for_context("worker-a", feature_enabled=False)
        self.check(
            "register_context_overrides.service_name",
            from_registered.service_name,
        )
        self.check(
            "for_context.registered.timeout_seconds",
            from_registered.timeout_seconds,
        )
        self.check("for_context.registered.max_workers", from_registered.max_workers)
        self.check(
            "for_context.runtime_override.feature_enabled",
            from_runtime.feature_enabled,
        )


if __name__ == "__main__":
    Ex02FlextSettings(caller_file=Path(__file__)).run()
