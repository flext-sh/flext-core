"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import override

from flext_core import FlextSettings, c, u

from .shared import Examples


class Ex02FlextSettings(Examples):
    """Golden-file tests for ``FlextSettings`` public API."""

    class _TestConfig(FlextSettings):
        """Subclass with extra fields used across namespace/context exercises."""

        service_name: str = "test-service"
        feature_enabled: bool = True

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        self._exercise_singleton_and_global()
        self._exercise_configuration_fields_and_validation()
        self._exercise_effective_log_level_and_override()
        self._exercise_get_global_and_provider()
        self._exercise_resolve_env_file_and_auto_config()
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
        global_instance = FlextSettings.get_global()
        self.check("get_global.identity", global_instance is first)
        getattr(FlextSettings, "_reset_instance")()
        third = FlextSettings()
        self.check("_reset_instance.recreates_singleton", third is not first)
        FlextSettings.reset_for_testing()
        fourth = FlextSettings.get_global()
        self.check("reset_for_testing.recreates_global", fourth is not third)
        getattr(self._TestConfig, "_reset_instance")()
        test_a = self._TestConfig()
        test_b = self._TestConfig()
        self.check("subclass.singleton_identity", test_a is test_b)

    def _exercise_configuration_fields_and_validation(self) -> None:
        """Exercise all configuration fields and validation."""
        self.section("configuration_fields_and_validation")
        FlextSettings.reset_for_testing()
        config = FlextSettings(
            app_name="demo-app",
            version="9.8.7",
            debug=True,
            trace=False,
            log_level=c.Settings.LogLevel.INFO,
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
            exception_failure_level=c.Exceptions.FAILURE_LEVEL_DEFAULT,
        )
        self.check("field.app_name", config.app_name)
        self.check("field.version", config.version)
        self.check("field.debug", config.debug)
        self.check("field.trace", config.trace)
        self.check("field.log_level", config.log_level)
        self.check("field.async_logging", config.async_logging)
        self.check("field.enable_caching", config.enable_caching)
        self.check("field.cache_ttl", config.cache_ttl)
        self.check("field.database_url", config.database_url)
        self.check("field.database_pool_size", config.database_pool_size)
        self.check("field.circuit_breaker_threshold", config.circuit_breaker_threshold)
        self.check("field.rate_limit_max_requests", config.rate_limit_max_requests)
        self.check("field.rate_limit_window_seconds", config.rate_limit_window_seconds)
        self.check("field.retry_delay", config.retry_delay)
        self.check("field.max_retry_attempts", config.max_retry_attempts)
        self.check("field.enable_timeout_executor", config.enable_timeout_executor)
        self.check("field.dispatcher_enable_logging", config.dispatcher_enable_logging)
        self.check("field.dispatcher_auto_context", config.dispatcher_auto_context)
        self.check(
            "field.dispatcher_timeout_seconds", config.dispatcher_timeout_seconds
        )
        self.check("field.dispatcher_enable_metrics", config.dispatcher_enable_metrics)
        self.check("field.executor_workers", config.executor_workers)
        self.check("field.timeout_seconds", config.timeout_seconds)
        self.check("field.max_workers", config.max_workers)
        self.check("field.max_batch_size", config.max_batch_size)
        self.check("field.api_key", config.api_key)
        self.check("field.exception_failure_level", config.exception_failure_level)
        validated = FlextSettings.model_validate(config.model_dump())
        self.check(
            "validate_configuration.indirect_via_model_validate", validated.app_name
        )

    def _exercise_effective_log_level_and_override(self) -> None:
        """Exercise effective_log_level property and apply_override."""
        self.section("effective_log_level_and_override")
        FlextSettings.reset_for_testing()
        config = FlextSettings(
            debug=False, trace=False, log_level=c.Settings.LogLevel.ERROR
        )
        self.check("effective_log_level.base", config.effective_log_level)
        valid_override = config.apply_override("debug", True)
        self.check("apply_override.valid_return", valid_override)
        self.check("apply_override.valid_applied", config.debug)
        self.check("effective_log_level.debug", config.effective_log_level)
        config.apply_override("trace", True)
        self.check("effective_log_level.trace", config.effective_log_level)
        invalid_override = config.apply_override("does_not_exist", "x")
        self.check("apply_override.invalid_return", invalid_override)

    def _exercise_get_global_and_provider(self) -> None:
        """Exercise get_global and DI config provider."""
        self.section("get_global_and_provider")
        FlextSettings.reset_for_testing()
        base = FlextSettings.get_global()
        cloned = FlextSettings.get_global()
        self.check("get_global.clone_same_values", cloned.app_name == base.app_name)
        self.check("get_global.clone_new_object", cloned is base)
        overridden = FlextSettings.get_global(
            overrides={"app_name": "materialized", "timeout_seconds": 55.0}
        )
        self.check("get_global.override.app_name", overridden.app_name)
        self.check("get_global.override.timeout_seconds", overridden.timeout_seconds)
        provider = overridden.get_di_config_provider()
        self.check("get_di_config_provider.type", type(provider).__name__)

    def _exercise_resolve_env_file_and_auto_config(self) -> None:
        """Exercise resolve_env_file and AutoConfig."""
        self.section("resolve_env_file_and_auto_config")
        FlextSettings.reset_for_testing()
        env_path = Path(sys.prefix) / "flext_settings_example.env"
        env_path.write_text("FLEXT_APP_NAME=from_env_file\n", encoding="utf-8")
        previous = self._set_env(c.Platform.ENV_FILE_ENV_VAR, str(env_path))
        try:
            resolved = u.resolve_env_file()
            self.check("resolve_env_file.custom_path", resolved)
            auto = FlextSettings.AutoConfig(
                config_class=self._TestConfig,
                env_prefix=c.Platform.ENV_PREFIX,
                env_file=resolved,
            )
            created = auto.create_config()
            self.check("AutoConfig.create_config.type", type(created).__name__)
            self.check(
                "AutoConfig.create_config.service_name",
                created.model_dump().get("service_name"),
            )
        finally:
            self._restore_env(c.Platform.ENV_FILE_ENV_VAR, previous)
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
        decorated_typed = base.get_namespace("decorated_ns", tc)
        registered_typed = base.get_namespace("registered_ns", tc)
        self.check("get_namespace.decorated.service_name", decorated_typed.service_name)
        self.check(
            "get_namespace.registered.service_name", registered_typed.service_name
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
            "register_context_overrides.service_name", from_registered.service_name
        )
        self.check(
            "for_context.registered.timeout_seconds", from_registered.timeout_seconds
        )
        self.check("for_context.registered.max_workers", from_registered.max_workers)
        self.check(
            "for_context.runtime_override.feature_enabled", from_runtime.feature_enabled
        )


if __name__ == "__main__":
    Ex02FlextSettings(__file__).run()
