"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from shared import Examples

from flext_core import FlextSettings, c


def _set_env(key: str, value: str | None) -> str | None:
    previous = os.environ.get(key)
    if value is None:
        _ = os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return previous


def _restore_env(key: str, previous: str | None) -> None:
    if previous is None:
        _ = os.environ.pop(key, None)
    else:
        os.environ[key] = previous


class _TestConfig(FlextSettings):
    service_name: str = "test-service"
    feature_enabled: bool = True


class Ex02FlextSettings(Examples):
    """Exercise FlextSettings public API."""

    def exercise(self) -> None:
        """Run all settings demonstrations and verify golden file."""
        self.demo_singleton_and_global()
        self.demo_configuration_fields_and_validation()
        self.demo_effective_log_level_and_override()
        self.demo_materialize_and_provider()
        self.demo_resolve_env_file_and_auto_config()
        self.demo_namespace_system()
        self.demo_context_system()
        FlextSettings.reset_global_instance()
        _TestConfig.reset_global_instance()

    def demo_singleton_and_global(self) -> None:
        """Exercise singleton pattern and global instance management."""
        self.section("singleton_and_global")
        FlextSettings.reset_global_instance()

        first = FlextSettings()
        second = FlextSettings()
        self.check("constructor.singleton_identity", first is second)

        global_instance = FlextSettings.get_global_instance()
        self.check("get_global_instance.identity", global_instance is first)

        getattr(FlextSettings, "_reset_instance")()
        third = FlextSettings()
        self.check("_reset_instance.recreates_singleton", third is not first)

        FlextSettings.reset_global_instance()
        fourth = FlextSettings.get_global_instance()
        self.check("reset_global_instance.recreates_global", fourth is not third)

        getattr(_TestConfig, "_reset_instance")()
        test_a = _TestConfig()
        test_b = _TestConfig()
        self.check("subclass.singleton_identity", test_a is test_b)

    def demo_configuration_fields_and_validation(self) -> None:
        """Exercise all configuration fields and validation."""
        self.section("configuration_fields_and_validation")
        FlextSettings.reset_global_instance()

        app_name = self.rand_str(8)
        version = f"{self.rand_int(1, 9)}.{self.rand_int(0, 9)}.{self.rand_int(0, 9)}"
        debug = self.rand_bool()
        trace = self.rand_bool()
        cache_ttl = self.rand_int(1, 1000)
        db_pool_size = self.rand_int(1, 50)
        cb_threshold = self.rand_int(1, 10)
        rate_limit_max = self.rand_int(1, 200)
        rate_limit_window = self.rand_int(1, 120)
        retry_delay = self.rand_int(1, 20)
        max_retry_attempts = self.rand_int(1, 10)
        dispatcher_timeout = float(self.rand_int(1, 30))
        executor_workers = self.rand_int(1, 16)
        timeout_seconds = float(self.rand_int(1, 120))
        max_workers = self.rand_int(1, 64)
        max_batch_size = self.rand_int(1, 200)
        api_key = self.rand_str(12)
        db_name = self.rand_str(8)
        database_url = f"sqlite:///tmp/{db_name}.db"

        config = FlextSettings(
            app_name=app_name,
            version=version,
            debug=debug,
            trace=trace,
            log_level=c.Settings.LogLevel.INFO,
            async_logging=False,
            enable_caching=False,
            cache_ttl=cache_ttl,
            database_url=database_url,
            database_pool_size=db_pool_size,
            circuit_breaker_threshold=cb_threshold,
            rate_limit_max_requests=rate_limit_max,
            rate_limit_window_seconds=rate_limit_window,
            retry_delay=retry_delay,
            max_retry_attempts=max_retry_attempts,
            enable_timeout_executor=False,
            dispatcher_enable_logging=False,
            dispatcher_auto_context=False,
            dispatcher_timeout_seconds=dispatcher_timeout,
            dispatcher_enable_metrics=False,
            executor_workers=executor_workers,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            max_batch_size=max_batch_size,
            api_key=api_key,
            exception_failure_level=c.Exceptions.FAILURE_LEVEL_DEFAULT,
        )

        self.check("field.app_name", config.app_name)
        self.check("field.app_name_matches", config.app_name == app_name)
        self.check("field.version", config.version)
        self.check("field.version_matches", config.version == version)
        self.check("field.debug", config.debug)
        self.check("field.debug_matches", config.debug == debug)
        self.check("field.trace", config.trace)
        self.check("field.trace_matches", config.trace == trace)
        self.check("field.log_level", config.log_level)
        self.check(
            "field.log_level_matches", config.log_level == c.Settings.LogLevel.INFO
        )
        self.check("field.async_logging", config.async_logging)
        self.check("field.enable_caching", config.enable_caching)
        self.check("field.cache_ttl", config.cache_ttl)
        self.check("field.cache_ttl_matches", config.cache_ttl == cache_ttl)
        self.check("field.database_url", config.database_url)
        self.check("field.database_url_matches", config.database_url == database_url)
        self.check("field.database_pool_size", config.database_pool_size)
        self.check(
            "field.database_pool_size_matches",
            config.database_pool_size == db_pool_size,
        )
        self.check("field.circuit_breaker_threshold", config.circuit_breaker_threshold)
        self.check(
            "field.circuit_breaker_threshold_matches",
            config.circuit_breaker_threshold == cb_threshold,
        )
        self.check("field.rate_limit_max_requests", config.rate_limit_max_requests)
        self.check(
            "field.rate_limit_max_requests_matches",
            config.rate_limit_max_requests == rate_limit_max,
        )
        self.check("field.rate_limit_window_seconds", config.rate_limit_window_seconds)
        self.check(
            "field.rate_limit_window_seconds_matches",
            config.rate_limit_window_seconds == rate_limit_window,
        )
        self.check("field.retry_delay", config.retry_delay)
        self.check("field.retry_delay_matches", config.retry_delay == retry_delay)
        self.check("field.max_retry_attempts", config.max_retry_attempts)
        self.check(
            "field.max_retry_attempts_matches",
            config.max_retry_attempts == max_retry_attempts,
        )
        self.check("field.enable_timeout_executor", config.enable_timeout_executor)
        self.check("field.dispatcher_enable_logging", config.dispatcher_enable_logging)
        self.check("field.dispatcher_auto_context", config.dispatcher_auto_context)
        self.check(
            "field.dispatcher_timeout_seconds", config.dispatcher_timeout_seconds
        )
        self.check(
            "field.dispatcher_timeout_seconds_matches",
            config.dispatcher_timeout_seconds == dispatcher_timeout,
        )
        self.check("field.dispatcher_enable_metrics", config.dispatcher_enable_metrics)
        self.check("field.executor_workers", config.executor_workers)
        self.check(
            "field.executor_workers_matches",
            config.executor_workers == executor_workers,
        )
        self.check("field.timeout_seconds", config.timeout_seconds)
        self.check(
            "field.timeout_seconds_matches", config.timeout_seconds == timeout_seconds
        )
        self.check("field.max_workers", config.max_workers)
        self.check("field.max_workers_matches", config.max_workers == max_workers)
        self.check("field.max_batch_size", config.max_batch_size)
        self.check(
            "field.max_batch_size_matches", config.max_batch_size == max_batch_size
        )
        self.check("field.api_key", config.api_key)
        self.check("field.api_key_matches", config.api_key == api_key)
        self.check("field.exception_failure_level", config.exception_failure_level)

        validated = FlextSettings.model_validate(config.model_dump())
        self.check(
            "validate_configuration.indirect_via_model_validate", validated.app_name
        )
        self.check(
            "validate_configuration.indirect_via_model_validate_matches",
            validated.app_name == app_name,
        )

    def demo_effective_log_level_and_override(self) -> None:
        """Exercise effective_log_level property and apply_override."""
        self.section("effective_log_level_and_override")
        FlextSettings.reset_global_instance()

        config = FlextSettings(
            debug=False,
            trace=False,
            log_level=c.Settings.LogLevel.ERROR,
        )
        self.check("effective_log_level.base", config.effective_log_level)
        self.check(
            "effective_log_level.base_matches",
            config.effective_log_level == c.Settings.LogLevel.ERROR,
        )

        valid_override = config.apply_override("debug", True)
        self.check("apply_override.valid_return", valid_override)
        self.check("apply_override.valid_applied", config.debug)
        self.check("effective_log_level.debug", config.effective_log_level)

        _ = config.apply_override("trace", True)
        self.check("effective_log_level.trace", config.effective_log_level)

        invalid_override_value = self.rand_str(4)
        invalid_override = config.apply_override(
            "does_not_exist", invalid_override_value
        )
        self.check("apply_override.invalid_return", invalid_override)

    def demo_materialize_and_provider(self) -> None:
        """Exercise materialize and DI config provider."""
        self.section("materialize_and_provider")
        FlextSettings.reset_global_instance()

        base = FlextSettings.get_global_instance()
        cloned = FlextSettings.materialize()
        self.check("materialize.clone_same_values", cloned.app_name == base.app_name)
        self.check("materialize.clone_new_object", cloned is not base)

        materialized_name = self.rand_str(10)
        materialized_timeout = float(self.rand_int(10, 100))
        overridden = FlextSettings.materialize(
            config_overrides={
                "app_name": materialized_name,
                "timeout_seconds": materialized_timeout,
            }
        )
        self.check("materialize.override.app_name", overridden.app_name)
        self.check(
            "materialize.override.app_name_matches",
            overridden.app_name == materialized_name,
        )
        self.check("materialize.override.timeout_seconds", overridden.timeout_seconds)
        self.check(
            "materialize.override.timeout_seconds_matches",
            overridden.timeout_seconds == materialized_timeout,
        )

        provider = overridden.get_di_config_provider()
        self.check("get_di_config_provider.type", type(provider).__name__)
        self.check(
            "get_di_config_provider.type_matches",
            type(provider).__name__ == "ConfigProvider",
        )

    def demo_resolve_env_file_and_auto_config(self) -> None:
        """Exercise resolve_env_file and AutoConfig."""
        self.section("resolve_env_file_and_auto_config")
        FlextSettings.reset_global_instance()

        env_path = Path(sys.prefix) / f"{self.rand_str(10)}.env"
        env_service_name = self.rand_str(10)
        _ = env_path.write_text(
            f"FLEXT_APP_NAME={env_service_name}\n",
            encoding="utf-8",
        )
        previous = _set_env(c.Platform.ENV_FILE_ENV_VAR, str(env_path))
        try:
            resolved = FlextSettings.resolve_env_file()
            self.check("resolve_env_file.custom_path", resolved)
            self.check("resolve_env_file.custom_path_matches", resolved == env_path)

            auto = FlextSettings.AutoConfig(
                config_class=_TestConfig,
                env_prefix=c.Platform.ENV_PREFIX,
                env_file=resolved,
            )
            created = auto.create_config()
            self.check("AutoConfig.create_config.type", type(created).__name__)
            self.check(
                "AutoConfig.create_config.type_matches",
                type(created).__name__ == "_TestConfig",
            )
            self.check(
                "AutoConfig.create_config.service_name",
                created.model_dump().get("service_name"),
            )
            self.check(
                "AutoConfig.create_config.service_name_matches",
                created.model_dump().get("service_name") == _TestConfig().service_name,
            )
        finally:
            _restore_env(c.Platform.ENV_FILE_ENV_VAR, previous)
            if env_path.exists():
                _ = env_path.unlink()

    def demo_namespace_system(self) -> None:
        """Exercise namespace registration and retrieval."""
        self.section("namespace_system")
        FlextSettings.reset_global_instance()

        decorated_service_name = self.rand_str(9)
        registered_service_name = self.rand_str(9)
        decorated_namespace = f"decorated_{self.rand_str(5)}"
        registered_namespace = f"registered_{self.rand_str(5)}"

        @FlextSettings.auto_register(namespace=decorated_namespace)
        class _DecoratedNamespace(_TestConfig):
            service_name: str = decorated_service_name

        class _RegisteredNamespace(_TestConfig):
            service_name: str = registered_service_name

        FlextSettings.register_namespace(registered_namespace, _RegisteredNamespace)
        decorated_raw = FlextSettings.get_namespace_config(decorated_namespace)
        registered_raw = FlextSettings.get_namespace_config(registered_namespace)

        decorated_name = decorated_raw.__name__ if decorated_raw is not None else "None"
        registered_name = (
            registered_raw.__name__ if registered_raw is not None else "None"
        )
        self.check("get_namespace_config.decorated", decorated_name)
        self.check("get_namespace_config.registered", registered_name)
        self.check("auto_register.class_used", _DecoratedNamespace.__name__)
        self.check(
            "auto_register.class_used_matches",
            _DecoratedNamespace.__name__ == decorated_name,
        )

        base = FlextSettings()
        decorated_typed = base.get_namespace(decorated_namespace, _TestConfig)
        registered_typed = base.get_namespace(registered_namespace, _TestConfig)
        self.check("get_namespace.decorated.service_name", decorated_typed.service_name)
        self.check(
            "get_namespace.decorated.service_name_matches",
            decorated_typed.service_name == decorated_service_name,
        )
        self.check(
            "get_namespace.registered.service_name", registered_typed.service_name
        )
        self.check(
            "get_namespace.registered.service_name_matches",
            registered_typed.service_name == registered_service_name,
        )

    def demo_context_system(self) -> None:
        """Exercise context override system."""
        self.section("context_system")
        _TestConfig.reset_global_instance()

        context_name = f"worker-{self.rand_str(4)}"
        context_service_name = self.rand_str(12)
        context_timeout = float(self.rand_int(5, 90))
        context_workers = self.rand_int(1, 8)
        runtime_feature_enabled = self.rand_bool()

        _TestConfig.register_context_overrides(
            context_name,
            service_name=context_service_name,
            timeout_seconds=context_timeout,
            max_workers=context_workers,
        )
        from_registered = _TestConfig.for_context(context_name)
        from_runtime = _TestConfig.for_context(
            context_name,
            feature_enabled=runtime_feature_enabled,
        )

        self.check(
            "register_context_overrides.service_name", from_registered.service_name
        )
        self.check(
            "register_context_overrides.service_name_matches",
            from_registered.service_name == context_service_name,
        )
        self.check(
            "for_context.registered.timeout_seconds", from_registered.timeout_seconds
        )
        self.check(
            "for_context.registered.timeout_seconds_matches",
            from_registered.timeout_seconds == context_timeout,
        )
        self.check("for_context.registered.max_workers", from_registered.max_workers)
        self.check(
            "for_context.registered.max_workers_matches",
            from_registered.max_workers == context_workers,
        )
        self.check(
            "for_context.runtime_override.feature_enabled",
            from_runtime.feature_enabled,
        )
        self.check(
            "for_context.runtime_override.feature_enabled_matches",
            from_runtime.feature_enabled == runtime_feature_enabled,
        )


if __name__ == "__main__":
    Ex02FlextSettings(__file__).run()
