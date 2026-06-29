"""Settings example field checks kept below the module LOC cap."""

from __future__ import annotations

from flext_core import FlextSettings, c

from .shared import ExamplesFlextShared


class Ex02FlextSettingsFieldChecks(ExamplesFlextShared):
    """Field and validation checks for the settings example."""

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
        self.audit_check("field.app_name", settings.app_name)
        self.audit_check("field.version", settings.version)
        self.audit_check("field.debug", settings.debug)
        self.audit_check("field.trace", settings.trace)
        self.audit_check("field.log_level", settings.log_level)
        self.audit_check("field.async_logging", settings.async_logging)
        self.audit_check("field.enable_caching", settings.enable_caching)
        self.audit_check("field.cache_ttl", settings.cache_ttl)
        self.audit_check("field.database_url", settings.database_url)
        self.audit_check("field.database_pool_size", settings.database_pool_size)
        self.audit_check(
            "field.circuit_breaker_threshold", settings.circuit_breaker_threshold
        )
        self.audit_check(
            "field.rate_limit_max_requests", settings.rate_limit_max_requests
        )
        self.audit_check(
            "field.rate_limit_window_seconds", settings.rate_limit_window_seconds
        )
        self.audit_check("field.retry_delay", settings.retry_delay)
        self.audit_check("field.max_retry_attempts", settings.max_retry_attempts)
        self.audit_check(
            "field.enable_timeout_executor", settings.enable_timeout_executor
        )
        self.audit_check(
            "field.dispatcher_enable_logging", settings.dispatcher_enable_logging
        )
        self.audit_check(
            "field.dispatcher_auto_context", settings.dispatcher_auto_context
        )
        self.audit_check(
            "field.dispatcher_timeout_seconds",
            settings.dispatcher_timeout_seconds,
        )
        self.audit_check(
            "field.dispatcher_enable_metrics", settings.dispatcher_enable_metrics
        )
        self.audit_check("field.executor_workers", settings.executor_workers)
        self.audit_check("field.timeout_seconds", settings.timeout_seconds)
        self.audit_check("field.max_workers", settings.max_workers)
        self.audit_check("field.max_batch_size", settings.max_batch_size)
        self.audit_check("field.api_key", settings.api_key)
        self.audit_check(
            "field.exception_failure_level", settings.exception_failure_level
        )
        validated = FlextSettings.model_validate(settings.model_dump())
        self.audit_check(
            "validate_configuration.indirect_via_model_validate",
            validated.app_name,
        )
