"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flext_core import FlextSettings, c, u

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: object) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(x) for x in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


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


def demo_singleton_and_global() -> None:
    """Exercise singleton pattern and global instance management."""
    _section("singleton_and_global")
    FlextSettings.reset_for_testing()

    first = FlextSettings()
    second = FlextSettings()
    _check("constructor.singleton_identity", first is second)

    global_instance = FlextSettings.get_global()
    _check("get_global.identity", global_instance is first)

    getattr(FlextSettings, "_reset_instance")()
    third = FlextSettings()
    _check("_reset_instance.recreates_singleton", third is not first)

    FlextSettings.reset_for_testing()
    fourth = FlextSettings.get_global()
    _check("reset_global_instance.recreates_global", fourth is not third)

    getattr(_TestConfig, "_reset_instance")()
    test_a = _TestConfig()
    test_b = _TestConfig()
    _check("subclass.singleton_identity", test_a is test_b)


def demo_configuration_fields_and_validation() -> None:
    """Exercise all configuration fields and validation."""
    _section("configuration_fields_and_validation")
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

    _check("field.app_name", config.app_name)
    _check("field.version", config.version)
    _check("field.debug", config.debug)
    _check("field.trace", config.trace)
    _check("field.log_level", config.log_level)
    _check("field.async_logging", config.async_logging)
    _check("field.enable_caching", config.enable_caching)
    _check("field.cache_ttl", config.cache_ttl)
    _check("field.database_url", config.database_url)
    _check("field.database_pool_size", config.database_pool_size)
    _check("field.circuit_breaker_threshold", config.circuit_breaker_threshold)
    _check("field.rate_limit_max_requests", config.rate_limit_max_requests)
    _check("field.rate_limit_window_seconds", config.rate_limit_window_seconds)
    _check("field.retry_delay", config.retry_delay)
    _check("field.max_retry_attempts", config.max_retry_attempts)
    _check("field.enable_timeout_executor", config.enable_timeout_executor)
    _check("field.dispatcher_enable_logging", config.dispatcher_enable_logging)
    _check("field.dispatcher_auto_context", config.dispatcher_auto_context)
    _check("field.dispatcher_timeout_seconds", config.dispatcher_timeout_seconds)
    _check("field.dispatcher_enable_metrics", config.dispatcher_enable_metrics)
    _check("field.executor_workers", config.executor_workers)
    _check("field.timeout_seconds", config.timeout_seconds)
    _check("field.max_workers", config.max_workers)
    _check("field.max_batch_size", config.max_batch_size)
    _check("field.api_key", config.api_key)
    _check("field.exception_failure_level", config.exception_failure_level)

    validated = FlextSettings.model_validate(config.model_dump())
    _check("validate_configuration.indirect_via_model_validate", validated.app_name)


def demo_effective_log_level_and_override() -> None:
    """Exercise effective_log_level property and apply_override."""
    _section("effective_log_level_and_override")
    FlextSettings.reset_for_testing()

    config = FlextSettings(
        debug=False, trace=False, log_level=c.Settings.LogLevel.ERROR
    )
    _check("effective_log_level.base", config.effective_log_level)

    valid_override = config.apply_override("debug", True)
    _check("apply_override.valid_return", valid_override)
    _check("apply_override.valid_applied", config.debug)
    _check("effective_log_level.debug", config.effective_log_level)

    _ = config.apply_override("trace", True)
    _check("effective_log_level.trace", config.effective_log_level)

    invalid_override = config.apply_override("does_not_exist", "x")
    _check("apply_override.invalid_return", invalid_override)


def demo_materialize_and_provider() -> None:
    """Exercise materialize and DI config provider."""
    _section("materialize_and_provider")
    FlextSettings.reset_for_testing()

    base = FlextSettings.get_global()
    cloned = FlextSettings.materialize()
    _check("materialize.clone_same_values", cloned.app_name == base.app_name)
    _check("materialize.clone_new_object", cloned is not base)

    overridden = FlextSettings.materialize(
        config_overrides={"app_name": "materialized", "timeout_seconds": 55.0}
    )
    _check("materialize.override.app_name", overridden.app_name)
    _check("materialize.override.timeout_seconds", overridden.timeout_seconds)

    provider = overridden.get_di_config_provider()
    _check("get_di_config_provider.type", type(provider).__name__)


def demo_resolve_env_file_and_auto_config() -> None:
    """Exercise resolve_env_file and AutoConfig."""
    _section("resolve_env_file_and_auto_config")
    FlextSettings.reset_for_testing()

    env_path = Path(sys.prefix) / "flext_settings_example.env"
    _ = env_path.write_text("FLEXT_APP_NAME=from_env_file\n", encoding="utf-8")
    previous = _set_env(c.Platform.ENV_FILE_ENV_VAR, str(env_path))
    try:
        resolved = FlextSettings.resolve_env_file()
        _check("resolve_env_file.custom_path", resolved)

        auto = FlextSettings.AutoConfig(
            config_class=_TestConfig,
            env_prefix=c.Platform.ENV_PREFIX,
            env_file=resolved,
        )
        created = auto.create_config()
        _check("AutoConfig.create_config.type", type(created).__name__)
        _check(
            "AutoConfig.create_config.service_name",
            created.model_dump().get("service_name"),
        )
    finally:
        _restore_env(c.Platform.ENV_FILE_ENV_VAR, previous)
        if env_path.exists():
            _ = env_path.unlink()


def demo_namespace_system() -> None:
    """Exercise namespace registration and retrieval."""
    _section("namespace_system")
    FlextSettings.reset_for_testing()

    @FlextSettings.auto_register(namespace="decorated_ns")
    class _DecoratedNamespace(_TestConfig):
        service_name: str = "decorated"

    class _RegisteredNamespace(_TestConfig):
        service_name: str = "registered"

    FlextSettings.register_namespace("registered_ns", _RegisteredNamespace)
    decorated_raw = FlextSettings.get_namespace_config("decorated_ns")
    registered_raw = FlextSettings.get_namespace_config("registered_ns")

    _check("get_namespace_config.decorated", decorated_raw)
    _check("get_namespace_config.registered", registered_raw)
    _check("auto_register.class_used", _DecoratedNamespace.__name__)

    base = FlextSettings()
    decorated_typed = base.get_namespace("decorated_ns", _TestConfig)
    registered_typed = base.get_namespace("registered_ns", _TestConfig)
    _check("get_namespace.decorated.service_name", decorated_typed.service_name)
    _check("get_namespace.registered.service_name", registered_typed.service_name)


def demo_context_system() -> None:
    """Exercise context override system."""
    _section("context_system")
    _TestConfig.reset_for_testing()

    _TestConfig.register_context_overrides(
        "worker-a",
        service_name="worker-service",
        timeout_seconds=41.0,
        max_workers=2,
    )
    from_registered = _TestConfig.for_context("worker-a")
    from_runtime = _TestConfig.for_context("worker-a", feature_enabled=False)

    _check("register_context_overrides.service_name", from_registered.service_name)
    _check("for_context.registered.timeout_seconds", from_registered.timeout_seconds)
    _check("for_context.registered.max_workers", from_registered.max_workers)
    _check("for_context.runtime_override.feature_enabled", from_runtime.feature_enabled)


def main() -> None:
    """Run all settings demonstrations and verify golden file."""
    demo_singleton_and_global()
    demo_configuration_fields_and_validation()
    demo_effective_log_level_and_override()
    demo_materialize_and_provider()
    demo_resolve_env_file_and_auto_config()
    demo_namespace_system()
    demo_context_system()
    FlextSettings.reset_for_testing()
    _TestConfig.reset_for_testing()
    _verify()


if __name__ == "__main__":
    main()
