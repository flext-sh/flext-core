"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import override

from flext_core import c, m, u
from flext_core._settings import FlextSettings

from .ex_02_flext_settings_helpers import Ex02FlextSettingsFieldChecks


class Ex02FlextSettings(Ex02FlextSettingsFieldChecks):
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
        self.audit_check("constructor.singleton_identity", first is second)
        self.audit_check("fetch_global.identity", global_instance is first)
        FlextSettings.reset_for_testing()
        third = FlextSettings()
        self.audit_check("reset_for_testing.recreates_singleton", third is not first)
        FlextSettings.reset_for_testing()
        self.audit_check("reset_for_testing.recreates_global", fourth is not third)
        self._TestConfig.reset_for_testing()
        test_a = self._TestConfig()
        test_b = self._TestConfig()
        self.audit_check("subclass.singleton_identity", test_a is test_b)

    def _exercise_effective_log_level_and_override(self) -> None:
        """Exercise effective_log_level property and update_global."""
        self.section("effective_log_level_and_override")
        FlextSettings.reset_for_testing()
        FlextSettings.update_global(
            debug=False,
            trace=False,
            log_level=c.LogLevel.ERROR,
        )
        base = FlextSettings.fetch_global()
        self.audit_check("effective_log_level.base", base.effective_log_level)
        FlextSettings.update_global(debug=True)
        self.audit_check(
            "effective_log_level.debug",
            FlextSettings.fetch_global().effective_log_level,
        )
        FlextSettings.update_global(trace=True)
        self.audit_check(
            "effective_log_level.trace",
            FlextSettings.fetch_global().effective_log_level,
        )

    def _exercise_fetch_global_and_provider(self) -> None:
        """Exercise fetch_global and DI settings provider."""
        self.section("fetch_global_and_provider")
        FlextSettings.reset_for_testing()
        base = FlextSettings.fetch_global()
        cloned = FlextSettings.fetch_global()
        self.audit_check(
            "fetch_global.clone_same_values",
            cloned.app_name == base.app_name,
        )
        self.audit_check("fetch_global.clone_new_object", cloned is base)
        overridden = FlextSettings.fetch_global(
            overrides={"app_name": "materialized", "timeout_seconds": 55.0},
        )
        self.audit_check("fetch_global.override.app_name", overridden.app_name)
        self.audit_check(
            "fetch_global.override.timeout_seconds",
            overridden.timeout_seconds,
        )
        provider = overridden.resolve_di_settings_provider()
        self.audit_check("resolve_di_settings_provider.type", type(provider).__name__)

    def _exercise_resolve_env_file_and_auto_settings(self) -> None:
        """Exercise resolve_env_file and AutoSettings."""
        self.section("resolve_env_file_and_auto_settings")
        FlextSettings.reset_for_testing()
        env_path = Path(__file__).with_name("flext_settings_example.env")
        env_path.write_text("FLEXT_APP_NAME=from_env_file\n", encoding="utf-8")
        previous = self._set_env(c.ENV_FILE_ENV_VAR, str(env_path))
        try:
            resolved = u.resolve_env_file()
            self.audit_check("resolve_env_file.custom_path", resolved)
            auto = m.AutoSettings(
                settings_class=self._TestConfig,
                env_prefix=c.ENV_PREFIX,
                env_file=resolved,
            )
            created = auto.create_settings()
            self.audit_check(
                "AutoSettings.create_settings.type",
                type(created).__name__,
            )
            self.audit_check(
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
        self.audit_check(
            "fetch_namespace.decorated.service_name",
            decorated_typed.service_name,
        )
        self.audit_check(
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
        self.audit_check(
            "register_context_overrides.service_name",
            from_registered.service_name,
        )
        self.audit_check(
            "for_context.registered.timeout_seconds",
            from_registered.timeout_seconds,
        )
        self.audit_check(
            "for_context.registered.max_workers",
            from_registered.max_workers,
        )
        self.audit_check(
            "for_context.runtime_override.feature_enabled",
            from_runtime.feature_enabled,
        )


if __name__ == "__main__":
    Ex02FlextSettings(caller_file=Path(__file__)).run()
