"""FlextSettings — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import override

from flext_core import FlextSettings, c, m, u

from .ex_02_flext_settings_helpers import Ex02FlextSettingsFieldChecks


class Ex02FlextSettings(Ex02FlextSettingsFieldChecks):
    """Golden-file tests for ``FlextSettings`` public API."""

    class _TestConfig(FlextSettings):
        """Subclass with extra fields used across override exercises."""

        service_name: str = "test-service"
        feature_enabled: bool = True

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        saved = self._strip_flext_env()
        try:
            self._exercise_singleton_and_global()
            self._exercise_settings_fields_and_validation()
            self._exercise_update_global_and_clone()
            self._exercise_resolve_env_file_and_auto_settings()
        finally:
            self._restore_flext_env(saved)
        FlextSettings.reset_for_testing()
        self._TestConfig.reset_for_testing()

    @staticmethod
    def _strip_flext_env() -> dict[str, str]:
        """Remove ``FLEXT_``-prefixed vars so field defaults are deterministic."""
        saved = {k: v for k, v in os.environ.items() if k.startswith("FLEXT_")}
        for key in saved:
            os.environ.pop(key, None)
        return saved

    @staticmethod
    def _restore_flext_env(saved: dict[str, str]) -> None:
        """Restore previously stripped ``FLEXT_`` env vars."""
        for key, value in saved.items():
            os.environ[key] = value

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
        """Exercise the per-class singleton and ``fetch_global``."""
        self.section("singleton_and_global")
        FlextSettings.reset_for_testing()
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        global_instance = FlextSettings.fetch_global()
        self.audit_check("constructor.singleton_identity", first is second)
        self.audit_check("fetch_global.identity", global_instance is first)
        FlextSettings.reset_for_testing()
        third = FlextSettings.fetch_global()
        self.audit_check("reset_for_testing.recreates_singleton", third is not first)
        test_a = self._TestConfig.fetch_global()
        test_b = self._TestConfig.fetch_global()
        self.audit_check("subclass.singleton_identity", test_a is test_b)
        self.audit_check("subclass.isolated_slot", test_a is not third)

    def _exercise_update_global_and_clone(self) -> None:
        """Exercise ``update_global``, ``fetch_global`` overrides and ``clone``."""
        self.section("update_global_and_clone")
        FlextSettings.reset_for_testing()
        base = FlextSettings.fetch_global()
        self.audit_check("update_global.before.log_level", base.log_level)
        updated = FlextSettings.update_global(log_level="ERROR", debug=True)
        self.audit_check("update_global.replaces_instance", updated is not base)
        self.audit_check(
            "update_global.installed_as_singleton",
            FlextSettings.fetch_global() is updated,
        )
        self.audit_check("update_global.after.log_level", updated.log_level)
        self.audit_check("update_global.after.debug", updated.debug)
        cloned = updated.clone(trace=True)
        self.audit_check("clone.new_object", cloned is not updated)
        self.audit_check("clone.override.trace", cloned.trace)
        self.audit_check("clone.inherits.log_level", cloned.log_level)
        via_fetch = FlextSettings.fetch_global(overrides={"timezone": "Asia/Tokyo"})
        self.audit_check("fetch_global.override.new_object", via_fetch is not updated)
        self.audit_check("fetch_global.override.timezone", via_fetch.timezone)

    def _exercise_resolve_env_file_and_auto_settings(self) -> None:
        """Exercise ``resolve_env_file`` and ``AutoSettings``."""
        self.section("resolve_env_file_and_auto_settings")
        FlextSettings.reset_for_testing()
        env_path = Path(__file__).with_name("flext_settings_example.env")
        env_path.write_text("FLEXT_LOG_LEVEL=WARNING\n", encoding="utf-8")
        previous = self._set_env(c.ENV_FILE_ENV_VAR, str(env_path))
        try:
            resolved = u.resolve_env_file()
            self.audit_check(
                "resolve_env_file.matches_requested", resolved == str(env_path)
            )
            auto = m.AutoSettings(
                settings_class=self._TestConfig,
                env_prefix=c.ENV_PREFIX,
                env_file=resolved,
            )
            created = auto.create_settings()
            self.audit_check(
                "AutoSettings.create_settings.type", type(created).__name__
            )
            self.audit_check(
                "AutoSettings.create_settings.service_name",
                created.model_dump().get("service_name"),
            )
        finally:
            self._restore_env(c.ENV_FILE_ENV_VAR, previous)
            if env_path.exists():
                env_path.unlink()


if __name__ == "__main__":
    Ex02FlextSettings(caller_file=Path(__file__)).run()
