"""Settings example field checks kept below the module LOC cap."""

from __future__ import annotations

from flext_core import FlextSettings

from .shared import ExamplesFlextShared


class Ex02FlextSettingsFieldChecks(ExamplesFlextShared):
    """Field and validation checks for the settings example."""

    def _exercise_settings_fields_and_validation(self) -> None:
        """Exercise every universal settings field and validation."""
        self.section("settings_fields_and_validation")
        FlextSettings.reset_for_testing()
        settings = FlextSettings(
            debug=True,
            trace=True,
            log_level="ERROR",
            timezone="America/Sao_Paulo",
            async_logging=False,
        )
        self.audit_check("field.debug", settings.debug)
        self.audit_check("field.trace", settings.trace)
        self.audit_check("field.log_level", settings.log_level)
        self.audit_check("field.timezone", settings.timezone)
        self.audit_check("field.async_logging", settings.async_logging)
        validated = FlextSettings.model_validate(settings.model_dump())
        self.audit_check(
            "validate_configuration.indirect_via_model_validate",
            validated.log_level,
        )
