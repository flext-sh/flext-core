"""Configuration loader/merger (new) with strict priorities.

Single-class module: defines `FlextConfigLoader` only.

Merges dicts from providers in a well-defined order and validates the final
result using `FlextConfigSchemaAppConfig`.
"""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from flext_core.config_new.schema_app import FlextConfigSchemaAppConfig
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfigLoader:
    """Merge config providers deterministically and validate with FlextConfigSchemaAppConfig."""

    PRIORITY: Final[tuple[str, ...]] = (
        FlextConstants.Config.ConfigSource.CLI.value,
        FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
        FlextConstants.Config.ConfigSource.DOTENV.value,
        FlextConstants.Config.ConfigSource.FILE.value,
        FlextConstants.Config.ConfigSource.DEFAULT.value,
    )

    @staticmethod
    def merge(
        base: FlextTypes.Config.ConfigDict, override: FlextTypes.Config.ConfigDict
    ) -> FlextTypes.Config.ConfigDict:
        merged: FlextTypes.Config.ConfigDict = {**base, **override}
        return merged

    @classmethod
    def load_and_validate(
        cls, configs: list[FlextTypes.Config.ConfigDict]
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        try:
            # Fold merge left to right (higher priority later in list)
            merged: FlextTypes.Config.ConfigDict = {}
            for cfg in configs:
                merged = cls.merge(merged, cfg)

            # Ensure sensible defaults for required keys when missing
            if "environment" not in merged:
                merged["environment"] = FlextConstants.Config.DEFAULT_ENVIRONMENT
            if "log_level" not in merged:
                merged["log_level"] = FlextConstants.Config.LogLevel.INFO.value

            # Optional normalization could happen here, but be permissive to
            # allow flexible sources like "constants" used in tests.

            # Drop non-schema helper keys (e.g., priority hints)
            merged.pop("priority", None)

            # Additional validation is handled by Pydantic enums in FlextConfigSchemaAppConfig

            model = FlextConfigSchemaAppConfig.model_validate(merged)
            return FlextResult[FlextConfigSchemaAppConfig].ok(model)
        except (ValidationError, ValueError, TypeError) as e:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                f"Configuration load failed: {e}"
            )

    # ------------------------------------------------------------------
    # Convenience helpers expected by tests/usage
    # ------------------------------------------------------------------

    @staticmethod
    def merge_configs(
        configs: list[FlextTypes.Config.ConfigDict],
    ) -> FlextTypes.Config.ConfigDict:
        """Merge a list of config dicts in order.

        Later entries override earlier ones.
        """
        merged: FlextTypes.Config.ConfigDict = {}
        for cfg in configs:
            merged = FlextConfigLoader.merge(merged, cfg)
        return merged

    @staticmethod
    def validate_config(
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Validate a config dict into FlextConfigSchemaAppConfig."""
        try:
            if not config:
                return FlextResult[FlextConfigSchemaAppConfig].fail(
                    "Empty configuration provided"
                )
            model = FlextConfigSchemaAppConfig.model_validate(config)
            return FlextResult[FlextConfigSchemaAppConfig].ok(model)
        except (ValidationError, ValueError, TypeError) as e:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                f"Invalid configuration: {e}"
            )
