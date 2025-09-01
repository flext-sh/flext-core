"""Config facade (new) for the consolidated configuration pipeline.

Single-class module: defines `FlextConfigCore` only.

Responsibilities:
- Coordinate providers
- Merge by priority
- Validate and return `FlextConfigSchemaAppConfig`
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import ValidationError

from flext_core.config_new.loader import FlextConfigLoader
from flext_core.config_new.providers import FlextConfigProviders
from flext_core.config_new.schema_app import FlextConfigSchemaAppConfig
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextConfigCore:
    """Facade for loading validated application configuration."""

    @staticmethod
    def from_defaults() -> FlextConfigSchemaAppConfig:
        return FlextConfigSchemaAppConfig.model_validate(
            FlextConfigProviders.from_constants()
        )

    @staticmethod
    def from_env() -> FlextConfigSchemaAppConfig:
        data = FlextConfigProviders.from_env()
        if not data:
            # fall back to defaults
            data = FlextConfigProviders.from_constants()
        return FlextConfigSchemaAppConfig.model_validate(data)

    @staticmethod
    def from_file(path: str | Path) -> FlextResult[FlextConfigSchemaAppConfig]:
        raw = FlextConfigProviders.from_file(path)
        if raw.is_failure:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                raw.error or "File load error"
            )
        try:
            model = FlextConfigSchemaAppConfig.model_validate(raw.value)
            return FlextResult[FlextConfigSchemaAppConfig].ok(model)
        except (ValidationError, ValueError, TypeError) as e:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                f"Invalid file configuration: {e}"
            )

    @staticmethod
    def merged(
        *,
        cli: dict[str, object] | None = None,
        env: bool = True,
        file_path: str | None = None,
        include_defaults: bool = True,
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        configs: list[FlextTypes.Config.ConfigDict] = []

        # Lowest priority first: defaults -> file -> env -> cli
        if include_defaults:
            configs.append(FlextConfigProviders.from_constants())
        if file_path:
            result = FlextConfigProviders.from_file(file_path)
            if result.is_failure:
                return FlextResult[FlextConfigSchemaAppConfig].fail(
                    result.error or "File not loaded"
                )
            configs.append(result.value)
        if env:
            configs.append(FlextConfigProviders.from_env())
        if cli:
            c = dict(cli)
            c["config_source"] = "cli"
            configs.append(cast("FlextTypes.Config.ConfigDict", c))

        return FlextConfigLoader.load_and_validate(configs)

    # ---------------------------------------------------------------------
    # Additional unified facade methods forwarding to underlying classes
    # ---------------------------------------------------------------------

    @staticmethod
    def validate_dict(
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Validate a raw config dict into FlextConfigSchemaAppConfig."""
        try:
            model = FlextConfigSchemaAppConfig.model_validate(config)
            return FlextResult[FlextConfigSchemaAppConfig].ok(model)
        except (ValidationError, ValueError, TypeError) as e:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                f"Invalid configuration: {e}"
            )

    @staticmethod
    def schema() -> dict[str, object]:
        """Return the JSON schema for FlextConfigSchemaAppConfig."""
        return FlextConfigSchemaAppConfig.model_json_schema()

    @staticmethod
    def dump(config: FlextConfigSchemaAppConfig) -> FlextTypes.Config.ConfigDict:
        """Dump FlextConfigSchemaAppConfig to a plain dict."""
        return cast("FlextTypes.Config.ConfigDict", config.model_dump())

    @staticmethod
    def dump_json(config: FlextConfigSchemaAppConfig) -> str:
        """Dump FlextConfigSchemaAppConfig to a JSON string."""
        return config.model_dump_json()

    @staticmethod
    def with_overrides(
        config: FlextConfigSchemaAppConfig, overrides: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Return a new FlextConfigSchemaAppConfig with overrides applied and validated."""
        merged = {**config.model_dump(), **overrides}
        return FlextConfigCore.validate_dict(
            cast("FlextTypes.Config.ConfigDict", merged)
        )

    @staticmethod
    def defaults_dict() -> FlextTypes.Config.ConfigDict:
        """Return the default configuration as a plain dict."""
        return FlextConfigProviders.from_constants()

    @staticmethod
    def env_dict(
        prefix: str = FlextConfigProviders.ENV_PREFIX,
    ) -> FlextTypes.Config.ConfigDict:
        """Return the environment-based configuration as a plain dict."""
        return FlextConfigProviders.from_env(prefix)

    # ------------------------------------------------------------------
    # Legacy-like signatures (real functionality; no compat shims)
    # ------------------------------------------------------------------

    @staticmethod
    def safe_load_json_file(
        path: str | Path,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Safely load a JSON config file and return a dict."""
        return FlextConfigProviders.from_file(path)

    @staticmethod
    def load_from_file(path: str | Path) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Load and validate FlextConfigSchemaAppConfig from a JSON file path."""
        return FlextConfigCore.from_file(path)

    @staticmethod
    def load_and_validate_from_file(
        path: str | Path, required_keys: list[str] | tuple[str, ...] | None = None
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Load from file, ensure required keys exist, and validate FlextConfigSchemaAppConfig."""
        raw = FlextConfigProviders.from_file(path)
        if raw.is_failure:
            return FlextResult[FlextConfigSchemaAppConfig].fail(
                raw.error or "File load error"
            )
        data = raw.value
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return FlextResult[FlextConfigSchemaAppConfig].fail(
                    f"Missing required keys: {', '.join(missing)}"
                )
        return FlextConfigCore.validate_dict(data)

    @staticmethod
    def safe_get_env_var(name: str, default: str | None = None) -> FlextResult[str]:
        """Safely retrieve environment variable value with default."""
        try:
            from os import environ

            value = environ.get(name, default)
            if value is None:
                return FlextResult[str].fail(f"Environment variable '{name}' not set")
            return FlextResult[str].ok(value)
        except Exception as e:
            return FlextResult[str].fail(f"Failed to read env var '{name}': {e}")

    @staticmethod
    def validate_business_rules(
        config_data: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Validate business rules by building FlextConfigSchemaAppConfig and returning normalized dict."""
        validated = FlextConfigCore.validate_dict(config_data)
        if validated.is_failure:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                validated.error or "Validation failed"
            )
        return FlextResult[FlextTypes.Config.ConfigDict].ok(
            FlextConfigCore.dump(validated.value)
        )

    @staticmethod
    def create_with_validation(
        config_data: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextConfigSchemaAppConfig]:
        """Create FlextConfigSchemaAppConfig from dict with full validation."""
        return FlextConfigCore.validate_dict(config_data)
