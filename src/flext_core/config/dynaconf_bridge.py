"""Bridge between Pydantic settings and Dynaconf.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides seamless integration between Dynaconf's configuration management
and Pydantic's type-safe settings with comprehensive error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from typing import TypeVar

from dynaconf import Dynaconf
from dynaconf import ValidationError as DynaconfValidationError
from pydantic import ValidationError as PydanticValidationError

from flext_core.config.base import BaseSettings
from flext_core.config.base import ConfigurationError

TSettings = TypeVar("TSettings", bound=BaseSettings)


class DynaconfBridge:
    """Bridge to integrate Dynaconf with Pydantic settings.

    This allows projects to use Dynaconf's advanced features while
    maintaining type safety with Pydantic models.
    """

    def __init__(
        self,
        settings_class: type[TSettings],
        *,
        env_prefix: str | None = None,
        settings_files: list[str] | None = None,
        environments: bool = True,
        env_switcher: str = "FLEXT_ENV",
        root_path: Path | None = None,
    ) -> None:
        """Initialize the Dynaconf bridge.

        Args:
            settings_class: Pydantic settings class to bridge
            env_prefix: Environment variable prefix
            settings_files: List of settings files to load
            environments: Whether to enable environment-specific configs
            env_switcher: Environment variable to switch environments
            root_path: Root path for relative file paths

        """
        self.settings_class = settings_class
        self.env_prefix = env_prefix or settings_class.get_env_prefix()

        # Default settings files if not provided:
        if settings_files is None:
            settings_files = [
                "settings.toml",
                "settings.yaml",
                "settings.json",
                ".secrets.toml",
            ]

        # Initialize Dynaconf
        self.dynaconf = Dynaconf(
            envvar_prefix=self.env_prefix,
            settings_files=settings_files,
            environments=environments,
            env_switcher=env_switcher,
            root_path=root_path or Path.cwd(),
            load_dotenv=True,
            merge_enabled=True,
        )

    def load_settings(self, settings_class: type[TSettings] | None = None) -> TSettings:
        """Load settings using Dynaconf and validate with Pydantic.

        Args:
            settings_class: Settings class to use (defaults to instance class)

        Returns:
            Validated settings instance

        Raises:
            ConfigurationError: If settings validation fails

        """
        target_class = settings_class or self.settings_class
        try:
            # Get all settings from Dynaconf as dict
            dynaconf_data = self.dynaconf.as_dict()

            # Remove Dynaconf internals
            cleaned_data = self._clean_dynaconf_data(dynaconf_data)

            # Create Pydantic settings instance
            return target_class(**cleaned_data)  # type: ignore[arg-type,return-value]

        except PydanticValidationError as e:
            msg = f"Configuration validation failed: {e}"
            raise ConfigurationError(msg) from e
        except DynaconfValidationError as e:
            msg = f"Dynaconf validation failed: {e}"
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading settings: {e}"
            raise ConfigurationError(msg) from e

    @staticmethod
    def _clean_dynaconf_data(data: dict[str, object]) -> dict[str, object]:
        """Clean Dynaconf internal data from settings dict.

        Args:
            data: Raw settings data from Dynaconf

        Returns:
            Cleaned settings data without internal keys

        """
        # Dynaconf internal keys to remove
        internal_keys = {
            "SETTINGS_FILE_FOR_DYNACONF",
            "ENV_FOR_DYNACONF",
            "FORCE_ENV_FOR_DYNACONF",
            "_loaded_files",
            "_fresh",
            "_kwargs",
        }

        return {
            k: v
            for k, v in data.items()
            if k not in internal_keys and not k.startswith("_")
        }

    def validate(self) -> None:
        """Validate settings using both Dynaconf and Pydantic.

        Raises:
            ConfigurationError: If validation fails at any stage

        """
        try:
            self.dynaconf.validators.validate()
        except DynaconfValidationError as e:
            msg = f"Dynaconf validation failed: {e}"
            raise ConfigurationError(msg) from e

        # Then validate with Pydantic
        self.load_settings()

    def get(self, key: str, *, default: object = None) -> object:
        """Get a configuration value.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default

        """
        return self.dynaconf.get(key, default)

    def reload(self) -> BaseSettings:
        """Reload settings from all sources.

        Returns:
            Reloaded and validated settings instance

        """
        self.dynaconf.reload()
        return self.load_settings()


class DynaconfSettings(BaseSettings):
    """Base settings class with Dynaconf integration.

    Extends BaseSettings to automatically load from Dynaconf.
    """

    # Dynaconf configuration - using ClassVar for mutable class attributes
    _dynaconf_settings_files: ClassVar[list[str]] = ["settings.toml", ".secrets.toml"]
    _dynaconf_environments: ClassVar[bool] = True
    _dynaconf_env_switcher: ClassVar[str] = "FLEXT_ENV"

    @classmethod
    def from_dynaconf(
        cls,
        *,
        settings_files: list[str] | None = None,
        env_prefix: str | None = None,
        root_path: Path | None = None,
    ) -> DynaconfSettings:
        """Create settings instance from Dynaconf.

        Args:
            settings_files: List of settings files to load
            env_prefix: Environment variable prefix
            root_path: Root path for relative file paths

        Returns:
            Settings instance loaded from Dynaconf sources

        """
        bridge = DynaconfBridge(
            settings_class=cls,
            env_prefix=env_prefix,
            settings_files=settings_files or cls._dynaconf_settings_files,
            environments=cls._dynaconf_environments,
            env_switcher=cls._dynaconf_env_switcher,
            root_path=root_path,
        )

        return bridge.load_settings()

    @classmethod
    def create_bridge(
        cls,
        *,
        settings_files: list[str] | None = None,
        env_prefix: str | None = None,
        root_path: Path | None = None,
    ) -> DynaconfBridge:
        """Create a Dynaconf bridge for this settings class.

        Args:
            settings_files: List of settings files to load
            env_prefix: Environment variable prefix
            root_path: Root path for relative file paths

        Returns:
            Configured Dynaconf bridge instance

        """
        return DynaconfBridge(
            settings_class=cls,
            env_prefix=env_prefix,
            settings_files=settings_files or cls._dynaconf_settings_files,
            environments=cls._dynaconf_environments,
            env_switcher=cls._dynaconf_env_switcher,
            root_path=root_path,
        )


def load_settings_with_dynaconf[TSettings: BaseSettings](
    settings_class: type[TSettings],
    *,
    settings_files: list[str] | None = None,
    env_prefix: str | None = None,
    validate: bool = True,
) -> TSettings:
    """Load settings using Dynaconf bridge.

    Args:
        settings_class: Pydantic settings class to load
        settings_files: List of settings files to load
        env_prefix: Environment variable prefix
        validate: Whether to run validation

    Returns:
        Loaded and validated settings instance

    """
    bridge = DynaconfBridge(
        settings_class=settings_class,
        env_prefix=env_prefix,
        settings_files=settings_files,
    )

    if validate:
        bridge.validate()

    return bridge.load_settings()


__all__ = [
    "DynaconfBridge",
    "DynaconfSettings",
    "load_settings_with_dynaconf",
]
