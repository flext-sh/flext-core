"""FlextConfig — frozen runtime config singleton (ADR-005 §7).

A thin layer over ``pydantic_settings.BaseSettings`` that depends on **nothing**
but stdlib + pydantic-settings. Declarative execution parametrization is loaded
from the project's local ``config/`` dir: every ``config/*.yaml`` file is
auto-discovered and deep-merged (YAML is the authoring format so rules stay
organized across multiple files). Frozen and static, like constants; a per-class
singleton exposed at the root as ``config``.

``FlextConfig`` is a **sibling** of ``FlextSettings`` — the two never expose each
other, and neither imports ``constants``/``c`` (constants import *these* as their
base). Access is lazy (``fetch_global`` reads the files at first call, never at
import), so ``c/t/p/m/u`` use it with zero import-time coupling.

Each library gets its own namespaced subclass (``class FlextCliConfig(FlextConfig)``)
with flat fields composed via MRO; a domain prefix (``cli_``/``mcp_``) is optional,
only to organize. Fields are unique at the subclass root.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import ClassVar, Self

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class FlextConfig(BaseSettings):
    """Frozen per-class config singleton auto-loaded from ``config/*.yaml``.

    Frozen: attribute mutation raises. Read-only — there is no ``update_global``;
    use ``fetch_global()``. Subclass per library and drop ``config/*.yaml`` files;
    no per-domain wiring is required.
    """

    CONFIG_DIR: ClassVar[str] = "config"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        extra="allow",
        env_prefix="FLEXT_CONFIG_",
    )

    _lock: ClassVar[RLock] = RLock()
    _instance: ClassVar[FlextConfig | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Give every concrete subclass its own isolated singleton slot."""
        super().__init_subclass__(**kwargs)
        cls._instance = None

    @classmethod
    def _config_files(cls) -> list[Path]:
        """Auto-discover every ``config/*.yaml`` (sorted for deterministic merge)."""
        config_dir = Path(cls.CONFIG_DIR)
        if not config_dir.is_dir():
            return []
        return sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env + every ``config/*.yaml`` deep-merged; no dotenv/secret sources."""
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=cls._config_files(),
                deep_merge=True,
            ),
        )

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the shared frozen singleton (lazy; built on first access)."""
        instance = cls._instance
        if isinstance(instance, cls):
            return instance
        with cls._lock:
            instance = cls._instance
            if isinstance(instance, cls):
                return instance
            created = cls()
            cls._instance = created
            return created

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the singleton slot for test isolation."""
        with cls._lock:
            cls._instance = None


config: FlextConfig = FlextConfig.fetch_global()
"""Pre-instantiated frozen config singleton — ``from flext_core import config``."""

__all__: list[str] = ["FlextConfig", "config"]
