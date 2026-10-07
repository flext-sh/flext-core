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

import inspect
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, ClassVar, Self, override

from pydantic import JsonValue
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from flext_core._constants import FlextConstantsConfig
from flext_core._settings import app_env_prefix, platform_config_root
from flext_core.config_sources import FlextStrictYamlConfigSource

if TYPE_CHECKING:
    from flext_core import t


class FlextConfig(BaseSettings):
    """Frozen per-class config singleton auto-loaded from ``config/*.yaml``.

    Frozen: attribute mutation raises. Read-only — there is no ``update_global``;
    use ``fetch_global()``. Subclass per library and drop ``config/*.yaml`` files;
    no per-domain wiring is required.
    """

    # NOTE (multi-agent): exact-file consumers declare their YAML surface here;
    # the empty default preserves deterministic directory auto-discovery.
    CONFIG_FILENAMES: ClassVar[t.VariadicTuple[str]] = ()

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        extra="allow",
        env_prefix="FLEXT_CONFIG_",
    )

    _lock: ClassVar[RLock] = RLock()
    _instance: ClassVar[FlextConfig | None] = None

    @classmethod
    def _package_namespace(cls) -> str:
        """Return the namespace segment owned by the declaring package.

        Derived from the concrete subclass's own import package, so every
        FLEXT distribution owns ``<import-package-with-dashes>`` without
        naming itself anywhere (``flext_core`` -> ``flext-core``,
        ``ai_hub`` -> ``ai-hub``).
        """
        package = cls.__module__.split(".", 1)[0]
        return package.replace("_", "-")

    @classmethod
    def _config_dir(cls) -> Path:
        """Resolve the packaged ``config/`` root, independent of the caller's CWD.

        ``CONFIG_DIR`` may be an absolute path (explicit override, used verbatim)
        or a relative name (default ``"config"``). When relative it is resolved
        against the concrete subclass's own module layout, trying in order:

        1. ``<pkg>/config`` — the packaged copy shipped via hatch force-include,
           present after ``pip install`` (``site-packages/<pkg>/config``).
        2. ``<project-root>/config`` — the editable/workspace source tree, where
           ``config/`` sits beside ``src/`` (``<root>/src/<pkg>/_config.py`` →
           ``parents[2]`` is the project root).

        An operator may relocate the root entirely with
        ``<PACKAGE>_CONFIG_DIR``. Library code must never depend on the process
        CWD, so the legacy CWD-relative lookup is gone.

        Returns:
            The resulting ``Path``.
        """
        namespace = cls._package_namespace()
        override = os.environ.get(f"{app_env_prefix(namespace)}CONFIG_DIR")
        if override:
            return Path(override)
        config_dir = Path(FlextConstantsConfig.CONFIG_DIR_NAME)
        if config_dir.is_absolute():
            return config_dir
        module_path = Path(inspect.getfile(cls)).resolve()
        packaged = module_path.parent / config_dir
        if packaged.is_dir():
            return packaged
        return module_path.parents[2] / config_dir

    @classmethod
    def _user_config_dir(cls) -> Path:
        """Return the operator's optional preference directory for this package.

        Lives under the platform config root scoped by the package namespace
        (``$XDG_CONFIG_HOME/<namespace>`` on Linux). Packaged defaults stay
        immutable; anything declared here overlays them.
        """
        return platform_config_root() / cls._package_namespace()

    @classmethod
    def _yaml_files_in(cls, directory: Path) -> list[Path]:
        """Return every YAML file in one directory, sorted for deterministic merge."""
        return sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))

    @classmethod
    def _config_files(cls) -> list[Path]:
        """Packaged ``config/*.yaml`` first, then the operator's overlay files.

        Later files win on key collision, so operator preferences override the
        packaged defaults while every undeclared key keeps shipping its default.

        Returns:
            The resulting ``list[Path]``.

        Raises:
            FileNotFoundError: If declared config directory does not exist; or if
                ``missing``.
            ValueError: If ``invalid``.
        """
        config_dir = cls._config_dir()
        user_files = cls._yaml_files_in(cls._user_config_dir())
        if not config_dir.is_dir():
            if cls.CONFIG_FILENAMES:
                msg = f"declared config directory does not exist: {config_dir}"
                raise FileNotFoundError(msg)
            return user_files
        if cls.CONFIG_FILENAMES:
            invalid = tuple(
                filename
                for filename in cls.CONFIG_FILENAMES
                if Path(filename).name != filename
                or Path(filename).suffix not in {".yaml", ".yml"}
            )
            if invalid:
                msg = "invalid declared config filenames: " + ", ".join(invalid)
                raise ValueError(msg)
            files = [config_dir / filename for filename in cls.CONFIG_FILENAMES]
            missing = tuple(str(path) for path in files if not path.is_file())
            if missing:
                msg = "declared config files do not exist: " + ", ".join(missing)
                raise FileNotFoundError(msg)
            return files + user_files
        return cls._yaml_files_in(config_dir) + user_files

    @classmethod
    def _transform_loaded_yaml(cls, data: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Hook for subclasses to transform merged YAML before validation.

        Default is identity (no transformation). Override to apply env
        expansion, section filtering, or any data reshaping without
        reimplementing ``settings_customise_sources`` or the YAML source.

        Returns:
            The resulting ``dict[str, JsonValue]``.
        """
        return data

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> t.VariadicTuple[PydanticBaseSettingsSource]:
        """Env + every ``config/*.yaml`` deep-merged; no dotenv/secret sources.

        Returns:
            The resulting ``t.VariadicTuple[PydanticBaseSettingsSource]``.
        """
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            # NOTE (multi-agent): one canonical loader rejects duplicate keys
            # before settings construction; consumers never add local parsers.
            FlextStrictYamlConfigSource(
                settings_cls,
                yaml_file=cls._config_files(),
                yaml_config_section=FlextConstantsConfig.YAML_CONFIG_SECTION,
                deep_merge=True,
                transform=cls._transform_loaded_yaml,
            ),
        )

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the shared frozen singleton (lazy; built on first access)."""
        instance = cls.__dict__.get("_instance")
        if isinstance(instance, cls):
            return instance
        with cls._lock:
            instance = cls.__dict__.get("_instance")
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
