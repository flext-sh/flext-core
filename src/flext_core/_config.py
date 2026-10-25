"""FlextConfig — frozen runtime config singleton (ADR-005 §7).

A thin layer over ``pydantic_settings.BaseSettings`` that depends on **nothing**
but stdlib + pydantic-settings. Declarative execution parametrization is loaded
from the declaring package's ``config/`` resource: every ``config/*.yaml`` file is
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

Only ``fetch_global`` publishes a root. Direct construction is isolated; once
published, the root identity is stable. ``reset_for_testing`` reloads current
sources into that root only after complete validation, keeping imported handles
live without making config writable to consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import os
from importlib.resources import files
from pathlib import Path
from threading import RLock
from typing import ClassVar, Self, override

from pydantic import JsonValue, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from flext_core._settings import FlextSettings
from flext_core.config_sources import FlextStrictYamlConfigSource


class FlextConfig(BaseSettings):
    """Frozen per-class config singleton auto-loaded from ``config/*.yaml``.

    Frozen: attribute mutation raises. Read-only — there is no ``update_global``;
    use ``fetch_global()``. Subclass per library and drop ``config/*.yaml`` files;
    no per-domain wiring is required.
    """

    # NOTE (multi-agent): exact-file consumers declare their YAML surface here;
    # the empty default preserves deterministic directory auto-discovery.
    CONFIG_FILENAMES: ClassVar[tuple[str, ...]] = ()
    CONFIG_DIR_NAME: ClassVar[str] = "config"
    """Packaged resource directory holding the declaring package's YAML files."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="FLEXT_CONFIG_",
    )

    _lock: ClassVar[RLock] = RLock()
    _instance: ClassVar[Self | None] = None

    @classmethod
    def _package_namespace(cls) -> str:
        """Return the namespace segment owned by the declaring package.

        Derived from the concrete subclass's own import package, so every
        FLEXT distribution owns ``<import-package-with-dashes>`` without
        naming itself anywhere (``flext_core`` -> ``flext-core``,
        ``ai_hub`` -> ``ai-hub``).

        Returns:
            The namespace segment owned by the declaring package.

        """
        package = cls.__module__.split(".", 1)[0]
        return package.replace("_", "-")

    @classmethod
    def _config_dir(cls) -> Path:
        """Resolve the packaged ``config/`` root, independent of the caller's CWD.

        The declaring import package owns its YAML as package data, read through
        ``importlib.resources`` without enumerating installed distributions.
        Repository overlays are separate explicit consumer inputs, never a
        replacement for a missing packaged default. ``<PACKAGE>_CONFIG_DIR``
        remains the explicit operator relocation.

        Returns:
            The resulting ``Path``.

        """
        override = os.environ.get(
            f"{FlextSettings.app_env_prefix(cls._package_namespace())}CONFIG_DIR"
        )
        if override:
            return Path(override)
        package = cls.__module__.partition(".")[0]
        return Path(str(files(package) / cls.CONFIG_DIR_NAME))

    @classmethod
    def _user_config_dir(cls) -> Path:
        """Return the operator's optional preference directory for this package.

        Lives under the platform config root scoped by the package namespace
        (``$XDG_CONFIG_HOME/<namespace>`` on Linux). Packaged defaults stay
        immutable; anything declared here overlays them.

        Returns:
            The operator's optional preference directory for this package.

        """
        return FlextSettings.platform_config_root() / cls._package_namespace()

    @classmethod
    def _yaml_files_in(cls, directory: Path) -> list[Path]:
        """Return every YAML file in one directory, sorted for deterministic merge.

        Returns:
            Every YAML file in one directory, sorted for deterministic merge.

        """
        return sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))

    @classmethod
    def config_files(cls) -> list[Path]:
        """Collect packaged ``config/*.yaml`` first, then the operator's overlay files.

        Later files win on key collision, so operator preferences override the
        packaged defaults while every undeclared key keeps shipping its default.

        Returns:
            The resulting ``list[Path]``.

        Raises:
            FileNotFoundError: If a declared config directory or declared config
                file does not exist.
            ValueError: If a declared config filename is not a bare ``.yaml`` or
                ``.yml`` name.

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
    def _transform_loaded_yaml(
        cls,
        data: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Transform merged YAML before validation on subclass override.

        Default is identity (no transformation). Override to apply env
        expansion, section filtering, or any data reshaping without
        reimplementing ``settings_customise_sources`` or the YAML source.

        Returns:
            The resulting ``dict[str, JsonValue]``.

        """
        return data

    @classmethod
    def _config_source_files(cls) -> list[Path]:
        """Compose each declaring owner's YAML in base-to-subclass precedence.

        Each hook is bound to its own class, so inherited namespaces retain their
        package resources. Shared files keep the most specific declaration's order
        and enter the existing list-aware merger once.

        Returns:
            Ordered, unique source paths for the complete config inheritance chain.

        """
        paths = tuple(
            path
            for owner in reversed(cls.__mro__)
            if issubclass(owner, FlextConfig)
            for path in owner.config_files()
        )
        return list(reversed(dict.fromkeys(reversed(paths))))

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env + every ``config/*.yaml`` deep-merged; no dotenv/secret sources.

        Returns:
            The resulting ``tuple[PydanticBaseSettingsSource, ...]``.

        """
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            # NOTE (multi-agent): one canonical loader rejects duplicate keys
            # before settings construction; consumers never add local parsers.
            FlextStrictYamlConfigSource(
                settings_cls,
                yaml_file=cls._config_source_files(),
                deep_merge=True,
                transform=cls._transform_loaded_yaml,
            ),
        )

    @classmethod
    def _load_sources(cls) -> Self:
        """Validate current sources with their declaring owner in failure notes.

        Returns:
            A fully validated, unpublished config candidate.

        Raises:
            ValidationError: If the configuration is invalid.

        """
        try:
            return cls()
        except ValidationError as error:
            sources = ", ".join(str(path) for path in cls._config_source_files())
            error.add_note(
                f"Configuration owner: {inspect.getfile(cls)}; sources: {sources}"
            )
            raise

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the shared frozen root, publishing only after validation."""
        with cls._lock:
            instance = cls.__dict__.get("_instance")
            if isinstance(instance, cls):
                return instance
            created = cls._load_sources()
            cls._instance = created
            return created

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reload a published root in place; keep unused concrete classes lazy.

        Repoint through the real environment seam, then restore and reset again.
        Invalid sources leave the published reference and validated state intact.
        """
        with cls._lock:
            instance = cls.__dict__.get("_instance")
            if isinstance(instance, cls):
                validated = cls._load_sources()
                instance.__setstate__(validated.__getstate__())


config: FlextConfig = FlextConfig.fetch_global()
"""Pre-instantiated frozen config singleton — ``from flext_core import config``."""

__all__: list[str] = ["FlextConfig", "config"]
