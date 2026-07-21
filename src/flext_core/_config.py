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
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, ClassVar, Self, cast, override

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from yaml import MappingNode, SafeLoader
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

if TYPE_CHECKING:
    from pydantic import JsonValue


class _UniqueKeySafeLoader(SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: SafeLoader, node: MappingNode, *, deep: bool = False
) -> dict[str, JsonValue]:
    """Construct one JSON mapping and fail before a duplicate can overwrite."""
    values: dict[str, JsonValue] = {}
    for key_node, value_node in node.value:
        key = cast("JsonValue", loader.construct_object(key_node, deep=deep))
        if not isinstance(key, str):
            context = "while constructing a config mapping"
            problem = "config mapping keys must be strings"
            raise ConstructorError(
                context, node.start_mark, problem, key_node.start_mark
            )
        if key in values:
            context = "while constructing a config mapping"
            problem = f"duplicate config key: {key}"
            raise ConstructorError(
                context, node.start_mark, problem, key_node.start_mark
            )
        values[key] = cast("JsonValue", loader.construct_object(value_node, deep=deep))
    return values


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


class _StrictYamlConfigSettingsSource(YamlConfigSettingsSource):
    """Pydantic settings source backed by the unique-key safe loader."""

    @override
    def _read_file(self, file_path: Path) -> dict[str, JsonValue]:
        """Parse one YAML config file exactly once with strict mapping keys."""
        with file_path.open(encoding=self.yaml_file_encoding) as yaml_file:
            loader = _UniqueKeySafeLoader(yaml_file)
            try:
                loaded = cast("JsonValue", loader.get_single_data())
            finally:
                loader.dispose()
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            msg = f"config YAML root must be a mapping: {file_path}"
            raise TypeError(msg)
        return loaded


class FlextConfig(BaseSettings):
    """Frozen per-class config singleton auto-loaded from ``config/*.yaml``.

    Frozen: attribute mutation raises. Read-only — there is no ``update_global``;
    use ``fetch_global()``. Subclass per library and drop ``config/*.yaml`` files;
    no per-domain wiring is required.
    """

    CONFIG_DIR: ClassVar[str] = "config"
    # NOTE (multi-agent): exact-file consumers declare their YAML surface here;
    # the empty default preserves deterministic directory auto-discovery.
    CONFIG_FILENAMES: ClassVar[tuple[str, ...]] = ()

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True, extra="allow", env_prefix="FLEXT_CONFIG_"
    )

    _lock: ClassVar[RLock] = RLock()
    _instance: ClassVar[FlextConfig | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Give every concrete subclass its own isolated singleton slot."""
        _ = kwargs
        super().__init_subclass__()
        cls._instance = None

    @classmethod
    def _config_dir(cls) -> Path:
        """Resolve ``config/`` deterministically, independent of the caller's CWD.

        ``CONFIG_DIR`` may be an absolute path (explicit override, used verbatim)
        or a relative name (default ``"config"``). When relative it is resolved
        against the concrete subclass's own module layout, trying in order:

        1. ``<pkg>/config`` — the packaged copy shipped via hatch force-include,
           present after ``pip install`` (``site-packages/<pkg>/config``).
        2. ``<project-root>/config`` — the editable/workspace source tree, where
           ``config/`` sits beside ``src/`` (``<root>/src/<pkg>/_config.py`` →
           ``parents[2]`` is the project root).

        Library code must never depend on the process CWD, so the legacy
        CWD-relative lookup is gone.
        """
        config_dir = Path(cls.CONFIG_DIR)
        if config_dir.is_absolute():
            return config_dir
        module_path = Path(inspect.getfile(cls)).resolve()
        packaged = module_path.parent / config_dir
        if packaged.is_dir():
            return packaged
        return module_path.parents[2] / config_dir

    @classmethod
    def _config_files(cls) -> list[Path]:
        """Auto-discover every ``config/*.yaml`` (sorted for deterministic merge)."""
        config_dir = cls._config_dir()
        if not config_dir.is_dir():
            if cls.CONFIG_FILENAMES:
                msg = f"declared config directory does not exist: {config_dir}"
                raise FileNotFoundError(msg)
            return []
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
            return files
        return sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))

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
        """Env + every ``config/*.yaml`` deep-merged; no dotenv/secret sources."""
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            # NOTE (multi-agent): one canonical loader rejects duplicate keys
            # before settings construction; consumers never add local parsers.
            _StrictYamlConfigSettingsSource(
                settings_cls, yaml_file=cls._config_files(), deep_merge=True
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
