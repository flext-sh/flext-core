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
from collections.abc import Callable
from importlib.resources.abc import Traversable
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Self, cast, override

from pydantic import JsonValue
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from pydantic_settings.sources import PathType
from yaml import MappingNode, SafeLoader
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from flext_core._settings import app_env_prefix, platform_config_root


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


class StrictYamlConfigSource(YamlConfigSettingsSource):
    """Pydantic settings source backed by the unique-key safe loader.

    Accepts an optional ``transform`` callable applied to the fully merged
    YAML data before pydantic validates it. Subclasses of ``FlextConfig``
    use this to apply env-expansion, filtering, or section reshaping without
    reinstantiating the source or overriding ``settings_customise_sources``.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: PathType | None = None,
        yaml_file_encoding: str | None = None,
        yaml_config_section: str | None = None,
        *,
        deep_merge: bool = False,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._transform = transform
        super().__init__(
            settings_cls,
            yaml_file=yaml_file,
            yaml_file_encoding=yaml_file_encoding,
            yaml_config_section=yaml_config_section,
            deep_merge=deep_merge,
        )

    @override
    def __call__(self) -> dict[str, Any]:
        """Return merged YAML data, applying the transform hook if set."""
        data = super().__call__()
        if self._transform is not None:
            data = self._transform(data)
        return data

    @override
    def _read_file(self, file_path: Path | Traversable) -> dict[str, JsonValue]:
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

    @override
    def _read_files(self, files: object, deep_merge: bool = False) -> dict[str, object]:
        """Read multiple YAML files with list-aware deep merge.

        The upstream ``deep_update`` only recurses into dicts; when two files
        declare the same list key (e.g. ``command_rules``), the second file
        *replaces* the first list entirely. This override concatenates lists
        instead, enabling domain-split config files to each contribute rules
        to the same list.
        """
        from collections.abc import Sequence as _Sequence
        from pathlib import Path as _Path

        if files is None:
            return {}
        if isinstance(files, str) or not isinstance(files, _Sequence):
            files = [files]
        merged: dict[str, object] = {}
        for file in files:
            raw_path = _Path(file) if isinstance(file, str) else file
            if not isinstance(raw_path, _Path):
                continue
            file_path = raw_path.expanduser()
            if not file_path.is_file():
                continue
            updating: dict[str, object] = dict(self._read_file(file_path))
            if deep_merge:
                merged = self._deep_merge_lists(merged, updating)
            else:
                merged.update(updating)
        return merged

    @staticmethod
    def _deep_merge_lists(
        base: dict[str, object], updating: dict[str, object]
    ) -> dict[str, object]:
        """Deep-merge two config dicts, concatenating list values."""
        result = dict(base)
        for key, value in updating.items():
            existing = result.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                result[key] = StrictYamlConfigSource._deep_merge_lists(existing, value)
            elif isinstance(existing, list) and isinstance(value, list):
                result[key] = [*existing, *value]
            else:
                result[key] = value
        return result


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
    YAML_CONFIG_SECTION: ClassVar[str | None] = None

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
        """
        namespace = cls._package_namespace()
        override = os.environ.get(f"{app_env_prefix(namespace)}CONFIG_DIR")
        if override:
            return Path(override)
        config_dir = Path(cls.CONFIG_DIR)
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
    def _transform_loaded_yaml(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Hook for subclasses to transform merged YAML before validation.

        Default is identity (no transformation). Override to apply env
        expansion, section filtering, or any data reshaping without
        reimplementing ``settings_customise_sources`` or the YAML source.
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
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env + every ``config/*.yaml`` deep-merged; no dotenv/secret sources."""
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            # NOTE (multi-agent): one canonical loader rejects duplicate keys
            # before settings construction; consumers never add local parsers.
            StrictYamlConfigSource(
                settings_cls,
                yaml_file=cls._config_files(),
                yaml_config_section=cls.YAML_CONFIG_SECTION,
                deep_merge=True,
                transform=cls._transform_loaded_yaml,
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
