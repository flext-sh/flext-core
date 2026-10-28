"""FlextSettings — flat, self-contained settings foundation (layer-0 SSOT).

ONE base class, no MRO field-mixins and no composed facade imports. Override
annotations use the lower settings protocol owner. ``FlextSettings`` carries the
per-class singleton lifecycle, the canonical helper API (``fetch_global``/``clone``/
``update_global``/``reset_for_testing``/``resolve_env_file``), the namespace
registry, and the five universal runtime fields every FLEXT app shares.

Canonical usage — always this shape:

    from flext_core import FlextSettings

    class FlextXSettings(FlextSettings):
        my_field: str = "default"

    settings = FlextXSettings.fetch_global()   # export the project singleton

Then ``from flext_x import settings`` and use it directly (``settings.debug``,
``settings.my_field``, ``settings.Cli`` via the namespace registry). The root
package exports its own ``settings`` singleton the same way.

The global lifecycle alone publishes a singleton; direct construction creates
an isolated instance. Once published, each concrete class keeps that root identity.
Updates and test resets validate detached candidates before publishing their
state in place, so already-imported handles remain live even after source repoints.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import logging
import os
import sys
import threading
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import singledispatchmethod
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Protocol, Self, Unpack

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretBytes,
    SecretStr,
    computed_field,
    model_validator,
)
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource as _PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PydanticBaseSettingsSource = _PydanticBaseSettingsSource
"""Source interface for Pydantic custom settings loaders.

Re-exported for layer-0 parity.
"""


class FlextSettings(BaseSettings):
    """Single settings base: singleton + helper API + registry + universal fields.

    Every project subclasses this directly and exports its own
    ``settings = FlextXSettings.fetch_global()`` singleton.
    """

    @staticmethod
    def app_env_prefix(namespace: str) -> str:
        """Return the environment prefix owned by one application namespace."""
        normalized = "".join(char if char.isalnum() else "_" for char in namespace)
        return f"{normalized.upper()}_"

    @staticmethod
    def _resolve_env_file(
        env_var: str, default: str, namespace: str | None = None
    ) -> str:
        """Resolve dotenv input before the settings class configuration is built.

        Returns:
            The dotenv file path selected for this settings class.

        """
        custom_env_file = os.environ.get(env_var)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        if namespace:
            scoped = Path.cwd() / f".env.flext-{namespace}"
            if scoped.exists():
                return str(scoped.resolve())
        default_path = Path.cwd() / default
        if default_path.exists():
            return str(default_path.resolve())
        return default

    ENV_FILE_ENV_VAR: ClassVar[str] = "FLEXT_ENV_FILE"
    ENV_FILE_DEFAULT: ClassVar[str] = ".env"
    ENV_PREFIX: ClassVar[str] = "FLEXT_"

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=_resolve_env_file(ENV_FILE_ENV_VAR, ENV_FILE_DEFAULT),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    @staticmethod
    def resolve_env_file(namespace: str | None = None) -> str:
        """Resolve the effective ``.env`` path honouring ``FLEXT_ENV_FILE``.

        Settings-layer canonical owner; the former utility duplicate was
        removed (chain law: ``c`` consumes ``settings``, never the reverse).

        Returns:
            The resulting ``str``.

        """
        return FlextSettings._resolve_env_file(
            FlextSettings.ENV_FILE_ENV_VAR, FlextSettings.ENV_FILE_DEFAULT, namespace
        )

    debug: Annotated[bool, Field(description="Enable debug mode")] = False
    trace: Annotated[
        bool, Field(description="Enable detailed stacks and high-volume diagnostics")
    ] = False
    log_level: str = Field(default="INFO", description="Log level")
    logging_profile: Literal["console", "json"] = Field(
        default="console", description="Diagnostic output presentation"
    )
    timezone: Annotated[
        str,
        Field(description="IANA timezone for datetime operations"),
    ] = "UTC"
    async_logging: Annotated[
        bool,
        Field(description="Enable asynchronous buffered logging"),
    ] = True

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _instance: ClassVar[Self | None] = None
    _app_namespace: ClassVar[str | None] = None

    type _FieldValue = (
        BaseModel
        | JsonValue
        | Path
        | datetime
        | bytes
        | SecretStr
        | SecretBytes
        | Mapping[str, JsonValue]
        | tuple[JsonValue, ...]
    )

    class _ModelValue(Protocol):
        """Native model operations accepted at the layer-zero settings boundary."""

        def model_dump(
            self,
            *,
            mode: str = "python",
            exclude: set[str] | None = None,
            by_alias: bool | None = None,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
        ) -> Mapping[str, FlextSettings._FieldValue]:
            """Expose model data without requiring an upper-layer facade import."""
            ...

        def model_copy(self, *, deep: bool = False) -> Self:
            """Copy a model while retaining its declared concrete type."""
            ...

    type _OverrideValue = (
        FlextSettings._FieldValue | FlextSettings._ModelValue | Sequence[JsonValue]
    )

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Give every concrete subclass its own isolated singleton slot."""
        super().__init_subclass__(**kwargs)
        cls._instance = None

    @classmethod
    @contextmanager
    def singleton_disabled(cls) -> Generator[None]:
        """Serialize isolated construction for existing snapshot consumers.

        Construction itself never publishes; the global lifecycle owns the root.
        """
        with cls._lock:
            yield

    @classmethod
    def _load_sources(cls, **overrides: FlextSettings._OverrideValue) -> Self:
        """Validate sources and overrides together, without publishing defaults.

        Returns:
            A settings instance validated from every source and override.

        """
        env_file = (
            cls.resolve_env_file()
            if cls.ENV_FILE_ENV_VAR in os.environ
            else cls.model_config.get("env_file")
        )
        sources, init_kwargs = cls._settings_init_sources(
            _env_file=env_file, _init_kwargs={}
        )
        source_values = cls._settings_build_values(sources, init_kwargs)
        values = cls._field_input(cls, source_values)
        # Reuse the native source merger without serializing model-valued input.
        combined = cls._settings_build_values(
            (
                InitSettingsSource(
                    cls, overrides, nested_model_default_partial_update=False
                ),
                InitSettingsSource(
                    cls, dict(values), nested_model_default_partial_update=False
                ),
            ),
            {},
        )
        values = cls._field_input(cls, combined)
        # Allocation alone neither initializes nor publishes the model. Pydantic
        # receives the complete input once, including required override-only fields.
        return cls._validate_candidate(cls.__new__(cls), deepcopy(values))

    @classmethod
    def _field_input(
        cls,
        model_type: type[BaseModel],
        source_values: Mapping[str, FlextSettings._FieldValue],
    ) -> MutableMapping[str, FlextSettings._FieldValue]:
        """Resolve native validation aliases into complete field-name input.

        Returns:
            The source values keyed by field name.

        """
        # Native sources may use validation aliases. Resolve those before applying
        # the owner's field-name overrides, including aliases with shared paths.
        field_values = {
            name: source_values[name]
            for name in model_type.model_fields
            if name in source_values
        }
        consumed: set[str] = set(field_values)
        for name, field in model_type.model_fields.items():
            alias = (
                field.validation_alias
                if model_type.model_config.get("validate_by_alias", True)
                else None
            )
            aliases = alias.choices if isinstance(alias, AliasChoices) else (alias,)
            for candidate_alias in aliases:
                if candidate_alias is None:
                    continue
                path = (
                    AliasPath(candidate_alias)
                    if isinstance(candidate_alias, str)
                    else candidate_alias
                )
                root = path.path[0]
                if not isinstance(root, str):
                    continue
                value = cls._alias_path_value(
                    source_values.get(root, PydanticUndefined), path.path[1:]
                )
                if not isinstance(value, PydanticUndefinedType):
                    field_values[name] = value
                    consumed.add(root)
                    break
        values = {
            key: value for key, value in source_values.items() if key not in consumed
        }
        values.update(field_values)
        return values

    @staticmethod
    def _alias_path_value(
        value: FlextSettings._FieldValue | PydanticUndefinedType,
        keys: Sequence[str | int],
    ) -> FlextSettings._FieldValue | PydanticUndefinedType:
        """Follow the remaining segments of one validation-alias path.

        Mapping segments select by key; integer segments index sequences within
        bounds. Any other step leaves the alias unresolved.

        Returns:
            The value the path reaches, or ``PydanticUndefined`` when it is absent.

        """
        for key in keys:
            if isinstance(value, Mapping):
                value = next(
                    (item for name, item in value.items() if name == key),
                    PydanticUndefined,
                )
            elif (
                isinstance(value, (list, tuple, bytes))
                and isinstance(key, int)
                and -len(value) <= key < len(value)
            ):
                value = value[key]
            else:
                return PydanticUndefined
        return value

    @classmethod
    def _validate_candidate[ModelT: BaseModel, ValueT](
        cls, candidate: ModelT, values: Mapping[str, ValueT]
    ) -> ModelT:
        """Validate detached stored state, never a presentation serialization.

        Returns:
            A validated copy of the candidate carrying the given values.

        """
        validated = {
            name: cls._validate_nested(value) for name, value in values.items()
        }
        candidate.__pydantic_validator__.validate_python(
            validated, self_instance=candidate, by_name=True
        )
        return candidate

    @singledispatchmethod
    @classmethod
    def _validate_nested[ValueT](cls, value: ValueT) -> ValueT:
        """Keep arbitrary native leaves intact while dispatching stored models.

        Returns:
            The value itself, unchanged.

        """
        return value

    @_validate_nested.register(BaseModel)
    @classmethod
    def _validate_model[ModelT: BaseModel](cls, value: ModelT) -> ModelT:
        """Validate a copy so a model resident in a hashed container stays intact.

        Returns:
            The validated deep copy of the model.

        """
        copied = value.model_copy(deep=True)
        model_type: type[BaseModel] = copied.__class__
        values = {
            name: stored
            for name, stored in copied.__dict__.items()
            if name in model_type.model_fields
        }
        values.update(copied.model_extra or {})
        return cls._validate_candidate(copied, values)

    @_validate_nested.register(Mapping)
    @_validate_nested.register(list)
    @_validate_nested.register(tuple)
    @_validate_nested.register(set)
    @_validate_nested.register(frozenset)
    @classmethod
    def _validate_collection[KeyT, ValueT](
        cls,
        value: Mapping[KeyT, ValueT]
        | list[ValueT]
        | tuple[ValueT, ...]
        | set[ValueT]
        | frozenset[ValueT],
    ) -> (
        Mapping[KeyT, ValueT]
        | list[ValueT]
        | tuple[ValueT, ...]
        | set[ValueT]
        | frozenset[ValueT]
    ):
        """Rebuild containers from validated copies without mutating resident keys.

        Returns:
            A new container of the same kind holding validated copies.

        """
        if isinstance(value, Mapping):
            return {
                cls._validate_nested(key): cls._validate_nested(nested)
                for key, nested in value.items()
            }
        nested_values = (cls._validate_nested(nested) for nested in value)
        if isinstance(value, list):
            return list(nested_values)
        if isinstance(value, tuple):
            return tuple(nested_values)
        if isinstance(value, set):
            return set(nested_values)
        return frozenset(nested_values)

    @classmethod
    def fetch_global(cls, *, overrides: Mapping[str, JsonValue] | None = None) -> Self:
        """Return the shared per-class singleton (lazy, thread-safe).

        With ``overrides`` return an isolated clone; the singleton is not
        mutated (use ``update_global`` for that).

        Returns:
            The shared per-class singleton (lazy, thread-safe).

        """
        with cls._lock:
            instance = cls._instance
            if overrides is not None:
                cls._validate_overrides(**overrides)
                if instance is None:
                    return cls._load_sources(**overrides)
                return instance.clone(**overrides)
            if instance is None:
                instance = cls._load_sources()
                cls._instance = instance
            return instance

    @classmethod
    def _merge_overrides[ValueT](
        cls,
        current: Mapping[str, ValueT],
        **overrides: FlextSettings._OverrideValue,
    ) -> MutableMapping[str, FlextSettings._OverrideValue]:
        """Merge partial nested-model overrides onto the current state.

        Returns:
            The resulting mapping of the merged override values.

        """
        cls._validate_overrides(**overrides)
        merged: MutableMapping[str, FlextSettings._OverrideValue] = {}
        for field_name, override_value in overrides.items():
            current_value = current.get(field_name)
            if isinstance(current_value, BaseModel) and isinstance(
                override_value,
                Mapping,
            ):
                current_model: BaseModel = current_value
                model_type: type[BaseModel] = current_model.__class__
                computed = set(model_type.model_computed_fields)
                stored = {
                    name: value
                    for name, value in current_model.__dict__.items()
                    if name in model_type.model_fields
                }
                stored.update(current_model.model_extra or {})
                stored.update({
                    k: v for k, v in override_value.items() if k not in computed
                })
                merged[field_name] = cls._validate_candidate(
                    current_model.model_copy(deep=True),
                    deepcopy(cls._field_input(model_type, stored)),
                )
                continue
            merged[field_name] = override_value
        return merged

    def clone(self, **overrides: FlextSettings._OverrideValue) -> Self:
        """Deep copy with optional field overrides + re-validation.

        Returns:
            The resulting ``Self``.

        """
        with self.__class__.singleton_disabled():
            if not overrides:
                return self.model_copy(deep=True)
            copied = self.model_copy(deep=True)
            values = {
                name: stored
                for name, stored in copied.__dict__.items()
                if name in type(copied).model_fields
            }
            values.update(copied.model_extra or {})
            values.update(deepcopy(self._merge_overrides(values, **overrides)))
            return self._validate_candidate(copied, values)

    @classmethod
    def update_global(cls, **overrides: FlextSettings._OverrideValue) -> Self:
        """Validate an isolated candidate, then publish into the stable root.

        Returns:
            The resulting ``Self``.

        """
        with cls._lock:
            cls._validate_overrides(**overrides)
            current = cls._instance
            if current is None:
                validated = cls._load_sources(**overrides)
                cls._instance = validated
                return validated
            if overrides:
                validated = current.clone(**overrides)
                current.__setstate__(validated.__getstate__())
            return current

    @classmethod
    def _validate_overrides(cls, **overrides: FlextSettings._OverrideValue) -> None:
        """Reject override keys that are not declared model fields.

        Raises:
            ValueError: If Unknown settings override(s) for.

        """
        unknown = sorted(set(overrides) - set(cls.model_fields))
        if unknown:
            msg = (
                f"Unknown settings override(s) for {cls.__name__}: {', '.join(unknown)}"
            )
            raise ValueError(msg)

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reload sources into a published root; leave an unused class lazy.

        Repoint the real environment before calling and restore it before calling
        again. Validation failure leaves the published state and reference intact.
        """
        with cls._lock:
            if cls._instance is not None:
                validated = cls._load_sources()
                cls._instance.__setstate__(validated.__getstate__())

    @staticmethod
    def _platform_root(
        apple_leaf: str,
        windows_env_var: str,
        windows_default: Path,
        xdg_env_var: str,
        xdg_default: Path,
    ) -> Path:
        """Return the OS-native user directory root for one platform convention.

        Linux/BSD honour the XDG environment variable and its declared default;
        macOS uses ``~/Library/<apple_leaf>``; Windows honours its environment
        variable and declared default. This owner depends only on the stdlib.

        Returns:
            The resulting ``Path``.

        """
        if sys.platform == "darwin":
            return Path.home() / "Library" / apple_leaf
        if sys.platform == "win32":
            windows_override = os.environ.get(windows_env_var)
            return (
                Path(windows_override)
                if windows_override
                else Path.home() / windows_default
            )
        xdg_override = os.environ.get(xdg_env_var)
        return Path(xdg_override) if xdg_override else Path.home() / xdg_default

    @staticmethod
    def _platform_cache_root() -> Path:
        """Return the OS-native cache root shared by cache and work directories.

        Returns:
            The platform cache root, including its existing environment override.

        """
        return FlextSettings._platform_root(
            "Caches",
            "LOCALAPPDATA",
            Path("AppData") / "Local",
            "XDG_CACHE_HOME",
            Path(".cache"),
        )

    @staticmethod
    def platform_config_root() -> Path:
        """Return the OS-native config root shared with configuration loading.

        Returns:
            The platform config root, including its existing environment override.

        """
        return FlextSettings._platform_root(
            "Application Support",
            "APPDATA",
            Path("AppData") / "Roaming",
            "XDG_CONFIG_HOME",
            Path(".config"),
        )

    @staticmethod
    def _validate_app_namespace(namespace: str) -> str:
        """Return one safe application namespace segment or fail validation.

        Raises:
            ValueError: If application namespace must be one non-empty path segment.

        """
        candidate = namespace.strip()
        if (
            not candidate
            or candidate in {".", ".."}
            or Path(candidate).name != candidate
        ):
            msg = "application namespace must be one non-empty path segment"
            raise ValueError(msg)
        return candidate

    @classmethod
    def apply_app_namespace(cls, namespace: str) -> None:
        """Set the outer application identity once for the current process."""
        candidate = FlextSettings._validate_app_namespace(namespace)
        with FlextSettings._lock:
            if FlextSettings._app_namespace is None:
                FlextSettings._app_namespace = candidate

    @classmethod
    def reset_app_namespace(cls) -> None:
        """Clear the process application identity for isolated tests."""
        with FlextSettings._lock:
            FlextSettings._app_namespace = None

    @classmethod
    def _owner_namespace(cls) -> str:
        """Return the owning project's own namespace, derived from ``env_prefix``.

        This is the default identity when no application registered one, so a
        standalone project (e.g. ``ai-hub``) transparently owns its own
        directories without being forced to call ``apply_app_namespace``.
        """
        env_prefix = cls.model_config.get("env_prefix") or FlextSettings.ENV_PREFIX
        return env_prefix.rstrip("_").lower().replace("_", "-")

    @classmethod
    def _current_app_namespace(cls) -> str:
        """Return the effective namespace.

        Precedence: an explicitly registered application identity
        (``apply_app_namespace``) wins so every library shares it; then the
        ``FLEXT_APP_NAMESPACE`` environment override; otherwise the owning
        project's own namespace prevails as the default (registration is never
        mandatory).
        """
        registered = FlextSettings._app_namespace or os.environ.get(
            "FLEXT_APP_NAMESPACE",
        )
        return FlextSettings._validate_app_namespace(
            registered or cls._owner_namespace()
        )

    @classmethod
    def _absolute_app_dir(cls, name: str, root: Path) -> Path:
        namespace = cls._current_app_namespace()
        override = os.environ.get(
            f"{FlextSettings.app_env_prefix(namespace)}{name.upper()}_DIR"
        )
        path = Path(override) if override else root / namespace
        if not path.is_absolute():
            msg = f"{name}_dir must be absolute"
            raise ValueError(msg)
        return path

    @computed_field
    @property
    def cache_dir(self) -> Path:
        """Consuming application's cache directory."""
        return self._absolute_app_dir("cache", FlextSettings._platform_cache_root())

    @computed_field
    @property
    def work_dir(self) -> Path:
        """Consuming application's scratch/work directory."""
        return self._absolute_app_dir("work", FlextSettings._platform_cache_root())

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Consuming application's durable data directory."""
        return self._absolute_app_dir(
            "data",
            FlextSettings._platform_root(
                "Application Support",
                "LOCALAPPDATA",
                Path("AppData") / "Local",
                "XDG_DATA_HOME",
                Path(".local") / "share",
            ),
        )

    @computed_field
    @property
    def state_dir(self) -> Path:
        """Consuming application's persistent state directory."""
        return self._absolute_app_dir(
            "state",
            FlextSettings._platform_root(
                "Application Support",
                "LOCALAPPDATA",
                Path("AppData") / "Local",
                "XDG_STATE_HOME",
                Path(".local") / "state",
            ),
        )

    @computed_field
    @property
    def config_dir(self) -> Path:
        """Consuming application's user configuration directory."""
        return self._absolute_app_dir("config", FlextSettings.platform_config_root())

    @computed_field
    @property
    def runtime_dir(self) -> Path:
        """Consuming application's ephemeral runtime directory.

        Raises:
            ValueError: If runtime_dir must be absolute.

        """
        namespace = self._current_app_namespace()
        override = os.environ.get(
            f"{FlextSettings.app_env_prefix(namespace)}RUNTIME_DIR"
        )
        if override:
            path = Path(override)
        else:
            xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
            path = (
                Path(xdg_runtime_dir) / namespace
                if xdg_runtime_dir
                else self.work_dir / "run"
            )
        if not path.is_absolute():
            msg = "runtime_dir must be absolute"
            raise ValueError(msg)
        return path

    @model_validator(mode="after")
    def _validate_logging(self) -> Self:
        """Validate severity without logging or importing higher-layer utilities.

        Returns:
            Self: The validated settings instance.

        Raises:
            ValueError: If the log level is invalid.

        """
        number = logging.getLevelNamesMapping().get(self.log_level.strip().upper())
        if number is None or number < logging.DEBUG:
            msg = f"Invalid log level: {self.log_level}"
            raise ValueError(msg)
        return self


settings: FlextSettings = FlextSettings.fetch_global()
"""Pre-instantiated root settings singleton — ``from flext_core import settings``."""

__all__: list[str] = [
    "FlextSettings",
    "PydanticBaseSettingsSource",
    "settings",
]
