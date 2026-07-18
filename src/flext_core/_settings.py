"""FlextSettings — flat, self-contained settings foundation (layer-0 SSOT).

ONE base class, no MRO field-mixins, zero imports from ``c``/``t``/``p``/``m``/
``u`` (only stdlib + pydantic-settings). ``FlextSettings`` carries the per-class
singleton lifecycle, the canonical helper API (``fetch_global``/``clone``/
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

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE_ENV_VAR = "FLEXT_ENV_FILE"
_ENV_FILE_DEFAULT = ".env"
_ERR_TRACE_REQUIRES_DEBUG = "trace mode requires debug mode to be enabled"


def _resolve_env_file(namespace: str | None = None) -> str:
    """Centralised .env discovery honouring ``FLEXT_ENV_FILE``.

    Module-level so it can seed ``model_config`` before the class body
    finishes evaluating.
    """
    custom_env_file = os.environ.get(_ENV_FILE_ENV_VAR)
    if custom_env_file:
        custom_path = Path(custom_env_file)
        if custom_path.exists():
            return str(custom_path.resolve())
        return custom_env_file
    if namespace:
        scoped = Path.cwd() / f".env.flext-{namespace}"
        if scoped.exists():
            return str(scoped.resolve())
    default_path = Path.cwd() / _ENV_FILE_DEFAULT
    if default_path.exists():
        return str(default_path.resolve())
    return _ENV_FILE_DEFAULT


def _platform_cache_root() -> Path:
    """Return the OS-native user cache root for scratch/work directories.

    Linux/BSD honour ``XDG_CACHE_HOME`` (default ``~/.cache``); macOS uses
    ``~/Library/Caches``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``). Module-level + stdlib-only so it can seed a field
    default without importing the facades (layer-0 purity).
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        return (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"


def _platform_data_root() -> Path:
    """Return the OS-native user data root for durable per-namespace data.

    Linux/BSD honour ``XDG_DATA_HOME`` (default ``~/.local/share``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``). Module-level + stdlib-only for layer-0 purity.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        return (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    return Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"


def _platform_config_root() -> Path:
    """Return the OS-native user config root for per-namespace configuration.

    Linux/BSD honour ``XDG_CONFIG_HOME`` (default ``~/.config``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%APPDATA%`` (default
    ``~/AppData/Roaming``). Module-level + stdlib-only for layer-0 purity.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if sys.platform == "win32":
        app_data = os.environ.get("APPDATA")
        return Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"


def _platform_state_root() -> Path:
    """Return the OS-native user state root for per-namespace state.

    Linux/BSD honour ``XDG_STATE_HOME`` (default ``~/.local/state``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``). Module-level + stdlib-only for layer-0 purity.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        return (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    return Path(xdg_state_home) if xdg_state_home else Path.home() / ".local" / "state"


def _namespace_dir_name(env_prefix: str) -> str:
    """Derive a filesystem directory name from a settings ``env_prefix``.

    ``FLEXT_`` -> ``flext``; ``AI_HUB_`` -> ``ai-hub``; empty -> ``flext``.
    """
    return env_prefix.rstrip("_").lower().replace("_", "-") or "flext"


class FlextSettings(BaseSettings):
    """Single settings base: singleton + helper API + registry + universal fields.

    Every project subclasses this directly and exports its own
    ``settings = FlextXSettings.fetch_global()`` singleton.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="FLEXT_",
        env_nested_delimiter="__",
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @staticmethod
    def _resolve_env_file(namespace: str | None = None) -> str:
        """Delegate to the module-level ``_resolve_env_file`` helper."""
        return _resolve_env_file(namespace)

    debug: Annotated[bool, Field(description="Enable debug mode")] = False
    trace: Annotated[bool, Field(description="Enable trace mode")] = False
    log_level: Annotated[str, Field(default="INFO", description="Log level")]
    timezone: Annotated[
        str, Field(description="IANA timezone for datetime operations")
    ] = "UTC"
    async_logging: Annotated[
        bool, Field(description="Enable asynchronous buffered logging")
    ] = True
    work_dir: Annotated[
        Path,
        Field(
            default=None,
            validate_default=True,
            description=(
                "Per-namespace scratch/work root; defaults to the OS cache home "
                "joined with the settings namespace (e.g. ~/.cache/flext)."
            ),
        ),
    ]
    data_dir: Annotated[
        Path,
        Field(
            default=None,
            validate_default=True,
            description="Per-namespace persistent application data directory.",
        ),
    ]
    state_dir: Annotated[
        Path,
        Field(
            default=None,
            validate_default=True,
            description="Per-namespace persistent runtime state directory.",
        ),
    ]
    config_dir: Annotated[
        Path,
        Field(
            default=None,
            validate_default=True,
            description="Per-namespace user configuration directory.",
        ),
    ]
    runtime_dir: Annotated[
        Path,
        Field(
            default=None,
            validate_default=True,
            description="Per-namespace ephemeral runtime directory.",
        ),
    ]

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _singleton_enabled: ClassVar[bool] = True
    _instance: ClassVar[FlextSettings | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Give every concrete subclass its own isolated singleton slot."""
        _ = kwargs
        super().__init_subclass__()
        cls._instance = None

    def __new__(cls, **kwargs: object) -> Self:
        """Singleton factory; unknown kwargs are ignored by consumer factories."""
        _ = kwargs
        if not cls._singleton_enabled:
            return super().__new__(cls)
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        raw_instance = cls._instance
        if not isinstance(raw_instance, cls):
            msg = f"Singleton instance is not of expected type {cls.__name__}"
            raise TypeError(msg)
        return raw_instance

    @classmethod
    def _initialized_instance(cls) -> Self | None:
        """Return the cached singleton only after Pydantic finished init."""
        existing = getattr(cls, "_instance", None)
        if isinstance(existing, cls) and hasattr(existing, "__pydantic_fields_set__"):
            return existing
        if existing is not None:
            with cls._lock:
                if getattr(cls, "_instance", None) is existing:
                    cls._instance = None
        return None

    @classmethod
    @contextmanager
    def singleton_disabled(cls) -> Generator[None]:
        """Temporarily disable singleton enforcement for clone operations."""
        with cls._lock:
            original = cls._singleton_enabled
            cls._singleton_enabled = False
            try:
                yield
            finally:
                cls._singleton_enabled = original

    @classmethod
    def fetch_global(cls, *, overrides: Mapping[str, object] | None = None) -> Self:
        """Return the shared per-class singleton (lazy, thread-safe).

        With ``overrides`` return an isolated clone; the singleton is not
        mutated (use ``update_global`` for that).
        """
        instance = cls._initialized_instance()
        if overrides:
            if instance is not None:
                return instance.clone(**overrides)
            with cls.singleton_disabled():
                return cls.model_validate(dict(overrides))
        if instance is not None:
            return instance
        with cls._lock:
            instance = cls._initialized_instance()
            if instance is not None:
                return instance
            with cls.singleton_disabled():
                created = cls()
            cls._instance = created
            return created

    @classmethod
    def _merge_overrides(cls, current: Self, **overrides: object) -> dict[str, object]:
        """Merge partial nested-model overrides onto the current state."""
        cls._validate_overrides(**overrides)
        merged: dict[str, object] = {}
        for field_name, override_value in overrides.items():
            current_value = getattr(current, field_name, None)
            if isinstance(current_value, BaseModel) and isinstance(
                override_value, Mapping
            ):
                computed = set(type(current_value).model_computed_fields)
                merged_dict = {
                    **current_value.model_dump(mode="python", exclude=computed),
                    **{k: v for k, v in override_value.items() if k not in computed},
                }
                merged[field_name] = type(current_value).model_validate(merged_dict)
                continue
            merged[field_name] = override_value
        return merged

    def clone(self, **overrides: object) -> Self:
        """Deep copy with optional field overrides + re-validation."""
        if not overrides:
            with self.__class__.singleton_disabled():
                return self.model_copy(deep=True)
        merged = self.__class__._merge_overrides(self, **overrides)  # noqa: SLF001  Why: internal helper on same class
        with self.__class__.singleton_disabled():
            copied = self.model_copy(update=merged, deep=True)
            return type(copied).model_validate(copied, from_attributes=True)

    @classmethod
    def update_global(cls, **overrides: object) -> Self:
        """Replace the singleton via ``model_copy(update=…)`` + revalidate."""
        if not overrides:
            return cls.fetch_global()
        current = cls.fetch_global()
        merged = cls._merge_overrides(current, **overrides)
        with cls.singleton_disabled():
            new_instance = current.model_copy(update=merged, deep=True)
            validated = cls.model_validate(new_instance, from_attributes=True)
        with cls._lock:
            cls._instance = validated
        return validated

    @classmethod
    def _validate_overrides(cls, **overrides: object) -> None:
        """Reject override keys that are not declared model fields."""
        unknown = sorted(set(overrides) - set(cls.model_fields))
        if unknown:
            msg = (
                f"Unknown settings override(s) for {cls.__name__}: {', '.join(unknown)}"
            )
            raise ValueError(msg)

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the singleton slot for test isolation."""
        with cls._lock:
            cls._instance = None

    @field_validator(
        "work_dir", "data_dir", "state_dir", "config_dir", "runtime_dir", mode="before"
    )
    @classmethod
    def _seed_namespace_dir(
        cls, value: Path | str | None, info: ValidationInfo
    ) -> Path | str:
        """Seed an unset directory from its platform root and project namespace."""
        if value:
            return value
        env_prefix = cls.model_config.get("env_prefix") or "FLEXT_"
        namespace = _namespace_dir_name(env_prefix)
        roots = {
            "work_dir": _platform_cache_root(),
            "data_dir": _platform_data_root(),
            "state_dir": _platform_state_root(),
            "config_dir": _platform_config_root(),
        }
        if info.field_name == "runtime_dir":
            xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
            if xdg_runtime_dir:
                return Path(xdg_runtime_dir) / namespace
            work_dir = info.data.get("work_dir")
            if not isinstance(work_dir, Path):
                msg = "work_dir must be resolved before runtime_dir"
                raise TypeError(msg)
            return work_dir / "run"
        root = roots.get(info.field_name or "")
        if root is None:
            msg = f"unexpected settings directory field: {info.field_name!r}"
            raise ValueError(msg)
        return root / namespace

    @field_validator("work_dir", "data_dir", "state_dir", "config_dir", "runtime_dir")
    @classmethod
    def _require_absolute_runtime_dir(cls, value: Path) -> Path:
        """Reject relative directory overrides at the settings boundary."""
        if not value.is_absolute():
            msg = "runtime directory paths must be absolute"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_settings(self) -> Self:
        """Enforce the trace-requires-debug invariant."""
        if self.trace and not self.debug:
            raise ValueError(_ERR_TRACE_REQUIRES_DEBUG)
        return self


settings: FlextSettings = FlextSettings.fetch_global()
"""Pre-instantiated root settings singleton — ``from flext_core import settings``."""

__all__: list[str] = ["FlextSettings", "settings"]
