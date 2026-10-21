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

import logging
import os
import sys
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import pydantic_settings
from pydantic import BaseModel, Field, TypeAdapter, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core._constants import FlextConstantsEnvironment, FlextConstantsSettings
from flext_core._protocols.settings import FlextProtocolsSettings

if TYPE_CHECKING:
    from flext_core import t

PydanticBaseSettingsSource = pydantic_settings.PydanticBaseSettingsSource
"""Upstream settings-source base exposed for consumer-defined loading sources."""


def _resolve_env_file(namespace: str | None = None) -> str:
    """Centralised .env discovery honouring ``FLEXT_ENV_FILE``.

    Module-level so it can seed ``model_config`` before the class body
    finishes evaluating.

    Returns:
        The resulting ``str``.

    """
    custom_env_file = os.environ.get(FlextConstantsSettings.ENV_FILE_ENV_VAR)
    if custom_env_file:
        custom_path = Path(custom_env_file)
        if custom_path.exists():
            return str(custom_path.resolve())
        return custom_env_file
    if namespace:
        scoped = Path.cwd() / f".env.flext-{namespace}"
        if scoped.exists():
            return str(scoped.resolve())
    default_path = Path.cwd() / FlextConstantsSettings.ENV_FILE_DEFAULT
    if default_path.exists():
        return str(default_path.resolve())
    return FlextConstantsSettings.ENV_FILE_DEFAULT


def _platform_root(
    apple_leaf: str,
    windows_env_var: str,
    windows_default: Path,
    xdg_env_var: str,
    xdg_default: Path,
) -> Path:
    """Return the OS-native user directory root for one platform convention.

    Linux/BSD honour the XDG environment variable (falling back to the XDG
    default under ``Path.home()``); macOS uses ``~/Library/<apple_leaf>``;
    Windows uses the Windows environment variable (falling back to the
    Windows default under ``Path.home()``). Module-level + stdlib-only so it
    can seed field defaults without importing the facades (layer-0 purity).

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


def _platform_cache_root() -> Path:
    """Return the OS-native user cache root for scratch/work directories.

    Linux/BSD honour ``XDG_CACHE_HOME`` (default ``~/.cache``); macOS uses
    ``~/Library/Caches``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``).

    Returns:
        The resulting ``Path``.

    """
    return _platform_root(
        "Caches",
        "LOCALAPPDATA",
        Path("AppData") / "Local",
        "XDG_CACHE_HOME",
        Path(".cache"),
    )


def _platform_data_root() -> Path:
    """Return the OS-native user data root for durable per-namespace data.

    Linux/BSD honour ``XDG_DATA_HOME`` (default ``~/.local/share``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``).

    Returns:
        The resulting ``Path``.

    """
    return _platform_root(
        "Application Support",
        "LOCALAPPDATA",
        Path("AppData") / "Local",
        "XDG_DATA_HOME",
        Path(".local") / "share",
    )


def platform_config_root() -> Path:
    """Return the OS-native user config root for per-namespace configuration.

    Linux/BSD honour ``XDG_CONFIG_HOME`` (default ``~/.config``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%APPDATA%`` (default
    ``~/AppData/Roaming``).

    Returns:
        The resulting ``Path``.

    """
    return _platform_root(
        "Application Support",
        "APPDATA",
        Path("AppData") / "Roaming",
        "XDG_CONFIG_HOME",
        Path(".config"),
    )


def _platform_state_root() -> Path:
    """Return the OS-native user state root for per-namespace state.

    Linux/BSD honour ``XDG_STATE_HOME`` (default ``~/.local/state``); macOS uses
    ``~/Library/Application Support``; Windows uses ``%LOCALAPPDATA%`` (default
    ``~/AppData/Local``).

    Returns:
        The resulting ``Path``.

    """
    return _platform_root(
        "Application Support",
        "LOCALAPPDATA",
        Path("AppData") / "Local",
        "XDG_STATE_HOME",
        Path(".local") / "state",
    )


def app_env_prefix(namespace: str) -> str:
    """Return the environment prefix owned by one application namespace."""
    normalized = "".join(char if char.isalnum() else "_" for char in namespace)
    return f"{normalized.upper()}_"


def _validate_app_namespace(namespace: str) -> str:
    """Return one safe application namespace segment or fail validation.

    Raises:
        ValueError: If application namespace must be one non-empty path segment.

    """
    candidate = namespace.strip()
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        msg = "application namespace must be one non-empty path segment"
        raise ValueError(msg)
    return candidate


def _namespace_dir_name(env_prefix: str) -> str:
    """Derive the owning project's namespace segment from its ``env_prefix``.

    ``FLEXT_`` -> ``flext``; ``AI_HUB_`` -> ``ai-hub``. This
    is the default when no application identity was registered.

    Returns:
        The resulting ``str``.

    """
    return env_prefix.rstrip("_").lower().replace("_", "-")


class FlextSettings(BaseSettings):
    """Single settings base: singleton + helper API + registry + universal fields.

    Every project subclasses this directly and exports its own
    ``settings = FlextXSettings.fetch_global()`` singleton.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix=FlextConstantsEnvironment.ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @staticmethod
    def resolve_env_file(namespace: str | None = None) -> str:
        """Resolve the effective ``.env`` path honouring ``FLEXT_ENV_FILE``.

        Settings-layer canonical owner; the former utility duplicate was
        removed (chain law: ``c`` consumes ``settings``, never the reverse).

        Returns:
            The resulting ``str``.

        """
        return _resolve_env_file(namespace)

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
    def _load_sources(cls) -> Self:
        """Validate current sources, resolving an explicit dotenv repoint now."""
        if FlextConstantsSettings.ENV_FILE_ENV_VAR in os.environ:
            return cls.model_validate({"_env_file": cls.resolve_env_file()})
        return cls()

    @classmethod
    def fetch_global(
        cls, *, overrides: t.MappingKV[str, t.JsonValue] | None = None
    ) -> Self:
        """Return the shared per-class singleton (lazy, thread-safe).

        With ``overrides`` return an isolated clone; the singleton is not
        mutated (use ``update_global`` for that).
        """
        with cls._lock:
            instance = cls._instance
            if overrides is not None:
                cls._validate_overrides(**overrides)
                current = instance if instance is not None else cls._load_sources()
                return current.clone(**overrides)
            if instance is None:
                instance = cls._load_sources()
                cls._instance = instance
            return instance

    @classmethod
    def _merge_overrides(
        cls, current: Self, **overrides: FlextProtocolsSettings.SettingsOverride | None
    ) -> t.MutableMappingKV[str, FlextProtocolsSettings.SettingsOverride | None]:
        """Merge partial nested-model overrides onto the current state.

        Returns:
            The resulting mapping of the merged override values.

        """
        cls._validate_overrides(**overrides)
        merged: t.MutableMappingKV[
            str, FlextProtocolsSettings.SettingsOverride | None
        ] = {}
        for field_name, override_value in overrides.items():
            current_value = getattr(current, field_name, None)
            if isinstance(current_value, BaseModel) and isinstance(
                override_value,
                Mapping,
            ):
                computed = set(type(current_value).model_computed_fields)
                merged_dict = {
                    **current_value.model_dump(mode="python", exclude=computed),
                    **{k: v for k, v in override_value.items() if k not in computed},
                }
                merged[field_name] = type(current_value).model_validate(
                    merged_dict, by_name=True
                )
                continue
            merged[field_name] = override_value
        return merged

    def clone(
        self, **overrides: FlextProtocolsSettings.SettingsOverride | None
    ) -> Self:
        """Deep copy with optional field overrides + re-validation.

        Returns:
            The resulting ``Self``.

        """
        with self.__class__.singleton_disabled():
            if not overrides:
                return self.model_copy(deep=True)
            copied = self.model_copy(deep=True)
            values = copied.model_dump(mode="python", round_trip=True)
            values.update(deepcopy(self._merge_overrides(copied, **overrides)))
            # Validate input, not a model instance (which may skip revalidation).
            # self_instance preserves field-name overrides without rerunning sources.
            type(self).__pydantic_validator__.validate_python(
                values, self_instance=copied, by_name=True
            )
            return copied

    @classmethod
    def update_global(
        cls, **overrides: FlextProtocolsSettings.SettingsOverride | None
    ) -> Self:
        """Validate an isolated candidate, then publish into the stable root.

        Returns:
            The resulting ``Self``.

        """
        with cls._lock:
            cls._validate_overrides(**overrides)
            current = cls._instance
            if current is None:
                current = cls._load_sources()
                validated = current.clone(**overrides) if overrides else current
                cls._instance = validated
                return validated
            if overrides:
                validated = current.clone(**overrides)
                current.__setstate__(validated.__getstate__())
            return current

    @classmethod
    def _validate_overrides(
        cls, **overrides: FlextProtocolsSettings.SettingsOverride | None
    ) -> None:
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

    @classmethod
    def apply_app_namespace(cls, namespace: str) -> None:
        """Set the outer application identity once for the current process."""
        candidate = _validate_app_namespace(namespace)
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
        env_prefix = (
            cls.model_config.get("env_prefix") or FlextConstantsEnvironment.ENV_PREFIX
        )
        return _namespace_dir_name(env_prefix)

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
        return _validate_app_namespace(registered or cls._owner_namespace())

    @classmethod
    def _absolute_app_dir(cls, name: str, root: Path) -> Path:
        namespace = cls._current_app_namespace()
        override = os.environ.get(f"{app_env_prefix(namespace)}{name.upper()}_DIR")
        path = Path(override) if override else root / namespace
        if not path.is_absolute():
            msg = f"{name}_dir must be absolute"
            raise ValueError(msg)
        return path

    @computed_field
    @property
    def cache_dir(self) -> Path:
        """Consuming application's cache directory."""
        return self._absolute_app_dir("cache", _platform_cache_root())

    @computed_field
    @property
    def work_dir(self) -> Path:
        """Consuming application's scratch/work directory."""
        return self._absolute_app_dir("work", _platform_cache_root())

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Consuming application's durable data directory."""
        return self._absolute_app_dir("data", _platform_data_root())

    @computed_field
    @property
    def state_dir(self) -> Path:
        """Consuming application's persistent state directory."""
        return self._absolute_app_dir("state", _platform_state_root())

    @computed_field
    @property
    def config_dir(self) -> Path:
        """Consuming application's user configuration directory."""
        return self._absolute_app_dir("config", platform_config_root())

    @computed_field
    @property
    def runtime_dir(self) -> Path:
        """Consuming application's ephemeral runtime directory.

        Raises:
            ValueError: If runtime_dir must be absolute.

        """
        namespace = self._current_app_namespace()
        override = os.environ.get(f"{app_env_prefix(namespace)}RUNTIME_DIR")
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
        TypeAdapter(
            type(self).model_fields["logging_profile"].rebuild_annotation()
        ).validate_python(self.logging_profile, strict=True)
        return self


settings: FlextSettings = FlextSettings.fetch_global()
"""Pre-instantiated root settings singleton — ``from flext_core import settings``."""

__all__: list[str] = [
    "FlextSettings",
    "PydanticBaseSettingsSource",
    "settings",
]
