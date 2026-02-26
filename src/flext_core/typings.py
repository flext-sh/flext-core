"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import re
import typing
from collections.abc import Callable, ItemsView, KeysView, Mapping, Sequence, ValuesView
from datetime import UTC, datetime
from enum import StrEnum
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from re import Pattern
from typing import (
    Annotated,
    ClassVar,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)
from urllib.parse import urlparse

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, RootModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

# LaxStr compatibility for external integrations (LDAP, etc.)

# ============================================================================
# Module-Level TypeVars - Used in flext-core src/
# ============================================================================
# All TypeVars are defined at module level for clarity and accessibility.
# Only TypeVars actually used in flext-core src/ are declared here.
# Note: ParamSpec cannot be used in type aliases within nested classes.

# ============================================================================
# Core Generic TypeVars
# ============================================================================
# Fundamental type variables used throughout flext-core.
T = TypeVar("T")
"""Generic type variable - most commonly used type parameter.
Used in: decorators, container, mixins, pagination, protocols."""
T_co = TypeVar("T_co", covariant=True)
"""Covariant generic type variable - for read-only types.
Used in: protocols, result."""
T_contra = TypeVar("T_contra", contravariant=True)
"""Contravariant generic type variable - for write-only types.
Used in: protocols (for future extensions)."""
E = TypeVar("E")
"""Element type - for collections and sequences.
Used in: collection, enum, args utilities."""
U = TypeVar("U")
"""Utility type - for utility functions and helpers.
Used in: result (for map/flat_map operations)."""
R = TypeVar("R")
"""Return type - for function return values and decorators.
Used in: decorators, args utilities."""
DictValueT = TypeVar("DictValueT")
"""Dictionary value type for RootModel dict-like mixins."""

# ============================================================================
# ParamSpec
# ============================================================================
P = ParamSpec("P")
"""ParamSpec for decorator patterns and variadic function signatures.
Used in: decorators, args utilities."""

# ============================================================================
# Handler TypeVars
# ============================================================================
# Type variables for CQRS handlers and message processing.
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
"""Contravariant message type - for message objects.
Used in: handlers (CQRS message processing)."""
ResultT = TypeVar("ResultT")
"""Result type - generic result type variable.
Used in: handlers (handler return types)."""

# ============================================================================
# Config/Model TypeVars
# ============================================================================
# Type variables for configuration and model types.
T_Model = TypeVar("T_Model", bound=BaseModel)
"""Model type - for Pydantic model types (bound to BaseModel).
Used in: configuration utilities."""
T_Namespace = TypeVar("T_Namespace")
"""Namespace type - for namespace objects.
Used in: config (namespace configuration)."""
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
"""Settings type - for Pydantic settings types (bound to BaseSettings).
Used in: config (settings configuration)."""


# ============================================================================
# Module-Level Types (PEP 695) - model-centric scalar/payload aliases
# ============================================================================

# Internal only: recursive container value for RootModel Dict/ConfigMap/ServiceMap.
# Not exported. Public API uses m.ConfigMap, m.Dict, ScalarValue.
type _ContainerValue = (
    str
    | int
    | float
    | bool
    | datetime
    | None
    | BaseModel
    | Path
    | Sequence[_ContainerValue]
    | Mapping[str, _ContainerValue]
)

type JsonPrimitive = str | int | float | bool | None

type JsonValue = (
    JsonPrimitive | Sequence[_ContainerValue] | Mapping[str, _ContainerValue]
)

type JsonDict = Mapping[str, JsonValue]

# Scalar at module level for use in callable types (not exported)
type _ScalarML = str | int | float | bool | datetime | None
type ScalarValue = _ScalarML
type PayloadValue = _ContainerValue | BaseModel | Path
type RegisterableService = (
    _ContainerValue | BindableLogger | Callable[..., _ContainerValue]
)

# ============================================================================
# Core Callables - strict DI-compatible service types
# ============================================================================
type ServiceInstanceType = _ContainerValue
type FactoryCallable = Callable[[], RegisterableService]
type ResourceCallable = Callable[[], _ContainerValue]

type FactoryRegistrationCallable = Callable[[], _ScalarML | Sequence[_ScalarML]]


_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_HOSTNAME_LABEL_PATTERN = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
_DN_COMPONENT_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9-]*\s*=\s*.+$")
_ORACLE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,29}$")
_SINGER_STREAM_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
_MAX_HOSTNAME_LENGTH = 253
_MAX_PORT = 65535
_MAX_PERCENTAGE = 100.0
_MAX_BOUNDED_LIST_ITEMS = 1000


def _ensure_string(value: str | bytes | bytearray) -> str:
    if not isinstance(value, str):
        msg = "value must be a string"
        raise TypeError(msg)
    return value


def _validate_non_empty_str(value: str) -> str:
    text = _ensure_string(value).strip()
    if not text:
        msg = "value cannot be empty"
        raise ValueError(msg)
    return text


def _validate_stripped_str(value: str) -> str:
    return _ensure_string(value).strip()


def _validate_lower_str(value: str) -> str:
    return _ensure_string(value).strip().lower()


def _validate_upper_str(value: str) -> str:
    return _ensure_string(value).strip().upper()


def _validate_slug_str(value: str) -> str:
    text = _ensure_string(value).strip().lower()
    if not _SLUG_PATTERN.fullmatch(text):
        msg = "value must be a valid slug"
        raise ValueError(msg)
    return text


def _validate_identifier_str(value: str) -> str:
    text = _ensure_string(value).strip()
    if not text.isidentifier():
        msg = "value must be a valid identifier"
        raise ValueError(msg)
    return text


def _validate_hostname(value: str) -> str:
    hostname = _ensure_string(value).strip()
    if not hostname or len(hostname) > _MAX_HOSTNAME_LENGTH:
        msg = "value must be a valid hostname"
        raise ValueError(msg)
    labels = hostname.split(".")
    if any(not _HOSTNAME_LABEL_PATTERN.fullmatch(label) for label in labels):
        msg = "value must be a valid hostname"
        raise ValueError(msg)
    return hostname.lower()


def _validate_port(value: int) -> int:
    if isinstance(value, bool):
        msg = "value must be an integer"
        raise TypeError(msg)
    if value < 1 or value > _MAX_PORT:
        msg = "value must be between 1 and 65535"
        raise ValueError(msg)
    return value


def _validate_uri(value: str) -> str:
    uri = _ensure_string(value).strip()
    parsed = urlparse(uri)
    if not parsed.scheme or not parsed.netloc:
        msg = "value must be a valid URI"
        raise ValueError(msg)
    return uri


def _validate_ipv4(value: str) -> str:
    address = ip_address(_ensure_string(value).strip())
    if not isinstance(address, IPv4Address):
        msg = "value must be a valid IPv4 address"
        raise TypeError(msg)
    return str(address)


def _validate_ipv6(value: str) -> str:
    address = ip_address(_ensure_string(value).strip())
    if not isinstance(address, IPv6Address):
        msg = "value must be a valid IPv6 address"
        raise TypeError(msg)
    return str(address)


def _validate_positive_int(value: int) -> int:
    if isinstance(value, bool):
        msg = "value must be an integer"
        raise TypeError(msg)
    if value <= 0:
        msg = "value must be positive"
        raise ValueError(msg)
    return value


def _validate_non_negative_int(value: int) -> int:
    if isinstance(value, bool):
        msg = "value must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = "value must be non-negative"
        raise ValueError(msg)
    return value


def _validate_percentage(value: float) -> float:
    if isinstance(value, bool):
        msg = "value must be numeric"
        raise TypeError(msg)
    percentage = float(value)
    if percentage < 0.0 or percentage > _MAX_PERCENTAGE:
        msg = "value must be between 0 and 100"
        raise ValueError(msg)
    return percentage


def _validate_bounded_float(value: float) -> float:
    if isinstance(value, bool):
        msg = "value must be numeric"
        raise TypeError(msg)
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        msg = "value must be between 0.0 and 1.0"
        raise ValueError(msg)
    return numeric


def _validate_existing_path(value: Path) -> Path:
    path = Path(value)
    if not path.exists():
        msg = "path must exist"
        raise ValueError(msg)
    return path


def _validate_existing_file_path(value: Path) -> Path:
    path = _validate_existing_path(value)
    if not path.is_file():
        msg = "path must be an existing file"
        raise ValueError(msg)
    return path


def _validate_existing_dir_path(value: Path) -> Path:
    path = _validate_existing_path(value)
    if not path.is_dir():
        msg = "path must be an existing directory"
        raise ValueError(msg)
    return path


def _validate_writable_path(value: Path) -> Path:
    path = Path(value)
    target = path if path.exists() else path.parent
    if not target.exists() or not target.is_dir() or not os.access(target, os.W_OK):
        msg = "path parent must exist and be writable"
        raise ValueError(msg)
    return path


def _normalize_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_past_datetime(value: datetime) -> datetime:
    dt = _normalize_utc_datetime(value)
    if dt >= datetime.now(UTC):
        msg = "value must be in the past"
        raise ValueError(msg)
    return dt


def _validate_future_datetime(value: datetime) -> datetime:
    dt = _normalize_utc_datetime(value)
    if dt <= datetime.now(UTC):
        msg = "value must be in the future"
        raise ValueError(msg)
    return dt


def _validate_iso_datetime_str(value: str) -> str:
    text = _ensure_string(value).strip()
    if not text:
        msg = "value cannot be empty"
        raise ValueError(msg)
    normalized = text.replace("Z", "+00:00")
    try:
        _ = datetime.fromisoformat(normalized)
    except ValueError as exc:
        msg = "value must be an ISO 8601 datetime string"
        raise ValueError(msg) from exc
    return text


def _validate_non_empty_list[T](value: list[T]) -> list[T]:
    if len(value) == 0:
        msg = "list cannot be empty"
        raise ValueError(msg)
    return value


def _validate_unique_list[T](value: list[T]) -> list[T]:
    seen: set[str] = set()
    for item in value:
        marker = repr(item)
        if marker in seen:
            msg = "list items must be unique"
            raise ValueError(msg)
        seen.add(marker)
    return value


def _validate_bounded_list[T](value: list[T]) -> list[T]:
    if len(value) > _MAX_BOUNDED_LIST_ITEMS:
        msg = "list cannot contain more than 1000 items"
        raise ValueError(msg)
    return value


def _validate_dn_str(value: str) -> str:
    text = _ensure_string(value).strip()
    if not text:
        msg = "DN cannot be empty"
        raise ValueError(msg)
    parts = [part.strip() for part in text.split(",")]
    if any(not _DN_COMPONENT_PATTERN.fullmatch(part) for part in parts):
        msg = "value must be a valid distinguished name"
        raise ValueError(msg)
    return text


def _validate_ldif_line(value: str) -> str:
    text = _ensure_string(value)
    if not text.strip():
        msg = "LDIF line cannot be empty"
        raise ValueError(msg)
    if "\n" in text or "\r" in text:
        msg = "LDIF line must not contain newlines"
        raise ValueError(msg)
    return text


def _validate_oracle_identifier(value: str) -> str:
    identifier = _ensure_string(value).strip().upper()
    if not _ORACLE_IDENTIFIER_PATTERN.fullmatch(identifier):
        msg = "value must be a valid Oracle identifier"
        raise ValueError(msg)
    return identifier


def _validate_singer_stream_name(value: str) -> str:
    stream = _ensure_string(value).strip()
    if not _SINGER_STREAM_PATTERN.fullmatch(stream):
        msg = "value must be a valid Singer stream name"
        raise ValueError(msg)
    return stream


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All complex types (type aliases) are organized within this class.
    TypeVars are defined at module level (see above).

    All aliases use ``TypeAlias`` annotation (not PEP 695 ``type`` statement)
    so that basedpyright resolves them as proper class attributes at workspace scope.
    """

    # =====================================================================
    # COMPLEX TYPE ALIASES (TypeAlias for basedpyright compatibility)
    # =====================================================================

    type LaxStr = str | bytes | bytearray
    """LaxStr compatibility for ldap3 integration."""

    # =========================================================================
    # ALIGNED TYPE HIERARCHY (Pydantic-safe, no dynamic top types)
    # =========================================================================

    # Tier 1: Scalar primitives (immutable, JSON-safe)
    type ScalarValue = str | int | float | bool | datetime | None
    type ScalarAlias = ScalarValue

    # Tier 2: Pydantic-safe metadata values
    type MetadataScalarValue = str | int | float | bool | None
    type MetadataListValue = list[str | int | float | bool | None]

    # Tier 2.5: Pydantic-safe config types for Field() annotations
    type PydanticConfigValue = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )

    # Tier 3: General value types (superset including BaseModel, Path, datetime)
    type GeneralScalarValue = str | int | float | bool | datetime | None
    type GeneralListValue = list[str | int | float | bool | datetime | None]

    # Input type for guard functions (flat union, no recursion; prefer models at API)
    type GuardInputValue = _ContainerValue

    # Recursive value type for m.ConfigMap / m.Dict root (no isinstance; use models)
    type ConfigMapValue = _ContainerValue

    type FlexibleValue = _ContainerValue

    # RegisterableService - fixed types only
    RegisterableService: TypeAlias = RegisterableService

    # RegistrablePlugin - fixed types only
    type RegistrablePlugin = (
        ScalarValue | BaseModel | Callable[..., ScalarValue | BaseModel]
    )

    # Constant value type - all possible constant types in FlextConstants
    type ConstantValue = (
        str
        | int
        | float
        | bool
        | ConfigDict
        | SettingsConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | Mapping[str, str | int]
        | StrEnum
        | type[StrEnum]
        | Pattern[str]
        | type
    )

    class ObjectList(RootModel[list[_ContainerValue]]):
        """Sequence of container values for batch operations. Use m.* models when possible."""

        root: list[_ContainerValue]

    # File content type supporting all serializable formats
    type FileContent = str | bytes | BaseModel | Sequence[Sequence[str]]

    # Sortable value type
    type SortableObjectType = str | int | float

    # =========================================================================
    # Conversion Mode Types
    # =========================================================================
    type ConversionMode = Literal[
        "to_str",
        "to_str_list",
        "normalize",
        "join",
    ]

    # MetadataAttributeValue - fixed scalar/mapping/list (Mapping, no dict)
    type MetadataAttributeValue = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | Mapping[
            str,
            str
            | int
            | float
            | bool
            | datetime
            | list[str | int | float | bool | datetime | None]
            | None,
        ]
        | list[str | int | float | bool | datetime | None]
    )

    # Type hint specifier - used for type introspection
    type TypeHintSpecifier = type | str | Callable[[ScalarValue], ScalarValue]

    # Generic type argument
    type GenericTypeArgument = str | type[ScalarValue]

    # Message type specifier
    type MessageTypeSpecifier = str | type

    # Type origin specifier
    type TypeOriginSpecifier = (
        str | type[ScalarValue] | Callable[[ScalarValue], ScalarValue]
    )

    # JSON types - re-exported from module level
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue
    JsonDict: TypeAlias = JsonDict

    # General value type for handlers and config boundaries
    type GeneralValueType = _ContainerValue

    # Single consolidated callable type for handlers and validators
    type HandlerCallable = Callable[[ScalarValue], ScalarValue]

    # Acceptable message types for handlers - scalars and models only
    type AcceptableMessageType = ScalarValue | BaseModel | Sequence[ScalarValue]

    # Conditional execution callable types
    type ConditionCallable = Callable[[ScalarValue], bool]

    # Handler type union
    type HandlerType = Callable[[ScalarValue], ScalarValue] | BaseModel

    # =========================================================================
    # GENERIC CONTAINERS (RootModels - use m.ConfigMap, m.Dict)
    # =========================================================================
    # Replaces raw dict aliases with strict typed models
    # =========================================================================

    @typing.runtime_checkable
    class _RootDictProtocol[RootValueT](typing.Protocol):
        root: dict[str, RootValueT]

    class _RootDictModel[DictValueT](RootModel[dict[str, DictValueT]]):
        def __getitem__(self, key: str) -> DictValueT:
            return self.root[key]

        def __setitem__(self, key: str, value: DictValueT) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __len__(self) -> int:
            return len(self.root)

        def __contains__(self, key: str) -> bool:
            return key in self.root

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.get(key, default)

        def items(self) -> ItemsView[str, DictValueT]:
            return self.root.items()

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def values(self) -> ValuesView[DictValueT]:
            return self.root.values()

        def update(self, other: Mapping[str, DictValueT]) -> None:
            self.root.update(other)

        def clear(self) -> None:
            self.root.clear()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            return self.root.setdefault(key, default)

    class Dict(_RootDictModel[_ContainerValue]):
        """Generic dictionary container. Prefer m.Dict in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ConfigMap(_RootDictModel[_ContainerValue]):
        """Configuration map container. Prefer m.ConfigMap in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    type ConfigurationMapping = ConfigMap
    type ConfigurationDict = ConfigMap

    class ServiceMap(_RootDictModel[_ContainerValue]):
        """Service registry map container. Prefer m.ServiceMap in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ErrorMap(_RootDictModel[int | str | dict[str, int]]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]] = Field(default_factory=dict)

    type IncEx = set[str] | Mapping[str, set[str] | bool]

    type DecoratorType = Callable[
        [Callable[[ScalarValue], ScalarValue]], Callable[[ScalarValue], ScalarValue]
    ]

    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], _ContainerValue]
    FactoryRegistrationCallable: TypeAlias = FactoryRegistrationCallable

    class FactoryMap(_RootDictModel[FactoryRegistrationCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, FactoryRegistrationCallable] = Field(default_factory=dict)

    class ResourceMap(_RootDictModel[ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, ResourceCallable] = Field(default_factory=dict)

    # =========================================================================
    # Validation mapping types (used in _models/validation.py)
    # =========================================================================
    class ValidatorCallable(
        RootModel[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]],
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]

        def __call__(self, value: _ScalarML | BaseModel) -> _ScalarML | BaseModel:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[
            dict[
                str,
                Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
            ]
        ],
    ):
        """Shared API for validator map containers."""

        def items(
            self,
        ) -> ItemsView[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]:
            """Get validator items."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.items()

        def values(
            self,
        ) -> ValuesView[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],]:
            """Get validator values."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ] = Field(default_factory=dict)

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ] = Field(default_factory=dict)

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ] = Field(default_factory=dict)

    # Error/Exception types (used in exceptions.py)
    # ErrorTypeMapping removed - Use m.ErrorMap
    type ExceptionKwargsType = str | int | float | bool | datetime | None

    # General nested dict type (used in guards.py)
    # GeneralNestedDict removed - Use m.Dict

    # ServiceMapping for service registration
    # ServiceMapping removed - Use m.ServiceMap

    # =====================================================================
    # Pydantic Models
    # =====================================================================
    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        results: list[_ScalarML] = Field(default_factory=list)
        errors: list[tuple[int, str]] = Field(default_factory=list)
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

    type NonEmptyStr = Annotated[str, AfterValidator(_validate_non_empty_str)]
    type StrippedStr = Annotated[str, AfterValidator(_validate_stripped_str)]
    type LowerStr = Annotated[str, AfterValidator(_validate_lower_str)]
    type UpperStr = Annotated[str, AfterValidator(_validate_upper_str)]
    type SlugStr = Annotated[str, AfterValidator(_validate_slug_str)]
    type IdentifierStr = Annotated[str, AfterValidator(_validate_identifier_str)]

    type ValidatedHostname = Annotated[str, AfterValidator(_validate_hostname)]
    type ValidatedPort = Annotated[int, AfterValidator(_validate_port)]
    type ValidatedURI = Annotated[str, AfterValidator(_validate_uri)]
    type ValidatedIPv4 = Annotated[str, AfterValidator(_validate_ipv4)]
    type ValidatedIPv6 = Annotated[str, AfterValidator(_validate_ipv6)]

    type PositiveInt = Annotated[int, AfterValidator(_validate_positive_int)]
    type NonNegativeInt = Annotated[int, AfterValidator(_validate_non_negative_int)]
    type Percentage = Annotated[float, AfterValidator(_validate_percentage)]
    type BoundedFloat = Annotated[float, AfterValidator(_validate_bounded_float)]

    type ExistingPath = Annotated[Path, AfterValidator(_validate_existing_path)]
    type ExistingFilePath = Annotated[
        Path, AfterValidator(_validate_existing_file_path)
    ]
    type ExistingDirPath = Annotated[Path, AfterValidator(_validate_existing_dir_path)]
    type WritablePath = Annotated[Path, AfterValidator(_validate_writable_path)]

    type UTCDatetime = Annotated[datetime, AfterValidator(_normalize_utc_datetime)]
    type PastDatetime = Annotated[datetime, AfterValidator(_validate_past_datetime)]
    type FutureDatetime = Annotated[datetime, AfterValidator(_validate_future_datetime)]
    type ISODatetimeStr = Annotated[str, AfterValidator(_validate_iso_datetime_str)]

    type NonEmptyList[T] = Annotated[list[T], AfterValidator(_validate_non_empty_list)]
    type UniqueList[T] = Annotated[list[T], AfterValidator(_validate_unique_list)]
    type BoundedList[T] = Annotated[list[T], AfterValidator(_validate_bounded_list)]

    type DNStr = Annotated[str, AfterValidator(_validate_dn_str)]
    type LDIFLine = Annotated[str, AfterValidator(_validate_ldif_line)]
    type OracleIdentifier = Annotated[str, AfterValidator(_validate_oracle_identifier)]
    type SingerStreamName = Annotated[str, AfterValidator(_validate_singer_stream_name)]

    # =====================================================================
    # VALIDATION TYPES (Annotated with Pydantic constraints)
    # =====================================================================

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        type PortNumber = Annotated[int, Field(ge=1, le=65535)]
        """Port number type alias (1-65535)."""

        type PositiveTimeout = Annotated[float, Field(gt=0.0, le=300.0)]
        """Positive timeout in seconds (0-300)."""

        type RetryCount = Annotated[int, Field(ge=0, le=10)]
        """Retry count (0-10)."""

        type WorkerCount = Annotated[int, Field(ge=1, le=100)]
        """Worker count (1-100)."""

        # =====================================================================
        # EXTENDED VALIDATION TYPES - Reusable Annotated aliases
        # =====================================================================

        type NonEmptyStr = Annotated[str, Field(min_length=1)]
        """Non-empty string (minimum 1 character)."""

        type StrippedStr = Annotated[str, Field(min_length=1)]
        """String with whitespace stripped (minimum 1 character after strip)."""

        type UriString = Annotated[str, Field(min_length=1)]
        """URI string with scheme validation (e.g., http://, https://)."""

        type HostnameStr = Annotated[str, Field(min_length=1)]
        """Hostname string with format validation."""

        type PositiveInt = Annotated[int, Field(gt=0)]
        """Positive integer (> 0)."""

        type NonNegativeInt = Annotated[int, Field(ge=0)]
        """Non-negative integer (>= 0)."""

        type BoundedStr = Annotated[str, Field(min_length=1, max_length=255)]
        """Bounded string (1-255 characters)."""

        type TimestampStr = Annotated[str, Field(min_length=1)]
        """ISO 8601 timestamp string."""


# ============================================================================
# FACTORY FUNCTIONS FOR DYNAMIC ANNOTATED TYPES
# ============================================================================


def bounded_int_factory(min_val: int, max_val: int) -> object:
    """Factory for bounded integer type.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Annotated type alias for bounded integer

    """
    return Annotated[int, Field(ge=min_val, le=max_val)]


def choice_str_factory(allowed_values: list[str]) -> object:
    """Factory for choice string type.

    Args:
        allowed_values: List of allowed string values

    Returns:
        Annotated type alias for choice string

    """
    return Annotated[str, Field(pattern=f"^({'|'.join(allowed_values)})$")]


def pattern_str_factory(pattern: str) -> object:
    """Factory for regex pattern string type.

    Args:
        pattern: Regular expression pattern

    Returns:
        Annotated type alias for pattern string

    """
    return Annotated[str, Field(pattern=pattern)]


t = FlextTypes

# ============================================================================
# Module-Level Re-exports of Validation Types
# ============================================================================
PortNumber = FlextTypes.Validation.PortNumber
NonEmptyStr = FlextTypes.NonEmptyStr
StrippedStr = FlextTypes.StrippedStr
LowerStr = FlextTypes.LowerStr
UpperStr = FlextTypes.UpperStr
SlugStr = FlextTypes.SlugStr
IdentifierStr = FlextTypes.IdentifierStr
ValidatedHostname = FlextTypes.ValidatedHostname
ValidatedPort = FlextTypes.ValidatedPort
ValidatedURI = FlextTypes.ValidatedURI
ValidatedIPv4 = FlextTypes.ValidatedIPv4
ValidatedIPv6 = FlextTypes.ValidatedIPv6
PositiveInt = FlextTypes.PositiveInt
NonNegativeInt = FlextTypes.NonNegativeInt
Percentage = FlextTypes.Percentage
BoundedFloat = FlextTypes.BoundedFloat
ExistingPath = FlextTypes.ExistingPath
ExistingFilePath = FlextTypes.ExistingFilePath
ExistingDirPath = FlextTypes.ExistingDirPath
WritablePath = FlextTypes.WritablePath
UTCDatetime = FlextTypes.UTCDatetime
PastDatetime = FlextTypes.PastDatetime
FutureDatetime = FlextTypes.FutureDatetime
ISODatetimeStr = FlextTypes.ISODatetimeStr
DNStr = FlextTypes.DNStr
LDIFLine = FlextTypes.LDIFLine
OracleIdentifier = FlextTypes.OracleIdentifier
SingerStreamName = FlextTypes.SingerStreamName

UriString = FlextTypes.Validation.UriString
HostnameStr = FlextTypes.Validation.HostnameStr
BoundedStr = FlextTypes.Validation.BoundedStr
TimestampStr = FlextTypes.Validation.TimestampStr
PositiveTimeout = FlextTypes.Validation.PositiveTimeout
RetryCount = FlextTypes.Validation.RetryCount
WorkerCount = FlextTypes.Validation.WorkerCount

__all__ = [
    "BoundedFloat",
    "BoundedStr",
    "DNStr",
    "ExistingDirPath",
    "ExistingFilePath",
    "ExistingPath",
    "FlextTypes",
    "FutureDatetime",
    "HostnameStr",
    "ISODatetimeStr",
    "IdentifierStr",
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
    "LDIFLine",
    "LowerStr",
    "MessageT_contra",
    "NonEmptyStr",
    "NonNegativeInt",
    "OracleIdentifier",
    "P",
    "PastDatetime",
    "Percentage",
    "PortNumber",
    "PositiveInt",
    "PositiveTimeout",
    "R",
    "ResultT",
    "RetryCount",
    "SingerStreamName",
    "SlugStr",
    "StrippedStr",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TimestampStr",
    "U",
    "UTCDatetime",
    "UpperStr",
    "UriString",
    "ValidatedHostname",
    "ValidatedIPv4",
    "ValidatedIPv6",
    "ValidatedPort",
    "ValidatedURI",
    "WorkerCount",
    "WritablePath",
    "bounded_int_factory",
    "choice_str_factory",
    "pattern_str_factory",
    "t",
]
