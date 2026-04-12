"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from flext_core import FlextProtocolsResult, FlextTypesServices


class FlextProtocolsBase:
    """Hierarchical protocol namespace organized by Interface Segregation Principle."""

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

    @runtime_checkable
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        model_fields: Mapping[str, object]

        def model_dump(
            self,
            *,
            mode: str = "python",
            include: FlextTypesServices.IncEx | None = None,
            exclude: FlextTypesServices.IncEx | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | str = True,
            fallback: Callable[[object], object] | None = None,
            serialize_as_any: bool = False,
        ) -> Mapping[str, object]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: object,
            *,
            strict: bool | None = None,
            extra: object | None = None,
            from_attributes: bool | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Self:
            """Validate arbitrary input into a concrete model instance."""
            ...

        def model_copy(
            self,
            *,
            update: Mapping[str, object] | None = None,
            deep: bool = False,
        ) -> Self:
            """Copy a validated model, optionally updating fields."""
            ...

    @runtime_checkable
    class ModelType[TModel: Model](Protocol):
        """Protocol for Pydantic-compatible model classes."""

        model_fields: Mapping[str, object]

        @classmethod
        def model_validate(
            cls,
            obj: object,
            *,
            strict: bool | None = None,
            extra: object | None = None,
            from_attributes: bool | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> TModel:
            """Validate arbitrary input into a concrete model instance."""
            ...

    @runtime_checkable
    class Routable(Protocol):
        """Protocol for messages that carry explicit route information."""

        @property
        def command_type(self) -> str | None:
            """Command type identifier."""
            ...

        @property
        def event_type(self) -> str | None:
            """Event type identifier."""
            ...

        @property
        def query_type(self) -> str | None:
            """Query type identifier."""
            ...

    @runtime_checkable
    class Executable(Base, Protocol):
        """Protocol for objects that can be executed and report service info."""

        def execute(
            self,
        ) -> FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]: ...

        def service_info(self) -> FlextTypesServices.FlatContainerMapping: ...

    @runtime_checkable
    class ConfigObject(Protocol):
        """Protocol for mapping-like configuration payloads."""

        def get(
            self,
            key: str,
            default: FlextTypesServices.RuntimeAtomic | None = None,
        ) -> FlextTypesServices.RuntimeAtomic | None:
            """Fetch a configuration value by key."""
            ...

        def keys(self) -> Iterable[str]:
            """Return known configuration keys."""
            ...

        def items(self) -> Iterable[tuple[str, FlextTypesServices.RuntimeAtomic]]:
            """Return key/value entries for configuration payloads."""
            ...

    @runtime_checkable
    class Flushable(Protocol):
        """Protocol for objects with a flush() method."""

        def flush(self) -> None: ...

    @runtime_checkable
    class Lock(Protocol):
        """Protocol for lock-like synchronization primitives."""

        def __enter__(self) -> object:
            """Context manager entry."""
            ...

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            value: BaseException | None,
            traceback: TracebackType | None,
            /,
        ) -> None:
            """Context manager exit."""
            ...


__all__: list[str] = ["FlextProtocolsBase"]
