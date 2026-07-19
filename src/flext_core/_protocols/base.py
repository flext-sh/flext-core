"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    # mro-wkii.17.26 (codex): p is composing this module; importing the root
    # m/p/t facades here re-enters protocols and models before t is available.
    from collections.abc import Iterable, MutableSequence
    from flext_core import p, t
    from pydantic.fields import FieldInfo
    from types import TracebackType


class FlextProtocolsBase:
    """Hierarchical protocol namespace organized by Interface Segregation Principle."""

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

    @runtime_checkable
    class AttributeProbe(Protocol):
        """Structural marker for values inspected only via ``hasattr``/``getattr``.

        Empty Protocol body — accepts any runtime value structurally. Used by
        ``FlextDecorators`` and other infrastructure helpers as a typed
        parameter annotation when the actual contract is "any value that
        may carry inspectable attributes" without loose top-type annotations.
        """

    @runtime_checkable
    class ModuleOwned(Protocol):
        """Structural contract for runtime symbols that expose owner module."""

        __module__: str

    @runtime_checkable
    class BaseModel(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        def model_dump(
            self,
            *,
            mode: str = "python",
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
        ) -> t.JsonDict:
            """Dump the validated model at an external serialization boundary."""
            ...

        def model_dump_json(
            self,
            *,
            indent: int | None = None,
            ensure_ascii: bool = False,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | Literal["none", "warn", "error"] = True,
            serialize_as_any: bool = False,
            polymorphic_serialization: bool | None = None,
        ) -> str:
            """Serialize the validated model to a JSON string."""
            ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[str, FlextProtocolsBase.AttributeProbe] | None = None,
            deep: bool = False,
        ) -> Self:
            """Copy a Pydantic-compatible model instance."""
            ...

        __pydantic_fields__: ClassVar[t.MappingKV[str, FieldInfo]]
        """Pydantic-internal mapping of field names to ``FieldInfo``."""

    @runtime_checkable
    class ArbitraryTypesModel(BaseModel, Protocol):
        """Structural protocol for Pydantic models allowing arbitrary types."""

    @runtime_checkable
    class ContractModel(BaseModel, Protocol):
        """Structural protocol for immutable strict Pydantic contract models."""

    # mro-dxrp.1.4 (agent: Sisyphus-Junior): expose the immutable value-object
    # capability required by runtime Pydantic annotations without model imports.
    @runtime_checkable
    class Value(ContractModel, Protocol):
        """Structural contract for immutable value-semantic models."""

        def __hash__(self) -> int: ...

    @runtime_checkable
    class Routable(Protocol):
        """Base protocol for messages that carry explicit route information."""

    @runtime_checkable
    class CommandRoutable(Routable, Protocol):
        """Protocol for command messages."""

        command_type: str
        """Command type identifier."""

    # mro-dxrp.1.4 (agent: Sisyphus-Junior): own the complete public command
    # contract used by runtime Pydantic annotations without importing models.
    @runtime_checkable
    class Command(ArbitraryTypesModel, CommandRoutable, Protocol):
        """Structural contract for validated CQRS command messages."""

        message_type: Literal["command"]
        command_id: str
        issuer_id: str | None

    @runtime_checkable
    class EventRoutable(Routable, Protocol):
        """Protocol for event messages."""

        event_type: str
        """Event type identifier."""

    @runtime_checkable
    class QueryRoutable(Routable, Protocol):
        """Protocol for query messages."""

        query_type: str
        """Query type identifier."""

    @runtime_checkable
    class Executable(Base, Protocol):
        """Protocol for objects that can be executed and report service info."""

        def execute(self) -> p.Result[t.JsonPayload]: ...

        def service_info(self) -> t.JsonMapping: ...

    @runtime_checkable
    class ConfigObject(Protocol):
        """Protocol for mapping-like configuration payloads."""

        def get(
            self, key: str, default: t.JsonPayload | None = None
        ) -> t.JsonPayload | None:
            """Fetch a configuration value by key."""
            ...

        def keys(self) -> Iterable[str]:
            """Return known configuration keys."""
            ...

        def items(self) -> Iterable[tuple[str, t.JsonPayload]]:
            """Return key/value entries for configuration payloads."""
            ...

    @runtime_checkable
    class Flushable(Protocol):
        """Protocol for objects with a flush() method."""

        def flush(self) -> None: ...

    @runtime_checkable
    class Lock(Protocol):
        """Protocol for lock-like synchronization primitives."""

        def __enter__(self) -> bool:
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

    @runtime_checkable
    class DomainEvent(BaseModel, Protocol):
        """Domain event identity and routing fields buffered by entities."""

        @property
        def event_type(self) -> str: ...

        @property
        def aggregate_id(self) -> str: ...

    @runtime_checkable
    class HasDomainEvents(Protocol):
        """Protocol for DDD aggregate roots that buffer uncommitted domain events.

        Every entity that carries a ``domain_events`` list and an identity
        satisfies this protocol. Use with ``u.add_domain_event`` utility.
        """

        unique_id: str
        domain_events: MutableSequence[FlextProtocolsBase.DomainEvent]


__all__: list[str] = ["FlextProtocolsBase"]
