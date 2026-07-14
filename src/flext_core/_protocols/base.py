"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableSequence
    from types import TracebackType

    from flext_core import m, p, t


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
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        # NOTE (multi-agent, mro-wkii.17 / agent: codex): declare only the
        # instance capability consumed by interfaces; Pydantic class APIs stay
        # on ``ModelType``.

        def model_dump(
            self,
            *,
            mode: str = "python",
            by_alias: bool | None = None,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
        ) -> t.MappingKV[str, t.JsonPayload | None]:
            """Dump the validated model at an external serialization boundary."""
            ...

        def model_copy(self, *, deep: bool = False) -> Self:
            """Copy a Pydantic-compatible model instance."""
            ...

    @runtime_checkable
    class Routable(Protocol):
        """Base protocol for messages that carry explicit route information."""

    @runtime_checkable
    class CommandRoutable(Routable, Protocol):
        """Protocol for command messages."""

        command_type: str
        """Command type identifier."""

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
    class HasDomainEvents(Protocol):
        """Protocol for DDD aggregate roots that buffer uncommitted domain events.

        Every entity that carries a ``domain_events`` list and an identity
        satisfies this protocol. Use with ``u.add_domain_event`` utility.
        """

        unique_id: str
        domain_events: MutableSequence[m.DomainEvent]


__all__: list[str] = ["FlextProtocolsBase"]
