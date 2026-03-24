"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from flext_core import r, t


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

        def model_dump(self, **kwargs: t.Container) -> Mapping[str, t.ValueOrModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.ValueOrModel,
            **kwargs: t.Container,
        ) -> Self:
            """Validate t.NormalizedValue against model."""
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

    @classmethod
    def check_protocol_compliance(
        cls,
        instance: t.NormalizedValue,
        protocol: type,
    ) -> bool:
        """Check protocol compliance via stdlib isinstance().

        Uses @runtime_checkable Protocol + isinstance() — the Python 3.13+
        standard way to do structural type checks at runtime.
        """
        try:
            return isinstance(instance, protocol)
        except TypeError:
            return False

    @classmethod
    def validate_protocol_compliance(
        cls,
        target_cls: type,
        protocol: type,
        class_name: str,
    ) -> None:
        """Validate that a class implements all required protocol members.

        Uses @runtime_checkable Protocol — no custom introspection needed.
        """
        try:
            compliant = issubclass(target_cls, protocol)
        except TypeError:
            compliant = False
        if not compliant:
            protocol_name = (
                protocol.__name__ if hasattr(protocol, "__name__") else str(protocol)
            )
            msg = f"Class '{class_name}' does not implement protocol '{protocol_name}'"
            raise TypeError(msg)

    @runtime_checkable
    class Executable(Base, Protocol):
        """Protocol for objects that can be executed and report service info."""

        def execute(self) -> r[t.RuntimeAtomic]: ...

        def get_service_info(self) -> t.FlatContainerMapping: ...

    @runtime_checkable
    class Flushable(Protocol):
        """Protocol for objects with a flush() method."""

        def flush(self) -> None: ...


__all__ = ["FlextProtocolsBase"]
