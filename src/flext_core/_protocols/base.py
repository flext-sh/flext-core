"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import FlextProtocolsResult


class FlextProtocolsBase:
    """Hierarchical protocol namespace organized by Interface Segregation Principle.

    Hierarchy follows architectural layers:
    - Base: Fundamental interfaces
    - Core: Result handling and model protocols
    - Configuration: Config and context management
    - Infrastructure: DI and container protocols
    - Domain: Business domain protocols
    - Application: CQRS and application layer protocols
    - Utility: Supporting utility protocols
    """

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

    @runtime_checkable
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        model_config: ClassVar[Mapping[str, t.Container]]
        model_fields: ClassVar[Mapping[str, type | str]]

        def model_dump(self, **kwargs: t.Container) -> Mapping[str, t.ValueOrModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.ValueOrModel,
            **kwargs: t.Container,
        ) -> Self:
            """Validate object against model."""
            ...

        def validate(self) -> FlextProtocolsResult.Result[bool]:
            """Validate model."""
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


__all__ = ["FlextProtocolsBase"]
