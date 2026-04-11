"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import p


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

        def model_dump(
            self,
            *,
            mode: str = "python",
            include: t.IncEx | None = None,
            exclude: t.IncEx | None = None,
            context: t.ValueOrModel | t.ContainerMapping | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | str = True,
            fallback: Callable[[t.ValueOrModel], t.ValueOrModel] | None = None,
            serialize_as_any: bool = False,
        ) -> Mapping[str, t.ValueOrModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.ModelInput,
            *,
            strict: bool | None = None,
            from_attributes: bool | None = None,
            context: t.ValueOrModel | t.ContainerMapping | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Self:
            """Validate a canonical value-or-model input against the model."""
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

        def execute(self) -> p.Result[t.RuntimeAtomic]: ...

        def service_info(self) -> t.FlatContainerMapping: ...

    @runtime_checkable
    class Flushable(Protocol):
        """Protocol for objects with a flush() method."""

        def flush(self) -> None: ...


__all__ = ["FlextProtocolsBase"]
