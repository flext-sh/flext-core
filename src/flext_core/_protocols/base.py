"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from pydantic import BaseModel

from flext_core import t

if TYPE_CHECKING:
    from flext_core._protocols.result import FlextProtocolsResult


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

        pass

    @runtime_checkable
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        model_config: ClassVar[Mapping[str, t.Container]]
        model_fields: ClassVar[Mapping[str, type | str]]

        def model_dump(
            self, **kwargs: t.Container
        ) -> Mapping[str, t.NormalizedValue | BaseModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.NormalizedValue | BaseModel,
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

    _protocol_specs: ClassVar[
        Mapping[str, Callable[[t.NormalizedValue], bool]] | None
    ] = None
    _protocol_type_map: ClassVar[Mapping[type, str] | None] = None

    @classmethod
    def get_protocol_specs(
        cls,
    ) -> Mapping[str, Callable[[t.NormalizedValue], bool]]:
        if cls._protocol_specs is None:
            from flext_core._protocols.config import FlextProtocolsConfig  # noqa: PLC0415  # Circular: _protocols/base ↔ siblings — lazy-cached, runs once
            from flext_core._protocols.context import FlextProtocolsContext  # noqa: PLC0415
            from flext_core._protocols.di import FlextProtocolsDI  # noqa: PLC0415
            from flext_core._protocols.handler import FlextProtocolsHandler  # noqa: PLC0415
            from flext_core._protocols.logging import FlextProtocolsLogging  # noqa: PLC0415
            from flext_core._protocols.result import FlextProtocolsResult  # noqa: PLC0415
            from flext_core._protocols.service import FlextProtocolsService  # noqa: PLC0415

            cls._protocol_specs = MappingProxyType({
                "config": lambda v: isinstance(v, FlextProtocolsConfig.Settings),
                "context": lambda v: isinstance(v, FlextProtocolsContext.Context),
                "container": lambda v: isinstance(v, FlextProtocolsDI.Container),
                "command_bus": lambda v: isinstance(
                    v, FlextProtocolsHandler.CommandBus
                ),
                "handler": lambda v: isinstance(v, FlextProtocolsHandler.Handler),
                "logger": lambda v: isinstance(v, FlextProtocolsLogging.Logger),
                "result": lambda v: isinstance(v, FlextProtocolsResult.Result),
                "service": lambda v: isinstance(v, FlextProtocolsService.Service),
                "middleware": lambda v: isinstance(v, FlextProtocolsHandler.Middleware),
            })
        return cls._protocol_specs

    @classmethod
    def get_protocol_type_map(cls) -> Mapping[type, str]:
        if cls._protocol_type_map is None:
            from flext_core._protocols.config import FlextProtocolsConfig
            from flext_core._protocols.context import FlextProtocolsContext
            from flext_core._protocols.di import FlextProtocolsDI
            from flext_core._protocols.handler import FlextProtocolsHandler
            from flext_core._protocols.logging import FlextProtocolsLogging
            from flext_core._protocols.result import FlextProtocolsResult
            from flext_core._protocols.service import FlextProtocolsService

            cls._protocol_type_map = MappingProxyType({
                FlextProtocolsConfig.Settings: "config",
                FlextProtocolsContext.Context: "context",
                FlextProtocolsDI.Container: "container",
                FlextProtocolsHandler.CommandBus: "command_bus",
                FlextProtocolsHandler.Handler: "handler",
                FlextProtocolsLogging.Logger: "logger",
                FlextProtocolsResult.Result: "result",
                FlextProtocolsService.Service: "service",
                FlextProtocolsHandler.Middleware: "middleware",
            })
        return cls._protocol_type_map

    @classmethod
    def check_protocol_compliance(
        cls,
        instance: object,
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
