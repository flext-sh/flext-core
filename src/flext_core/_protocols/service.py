"""FlextProtocolsService - service and repository protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from flext_core import FlextProtocolsBase, FlextProtocolsResult, t


class FlextProtocolsService:
    """Protocols for service execution and repository access."""

    @runtime_checkable
    class Service[T](FlextProtocolsBase.Base, Protocol):
        """FlextProtocolsBase.Base domain service interface.

        Reflects real implementations like FlextService which executes
        domain logic without requiring command parameters (services are
        self-contained with their own configuration).
        """

        def execute(self) -> FlextProtocolsResult.Result[T]:
            """Execute domain service logic.

            Reflects real implementations like FlextService which don't
            require command parameters - services are self-contained with
            their own configuration and context.
            """
            ...

        def get_service_info(self) -> Mapping[str, t.Scalar]:
            """Get service metadata and configuration information.

            Reflects real implementations like FlextService which provide
            service metadata for observability and debugging.
            """
            ...

        def is_valid(self) -> bool:
            """Check if service is in valid state for execution.

            Reflects real implementations like FlextService which check
            validity based on internal state and business rules.
            """
            ...

        def validate_business_rules(self) -> FlextProtocolsResult.Result[bool]:
            """Validate business rules with extensible validation pipeline.

            Reflects real implementations like FlextService which perform
            business rule validation without external command parameters.
            """
            ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable service objects in the DI container.

        Matches FlextDispatcher and similar services that expose a dispatch method.
        Parameter uses Protocol bound since dispatch implementations accept varying
        message protocols (Routable, Command, Query).
        """

        def dispatch(self, message: BaseModel, /) -> BaseModel:
            """Dispatch a message and return the result."""
            ...


__all__ = ["FlextProtocolsService"]
