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
        """FlextProtocolsBase.Base domain service interface."""

        def execute(self) -> FlextProtocolsResult.Result[T]:
            """Execute domain service logic."""
            ...

        def get_service_info(self) -> Mapping[str, t.Scalar]:
            """Get service metadata and configuration information."""
            ...

        def is_valid(self) -> bool:
            """Check if service is in valid state for execution."""
            ...

        def validate_business_rules(self) -> FlextProtocolsResult.Result[bool]:
            """Validate business rules with extensible validation pipeline.business rule validation without external command parameters."""
            ...

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable service objects in the DI container."""

        def dispatch(self, message: BaseModel, /) -> BaseModel:
            """Dispatch a message and return the result."""
            ...


__all__ = ["FlextProtocolsService"]
