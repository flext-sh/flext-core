"""FlextProtocolsService - service and repository protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult

Base = FlextProtocolsBase.Base


class FlextProtocolsService:
    """Protocols for service execution and repository access."""

    @runtime_checkable
    class Service[T](Base, Protocol):
        """Base domain service interface.

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
    class Repository[T](Base, Protocol):
        """Data access interface."""

        def delete(self, entity_id: str) -> FlextProtocolsResult.Result[bool]:
            """Delete entity."""
            ...

        def find_all(self) -> FlextProtocolsResult.Result[Sequence[T]]:
            """Find all entities."""
            ...

        def get_by_id(self, entity_id: str) -> FlextProtocolsResult.Result[T]:
            """Get entity by ID."""
            ...

        def save(self, entity: T) -> FlextProtocolsResult.Result[T]:
            """Save entity."""
            ...


__all__ = ["FlextProtocolsService"]
