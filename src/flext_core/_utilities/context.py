"""Context utility helpers for creating and managing context variables.

These helpers provide factory functions for creating StructlogProxyContextVar
instances with proper typing, ensuring consistent context variable creation
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from flext_core import FlextModelsContext, t


class FlextUtilitiesContextVariables:
    """Context variables using structlog as single source of truth.

    Centralized, immutable container for all context variable proxies.
    Access via u.ContextVariables.VARIABLE_NAME for standard context vars.
    """

    # Correlation variables for distributed tracing
    CORRELATION_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]]
    PARENT_CORRELATION_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]]

    # Service context variables for identification
    SERVICE_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]]
    SERVICE_VERSION: Final[FlextModelsContext.StructlogProxyContextVar[str]]

    # Request context variables for metadata
    USER_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]]
    REQUEST_ID: Final[FlextModelsContext.StructlogProxyContextVar[str]]
    REQUEST_TIMESTAMP: Final[FlextModelsContext.StructlogProxyContextVar[datetime]]

    # Performance context variables for timing
    OPERATION_NAME: Final[FlextModelsContext.StructlogProxyContextVar[str]]
    OPERATION_START_TIME: Final[FlextModelsContext.StructlogProxyContextVar[datetime]]
    OPERATION_METADATA: Final[FlextModelsContext.StructlogProxyContextVar[t.JsonMapping]]


class FlextUtilitiesContext:
    """Context utility helpers for creating and managing context variables."""

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[str]:
        """Create StructlogProxyContextVar[str] instance."""
        return FlextModelsContext.StructlogProxyContextVar(
            key,
            default=default,
        )

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[datetime]:
        """Create StructlogProxyContextVar[datetime] instance."""
        return FlextModelsContext.StructlogProxyContextVar(
            key,
            default=default,
        )

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: t.JsonMapping | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[t.JsonMapping]:
        """Create StructlogProxyContextVar[dict] instance."""
        return FlextModelsContext.StructlogProxyContextVar(
            key,
            default=default,
        )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContext"]
