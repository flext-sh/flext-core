"""Context utility helpers for creating and managing context variables.

These helpers provide factory functions for creating StructlogProxyContextVar
instances with proper typing, ensuring consistent context variable creation
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime

from flext_core._models.context import FlextModelsContext

# Import FlextTypes class directly for type aliases access
from flext_core.typings import FlextTypes


class FlextUtilitiesContext:
    """Context utility helpers for creating and managing context variables."""

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[str]:
        """Create StructlogProxyContextVar[str] instance.

        Helper factory for creating string-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[str] instance

        Example:
            >>> var = uContext.create_str_proxy("correlation_id")
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[str] = (
            FlextModelsContext.StructlogProxyContextVar[str](key, default=default)
        )
        return proxy

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[datetime]:
        """Create StructlogProxyContextVar[datetime] instance.

        Helper factory for creating datetime-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[datetime] instance

        Example:
            >>> from datetime import datetime
            >>> var = uContext.create_datetime_proxy("start_time")
            >>> var.set(datetime.now())
            >>> var.get()  # Returns datetime instance

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[datetime] = (
            FlextModelsContext.StructlogProxyContextVar[datetime](
                key,
                default=default,
            )
        )
        return proxy

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: FlextTypes.ConfigurationDict | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[FlextTypes.ConfigurationDict]:
        """Create StructlogProxyContextVar[dict] instance.

        Helper factory for creating dict-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[FlextTypes.ConfigurationDict] instance

        Example:
            >>> var = uContext.create_dict_proxy("metadata")
            >>> var.set({"key": "value"})
            >>> var.get()  # Returns dict

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[
            FlextTypes.ConfigurationDict
        ] = FlextModelsContext.StructlogProxyContextVar[FlextTypes.ConfigurationDict](
            key,
            default=default,
        )
        return proxy


uContext = FlextUtilitiesContext

__all__ = [
    "FlextUtilitiesContext",
    "uContext",
]
