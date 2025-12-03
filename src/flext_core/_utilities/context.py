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
from flext_core.typings import t


class FlextContext:
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
        return FlextModelsContext.StructlogProxyContextVar[str](key, default=default)

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
        return FlextModelsContext.StructlogProxyContextVar[datetime](
            key,
            default=default,
        )

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: dict[str, t.GeneralValueType] | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[dict[str, t.GeneralValueType]]:
        """Create StructlogProxyContextVar[dict] instance.

        Helper factory for creating dict-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[dict[str, t.GeneralValueType]] instance

        Example:
            >>> var = uContext.create_dict_proxy("metadata")
            >>> var.set({"key": "value"})
            >>> var.get()  # Returns dict

        """
        return FlextModelsContext.StructlogProxyContextVar[
            dict[str, t.GeneralValueType]
        ](
            key,
            default=default,
        )


uContext = FlextContext  # noqa: N816

__all__ = [
    "FlextContext",
    "uContext",
]
