"""Context utility helpers for creating and managing context variables.

These helpers provide factory functions for creating StructlogProxyContextVar
instances with proper typing, ensuring consistent context variable creation
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime

from flext_core import FlextModelsContextProxyVar, t


class FlextUtilitiesContext:
    """Context utility helpers for creating and managing context variables."""

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> FlextModelsContextProxyVar.StructlogProxyContextVar[str]:
        """Create StructlogProxyContextVar[str] instance."""
        return FlextModelsContextProxyVar.StructlogProxyContextVar(
            key,
            default=default,
        )

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> FlextModelsContextProxyVar.StructlogProxyContextVar[datetime]:
        """Create StructlogProxyContextVar[datetime] instance."""
        return FlextModelsContextProxyVar.StructlogProxyContextVar(
            key,
            default=default,
        )

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: t.JsonMapping | None = None,
    ) -> FlextModelsContextProxyVar.StructlogProxyContextVar[t.JsonMapping]:
        """Create StructlogProxyContextVar[dict] instance."""
        return FlextModelsContextProxyVar.StructlogProxyContextVar(
            key,
            default=default,
        )


__all__: list[str] = ["FlextUtilitiesContext"]
