"""Context utility helpers for creating and managing context variables.

Factory functions and singleton context-variable proxies. Both are exposed via
u.* (FlextUtilitiesContext is part of FlextUtilities MRO) so callers write
u.create_str_proxy(...) and u.CORRELATION_ID.set(...) instead of importing
internal classes directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from flext_core import FlextModelsContext, FlextTypes as t
from flext_core._constants.infrastructure import FlextConstantsInfrastructure as _c


class FlextUtilitiesContext:
    """Context utility factories + singleton proxy instances.

    Per AGENTS.md §0.7: single concern (context I/O). Proxy ClassVars are created
    once at class-definition time using the factory @staticmethod helpers defined
    in this same class. Access via u.CORRELATION_ID, u.SERVICE_NAME, etc.
    """

    # --- Proxy instances (ClassVar, created once) ---

    CORRELATION_ID: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(
            _c.ContextKey.CORRELATION_ID, default=None
        )
    )
    PARENT_CORRELATION_ID: ClassVar[
        FlextModelsContext.StructlogProxyContextVar[str]
    ] = FlextModelsContext.StructlogProxyContextVar(
        _c.ContextKey.PARENT_CORRELATION_ID, default=None
    )
    SERVICE_NAME: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(
            _c.ContextKey.SERVICE_NAME, default=None
        )
    )
    SERVICE_VERSION: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(
            _c.ContextKey.SERVICE_VERSION, default=None
        )
    )
    USER_ID: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(_c.ContextKey.USER_ID, default=None)
    )
    REQUEST_ID: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(
            _c.ContextKey.REQUEST_ID, default=None
        )
    )
    REQUEST_TIMESTAMP: ClassVar[
        FlextModelsContext.StructlogProxyContextVar[datetime]
    ] = FlextModelsContext.StructlogProxyContextVar(
        _c.ContextKey.REQUEST_TIMESTAMP, default=None
    )
    OPERATION_NAME: ClassVar[FlextModelsContext.StructlogProxyContextVar[str]] = (
        FlextModelsContext.StructlogProxyContextVar(
            _c.ContextKey.OPERATION_NAME, default=None
        )
    )
    OPERATION_START_TIME: ClassVar[
        FlextModelsContext.StructlogProxyContextVar[datetime]
    ] = FlextModelsContext.StructlogProxyContextVar(
        _c.ContextKey.OPERATION_START_TIME, default=None
    )
    OPERATION_METADATA: ClassVar[
        FlextModelsContext.StructlogProxyContextVar[t.JsonMapping]
    ] = FlextModelsContext.StructlogProxyContextVar(
        _c.ContextKey.OPERATION_METADATA, default=None
    )

    # --- Proxy factories (create new instances on demand) ---

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[str]:
        """Create a new StructlogProxyContextVar[str] instance."""
        return FlextModelsContext.StructlogProxyContextVar(key, default=default)

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[datetime]:
        """Create a new StructlogProxyContextVar[datetime] instance."""
        return FlextModelsContext.StructlogProxyContextVar(key, default=default)

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: t.JsonMapping | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[t.JsonMapping]:
        """Create a new StructlogProxyContextVar[dict] instance."""
        return FlextModelsContext.StructlogProxyContextVar(key, default=default)


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContext"]
