"""Context tracing, service, and performance inner classes.

Extracted from FlextContext as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Final

from flext_core import FlextUtilitiesContextLifecycle, c, m, t, u


class FlextUtilitiesContextTracing(FlextUtilitiesContextLifecycle):
    """Tracing, service, performance, and serialization inner classes for FlextContext."""

    _container_state: ClassVar[m.ContextContainerState]

    class Variables:
        """Context variables using structlog as single source of truth."""

        # Correlation variables for distributed tracing
        CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
            c.ContextKey.CORRELATION_ID,
            default=None,
        )
        PARENT_CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = (
            u.create_str_proxy(c.ContextKey.PARENT_CORRELATION_ID, default=None)
        )

        # Service context variables for identification
        SERVICE_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
            c.ContextKey.SERVICE_NAME,
            default=None,
        )
        SERVICE_VERSION: Final[m.StructlogProxyContextVar[str]] = (
            u.create_str_proxy(c.ContextKey.SERVICE_VERSION, default=None)
        )

        # Request context variables for metadata
        USER_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
            c.ContextKey.USER_ID,
            default=None,
        )
        REQUEST_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
            c.ContextKey.REQUEST_ID,
            default=None,
        )
        REQUEST_TIMESTAMP: Final[m.StructlogProxyContextVar[datetime]] = (
            u.create_datetime_proxy(c.ContextKey.REQUEST_TIMESTAMP, default=None)
        )

        # Performance context variables for timing
        OPERATION_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
            c.ContextKey.OPERATION_NAME,
            default=None,
        )
        OPERATION_START_TIME: Final[m.StructlogProxyContextVar[datetime]] = (
            u.create_datetime_proxy(
                c.ContextKey.OPERATION_START_TIME,
                default=None,
            )
        )
        OPERATION_METADATA: Final[m.StructlogProxyContextVar[t.JsonMapping]] = (
            u.create_dict_proxy(c.ContextKey.OPERATION_METADATA, default=None)
        )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContextTracing"]
