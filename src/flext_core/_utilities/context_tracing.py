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

        class Correlation:
            """Correlation variables for distributed tracing."""

            CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.CORRELATION_ID,
                default=None,
            )
            PARENT_CORRELATION_ID: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(c.ContextKey.PARENT_CORRELATION_ID, default=None)
            )

        class Service:
            """Service context variables for identification."""

            SERVICE_NAME: Final[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
                c.ContextKey.SERVICE_NAME,
                default=None,
            )
            SERVICE_VERSION: Final[m.StructlogProxyContextVar[str]] = (
                u.create_str_proxy(c.ContextKey.SERVICE_VERSION, default=None)
            )

        class Request:
            """Request context variables for metadata."""

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

        class Performance:
            """Performance context variables for timing."""

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
            OPERATION_METADATA: Final[
                m.StructlogProxyContextVar[t.FlatContainerMapping]
            ] = u.create_dict_proxy(c.ContextKey.OPERATION_METADATA, default=None)

        CorrelationId: Final[m.StructlogProxyContextVar[str]] = (
            Correlation.CORRELATION_ID
        )
        ParentCorrelationId: Final[m.StructlogProxyContextVar[str]] = (
            Correlation.PARENT_CORRELATION_ID
        )
        ServiceName: Final[m.StructlogProxyContextVar[str]] = Service.SERVICE_NAME
        ServiceVersion: Final[m.StructlogProxyContextVar[str]] = Service.SERVICE_VERSION
        UserId: Final[m.StructlogProxyContextVar[str]] = Request.USER_ID
        RequestId: Final[m.StructlogProxyContextVar[str]] = Request.REQUEST_ID
        RequestTimestamp: Final[m.StructlogProxyContextVar[datetime]] = (
            Request.REQUEST_TIMESTAMP
        )
        OperationName: Final[m.StructlogProxyContextVar[str]] = (
            Performance.OPERATION_NAME
        )
        OperationStartTime: Final[m.StructlogProxyContextVar[datetime]] = (
            Performance.OPERATION_START_TIME
        )
        OperationMetadata: Final[m.StructlogProxyContextVar[t.FlatContainerMapping]] = (
            Performance.OPERATION_METADATA
        )


__all__: list[str] = ["FlextUtilitiesContextTracing"]
