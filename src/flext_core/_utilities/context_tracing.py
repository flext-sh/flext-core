"""Context tracing utilities - Distributed correlation and service identification.

Provides utilities for managing correlation IDs, service metadata, request data,
and operation timing through dispatcher pipelines using structlog context variables.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from flext_core import FlextUtilitiesContextLifecycle, c, m, t, u


class FlextUtilitiesContextTracing(FlextUtilitiesContextLifecycle):
    """Tracing and service context utilities for FlextContext.

    Per AGENTS.md §0.7 (one nested class per facade) and §3.5 (no compat aliases):
    context variables are flat ClassVar[m.StructlogProxyContextVar[T]] attributes —
    no intermediate namespace class, no camelCase aliases. Created once at class-
    definition time via u.create_*_proxy factories; reused across the application.
    """

    _container_state: ClassVar[m.ContextContainerState]

    CORRELATION_ID: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.CORRELATION_ID, default=None
    )
    PARENT_CORRELATION_ID: ClassVar[m.StructlogProxyContextVar[str]] = (
        u.create_str_proxy(c.ContextKey.PARENT_CORRELATION_ID, default=None)
    )
    SERVICE_NAME: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.SERVICE_NAME, default=None
    )
    SERVICE_VERSION: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.SERVICE_VERSION, default=None
    )
    USER_ID: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.USER_ID, default=None
    )
    REQUEST_ID: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.REQUEST_ID, default=None
    )
    REQUEST_TIMESTAMP: ClassVar[m.StructlogProxyContextVar[datetime]] = (
        u.create_datetime_proxy(c.ContextKey.REQUEST_TIMESTAMP, default=None)
    )
    OPERATION_NAME: ClassVar[m.StructlogProxyContextVar[str]] = u.create_str_proxy(
        c.ContextKey.OPERATION_NAME, default=None
    )
    OPERATION_START_TIME: ClassVar[m.StructlogProxyContextVar[datetime]] = (
        u.create_datetime_proxy(c.ContextKey.OPERATION_START_TIME, default=None)
    )
    OPERATION_METADATA: ClassVar[m.StructlogProxyContextVar[t.JsonMapping]] = (
        u.create_dict_proxy(c.ContextKey.OPERATION_METADATA, default=None)
    )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContextTracing"]
