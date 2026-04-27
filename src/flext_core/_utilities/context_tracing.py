"""Context tracing utilities - Distributed correlation and service identification.

Provides utilities for managing correlation IDs, service metadata, request data,
and operation timing through dispatcher pipelines using structlog context variables.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilitiesContextLifecycle, c, m, t, u


class FlextUtilitiesContextTracing(FlextUtilitiesContextLifecycle):
    """Tracing and service context utilities for FlextContext.

    Per AGENTS.md §149 (single nested-class law) and §3.5 (no compat aliases),
    context variables are exposed as flat ClassVar attributes — no intermediate
    ``Variables`` namespace, no camelCase aliases. Variables are created once at
    class-definition time via ``u.create_*_proxy`` factories and reused across
    the application lifecycle.
    """

    _container_state: m.ContextContainerState

    CORRELATION_ID = u.create_str_proxy(
        c.ContextKey.CORRELATION_ID, default=None
    )
    PARENT_CORRELATION_ID = u.create_str_proxy(
        c.ContextKey.PARENT_CORRELATION_ID, default=None
    )
    SERVICE_NAME = u.create_str_proxy(
        c.ContextKey.SERVICE_NAME, default=None
    )
    SERVICE_VERSION = u.create_str_proxy(
        c.ContextKey.SERVICE_VERSION, default=None
    )
    USER_ID = u.create_str_proxy(c.ContextKey.USER_ID, default=None)
    REQUEST_ID = u.create_str_proxy(c.ContextKey.REQUEST_ID, default=None)
    REQUEST_TIMESTAMP = u.create_datetime_proxy(
        c.ContextKey.REQUEST_TIMESTAMP, default=None
    )
    OPERATION_NAME = u.create_str_proxy(
        c.ContextKey.OPERATION_NAME, default=None
    )
    OPERATION_START_TIME = u.create_datetime_proxy(
        c.ContextKey.OPERATION_START_TIME, default=None
    )
    OPERATION_METADATA = u.create_dict_proxy(
        c.ContextKey.OPERATION_METADATA, default=None
    )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContextTracing"]
