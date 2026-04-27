"""Context tracing utilities - Distributed correlation and service identification.

Provides utilities for managing correlation IDs, service metadata, and operation
timing through dispatcher pipelines using structlog context variables.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Final

from flext_core import FlextUtilitiesContextLifecycle, c, m, t, u


class FlextUtilitiesContextTracing(FlextUtilitiesContextLifecycle):
    """Tracing and service context utilities for FlextContext.

    Correlation IDs, service metadata, and timing variables are created on-demand
    via factory functions in FlextUtilitiesContext.
    """

    _container_state: ClassVar[m.ContextContainerState]

    # Context variables - created once, reused throughout application
    CORRELATION_ID: ClassVar = u.create_str_proxy(c.ContextKey.CORRELATION_ID, default=None)
    PARENT_CORRELATION_ID: ClassVar = u.create_str_proxy(
        c.ContextKey.PARENT_CORRELATION_ID, default=None
    )
    SERVICE_NAME: ClassVar = u.create_str_proxy(c.ContextKey.SERVICE_NAME, default=None)
    SERVICE_VERSION: ClassVar = u.create_str_proxy(c.ContextKey.SERVICE_VERSION, default=None)
    OPERATION_NAME: ClassVar = u.create_str_proxy(c.ContextKey.OPERATION_NAME, default=None)
    OPERATION_START_TIME: ClassVar = u.create_datetime_proxy(
        c.ContextKey.OPERATION_START_TIME, default=None
    )
    OPERATION_METADATA: ClassVar = u.create_dict_proxy(
        c.ContextKey.OPERATION_METADATA, default=None
    )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesContextTracing"]
