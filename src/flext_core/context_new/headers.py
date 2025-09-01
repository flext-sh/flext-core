"""Context header names (new).

Single-class module: defines `FlextContextHeaders` only.
"""

from __future__ import annotations


class FlextContextHeaders:
    """Standard header names used for correlation and service context."""

    CORRELATION_ID = "X-Correlation-Id"
    PARENT_CORRELATION_ID = "X-Parent-Correlation-Id"
    SERVICE_NAME = "X-Service-Name"
    USER_ID = "X-User-Id"
