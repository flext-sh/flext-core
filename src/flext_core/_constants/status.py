"""FlextConstantsStatus - domain lifecycle and status enumerations (SSOT).

Merges Status (domain), CommonStatus/OperationStatus/SpecialStatus (cqrs),
CheckStatus (mixins), and HealthStatus into a single flat enum family
exposed via `c.Status`. Health-specific values stay on a separate
`c.HealthStatus` because they carry a distinct semantic meaning (health
check state vs. lifecycle state).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique


class FlextConstantsStatus:
    """SSOT for domain, lifecycle, and health status enumerations."""

    @unique
    class ContextOperation(StrEnum):
        """Context operation types enumeration."""

        BIND = "bind"
        UNBIND = "unbind"
        CLEAR = "clear"
        GET = "get"
        REMOVE = "remove"
        SET = "set"

    @unique
    class Currency(StrEnum):
        """Currency enumeration for monetary operations."""

        USD = "USD"
        EUR = "EUR"
        GBP = "GBP"
        BRL = "BRL"

    @unique
    class OrderStatus(StrEnum):
        """Order status enumeration for order lifecycle."""

        PENDING = "pending"
        CONFIRMED = "confirmed"
        SHIPPED = "shipped"
        DELIVERED = "delivered"
        CANCELLED = "cancelled"

    @unique
    class Status(StrEnum):
        """Unified lifecycle + operation status values."""

        ACTIVE = "active"
        INACTIVE = "inactive"
        ARCHIVED = "archived"
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        COMPENSATING = "compensating"
        SUCCESS = "success"
        FAILURE = "failure"
        PARTIAL = "partial"
        SENT = "sent"
        IDLE = "idle"
        PROCESSING = "processing"
        STOPPED = "stopped"

    @unique
    class HealthStatus(StrEnum):
        """Health check state."""

        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        UNKNOWN = "unknown"
        ERROR = "error"
