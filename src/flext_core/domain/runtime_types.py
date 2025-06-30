"""Runtime-compatible type constructors for FLEXT Core."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field


# Runtime-compatible type constructors
def UserId(value: UUID | str) -> UUID:
    """Create a UserId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def TenantId(value: UUID | str) -> UUID:
    """Create a TenantId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def CorrelationId(value: UUID | str) -> UUID:
    """Create a CorrelationId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def TraceId(value: UUID | str) -> UUID:
    """Create a TraceId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def CommandId(value: UUID | str) -> UUID:
    """Create a CommandId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def QueryId(value: UUID | str) -> UUID:
    """Create a QueryId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


def EventId(value: UUID | str) -> UUID:
    """Create an EventId from UUID or string."""
    if isinstance(value, str):
        return UUID(value)
    return value


# Type aliases for static analysis
UserIdType = Annotated[UUID, Field(description="User identification value object")]
TenantIdType = Annotated[UUID, Field(description="Multi-tenant identification value object")]
CorrelationIdType = Annotated[UUID, Field(description="Request correlation value object")]
TraceIdType = Annotated[UUID, Field(description="Distributed tracing value object")]
CommandIdType = Annotated[UUID, Field(description="Command identification value object")]
QueryIdType = Annotated[UUID, Field(description="Query identification value object")]
EventIdType = Annotated[UUID, Field(description="Event identification value object")]
