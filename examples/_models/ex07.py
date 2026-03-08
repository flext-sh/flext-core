"""Example 07 registry/dispatcher models."""

from __future__ import annotations

from pydantic import BaseModel

from flext_core import m


class Ex07CreateUserCommand(m.Command):
    """Create user command model."""

    name: str
    email: str


class Ex07UserCreatedEvent(m.Event):
    """User-created event model."""

    event_type: str
    aggregate_id: str
    name: str


class Ex07GetUserQuery(m.Query):
    """Get user query model."""

    user_id: str


class Ex07DemoPlugin(BaseModel):
    """Demo plugin model."""

    name: str
