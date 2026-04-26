"""Example 07 registry/dispatcher models."""

from __future__ import annotations

from flext_core import m, u


class ExamplesFlextCoreModelsEx07:
    """Example 07 model namespace."""

    class Ex07CreateUserCommand(m.Command):
        """Create user command model."""

        name: str = u.Field(description="User display name")
        email: str = u.Field(description="User email address")

    class Ex07UserCreatedEvent(m.Event):
        """User-created event model."""

        event_type: str = u.Field(description="Event type identifier")
        aggregate_id: str = u.Field(description="Aggregate root identifier")
        name: str = u.Field(description="User name from event")

    class Ex07GetUserQuery(m.Query):
        """Get user query model."""

        user_id: str = u.Field(description="User identifier to query")

    class Ex07DemoPlugin(m.Value):
        """Demo plugin model."""

        name: str = u.Field(description="Plugin name")
