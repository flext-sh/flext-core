"""Example 07 registry/dispatcher models."""

from __future__ import annotations

from flext_core import m


class ExamplesFlextModelsEx07:
    """Example 07 model namespace."""

    class CreateUserCommand(m.Command):
        """Create user command model."""

        name: str = m.Field(description="User display name")
        email: str = m.Field(description="User email address")

    class UserCreatedEvent(m.Event):
        """User-created event model."""

        event_type: str = m.Field(description="Event type identifier")
        aggregate_id: str = m.Field(description="Aggregate root identifier")
        name: str = m.Field(description="User name from event")

    class GetUserQuery(m.Query):
        """Get user query model."""

        user_id: str = m.Field(description="User identifier to query")

    class DemoPlugin(m.Value):
        """Demo plugin model."""

        name: str = m.Field(description="Plugin name")
