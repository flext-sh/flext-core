"""Example 14 handlers models."""

from __future__ import annotations

from flext_core import m


class ExamplesFlextModelsEx14:
    """Example 14 model namespace."""

    class HandlerCreateUserCommand(m.Command):
        """Create user command payload."""

        user_id: str = m.Field(
            description="Unique identifier for the user being created."
        )
        name: str = m.Field(description="Full name of the user.")
        email: str = m.Field(description="Email address of the user.")

    class HandlerGetUserQuery(m.Query):
        """Get user query payload."""

        user_id: str = m.Field(
            description="Unique identifier for the user to retrieve."
        )

    class UserDTO(m.Value):
        """Data transfer object for user information."""

        id: str = m.Field(description="Unique identifier for the user.")
        name: str = m.Field(description="Full name of the user.")
        email: str = m.Field(description="Email address of the user.")
