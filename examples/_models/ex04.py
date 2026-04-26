"""Example 04 dispatcher models."""

from __future__ import annotations

from flext_core import m, t, u


class ExamplesFlextCoreModelsEx04:
    """Example 04 dispatcher model namespace."""

    class Examples:
        """Examples namespace for example 04 dispatcher models."""

        class Ex04CreateUser(m.Command):
            """Command to create a new user in example 04."""

            command_type: t.NonEmptyStr = u.Field(default="ex04_create_user", description="Command type identifier for create user operation")
            username: str = u.Field(description="Username for the user to create")

        class Ex04GetUser(m.Query):
            """Query to get a user record by username in example 04."""

            query_type: str | None = u.Field(default="ex04_get_user", description="Query type identifier for get user operation")
            username: str = u.Field(description="Username to retrieve")

        class Ex04DeleteUser(m.Command):
            """Command to delete a user in example 04."""

            command_type: t.NonEmptyStr = u.Field(default="ex04_delete_user", description="Command type identifier for delete user operation")
            username: str = u.Field(description="Username of the user to delete")

        class Ex04FailingDelete(m.Command):
            """Command deliberately fails to demonstrate error handling in example 04."""

            command_type: t.NonEmptyStr = u.Field(default="ex04_failing_delete", description="Command type identifier for deliberately failing delete operation")
            username: str = u.Field(description="Username for the failing delete operation")

        class Ex04AutoCommand(m.Command):
            command_type: t.NonEmptyStr = u.Field(default="ex04_auto_command", description="Command type identifier for auto command")
            payload: str = u.Field(description="Payload data for the auto command")

        class Ex04Ping(m.Command):
            command_type: t.NonEmptyStr = u.Field(default="ex04_ping", description="Command type identifier for ping operation")
            value: str = u.Field(description="Value data for the ping command")

        class Ex04UnknownQuery(m.Query):
            query_type: str | None = u.Field(default="ex04_unknown_query", description="Query type identifier for unknown query operation")
            payload: str = u.Field(description="Payload data for the unknown query")

        class Ex04UserCreated(m.Event):
            username: str = u.Field(description="Username of the user that was created")
            event_type: str = u.Field(default="user_created", description="Event type identifier for user creation")
            aggregate_id: str = u.Field(default="users", description="Aggregate ID for users")

        class Ex04NoSubscriberEvent(m.Event):
            marker: str = u.Field(description="Marker identifying the event as having no subscribers")
            event_type: str = u.Field(default="no_subscribers", description="Event type identifier for no subscribers event")
            aggregate_id: str = u.Field(default="events", description="Aggregate ID for events")


# Module-level re-exports for package __init__.py API
Ex04CreateUser = ExamplesFlextCoreModelsEx04.Examples.Ex04CreateUser
Ex04GetUser = ExamplesFlextCoreModelsEx04.Examples.Ex04GetUser
Ex04DeleteUser = ExamplesFlextCoreModelsEx04.Examples.Ex04DeleteUser
Ex04FailingDelete = ExamplesFlextCoreModelsEx04.Examples.Ex04FailingDelete
Ex04AutoCommand = ExamplesFlextCoreModelsEx04.Examples.Ex04AutoCommand
Ex04Ping = ExamplesFlextCoreModelsEx04.Examples.Ex04Ping
Ex04UnknownQuery = ExamplesFlextCoreModelsEx04.Examples.Ex04UnknownQuery
Ex04UserCreated = ExamplesFlextCoreModelsEx04.Examples.Ex04UserCreated
Ex04NoSubscriberEvent = ExamplesFlextCoreModelsEx04.Examples.Ex04NoSubscriberEvent
