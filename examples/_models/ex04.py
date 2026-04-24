"""Example 04 dispatcher models."""

from __future__ import annotations

from flext_core import m, t


class ExamplesFlextCoreModelsEx04:
    """Example 04 dispatcher model namespace."""

    class Examples:
        """Examples namespace for example 04 dispatcher models."""

        class Ex04CreateUser(m.Command):
            """Command to create a new user in example 04."""

            command_type: t.NonEmptyStr = "ex04_create_user"
            username: str

        class Ex04GetUser(m.Query):
            """Query to get a user record by username in example 04."""

            query_type: str | None = "ex04_get_user"
            username: str

        class Ex04DeleteUser(m.Command):
            """Command to delete a user in example 04."""

            command_type: t.NonEmptyStr = "ex04_delete_user"
            username: str

        class Ex04FailingDelete(m.Command):
            """Command deliberately fails to demonstrate error handling in example 04."""

            command_type: t.NonEmptyStr = "ex04_failing_delete"
            username: str

        class Ex04AutoCommand(m.Command):
            command_type: t.NonEmptyStr = "ex04_auto_command"
            payload: str

        class Ex04Ping(m.Command):
            command_type: t.NonEmptyStr = "ex04_ping"
            value: str

        class Ex04UnknownQuery(m.Query):
            query_type: str | None = "ex04_unknown_query"
            payload: str

        class Ex04UserCreated(m.Event):
            username: str
            event_type: str = "user_created"
            aggregate_id: str = "users"

        class Ex04NoSubscriberEvent(m.Event):
            marker: str
            event_type: str = "no_subscribers"
            aggregate_id: str = "events"


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
