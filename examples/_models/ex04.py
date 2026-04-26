"""Example 04 dispatcher models."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, t, u


class ExamplesFlextCoreModelsEx04:
    """Example 04 dispatcher model namespace."""

    class Ex04CreateUser(m.Command):
        """Command to create a new user in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for create user operation"),
        ] = "ex04_create_user"
        username: Annotated[
            str, u.Field(description="Username for the user to create")
        ]

    class Ex04GetUser(m.Query):
        """Query to get a user record by username in example 04."""

        query_type: Annotated[
            str | None,
            u.Field(description="Query type identifier for get user operation"),
        ] = "ex04_get_user"
        username: Annotated[
            str, u.Field(description="Username to retrieve")
        ]

    class Ex04DeleteUser(m.Command):
        """Command to delete a user in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for delete user operation"),
        ] = "ex04_delete_user"
        username: Annotated[
            str, u.Field(description="Username of the user to delete")
        ]

    class Ex04FailingDelete(m.Command):
        """Command deliberately fails to demonstrate error handling in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for deliberately failing delete operation"),
        ] = "ex04_failing_delete"
        username: Annotated[
            str, u.Field(description="Username for the failing delete operation")
        ]

    class Ex04AutoCommand(m.Command):
        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for auto command"),
        ] = "ex04_auto_command"
        payload: Annotated[
            str, u.Field(description="Payload data for the auto command")
        ]

    class Ex04Ping(m.Command):
        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for ping operation"),
        ] = "ex04_ping"
        value: Annotated[
            str, u.Field(description="Value data for the ping command")
        ]

    class Ex04UnknownQuery(m.Query):
        query_type: Annotated[
            str | None,
            u.Field(description="Query type identifier for unknown query operation"),
        ] = "ex04_unknown_query"
        payload: Annotated[
            str, u.Field(description="Payload data for the unknown query")
        ]

    class Ex04UserCreated(m.Event):
        username: Annotated[
            str, u.Field(description="Username of the user that was created")
        ]
        event_type: Annotated[
            str,
            u.Field(description="Event type identifier for user creation"),
        ] = "user_created"
        aggregate_id: Annotated[
            str, u.Field(description="Aggregate ID for users")
        ] = "users"

    class Ex04NoSubscriberEvent(m.Event):
        marker: Annotated[
            str, u.Field(description="Marker identifying the event as having no subscribers")
        ]
        event_type: Annotated[
            str,
            u.Field(description="Event type identifier for no subscribers event"),
        ] = "no_subscribers"
        aggregate_id: Annotated[
            str, u.Field(description="Aggregate ID for events")
        ] = "events"
