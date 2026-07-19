"""Example 04 dispatcher models."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, t, u


class ExamplesFlextModelsEx04:
    """Example 04 dispatcher model namespace."""

    class CreateUser(m.Command):
        """Command to create a new user in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for create user operation"),
        ] = "ex04_create_user"
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for create user operation")
        ] = ""
        event_type: Annotated[
            str, u.Field(description="Event type placeholder for create user operation")
        ] = "ex04_create_user_event"
        username: Annotated[str, u.Field(description="Username for the user to create")]

    class GetUser(m.Query):
        """Query to get a user record by username in example 04."""

        command_type: Annotated[
            str, u.Field(description="Command type placeholder for get user operation")
        ] = ""
        query_type: Annotated[
            str, u.Field(description="Query type identifier for get user operation")
        ] = "ex04_get_user"
        event_type: Annotated[
            str, u.Field(description="Event type placeholder for get user operation")
        ] = "ex04_get_user_event"
        username: Annotated[str, u.Field(description="Username to retrieve")]

    class DeleteUser(m.Command):
        """Command to delete a user in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for delete user operation"),
        ] = "ex04_delete_user"
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for delete user operation")
        ] = ""
        event_type: Annotated[
            str, u.Field(description="Event type placeholder for delete user operation")
        ] = "ex04_delete_user_event"
        username: Annotated[str, u.Field(description="Username of the user to delete")]

    class FailingDelete(m.Command):
        """Command deliberately fails to demonstrate error handling in example 04."""

        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(
                description="Command type identifier for deliberately failing delete operation"
            ),
        ] = "ex04_failing_delete"
        query_type: Annotated[
            str,
            u.Field(
                description="Query type placeholder for deliberately failing delete operation"
            ),
        ] = ""
        event_type: Annotated[
            str,
            u.Field(
                description="Event type placeholder for deliberately failing delete operation"
            ),
        ] = "ex04_failing_delete_event"
        username: Annotated[
            str, u.Field(description="Username for the failing delete operation")
        ]

    class AutoCommand(m.Command):
        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for auto command"),
        ] = "ex04_auto_command"
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for auto command")
        ] = ""
        event_type: Annotated[
            str, u.Field(description="Event type placeholder for auto command")
        ] = "ex04_auto_command_event"
        payload: Annotated[
            str, u.Field(description="Payload data for the auto command")
        ]

    class Ping(m.Command):
        command_type: Annotated[
            t.NonEmptyStr,
            u.Field(description="Command type identifier for ping operation"),
        ] = "ex04_ping"
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for ping operation")
        ] = ""
        event_type: Annotated[
            str, u.Field(description="Event type placeholder for ping operation")
        ] = "ex04_ping_event"
        value: Annotated[str, u.Field(description="Value data for the ping command")]

    class UnknownQuery(m.Query):
        command_type: Annotated[
            str,
            u.Field(description="Command type placeholder for unknown query operation"),
        ] = ""
        query_type: Annotated[
            str,
            u.Field(description="Query type identifier for unknown query operation"),
        ] = "ex04_unknown_query"
        event_type: Annotated[
            str,
            u.Field(description="Event type placeholder for unknown query operation"),
        ] = "ex04_unknown_query_event"
        payload: Annotated[
            str, u.Field(description="Payload data for the unknown query")
        ]

    class UserCreated(m.Event):
        command_type: Annotated[
            str, u.Field(description="Command type placeholder for user created event")
        ] = ""
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for user created event")
        ] = ""
        username: Annotated[
            str, u.Field(description="Username of the user that was created")
        ]
        event_type: Annotated[
            str, u.Field(description="Event type identifier for user creation")
        ] = "user_created"
        aggregate_id: Annotated[str, u.Field(description="Aggregate ID for users")] = (
            "users"
        )

    class NoSubscriberEvent(m.Event):
        command_type: Annotated[
            str,
            u.Field(description="Command type placeholder for no subscribers event"),
        ] = ""
        query_type: Annotated[
            str, u.Field(description="Query type placeholder for no subscribers event")
        ] = ""
        marker: Annotated[
            str,
            u.Field(
                description="Marker identifying the event as having no subscribers"
            ),
        ]
        event_type: Annotated[
            str, u.Field(description="Event type identifier for no subscribers event")
        ] = "no_subscribers"
        aggregate_id: Annotated[str, u.Field(description="Aggregate ID for events")] = (
            "events"
        )
