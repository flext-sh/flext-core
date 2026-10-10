"""Example 04 dispatcher models.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import m, t


class ExamplesFlextModelsEx04:
    """Example 04 dispatcher model namespace."""

    class _CommandEnvelope(m.Command):
        """Shared envelope fields for example 04 command models."""

        command_type: Annotated[
            t.NonEmptyStr,
            m.Field(description="Command type identifier for the ex04 operation"),
        ] = "ex04_command"
        query_type: Annotated[
            str,
            m.Field(description="Query type placeholder for the ex04 operation"),
        ] = ""
        event_type: Annotated[
            str,
            m.Field(description="Event type placeholder for the ex04 operation"),
        ] = ""

    class _QueryEnvelope(m.Query):
        """Shared envelope fields for example 04 query models."""

        command_type: Annotated[
            str,
            m.Field(description="Command type placeholder for the ex04 operation"),
        ] = ""
        query_type: Annotated[
            str | None,
            m.Field(description="Query type identifier for the ex04 operation"),
        ] = "ex04_query"
        event_type: Annotated[
            str,
            m.Field(description="Event type placeholder for the ex04 operation"),
        ] = ""

    class _EventEnvelope(m.Event):
        """Shared envelope fields for example 04 event models."""

        command_type: Annotated[
            str,
            m.Field(description="Command type placeholder for the ex04 operation"),
        ] = ""
        query_type: Annotated[
            str,
            m.Field(description="Query type placeholder for the ex04 operation"),
        ] = ""
        event_type: Annotated[
            str,
            m.Field(description="Event type identifier for the ex04 operation"),
        ] = "ex04_event"
        aggregate_id: Annotated[str, m.Field(description="Aggregate ID for events")] = (
            "events"
        )

    class CreateUser(_CommandEnvelope):
        """Command to create a new user in example 04."""

        command_type = "ex04_create_user"
        event_type = "ex04_create_user_event"
        username: Annotated[str, m.Field(description="Username for the user to create")]

    class GetUser(_QueryEnvelope):
        """Query to get a user record by username in example 04."""

        query_type = "ex04_get_user"
        event_type = "ex04_get_user_event"
        username: Annotated[str, m.Field(description="Username to retrieve")]

    class DeleteUser(_CommandEnvelope):
        """Command to delete a user in example 04."""

        command_type = "ex04_delete_user"
        event_type = "ex04_delete_user_event"
        username: Annotated[str, m.Field(description="Username of the user to delete")]

    class FailingDelete(_CommandEnvelope):
        """Command deliberately fails to demonstrate error handling in example 04."""

        command_type = "ex04_failing_delete"
        event_type = "ex04_failing_delete_event"
        username: Annotated[
            str,
            m.Field(description="Username for the failing delete operation"),
        ]

    class AutoCommand(_CommandEnvelope):
        """Command carrying an arbitrary payload for auto-discovery in example 04."""

        command_type = "ex04_auto_command"
        event_type = "ex04_auto_command_event"
        payload: Annotated[
            str,
            m.Field(description="Payload data for the auto command"),
        ]

    class Ping(_CommandEnvelope):
        """Command carrying a value for the ping operation in example 04."""

        command_type = "ex04_ping"
        event_type = "ex04_ping_event"
        value: Annotated[str, m.Field(description="Value data for the ping command")]

    class UnknownQuery(_QueryEnvelope):
        """Query that has no registered handler in example 04."""

        query_type = "ex04_unknown_query"
        event_type = "ex04_unknown_query_event"
        payload: Annotated[
            str,
            m.Field(description="Payload data for the unknown query"),
        ]

    class UserCreated(_EventEnvelope):
        """Event emitted after a user was created in example 04."""

        event_type = "user_created"
        aggregate_id = "users"
        username: Annotated[
            str,
            m.Field(description="Username of the user that was created"),
        ]

    class NoSubscriberEvent(_EventEnvelope):
        """Event emitted with no registered subscribers in example 04."""

        event_type = "no_subscribers"
        marker: Annotated[
            str,
            m.Field(
                description="Marker identifying the event as having no subscribers",
            ),
        ]
