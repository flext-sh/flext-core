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
            m.Field(
                default=...,
                description="Command type identifier for the ex04 operation",
            ),
        ] = "ex04_command"
        query_type: Annotated[
            str,
            m.Field(
                default=..., description="Query type placeholder for the ex04 operation"
            ),
        ] = ""
        event_type: Annotated[
            str,
            m.Field(
                default=..., description="Event type placeholder for the ex04 operation"
            ),
        ] = ""

    class _QueryEnvelope(m.Query):
        """Shared envelope fields for example 04 query models."""

        command_type: Annotated[
            str,
            m.Field(
                default=...,
                description="Command type placeholder for the ex04 operation",
            ),
        ] = ""
        query_type: Annotated[
            str | None,
            m.Field(
                default=..., description="Query type identifier for the ex04 operation"
            ),
        ] = "ex04_query"
        event_type: Annotated[
            str,
            m.Field(
                default=..., description="Event type placeholder for the ex04 operation"
            ),
        ] = ""

    class _EventEnvelope(m.Event):
        """Shared envelope fields for example 04 event models."""

        command_type: Annotated[
            str,
            m.Field(
                default=...,
                description="Command type placeholder for the ex04 operation",
            ),
        ] = ""
        query_type: Annotated[
            str,
            m.Field(
                default=..., description="Query type placeholder for the ex04 operation"
            ),
        ] = ""
        event_type: Annotated[
            str,
            m.Field(
                default=..., description="Event type identifier for the ex04 operation"
            ),
        ] = "ex04_event"
        aggregate_id: Annotated[
            str, m.Field(default=..., description="Aggregate ID for events")
        ] = "events"

    class CreateUser(_CommandEnvelope):
        """Command to create a new user in example 04."""

        command_type: t.NonEmptyStr = m.Field(
            default="ex04_create_user", description="Create-user command identifier"
        )
        event_type: str = m.Field(
            default="ex04_create_user_event", description="Create-user event identifier"
        )
        username: Annotated[
            str, m.Field(description="Username for the user to create")
        ] = ...

    class GetUser(_QueryEnvelope):
        """Query to get a user record by username in example 04."""

        query_type: str | None = m.Field(
            default="ex04_get_user", description="Get-user query identifier"
        )
        event_type: str = m.Field(
            default="ex04_get_user_event", description="Get-user event identifier"
        )
        username: Annotated[str, m.Field(description="Username to retrieve")] = ...

    class DeleteUser(_CommandEnvelope):
        """Command to delete a user in example 04."""

        command_type: t.NonEmptyStr = m.Field(
            default="ex04_delete_user", description="Delete-user command identifier"
        )
        event_type: str = m.Field(
            default="ex04_delete_user_event", description="Delete-user event identifier"
        )
        username: Annotated[
            str, m.Field(description="Username of the user to delete")
        ] = ...

    class FailingDelete(_CommandEnvelope):
        """Command deliberately fails to demonstrate error handling in example 04."""

        command_type: t.NonEmptyStr = m.Field(
            default="ex04_failing_delete",
            description="Failing-delete command identifier",
        )
        event_type: str = m.Field(
            default="ex04_failing_delete_event",
            description="Failing-delete event identifier",
        )
        username: Annotated[
            str, m.Field(description="Username for the failing delete operation")
        ] = ...

    class AutoCommand(_CommandEnvelope):
        """Command carrying an arbitrary payload for auto-discovery in example 04."""

        command_type: t.NonEmptyStr = m.Field(
            default="ex04_auto_command",
            description="Auto-discovered command identifier",
        )
        event_type: str = m.Field(
            default="ex04_auto_command_event",
            description="Auto-discovered event identifier",
        )
        payload: Annotated[
            str, m.Field(description="Payload data for the auto command")
        ] = ...

    class Ping(_CommandEnvelope):
        """Command carrying a value for the ping operation in example 04."""

        command_type: t.NonEmptyStr = m.Field(
            default="ex04_ping", description="Ping command identifier"
        )
        event_type: str = m.Field(
            default="ex04_ping_event", description="Ping event identifier"
        )
        value: Annotated[
            str, m.Field(description="Value data for the ping command")
        ] = ...

    class UnknownQuery(_QueryEnvelope):
        """Query that has no registered handler in example 04."""

        query_type: str | None = m.Field(
            default="ex04_unknown_query", description="Unregistered query identifier"
        )
        event_type: str = m.Field(
            default="ex04_unknown_query_event",
            description="Unregistered-query event identifier",
        )
        payload: Annotated[
            str, m.Field(description="Payload data for the unknown query")
        ] = ...

    class UserCreated(_EventEnvelope):
        """Event emitted after a user was created in example 04."""

        event_type: str = m.Field(
            default="user_created", description="User-created event identifier"
        )
        aggregate_id: str = m.Field(
            default="users", description="User aggregate identifier"
        )
        username: Annotated[
            str, m.Field(description="Username of the user that was created")
        ] = ...

    class NoSubscriberEvent(_EventEnvelope):
        """Event emitted with no registered subscribers in example 04."""

        event_type: str = m.Field(
            default="no_subscribers", description="Unsubscribed event identifier"
        )
        marker: Annotated[
            str,
            m.Field(
                description="Marker identifying the event as having no subscribers",
            ),
        ] = ...
