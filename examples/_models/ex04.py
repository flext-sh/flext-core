"""Example 04 dispatcher models."""

from __future__ import annotations

from flext_core import m, t, u


class Ex04CreateUser(m.Command):
    """Command to create a new user in example 04."""

    command_type: t.NonEmptyStr = "ex04_create_user"
    username: str


class Ex04GetUser(m.Query):
    """Query to get a user record by username in example 04."""

    pagination: m.Pagination | t.Dict = u.Field(default_factory=t.Dict)
    query_type: str = "ex04_get_user"
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
    query_type: str = "ex04_unknown_query"
    payload: str


class Ex04UserCreated(m.Event):
    username: str
    event_type: str = "user_created"
    aggregate_id: str = "users"


class Ex04NoSubscriberEvent(m.Event):
    marker: str
    event_type: str = "no_subscribers"
    aggregate_id: str = "events"
