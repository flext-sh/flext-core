"""Example 04 dispatcher models."""

from __future__ import annotations

from flext_core import m


class Ex04CreateUser(m.Command):
    username: str


class Ex04GetUser(m.Query):
    username: str


class Ex04DeleteUser(m.Command):
    username: str


class Ex04FailingDelete(m.Command):
    username: str


class Ex04AutoCommand(m.Command):
    payload: str


class Ex04Ping(m.Command):
    value: str


class Ex04UnknownQuery(m.Query):
    payload: str


class Ex04UserCreated(m.Event):
    username: str
    event_type: str = "user_created"
    aggregate_id: str = "users"


class Ex04NoSubscriberEvent(m.Event):
    marker: str
    event_type: str = "no_subscribers"
    aggregate_id: str = "events"
