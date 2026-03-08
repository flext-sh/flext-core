"""Example 04 dispatcher models."""

from __future__ import annotations

from pydantic import BaseModel


class Ex04CreateUser(BaseModel):
    """Create command payload."""

    username: str
    command_type: str = "create_user"
    query_type: str = ""
    event_type: str = ""


class Ex04GetUser(BaseModel):
    """Get query payload."""

    username: str
    command_type: str = ""
    query_type: str = "get_user"
    event_type: str = ""


class Ex04DeleteUser(BaseModel):
    """Delete command payload."""

    username: str
    command_type: str = "delete_user"
    query_type: str = ""
    event_type: str = ""


class Ex04FailingDelete(BaseModel):
    """Delete command that intentionally fails."""

    username: str
    command_type: str = "failing_delete"
    query_type: str = ""
    event_type: str = ""


class Ex04AutoCommand(BaseModel):
    """Auto command payload."""

    payload: str
    command_type: str = "auto_command"
    query_type: str = ""
    event_type: str = ""


class Ex04Ping(BaseModel):
    """Ping command payload."""

    value: str
    command_type: str = "ping"
    query_type: str = ""
    event_type: str = ""


class Ex04UnknownQuery(BaseModel):
    """Unknown query payload."""

    payload: str
    command_type: str = ""
    query_type: str = "unknown_query"
    event_type: str = ""


class Ex04UserCreated(BaseModel):
    """User-created event payload."""

    username: str
    command_type: str = ""
    query_type: str = ""
    event_type: str = "user_created"


class Ex04NoSubscriberEvent(BaseModel):
    """Event with no subscribers."""

    marker: str
    command_type: str = ""
    query_type: str = ""
    event_type: str = "no_subscribers"
