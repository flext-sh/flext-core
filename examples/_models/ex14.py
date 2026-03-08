"""Example 14 handlers models."""

from __future__ import annotations

from pydantic import BaseModel

from flext_core import m


class Ex14CreateUserCommand(m.Command):
    """Create user command payload."""

    user_id: str
    name: str
    email: str


class Ex14GetUserQuery(m.Query):
    """Get user query payload."""

    user_id: str


class Ex14UserDTO(BaseModel):
    """User DTO payload."""

    id: str
    name: str
    email: str
