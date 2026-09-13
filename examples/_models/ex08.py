"""Example models for ex08."""

from __future__ import annotations

from typing import Annotated

from flext_core import m


class ExamplesFlextModelsEx08:
    """Examples namespace wrapper for ex08 models."""

    class UserEntity(m.Entity):
        name: Annotated[str, m.Field(description="User display name")]

    class OrderEntity(m.Entity):
        status: Annotated[str, m.Field(description="Order lifecycle status")] = "active"
