"""Example models for ex12."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, u


class ExamplesFlextCoreModelsEx12(m):
    """Examples namespace wrapper for ex12 models."""

    class CommandA(m.Command):
        command_type: str = "ex12_command_a"
        query_type: str = ""
        event_type: Annotated[str, u.Field(description="Event type for command A")] = (
            "ex12_event_a"
        )
        value: Annotated[str, u.Field(description="Command A payload")]

    class CommandB(m.Command):
        command_type: str = "ex12_command_b"
        query_type: str = ""
        event_type: Annotated[str, u.Field(description="Event type for command B")] = (
            "ex12_event_b"
        )
        amount: Annotated[int, u.Field(description="Command B payload")]
