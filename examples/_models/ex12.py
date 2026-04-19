"""Example models for ex12."""

from __future__ import annotations

from examples import m, u


class Ex12CommandA(m.Command):
    command_type: str = "ex12_command_a"
    query_type: str = ""
    event_type: str = "ex12_event_a"
    value: str = u.Field(description="Command A payload")


class Ex12CommandB(m.Command):
    command_type: str = "ex12_command_b"
    query_type: str = ""
    event_type: str = "ex12_event_b"
    amount: int = u.Field(description="Command B payload")


class ExamplesFlextCoreModelsEx12(m):
    """Examples namespace wrapper for ex12 models."""
