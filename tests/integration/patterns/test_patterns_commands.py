"""Command pattern integration smoke tests."""

from __future__ import annotations

from tests import m


class TestsFlextCorePatternsCommands:
    def test_command_model_instantiation(self) -> None:
        class DemoCommand(m.Command):
            name: str

        command = DemoCommand(name="cmd", command_id="c1")
        assert command.name == "cmd"
