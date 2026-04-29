"""CQRS model smoke tests for internal model tier."""

from __future__ import annotations

from tests import m


class TestsFlextModelsCQRS:
    def test_command_instantiation(self) -> None:
        class CreateItem(m.Command):
            name: str

        cmd = CreateItem(name="item", command_id="cmd-1")
        assert cmd.name == "item"

    def test_query_instantiation(self) -> None:
        class GetItem(m.Query):
            item_id: str

        qry = GetItem(item_id="1")
        assert qry.item_id == "1"
