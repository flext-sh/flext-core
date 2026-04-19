"""Entity smoke coverage aligned to current model facade."""

from __future__ import annotations

from tests import m


class TestEntityCoverage:
    def test_entity_creation(self) -> None:
        class User(m.Entity):
            name: str

        user = User(name="alice", domain_events=[])
        assert user.name == "alice"
        assert user.unique_id is not None

    def test_entity_model_dump(self) -> None:
        class Item(m.Entity):
            code: str

        item = Item(code="IT-1", domain_events=[])
        dumped = item.model_dump(mode="python")
        assert dumped["code"] == "IT-1"
