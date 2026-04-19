"""Entity model smoke tests for internal model tier."""

from __future__ import annotations

from tests import m


class TestModelsEntity:
    def test_entity_instantiation(self) -> None:
        class EntitySample(m.Entity):
            name: str

        entity = EntitySample(name="sample", domain_events=[])
        assert entity.name == "sample"
        assert entity.unique_id is not None

    def test_entity_dump_contains_field(self) -> None:
        class EntitySample(m.Entity):
            code: str

        entity = EntitySample(code="E-1", domain_events=[])
        dumped = entity.model_dump(mode="python")
        assert dumped["code"] == "E-1"
