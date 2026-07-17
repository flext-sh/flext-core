"""Behavioral tests for the public FlextModels facade contract.

Exercises only observable public behavior of the model building blocks
exposed through the ``m`` facade (value objects, entities, mapping
containers, pagination) — value/identity semantics, immutability,
serialization, and mapping access — never private internals.
"""

from __future__ import annotations

import pytest

from tests.models import m
from tests.typings import p, t


class TestsFlextCoreModels:
    """Behavioral contract tests for FlextModels public building blocks."""

    class _SampleValue(m.Value):
        """Value object used to exercise value-object semantics."""

        x: int = m.Field(description="Numeric component of the value object")
        y: str = m.Field(description="String component of the value object")

    class _SampleEntity(m.Entity):
        """Entity used to exercise identity semantics."""

        name: str = m.Field(description="Human-readable entity name")

    # ----- Value object: value-based equality --------------------------------

    def test_value_objects_with_equal_fields_are_equal(self) -> None:
        left = self._SampleValue(x=1, y="a")
        right = self._SampleValue(x=1, y="a")

        assert left == right
        assert hash(left) == hash(right)

    @pytest.mark.parametrize(
        ("x", "y"),
        [(2, "a"), (1, "b"), (9, "z")],
    )
    def test_value_objects_with_different_fields_are_unequal(
        self, x: int, y: str
    ) -> None:
        base = self._SampleValue(x=1, y="a")

        other = self._SampleValue(x=x, y=y)

        assert base != other

    def test_value_object_is_immutable(self) -> None:
        value = self._SampleValue(x=1, y="a")

        with pytest.raises(m.ValidationError):
            setattr(value, "x", 5)

    def test_value_object_model_dump_exposes_public_fields(self) -> None:
        value = self._SampleValue(x=7, y="ok")

        dumped = value.model_dump()

        assert dumped == {"x": 7, "y": "ok"}

    def test_value_object_rejects_wrong_field_type(self) -> None:
        with pytest.raises(m.ValidationError):
            self._SampleValue.model_validate({"x": "not-an-int", "y": "a"})

    # ----- Entity: identity-based semantics ----------------------------------

    def test_entities_receive_distinct_generated_identities(self) -> None:
        first = self._SampleEntity(name="a")
        second = self._SampleEntity(name="a")

        assert first.unique_id != second.unique_id

    def test_entity_equality_is_identity_based(self) -> None:
        first = self._SampleEntity(name="same")
        same_reference = first
        second = self._SampleEntity(name="same")

        assert same_reference == first
        assert first != second

    def test_entity_hash_is_stable_for_identity(self) -> None:
        entity = self._SampleEntity(name="a")

        assert hash(entity) == hash(entity)

    def test_entity_not_equal_to_non_entity_object(self) -> None:
        entity = self._SampleEntity(name="a")

        assert entity != object()

    def test_entity_populates_lifecycle_defaults(self) -> None:
        entity = self._SampleEntity(name="a")

        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert entity.version == 1

    def test_entity_domain_events_buffer_starts_empty_and_appends(self) -> None:
        entity = self._SampleEntity(name="a")
        assert list(entity.domain_events) == []

        entry = m.Entry(event_type="created", aggregate_id=entity.unique_id)
        entity.domain_events.append(entry)

        assert list(entity.domain_events) == [entry]

    def test_entity_model_dump_exposes_public_fields(self) -> None:
        entity = self._SampleEntity(name="named")

        dumped = entity.model_dump()

        assert dumped["name"] == "named"
        assert dumped["version"] == 1
        assert "unique_id" in dumped

    # ----- Mapping containers ------------------------------------------------

    def test_config_map_supports_read_only_mapping_access(self) -> None:
        data: t.ScalarMapping = {"a": 1, "b": 2}
        config = m.ConfigMap(root=dict(data))

        assert config["a"] == 1
        assert len(config) == 2
        assert "b" in config
        assert set(config.keys()) == {"a", "b"}

    def test_dict_supports_item_assignment_and_lookup(self) -> None:
        payload = m.Dict(root={"name": "flext"})

        payload["version"] = "0.20.0"

        assert payload["version"] == "0.20.0"
        assert len(payload) == 2
        assert "name" in payload

    # ----- Pagination value object -------------------------------------------

    @pytest.mark.parametrize(
        ("page", "size"),
        [(1, 10), (3, 25), (5, 100)],
    )
    def test_pagination_exposes_page_and_size(self, page: int, size: int) -> None:
        pagination = m.Pagination(page=page, size=size)

        assert pagination.page == page
        assert pagination.size == size
