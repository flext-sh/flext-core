"""Behavioral tests for the public FlextModels facade contract.

Exercises only observable public behavior of the model building blocks
exposed through the ``m`` facade (value objects, entities, mapping
containers, pagination) — value/identity semantics, immutability,
serialization, and mapping access — never private internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from tests import m, t, tm


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
        """Test value objects with equal fields are equal."""
        left = self._SampleValue(x=1, y="a")
        right = self._SampleValue(x=1, y="a")

        tm.that(left, eq=right)
        tm.that(hash(left), eq=hash(right))

    @pytest.mark.parametrize(("x", "y"), [(2, "a"), (1, "b"), (9, "z")])
    def test_value_objects_with_different_fields_are_unequal(
        self,
        x: int,
        y: str,
    ) -> None:
        """Test value objects with different fields are unequal."""
        base = self._SampleValue(x=1, y="a")

        other = self._SampleValue(x=x, y=y)

        tm.that(base, ne=other)

    def test_value_object_is_immutable(self) -> None:
        """Test value object is immutable."""
        value = self._SampleValue(x=1, y="a")

        tm.rejects_assignment(value, "x", 5, expected=m.ValidationError)

    def test_value_object_model_dump_exposes_public_fields(self) -> None:
        """Test value object model dump exposes public fields."""
        value = self._SampleValue(x=7, y="ok")

        dumped = value.model_dump()

        tm.that(dumped, eq={"x": 7, "y": "ok"})

    def test_value_object_rejects_wrong_field_type(self) -> None:
        """Test value object rejects wrong field type."""
        with pytest.raises(m.ValidationError):
            self._SampleValue.model_validate({"x": "not-an-int", "y": "a"})

    # ----- Entity: value equality, explicit identity ---------------------------

    def test_entities_receive_distinct_generated_identities(self) -> None:
        """Test entities receive distinct generated identities."""
        first = self._SampleEntity(name="a")
        second = self._SampleEntity(name="a")

        tm.that(first.unique_id, ne=second.unique_id)

    def test_entity_equality_is_value_based(self) -> None:
        """Test entity equality compares field values, identity included."""
        first = self._SampleEntity(name="same")
        second = self._SampleEntity(name="same")

        tm.that(first, eq=first.model_copy())
        tm.that(first, ne=second)

    def test_mutable_entity_is_not_hashable(self) -> None:
        """Test a mutable entity is not hashable."""
        entity = self._SampleEntity(name="a")

        with pytest.raises(TypeError):
            hash(entity)

    def test_entity_not_equal_to_non_entity_object(self) -> None:
        """Test entity not equal to non entity object."""
        entity = self._SampleEntity(name="a")

        tm.that(entity, ne=object())

    def test_entity_populates_lifecycle_defaults(self) -> None:
        """Test entity populates lifecycle defaults."""
        entity = self._SampleEntity(name="a")

        tm.that(entity.created_at, none=False)
        tm.that(entity.updated_at, none=False)
        tm.that(entity.version, eq=1)

    def test_entity_domain_events_buffer_starts_empty_and_appends(self) -> None:
        """Test entity domain events buffer starts empty and appends."""
        entity = self._SampleEntity(name="a")
        tm.that(list(entity.domain_events), eq=[])

        entry = m.DomainEvent(event_type="created", aggregate_id=entity.unique_id)
        entity.domain_events.append(entry)

        tm.that(list(entity.domain_events), eq=[entry])

    def test_entity_model_dump_exposes_public_fields(self) -> None:
        """Test entity model dump exposes public fields."""
        entity = self._SampleEntity(name="named")

        dumped = entity.model_dump()

        tm.that(dumped["name"], eq="named")
        tm.that(dumped["version"], eq=1)
        tm.that("unique_id" in dumped, truth=True)

    # ----- Mapping containers ------------------------------------------------

    @staticmethod
    def test_config_map_supports_read_only_mapping_access() -> None:
        """Test config map supports read only mapping access."""
        data: Mapping[str, t.Scalar] = {"a": 1, "b": 2}
        config = m.ConfigMap(root=dict(data))

        tm.that(config["a"], eq=1)
        tm.that(len(config), eq=2)
        tm.that("b" in config, truth=True)
        tm.that(set(config.keys()), eq={"a", "b"})

    @staticmethod
    def test_dict_supports_item_assignment_and_lookup() -> None:
        """Test dict supports item assignment and lookup."""
        payload = m.Dict(root={"name": "flext"})

        payload["version"] = "0.12.0"

        tm.that(payload["version"], eq="0.12.0")
        tm.that(len(payload), eq=2)
        tm.that("name" in payload, truth=True)

    # ----- Pagination value object -------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(("page", "size"), [(1, 10), (3, 25), (5, 100)])
    def test_pagination_exposes_page_and_size(page: int, size: int) -> None:
        """Test pagination exposes page and size."""
        pagination = m.Pagination(page=page, size=size)

        tm.that(pagination.page, eq=page)
        tm.that(pagination.size, eq=size)
