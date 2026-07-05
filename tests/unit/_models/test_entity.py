"""Behavioral contract tests for the Entity / Value / AggregateRoot models.

These tests exercise only the public model contract exposed through the
``m`` facade: auto-populated identity/lifecycle fields, identity-based
equality and hashing for entities, value-based semantics for value objects,
mutable event buffer isolation, and public serialization via ``model_dump``.
No private attributes, internal collaborators, or implementation details
are inspected.
"""

from __future__ import annotations

import pytest

from tests.models import m


class TestsFlextCoreEntity:
    """Behavioral tests for the Entity DDD model family."""

    class _Sample(m.Entity):
        """Concrete entity used to exercise the public Entity contract."""

        name: str = m.Field(default="sample", description="Sample entity name")

    class _Coord(m.Value):
        """Concrete value object used to exercise value semantics."""

        x: int = m.Field(default=0, description="X coordinate")
        y: int = m.Field(default=0, description="Y coordinate")

    def test_entity_exposes_declared_field(self) -> None:
        entity = self._Sample(name="alpha")

        assert entity.name == "alpha"

    def test_entity_auto_generates_unique_id(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="b")

        assert first.unique_id is not None
        assert first.unique_id != second.unique_id

    def test_entity_lifecycle_fields_are_populated(self) -> None:
        entity = self._Sample(name="a")

        assert entity.version == 1
        assert entity.created_at is not None
        assert entity.updated_at is not None

    def test_entity_domain_events_default_empty(self) -> None:
        entity = self._Sample(name="a")

        assert entity.domain_events == []

    def test_entity_domain_events_are_not_shared_between_instances(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="b")

        assert first.domain_events is not second.domain_events

    def test_entities_with_same_identity_are_equal(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="different", unique_id=original.unique_id)

        assert original == same_identity

    def test_entities_with_different_identity_are_not_equal(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="a")

        assert first != second

    def test_entity_equality_with_non_entity_is_false(self) -> None:
        entity = self._Sample(name="a")

        assert entity != 42
        assert entity != "a"

    def test_entity_hash_is_consistent_with_identity_equality(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="other", unique_id=original.unique_id)

        assert hash(original) == hash(same_identity)

    def test_entities_with_same_identity_collapse_in_set(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="other", unique_id=original.unique_id)

        assert len({original, same_identity}) == 1

    def test_entity_model_dump_exposes_public_fields(self) -> None:
        entity = self._Sample(name="alpha")

        dumped = entity.model_dump(mode="python")

        assert dumped["name"] == "alpha"
        assert dumped["unique_id"] == entity.unique_id
        assert dumped["version"] == 1
        assert "created_at" in dumped
        assert "domain_events" in dumped

    def test_aggregate_root_is_an_entity(self) -> None:
        assert issubclass(m.AggregateRoot, m.Entity)

    def test_aggregate_root_uses_identity_equality(self) -> None:
        root = m.AggregateRoot()
        twin = m.AggregateRoot(unique_id=root.unique_id)

        assert root == twin
        assert hash(root) == hash(twin)

    def test_values_with_equal_fields_are_equal(self) -> None:
        assert self._Coord(x=1, y=2) == self._Coord(x=1, y=2)

    @pytest.mark.parametrize(("x_a", "y_a", "x_b", "y_b"), [(1, 2, 9, 2), (1, 2, 1, 9)])
    def test_values_with_differing_fields_are_not_equal(
        self, x_a: int, y_a: int, x_b: int, y_b: int
    ) -> None:
        assert self._Coord(x=x_a, y=y_a) != self._Coord(x=x_b, y=y_b)

    def test_equal_values_share_a_hash(self) -> None:
        assert hash(self._Coord(x=3, y=4)) == hash(self._Coord(x=3, y=4))

    def test_value_equality_with_non_value_is_false(self) -> None:
        assert self._Coord(x=1, y=1) != (1, 1)
