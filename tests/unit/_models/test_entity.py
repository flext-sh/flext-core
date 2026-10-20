"""Behavioral contract tests for the Entity / Value / AggregateRoot models.

These tests exercise only the public model contract exposed through the
``m`` facade: auto-populated identity/lifecycle fields, identity-based
equality and hashing for entities, value-based semantics for value objects,
mutable event buffer isolation, and public serialization via ``model_dump``.
No private attributes, internal collaborators, or implementation details
are inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import m, tm


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

        tm.that(entity.name, eq="alpha")

    def test_entity_auto_generates_unique_id(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="b")

        tm.that(first.unique_id, none=False)
        tm.that(first.unique_id, ne=second.unique_id)

    def test_entity_lifecycle_fields_are_populated(self) -> None:
        entity = self._Sample(name="a")

        tm.that(entity.version, eq=1)
        tm.that(entity.created_at, none=False)
        tm.that(entity.updated_at, none=False)

    def test_entity_domain_events_default_empty(self) -> None:
        entity = self._Sample(name="a")

        tm.that(entity.domain_events, eq=[])

    def test_entity_domain_events_are_not_shared_between_instances(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="b")

        tm.that(first.domain_events is not second.domain_events, truth=True)

    def test_entities_with_same_identity_are_equal(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="different", unique_id=original.unique_id)

        tm.that(original, eq=same_identity)

    def test_entities_with_different_identity_are_not_equal(self) -> None:
        first = self._Sample(name="a")
        second = self._Sample(name="a")

        tm.that(first, ne=second)

    def test_entity_equality_with_non_entity_is_false(self) -> None:
        entity = self._Sample(name="a")

        tm.that(entity, ne=42)
        tm.that(entity, ne="a")

    def test_entity_hash_is_consistent_with_identity_equality(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="other", unique_id=original.unique_id)

        tm.that(hash(original), eq=hash(same_identity))

    def test_entities_with_same_identity_share_hash(self) -> None:
        original = self._Sample(name="a")
        same_identity = self._Sample(name="other", unique_id=original.unique_id)

        tm.that(original, eq=same_identity)
        tm.that(len({hash(original), hash(same_identity)}), eq=1)

    def test_entity_model_dump_exposes_public_fields(self) -> None:
        entity = self._Sample(name="alpha")

        dumped = entity.model_dump(mode="python")

        tm.that(dumped["name"], eq="alpha")
        tm.that(dumped["unique_id"], eq=entity.unique_id)
        tm.that(dumped["version"], eq=1)
        tm.that("created_at" in dumped, truth=True)
        tm.that("domain_events" in dumped, truth=True)

    @staticmethod
    def test_aggregate_root_is_an_entity() -> None:
        tm.that(issubclass(m.AggregateRoot, m.Entity), truth=True)

    @staticmethod
    def test_aggregate_root_uses_identity_equality() -> None:
        root = m.AggregateRoot()
        twin = m.AggregateRoot(unique_id=root.unique_id)

        tm.that(root, eq=twin)
        tm.that(hash(root), eq=hash(twin))

    def test_values_with_equal_fields_are_equal(self) -> None:
        tm.that(self._Coord(x=1, y=2), eq=self._Coord(x=1, y=2))

    @pytest.mark.parametrize(("x_a", "y_a", "x_b", "y_b"), [(1, 2, 9, 2), (1, 2, 1, 9)])
    def test_values_with_differing_fields_are_not_equal(
        self,
        x_a: int,
        y_a: int,
        x_b: int,
        y_b: int,
    ) -> None:
        tm.that(self._Coord(x=x_a, y=y_a), ne=self._Coord(x=x_b, y=y_b))

    def test_equal_values_share_a_hash(self) -> None:
        tm.that(hash(self._Coord(x=3, y=4)), eq=hash(self._Coord(x=3, y=4)))

    def test_value_equality_with_non_value_is_false(self) -> None:
        tm.that(self._Coord(x=1, y=1), ne=(1, 1))
