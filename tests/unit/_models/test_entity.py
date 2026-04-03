"""Tests for Entity, Value, AggregateRoot, and DomainEvent via FlextModels facade.

Covers entity.py (244 LOC) and domain_event.py (175 LOC) through the m facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import pytest
from pydantic import ValidationError

from flext_tests import tm
from tests import c, m, t


class TestFlextModelsEntity:
    """Tests for entity, value object, aggregate root, and domain event models."""

    # ── DomainEvent.Entry creation ─────────────────────────────

    def test_domain_event_entry_defaults(self) -> None:
        event = m.Entry(event_type="created", aggregate_id="agg-1")
        tm.that(event.event_type, eq="created")
        tm.that(event.aggregate_id, eq="agg-1")
        tm.that(event.message_type, eq="event")
        tm.that(event.data.root, eq={})
        tm.that(event.unique_id, is_=str, empty=False)
        tm.that(event.created_at.tzinfo, eq=UTC)

    def test_domain_event_entry_with_data(self) -> None:
        data = m.ComparableConfigMap(root={"key": "value", "count": "42"})
        event = m.Entry(
            event_type="updated",
            aggregate_id="agg-2",
            data=data,
        )
        tm.that(event.data.root, eq={"key": "value", "count": "42"})

    def test_domain_event_entry_validates_dict_data(self) -> None:
        event = m.Entry(
            event_type="evt",
            aggregate_id="agg-1",
            data={"status": "active"},
        )
        tm.that(event.data.root, eq={"status": "active"})

    def test_domain_event_entry_none_data_becomes_empty(self) -> None:
        event = m.Entry(
            event_type="evt",
            aggregate_id="agg-1",
            data=None,
        )
        tm.that(event.data.root, eq={})

    def test_domain_event_entry_rejects_non_dict_data(self) -> None:
        with pytest.raises(TypeError, match="Domain event data must be a dictionary"):
            m.Entry.model_validate(
                {"event_type": "evt", "aggregate_id": "agg-1", "data": 42},
            )

    def test_domain_event_entry_requires_non_empty_event_type(self) -> None:
        with pytest.raises(ValidationError):
            m.Entry(event_type="", aggregate_id="agg-1")

    def test_domain_event_entry_requires_non_empty_aggregate_id(self) -> None:
        with pytest.raises(ValidationError):
            m.Entry(event_type="evt", aggregate_id="")

    def test_domain_event_alias_matches_entry(self) -> None:
        tm.that(m.DomainEvent, eq=m.Entry)

    # ── ComparableConfigMap equality ──────────────────────────

    def test_comparable_config_map_eq_dict(self) -> None:
        cfg = m.ComparableConfigMap(root={"a": 1})
        tm.that(cfg == {"a": 1}, eq=True)
        tm.that(cfg == {"a": 2}, eq=False)

    def test_comparable_config_map_eq_non_mapping(self) -> None:
        cfg = m.ComparableConfigMap(root={"a": 1})
        tm.that(cfg == 42, eq=False)

    def test_comparable_config_map_eq_mapping(self) -> None:
        cfg = m.ComparableConfigMap(root={"x": "y"})
        other = m.ComparableConfigMap(root={"x": "y"})
        tm.that(cfg == other, eq=True)

    # ── to_config_map helper ──────────────────────────────────

    def test_to_config_map_none_returns_empty(self) -> None:
        result = m.to_config_map(None)
        tm.that(result.root, eq={})

    def test_to_config_map_with_data(self) -> None:
        data = t.ConfigMap(root={"key": "val"})
        result = m.to_config_map(data)
        tm.that(result.root, eq={"key": "val"})
        tm.that(result, is_=m.ComparableConfigMap)

    # ── metadata_to_normalized ────────────────────────────────

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (None, None),
            ("hello", "hello"),
            (42, 42),
            (math.pi, math.pi),
            (True, True),
        ],
        ids=["none", "str", "int", "float", "bool"],
    )
    def test_metadata_to_normalized_scalars(
        self,
        input_val: t.MetadataOrValue | None,
        expected: t.RecursiveContainer,
    ) -> None:
        result = m.metadata_to_normalized(input_val)
        tm.that(result, eq=expected)

    def test_metadata_to_normalized_mapping(self) -> None:
        result = m.metadata_to_normalized({"k": "v"})
        tm.that(result, eq={"k": "v"})

    def test_metadata_to_normalized_sequence(self) -> None:
        result = m.metadata_to_normalized(["a", "b"])
        tm.that(result, eq=["a", "b"])

    def test_metadata_to_normalized_datetime(self) -> None:
        now = datetime.now(UTC)
        result = m.metadata_to_normalized(now)
        tm.that(result, eq=now)

    # ── Entity creation ───────────────────────────────────────

    def test_entity_creation_defaults(self) -> None:
        entity = m.Entity(unique_id="e-1")
        tm.that(entity.unique_id, eq="e-1")
        tm.that(entity.entity_id, eq="e-1")
        tm.that(entity.version, eq=1)
        tm.that(entity.created_at.tzinfo, eq=UTC)
        tm.that(entity.updated_at, none=False)
        tm.that(entity.domain_events, eq=[])

    def test_entity_updated_at_set_by_model_post_init(self) -> None:
        entity = m.Entity(unique_id="e-1")
        tm.that(entity.updated_at, is_=datetime)
        tm.that(cast("datetime", entity.updated_at).tzinfo, eq=UTC)

    # ── Entity identity equality ──────────────────────────────

    def test_entity_equality_by_identity(self) -> None:
        e1 = m.Entity(unique_id="same-id")
        e2 = m.Entity(unique_id="same-id")
        tm.that(e1 == e2, eq=True)

    def test_entity_inequality_different_ids(self) -> None:
        e1 = m.Entity(unique_id="id-1")
        e2 = m.Entity(unique_id="id-2")
        tm.that(e1 == e2, eq=False)

    def test_entity_eq_returns_not_implemented_for_non_model(self) -> None:
        entity = m.Entity(unique_id="e-1")
        tm.that(entity.__eq__("not-a-model"), eq=NotImplemented)

    def test_entity_hash_by_identity(self) -> None:
        e1 = m.Entity(unique_id="hash-id")
        e2 = m.Entity(unique_id="hash-id")
        tm.that(hash(e1), eq=hash(e2))

    def test_entity_hash_differs_for_different_ids(self) -> None:
        e1 = m.Entity(unique_id="a")
        e2 = m.Entity(unique_id="b")
        tm.that(hash(e1) != hash(e2), eq=True)

    def test_entities_usable_in_sets(self) -> None:
        e1 = m.Entity(unique_id="x")
        e2 = m.Entity(unique_id="x")
        e3 = m.Entity(unique_id="y")
        s = {e1, e2, e3}
        tm.that(len(s), eq=2)

    # ── Entity event collection ───────────────────────────────

    def test_add_domain_event_success(self) -> None:
        entity = m.Entity(unique_id="e-1")
        result = entity.add_domain_event("order_placed", {"item": "book"})
        event = tm.ok(result)
        tm.that(event.event_type, eq="order_placed")
        tm.that(event.aggregate_id, eq="e-1")
        tm.that(len(entity.domain_events), eq=1)

    def test_add_domain_event_empty_type_fails(self) -> None:
        entity = m.Entity(unique_id="e-1")
        result = entity.add_domain_event("")
        tm.fail(result, has="non-empty string")

    def test_add_domain_event_none_data_defaults_to_empty(self) -> None:
        entity = m.Entity(unique_id="e-1")
        result = entity.add_domain_event("evt")
        event = tm.ok(result)
        tm.that(event.data.root, eq={})

    def test_uncommitted_events_returns_copy(self) -> None:
        entity = m.Entity(unique_id="e-1")
        entity.add_domain_event("evt1")
        entity.add_domain_event("evt2")
        events = entity.uncommitted_events
        tm.that(len(events), eq=2)
        # Verify it's a copy, not the original list
        tm.that(events is not entity.domain_events, eq=True)

    def test_clear_domain_events(self) -> None:
        entity = m.Entity(unique_id="e-1")
        entity.add_domain_event("evt1")
        entity.add_domain_event("evt2")
        cleared = entity.clear_domain_events()
        tm.that(len(cleared), eq=2)
        tm.that(len(entity.domain_events), eq=0)

    def test_mark_events_as_committed(self) -> None:
        entity = m.Entity(unique_id="e-1")
        entity.add_domain_event("evt1")
        entity.add_domain_event("evt2")
        result = entity.mark_events_as_committed()
        committed = tm.ok(result)
        tm.that(len(committed), eq=2)
        tm.that(len(entity.domain_events), eq=0)

    def test_add_domain_event_max_limit(self) -> None:
        entity = m.Entity(unique_id="e-1")
        for i in range(c.HTTP_STATUS_MIN):
            tm.ok(entity.add_domain_event(f"evt_{i}"))
        result = entity.add_domain_event("one_too_many")
        tm.fail(result, has="exceed max events limit")

    # ── Bulk domain events ────────────────────────────────────

    def test_add_domain_events_bulk_success(self) -> None:
        entity = m.Entity(unique_id="e-1")
        events_input: Sequence[tuple[str, t.ConfigMap | None]] = [
            ("created", None),
            ("updated", t.ConfigMap(root={"field": "name"})),
        ]
        result = entity.add_domain_events_bulk(events_input)
        created = tm.ok(result)
        tm.that(len(created), eq=2)
        tm.that(len(entity.domain_events), eq=2)

    def test_add_domain_events_bulk_rejects_non_sequence(self) -> None:
        entity = m.Entity(unique_id="e-1")
        result = entity.add_domain_events_bulk(
            cast("Sequence[tuple[str, t.ConfigMap | None]]", "invalid")
        )
        tm.fail(result, has="must be a list or tuple")

    def test_add_domain_events_bulk_empty_event_type_fails(self) -> None:
        entity = m.Entity(unique_id="e-1")
        events: Sequence[tuple[str, t.ConfigMap | None]] = [("", None)]
        result = entity.add_domain_events_bulk(events)
        tm.fail(result, has="non-empty string")

    def test_add_domain_events_bulk_exceeds_max(self) -> None:
        entity = m.Entity(unique_id="e-1")
        for i in range(c.HTTP_STATUS_MIN - 1):
            entity.add_domain_event(f"evt_{i}")
        events: Sequence[tuple[str, t.ConfigMap | None]] = [
            ("a", None),
            ("b", None),
        ]
        result = entity.add_domain_events_bulk(events)
        tm.fail(result, has="exceed max events limit")

    # ── Entity event handler dispatch ─────────────────────────

    def test_add_domain_event_calls_apply_handler(self) -> None:
        """Verify _apply_{event_type} is called if present."""

        class _TrackedEntity(m.Entity):
            applied: list[str] = []

            def _apply_order_placed(self, data: t.ContainerMapping) -> None:
                self.applied.append("order_placed")

        entity = _TrackedEntity(unique_id="e-1")
        entity.add_domain_event("order_placed", {"item": "book"})
        tm.that(entity.applied, eq=["order_placed"])

    def test_add_domain_event_handler_exception_suppressed(self) -> None:
        """Apply handler exceptions are suppressed, event still added."""

        class _FailingEntity(m.Entity):
            def _apply_boom(self, data: t.ContainerMapping) -> None:
                msg = "handler error"
                raise RuntimeError(msg)

        entity = _FailingEntity(unique_id="e-1")
        result = entity.add_domain_event("boom")
        tm.ok(result)
        tm.that(len(entity.domain_events), eq=1)

    def test_add_domain_event_uses_data_event_type_for_handler(self) -> None:
        """If data contains event_type key, use that for handler lookup."""

        class _OverrideEntity(m.Entity):
            applied: list[str] = []

            def _apply_real_type(self, data: t.ContainerMapping) -> None:
                self.applied.append("real_type")

        entity = _OverrideEntity(unique_id="e-1")
        data = t.ConfigMap(root={"event_type": "real_type"})
        entity.add_domain_event("original_type", data)
        tm.that(entity.applied, eq=["real_type"])

    # ── Entity serialization round-trip ───────────────────────

    def test_entity_serialization_round_trip(self) -> None:
        entity = m.Entity(unique_id="rt-1")
        entity.add_domain_event("evt", {"k": "v"})
        dumped = entity.model_dump()
        tm.that(dumped["unique_id"], eq="rt-1")
        tm.that(dumped["entity_id"], eq="rt-1")
        tm.that(len(dumped["domain_events"]), eq=1)

    def test_entity_json_round_trip(self) -> None:
        entity = m.Entity(unique_id="json-1")
        json_str = entity.model_dump_json()
        tm.that(json_str, is_=str, has="json-1")

    # ── Value object ──────────────────────────────────────────

    def test_value_object_equality_by_value(self) -> None:
        class _Money(m.Value):
            amount: int
            currency: str

        v1 = _Money(amount=100, currency="USD")
        v2 = _Money(amount=100, currency="USD")
        v3 = _Money(amount=200, currency="USD")
        tm.that(v1 == v2, eq=True)
        tm.that(v1 == v3, eq=False)

    def test_value_object_hash_by_value(self) -> None:
        class _Point(m.Value):
            x: int
            y: int

        p1 = _Point(x=1, y=2)
        p2 = _Point(x=1, y=2)
        tm.that(hash(p1), eq=hash(p2))

    def test_value_object_usable_in_sets(self) -> None:
        class _Tag(m.Value):
            name: str

        t1 = _Tag(name="a")
        t2 = _Tag(name="a")
        t3 = _Tag(name="b")
        tm.that(hash(t1), eq=hash(t2))
        tm.that(hash(t1) != hash(t3), eq=True)

    def test_value_object_immutable(self) -> None:
        class _Frozen(m.Value):
            name: str

        v = _Frozen(name="fixed")
        with pytest.raises(ValidationError):
            v.name = "changed"

    def test_value_object_eq_non_model_returns_not_implemented(self) -> None:
        class _Val(m.Value):
            x: int

        v = _Val(x=1)
        tm.that(v.__eq__("not-model"), eq=NotImplemented)

    # ── AggregateRoot ─────────────────────────────────────────

    def test_aggregate_root_inherits_entity(self) -> None:
        agg = m.AggregateRoot(unique_id="agg-1")
        tm.that(agg, is_=m.Entity)
        tm.that(agg.unique_id, eq="agg-1")
        tm.that(agg.entity_id, eq="agg-1")

    def test_aggregate_root_check_invariants_passes(self) -> None:
        agg = m.AggregateRoot(unique_id="agg-1")
        agg.check_invariants()  # no-op, no invariants defined

    def test_aggregate_root_check_invariants_fails(self) -> None:
        class _StrictAggregate(m.AggregateRoot):
            _invariants = [lambda: False]

        with pytest.raises(ValueError, match="Invariant violated"):
            _StrictAggregate(unique_id="agg-1")

    def test_aggregate_root_event_collection(self) -> None:
        agg = m.AggregateRoot(unique_id="agg-1")
        agg.add_domain_event("created")
        agg.add_domain_event("updated")
        tm.that(len(agg.domain_events), eq=2)
        committed = tm.ok(agg.mark_events_as_committed())
        tm.that(len(committed), eq=2)
        tm.that(len(agg.domain_events), eq=0)

    def test_aggregate_root_too_many_events_at_construction(self) -> None:
        """AggregateRoot validator rejects > HTTP_STATUS_MIN events."""
        events = [
            m.Entry(event_type=f"e{i}", aggregate_id="agg-1")
            for i in range(c.HTTP_STATUS_MIN + 1)
        ]
        with pytest.raises(ValueError, match="Too many uncommitted domain events"):
            m.AggregateRoot(unique_id="agg-1", domain_events=events)

    # ── Entity logger property ────────────────────────────────

    def test_entity_logger_property(self) -> None:
        entity = m.Entity(unique_id="e-1")
        logger = entity.logger
        tm.that(logger, none=False)
