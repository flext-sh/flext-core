#!/usr/bin/env python3
"""Isolated test for Level 1 pydantic_base.py - Direct validation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import UTC, datetime
from uuid import UUID

from flx_core.domain.pydantic_base import (
    DomainAggregateRoot,
    DomainBaseModel,
    DomainEntity,
    DomainSpecification,
    DomainValueObject,
)


def test_domain_base_model() -> None:
    """Test DomainBaseModel basic functionality."""

    class TestModel(DomainBaseModel):
        name: str
        value: int = 42

    model = TestModel(name="test")
    assert model.name == "test"
    assert model.value == 42


def test_domain_value_object() -> None:
    """Test DomainValueObject immutability and equality."""

    class TestValueObject(DomainValueObject):
        value: int
        name: str

    vo1 = TestValueObject(value=42, name="test")
    vo2 = TestValueObject(value=42, name="test")

    assert vo1 == vo2  # Value-based equality
    assert hash(vo1) == hash(vo2)  # Value-based hashing


def test_domain_entity() -> None:
    """Test DomainEntity identity-based equality."""

    class TestEntity(DomainEntity):
        name: str

    entity1 = TestEntity(name="test1")
    entity2 = TestEntity(name="test2")
    entity3 = TestEntity(id=entity1.id, name="different_name")

    assert entity1 != entity2  # Different IDs
    assert entity1 == entity3  # Same ID (identity-based)
    assert isinstance(entity1.id, UUID)
    assert isinstance(entity1.created_at, datetime)
    assert entity1.created_at.tzinfo == UTC


def test_domain_aggregate_root() -> None:
    """Test DomainAggregateRoot event handling."""

    class TestAggregate(DomainAggregateRoot):
        name: str

    from flx_core.domain.pydantic_base import DomainEvent

    class TestEvent(DomainEvent):
        aggregate_id: UUID
        event_type: str = "test_event"

    aggregate = TestAggregate(name="test")
    event = TestEvent(aggregate_id=aggregate.id)

    initial_version = aggregate.aggregate_version
    aggregate.add_domain_event(event)

    assert len(aggregate.domain_events_list) == 1
    assert aggregate.aggregate_version == initial_version + 1


def test_domain_specification() -> None:
    """Test DomainSpecification composition."""

    class PositiveSpecification(DomainSpecification):
        specification_name: str = "positive"

        def is_satisfied_by(self, candidate: object) -> bool:
            return isinstance(candidate, (int, float)) and candidate > 0

    class EvenSpecification(DomainSpecification):
        specification_name: str = "even"

        def is_satisfied_by(self, candidate: object) -> bool:
            return isinstance(candidate, int) and candidate % 2 == 0

    positive_spec = PositiveSpecification()
    even_spec = EvenSpecification()

    # Test AND composition
    positive_and_even = positive_spec & even_spec
    assert positive_and_even.is_satisfied_by(4) is True  # Positive and even
    assert positive_and_even.is_satisfied_by(3) is False  # Positive but odd

    # Test OR composition
    positive_or_even = positive_spec | even_spec
    assert positive_or_even.is_satisfied_by(4) is True  # Both
    assert positive_or_even.is_satisfied_by(3) is True  # Positive only
    assert positive_or_even.is_satisfied_by(-2) is True  # Even only
    assert positive_or_even.is_satisfied_by(-3) is False  # Neither

    # Test NOT composition
    not_positive = ~positive_spec
    assert not_positive.is_satisfied_by(5) is False  # Positive (negated)
    assert not_positive.is_satisfied_by(-3) is True  # Not positive


def test_serialization() -> None:
    """Test serialization functionality."""

    class TestModel(DomainBaseModel):
        name: str
        value: int

    model = TestModel(name="test", value=42)
    json_data = model.model_dump_json_safe()

    assert json_data["name"] == "test"
    assert json_data["value"] == 42


if __name__ == "__main__":
    try:
        test_domain_base_model()
        test_domain_value_object()
        test_domain_entity()
        test_domain_aggregate_root()
        test_domain_specification()
        test_serialization()

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
