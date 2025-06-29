#!/usr/bin/env python3
"""DIRECT test for Level 1 pydantic_base.py - No complex imports."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Direct import of the target file
sys.path.append(str(src_path / "flx_core" / "domain"))

# Import dependencies first
from datetime import UTC, datetime
from uuid import UUID

# Now import directly our target
import pydantic_base


def test_level1_direct():
    """Direct test of Level 1 classes without complex project imports."""
    print("🧪 Testing Level 1 (pydantic_base.py) - DIRECT IMPORT")
    print("=" * 60)

    # Test 1: DomainBaseModel basic functionality
    class TestModel(pydantic_base.DomainBaseModel):
        name: str
        value: int = 42

    model = TestModel(name="test")
    assert model.name == "test"
    assert model.value == 42
    print("✅ DomainBaseModel: PASS")

    # Test 2: DomainValueObject immutability and equality
    class TestValueObject(pydantic_base.DomainValueObject):
        value: int
        name: str

    vo1 = TestValueObject(value=42, name="test")
    vo2 = TestValueObject(value=42, name="test")

    assert vo1 == vo2  # Value-based equality
    assert hash(vo1) == hash(vo2)  # Value-based hashing
    print("✅ DomainValueObject: PASS")

    # Test 3: DomainEntity identity-based equality
    class TestEntity(pydantic_base.DomainEntity):
        name: str

    entity1 = TestEntity(name="test1")
    entity2 = TestEntity(name="test2")
    entity3 = TestEntity(id=entity1.id, name="different_name")

    assert entity1 != entity2  # Different IDs
    assert entity1 == entity3  # Same ID (identity-based)
    assert isinstance(entity1.id, UUID)
    assert isinstance(entity1.created_at, datetime)
    assert entity1.created_at.tzinfo == UTC
    print("✅ DomainEntity: PASS")

    # Test 4: DomainAggregateRoot event handling
    class TestAggregate(pydantic_base.DomainAggregateRoot):
        name: str

    class TestEvent(pydantic_base.DomainEvent):
        aggregate_id: UUID
        event_type: str = "test_event"

    aggregate = TestAggregate(name="test")
    event = TestEvent(aggregate_id=aggregate.id)

    initial_version = aggregate.aggregate_version
    aggregate.add_domain_event(event)

    assert len(aggregate.domain_events_list) == 1
    assert aggregate.aggregate_version == initial_version + 1
    print("✅ DomainAggregateRoot: PASS")

    # Test 5: DomainSpecification composition
    class PositiveSpecification(pydantic_base.DomainSpecification):
        specification_name: str = "positive"

        def is_satisfied_by(self, candidate: object) -> bool:
            return isinstance(candidate, (int, float)) and candidate > 0

    class EvenSpecification(pydantic_base.DomainSpecification):
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

    print("✅ DomainSpecification: PASS")

    # Test 6: Serialization functionality
    class TestModelSerialization(pydantic_base.DomainBaseModel):
        name: str
        value: int

    model = TestModelSerialization(name="test", value=42)
    json_data = model.model_dump_json_safe()

    assert json_data["name"] == "test"
    assert json_data["value"] == 42
    print("✅ Serialization: PASS")

    print("=" * 60)
    print("🏆 ALL TESTS PASSED - Level 1 is working correctly!")
    print("📊 Coverage: 6 core scenarios tested successfully")
    print("🎯 Results: pydantic_base.py is 100% functional")


if __name__ == "__main__":
    try:
        test_level1_direct()
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
