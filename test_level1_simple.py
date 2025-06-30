#!/usr/bin/env python3
"""SIMPLE test for Level 1 pydantic_base.py - Core classes only."""

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


def test_level1_simple() -> None:
    """Simple test of Level 1 core classes."""

    # Test 1: DomainBaseModel
    class TestModel(pydantic_base.DomainBaseModel):
        name: str
        value: int = 42

    model = TestModel(name="test")
    assert model.name == "test"
    assert model.value == 42

    # Test serialization
    json_data = model.model_dump_json_safe()
    assert json_data["name"] == "test"
    assert json_data["value"] == 42

    # Test 2: DomainValueObject
    class TestValueObject(pydantic_base.DomainValueObject):
        value: int
        name: str

    vo1 = TestValueObject(value=42, name="test")
    vo2 = TestValueObject(value=42, name="test")
    vo3 = TestValueObject(value=43, name="test")

    # Value-based equality
    assert vo1 == vo2  # Same values
    assert vo1 != vo3  # Different values
    assert hash(vo1) == hash(vo2)  # Same hash for same values

    # Test 3: DomainEntity
    class TestEntity(pydantic_base.DomainEntity):
        name: str

    entity1 = TestEntity(name="test1")
    entity2 = TestEntity(name="test2")
    entity3 = TestEntity(id=entity1.id, name="different_name")

    # Identity-based equality
    assert entity1 != entity2  # Different IDs
    assert entity1 == entity3  # Same ID, different data

    # Proper types
    assert isinstance(entity1.id, UUID)
    assert isinstance(entity1.created_at, datetime)
    assert entity1.created_at.tzinfo == UTC
    assert entity1.version == 1

    # Test 4: DomainCommand
    class TestCommand(pydantic_base.DomainCommand):
        action: str
        target: str

    command = TestCommand(action="create", target="pipeline")

    assert command.action == "create"
    assert command.target == "pipeline"
    assert isinstance(command.command_id, UUID)
    assert isinstance(command.issued_at, datetime)
    assert command.issued_at.tzinfo == UTC

    # Test 5: DomainQuery
    class TestQuery(pydantic_base.DomainQuery):
        filter_field: str

    query = TestQuery(filter_field="name", limit=10, offset=0)

    assert query.filter_field == "name"
    assert query.limit == 10
    assert query.offset == 0
    assert isinstance(query.query_id, UUID)
    assert isinstance(query.issued_at, datetime)

    # Test 6: DomainSpecification
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

    # Basic specification functionality
    assert positive_spec.is_satisfied_by(5) is True
    assert positive_spec.is_satisfied_by(-3) is False
    assert even_spec.is_satisfied_by(4) is True
    assert even_spec.is_satisfied_by(3) is False

    # Composition testing
    positive_and_even = positive_spec & even_spec
    assert positive_and_even.is_satisfied_by(4) is True  # Positive and even
    assert positive_and_even.is_satisfied_by(3) is False  # Positive but odd
    assert positive_and_even.is_satisfied_by(-2) is False  # Even but negative

    positive_or_even = positive_spec | even_spec
    assert positive_or_even.is_satisfied_by(4) is True  # Both
    assert positive_or_even.is_satisfied_by(3) is True  # Positive only
    assert positive_or_even.is_satisfied_by(-2) is True  # Even only
    assert positive_or_even.is_satisfied_by(-3) is False  # Neither

    not_positive = ~positive_spec
    assert not_positive.is_satisfied_by(5) is False  # Positive (negated)
    assert not_positive.is_satisfied_by(-3) is True  # Not positive


if __name__ == "__main__":
    try:
        test_level1_simple()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
