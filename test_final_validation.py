#!/usr/bin/env python3
"""FINAL VALIDATION: Level 1 pydantic_base.py complete compliance test."""

import importlib.util
import sys
from pathlib import Path


def test_final_level1_validation() -> bool:
    """Final validation of Level 1 implementation."""
    # Direct module import without project complexity
    spec = importlib.util.spec_from_file_location(
        "pydantic_base",
        Path(__file__).parent / "src" / "flx_core" / "domain" / "pydantic_base.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pydantic_base"] = module
    spec.loader.exec_module(module)

    # Check all required classes exist
    required_classes = [
        "DomainBaseModel",
        "DomainValueObject",
        "DomainEntity",
        "DomainAggregateRoot",
        "DomainCommand",
        "DomainQuery",
        "DomainEvent",
        "DomainSpecification",
        "AndSpecification",
        "OrSpecification",
        "NotSpecification",
    ]

    for class_name in required_classes:
        assert hasattr(module, class_name), f"Missing class: {class_name}"

    # Test basic functionality
    TestModel = type(
        "TestModel",
        (module.DomainBaseModel,),
        {"__annotations__": {"name": str, "value": int}, "value": 42},
    )

    model = TestModel(name="test")
    assert model.name == "test"
    assert model.value == 42

    # Test serialization
    data = model.model_dump_json_safe()
    assert data["name"] == "test"
    assert data["value"] == 42

    # Test Value Object
    TestValueObject = type(
        "TestValueObject",
        (module.DomainValueObject,),
        {
            "__annotations__": {"value": int},
        },
    )

    vo1 = TestValueObject(value=42)
    vo2 = TestValueObject(value=42)
    assert vo1 == vo2
    assert hash(vo1) == hash(vo2)

    # Test Entity
    TestEntity = type(
        "TestEntity",
        (module.DomainEntity,),
        {
            "__annotations__": {"name": str},
        },
    )

    entity1 = TestEntity(name="test1")
    entity2 = TestEntity(name="test2")
    entity3 = TestEntity(id=entity1.id, name="different")

    assert entity1 != entity2  # Different IDs
    assert entity1 == entity3  # Same ID

    # Test Command/Query
    TestCommand = type(
        "TestCommand",
        (module.DomainCommand,),
        {
            "__annotations__": {"action": str},
        },
    )

    TestQuery = type(
        "TestQuery",
        (module.DomainQuery,),
        {
            "__annotations__": {"filter": str},
        },
    )

    command = TestCommand(action="create")
    query = TestQuery(filter="active")

    assert hasattr(command, "command_id")
    assert hasattr(query, "query_id")

    # Test Specification Pattern
    TestSpecification = type(
        "TestSpecification",
        (module.DomainSpecification,),
        {
            "specification_name": "test",
            "is_satisfied_by": lambda self, candidate: isinstance(candidate, int)
            and candidate > 0,
        },
    )

    spec = TestSpecification()
    assert spec.is_satisfied_by(5) is True
    assert spec.is_satisfied_by(-3) is False

    # Test composition
    spec2 = TestSpecification()
    and_spec = spec & spec2
    or_spec = spec | spec2
    not_spec = ~spec

    assert isinstance(and_spec, module.AndSpecification)
    assert isinstance(or_spec, module.OrSpecification)
    assert isinstance(not_spec, module.NotSpecification)

    # Test type aliases
    type_aliases = ["EntityId", "DomainEventData", "MetadataDict", "ConfigurationValue"]
    for alias in type_aliases:
        assert hasattr(module, alias), f"Missing type alias: {alias}"

    # Test __all__ export
    assert hasattr(module, "__all__")
    assert len(module.__all__) > 10

    return True


if __name__ == "__main__":
    try:
        success = test_final_level1_validation()
        if success:
            sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
