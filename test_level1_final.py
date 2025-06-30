#!/usr/bin/env python3
"""FINAL: Level 1 pydantic_base.py validation without dynamic classes."""

import importlib.util
import sys
from pathlib import Path


def test_level1_final() -> None:
    """Final Level 1 validation."""
    # Import module directly
    spec = importlib.util.spec_from_file_location(
        "pydantic_base",
        Path(__file__).parent / "src" / "flx_core" / "domain" / "pydantic_base.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Verify all classes are importable
    classes = [
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

    for cls_name in classes:
        cls = getattr(module, cls_name)
        assert cls is not None

    # Test type aliases
    aliases = ["EntityId", "DomainEventData", "MetadataDict", "ConfigurationValue"]
    for alias in aliases:
        type_alias = getattr(module, alias)
        assert type_alias is not None

    # Test __all__ exports
    assert hasattr(module, "__all__")
    assert len(module.__all__) >= 11

    # Basic class instantiation tests using actual class definitions
    try:
        # Create a simple model class
        exec(
            """
class SimpleModel(module.DomainBaseModel):
    name: str = "test"

model = SimpleModel()
assert model.name == "test"
""",
            {"module": module},
        )
    except Exception:
        raise

    try:
        # Test value object
        exec(
            """
class SimpleValueObject(module.DomainValueObject):
    value: int = 42

vo1 = SimpleValueObject()
vo2 = SimpleValueObject()
assert vo1 == vo2
assert hash(vo1) == hash(vo2)
""",
            {"module": module},
        )
    except Exception:
        raise

    try:
        # Test entity
        exec(
            """
class SimpleEntity(module.DomainEntity):
    name: str = "test"

entity = SimpleEntity()
assert hasattr(entity, 'id')
assert hasattr(entity, 'created_at')
assert entity.version == 1
""",
            {"module": module},
        )
    except Exception:
        raise

    try:
        # Test command
        exec(
            """
class SimpleCommand(module.DomainCommand):
    action: str = "test"

command = SimpleCommand()
assert hasattr(command, 'command_id')
assert hasattr(command, 'issued_at')
assert command.action == "test"
""",
            {"module": module},
        )
    except Exception:
        raise

    try:
        # Test query
        exec(
            """
class SimpleQuery(module.DomainQuery):
    filter_name: str = "test"

query = SimpleQuery()
assert hasattr(query, 'query_id')
assert hasattr(query, 'issued_at')
assert query.filter_name == "test"
""",
            {"module": module},
        )
    except Exception:
        raise

    # Test specification pattern
    try:
        exec(
            """
class PositiveSpec(module.DomainSpecification):
    specification_name: str = "positive"

    def is_satisfied_by(self, candidate: object) -> bool:
        return isinstance(candidate, int) and candidate > 0

spec = PositiveSpec()
assert spec.is_satisfied_by(5) is True
assert spec.is_satisfied_by(-3) is False
""",
            {"module": module},
        )
    except Exception:
        raise


if __name__ == "__main__":
    try:
        test_level1_final()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
