"""00 - FlextSingle-Import Pattern Demo.

This example demonstrates the NEW recommended single-import pattern
where ALL framework functionality is accessed through Flext

Key Benefits:
- Complete framework access via namespace pattern
- Python 3.13+ modern syntax with type parameters
- Zero additional imports needed for basic usage
- Cleaner example code with reduced import boilerplate

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextDecorators,
    FlextExceptions,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextRuntime,
    FlextUtilities,
)

# Relative import setup: Add src/ to path for running examples without PYTHONPATH
# (e.g., python -m examples.00_single_import_demo or direct execution)
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

os.environ.setdefault("FLEXT_DEBUG", "false")
os.environ.setdefault("FLEXT_TRACE", "false")


def demonstrate_single_import_pattern() -> None:
    """Demonstrate complete framework access via Flext."""
    print("\n" + "=" * 60)
    print("FLEXTCORE SINGLE-IMPORT PATTERN DEMONSTRATION")
    print("Complete framework access with zero additional imports")
    print("=" * 60)

    # ========================================
    # 1. RAILWAY PATTERN - Shorthand Methods
    # ========================================
    print("\n1. Railway Pattern - Shorthand Factory Methods:")

    # Create success result
    success_result = FlextResult[str].ok("Operation successful")
    print(f"   ✅ FlextResult.ok(): {success_result.value}")

    # Create failure result
    error_result = FlextResult[str].fail("Validation failed")
    print(f"   ❌ FlextResult.fail(): {error_result.error}")

    # Chain operations using railway pattern
    min_length = 3

    def validate_length(s: str) -> FlextResult[str]:
        if len(s) < min_length:
            return FlextResult[str].fail("Too short")
        return FlextResult[str].ok(s)

    def to_upper(s: str) -> str:
        return s.upper()

    chain_result = FlextResult[str].ok("hello").flat_map(validate_length).map(to_upper)
    print(f"   🚂 Railway chain: {chain_result.unwrap()}")

    # ========================================
    # 2. COMPONENTS VIA CONVENIENCE METHODS
    # ========================================
    print("\n2. Components via Convenience Methods:")

    # Logger - using FlextLogger factory method
    logger = FlextLogger.create_module_logger(__name__)
    logger.info("Single-import pattern demonstration")
    print(f"   ✅ create_module_logger: {type(logger).__name__}")

    # Container - simplified accessor
    container = FlextContainer.get_global()
    print(f"   ✅ get_container: {type(container).__name__}")

    # Runtime type guards
    runtime = FlextRuntime()
    print(f"   ✅ FlextRuntime: {type(runtime).__name__}")

    # ========================================
    # 3. CONSTANTS AND TYPES
    # ========================================
    print("\n3. Constants and Types:")

    # Access constants
    timeout = FlextConstants.Defaults.TIMEOUT
    print(f"   ⏱️  Timeout: {timeout}s")

    validation_error = FlextConstants.Errors.VALIDATION_ERROR
    print(f"   📋 Validation Error Code: {validation_error}")

    # Type aliases
    data: dict[str, object] = {"key": "value"}
    print(f"   📦 Type alias: {type(data).__name__}")

    # ========================================
    # 4. DECORATORS
    # ========================================
    print("\n4. Decorators:")

    @FlextDecorators.railway()
    def risky_operation(x: int) -> FlextResult[int]:
        """Decorated operation with automatic railway handling."""
        if x <= 0:
            return FlextResult[int].fail("Must be positive")  # Type inferred
        return FlextResult[int].ok(x * 2)

    result = risky_operation(5)
    print(f"   ✅ @railway decorator: {result.value}")

    # Demonstrate decorator availability
    print("   📦 Available decorators:")
    print(f"      - railway: {FlextDecorators.railway.__name__}")
    print(f"      - inject: {FlextDecorators.inject.__name__}")
    print(f"      - log_operation: {FlextDecorators.log_operation.__name__}")
    print(f"      - track_performance: {FlextDecorators.track_performance.__name__}")

    # ========================================
    # 5. VALIDATION UTILITIES
    # ========================================
    print("\n5. Validation Utilities:")

    # Cache management validation
    test_obj = FlextModels.Entity(unique_id="test-cache")
    cache_result = FlextUtilities.Cache.clear_object_cache(test_obj)
    print(f"   ✅ Cache clearing: {cache_result.is_success}")

    # ID generation utilities
    correlation_id = FlextUtilities.Generators.generate_correlation_id()
    print(f"   ✅ Correlation ID: {correlation_id[:8]}...")

    # Short ID generation
    short_id = FlextUtilities.Generators.generate_short_id()
    print(f"   ✅ Short ID: {short_id[:8]}...")

    # ========================================
    # 6. DOMAIN MODELS (DDD)
    # ========================================
    print("\n6. Domain Models (DDD Patterns):")

    # Entity - using base Entity with unique_id
    entity = FlextModels.Entity(unique_id="entity-123")
    print(f"   ✅ Entity: {entity.entity_id}")

    # Value Object
    class Email(FlextModels.Value):
        address: str

    email = Email(address="test@example.com")
    print(f"   ✅ Value Object: {email.address}")

    # ========================================
    # 7. EXCEPTIONS
    # ========================================
    print("\n7. Structured Exceptions:")

    try:
        error_msg = "Invalid input"
        raise FlextExceptions.ValidationError(
            error_msg,
            field="email",
            value="invalid",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )
    except FlextExceptions.ValidationError as e:
        print(f"   ❌ ValidationError caught: {e.message}")
        print(f"      Field: {e.field}, Code: {e.error_code}")

    # ========================================
    # 8. EXTENDING FLEXT PATTERNS VIA FLEEXTBASE
    # ========================================
    print("\n8. Extending Patterns with Flext")

    class DemoBase(FlextModels.ArbitraryTypesModel):
        """Demonstrate how domain libraries can extend Flext."""

    # Use the proper method for creating success results
    print("\n" + "=" * 60)
    print("✨ SINGLE-IMPORT PATTERN COMPLETE!")
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.BaseModel)   class MyBase(FlextModels.BaseModel):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.ArbitraryTypesModel)   class MyBase(FlextModels.ArbitraryTypesModel):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.ValueObject)   class MyBase(FlextModels.ValueObject):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Entity)   class MyBase(FlextModels.Entity):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.AggregateRoot)   class MyBase(FlextModels.AggregateRoot):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Cqrs.Command)   class MyBase(FlextModels.Cqrs.Command):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Cqrs.Query)   class MyBase(FlextModels.Cqrs.Query):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.DomainEvent)   class MyBase(FlextModels.DomainEvent):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Validation)   class MyBase(FlextModels.Validation):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Mixin)   class MyBase(FlextModels.Mixin):"
    )
    print(
        "🎯 Domain extension ready via: class MyBase(FlextModels.Mixin)   class MyBase(FlextModels.Mixin):"
    )
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    demonstrate_single_import_pattern()

    print("\n📚 Next Steps:")
    print("   - See 01_basic_result.py for complete FlextResult API")
    print("   - See 02_dependency_injection.py for DI patterns")
    print("   - See other examples for advanced patterns")


if __name__ == "__main__":
    main()
