"""00 - FlextCore Single-Import Pattern Demo.

This example demonstrates the NEW recommended single-import pattern
where ALL framework functionality is accessed through FlextCore.

Key Benefits:
- Single import statement: from flext_core import FlextCore
- Complete framework access via namespace pattern
- Python 3.13+ modern syntax with type parameters
- Zero additional imports needed for basic usage
- Cleaner example code with reduced import boilerplate

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os

os.environ.setdefault("FLEXT_DEBUG", "false")
os.environ.setdefault("FLEXT_TRACE", "false")

from flext_core import FlextBase, FlextCore


def demonstrate_single_import_pattern() -> None:
    """Demonstrate complete framework access via FlextCore."""
    print("\n" + "=" * 60)
    print("FLEXTCORE SINGLE-IMPORT PATTERN DEMONSTRATION")
    print("Complete framework access with zero additional imports")
    print("=" * 60)

    # ========================================
    # 1. RAILWAY PATTERN - Shorthand Methods
    # ========================================
    print("\n1. Railway Pattern - Shorthand Factory Methods:")

    # Create success result
    success_result = FlextCore.Result[str].ok("Operation successful")
    print(f"   ✅ FlextCore.Result.ok(): {success_result.value}")

    # Create failure result
    error_result = FlextCore.Result[str].fail("Validation failed")
    print(f"   ❌ FlextCore.Result.fail(): {error_result.error}")

    # Chain operations using railway pattern
    min_length = 3

    def validate_length(s: str) -> FlextCore.Result[str]:
        if len(s) < min_length:
            return FlextCore.Result[str].fail("Too short")
        return FlextCore.Result[str].ok(s)

    def to_upper(s: str) -> str:
        return s.upper()

    chain_result = (
        FlextCore.Result[str].ok("hello").flat_map(validate_length).map(to_upper)
    )
    print(f"   🚂 Railway chain: {chain_result.unwrap()}")

    # ========================================
    # 2. COMPONENTS VIA NAMESPACE
    # ========================================
    print("\n2. Components via Namespace Access:")

    # Logger
    logger = FlextCore.Logger(__name__)
    logger.info("Single-import pattern demonstration")  # type: ignore[attr-defined]
    print(f"   ✅ FlextCore.Logger: {type(logger).__name__}")

    # Container
    container = FlextCore.Container.get_global()
    print(f"   ✅ FlextCore.Container: {type(container).__name__}")

    # Runtime type guards
    runtime = FlextCore.Runtime()
    print(f"   ✅ FlextCore.Runtime: {type(runtime).__name__}")

    # ========================================
    # 3. CONSTANTS AND TYPES
    # ========================================
    print("\n3. Constants and Types:")

    # Access constants
    timeout = FlextCore.Constants.Defaults.TIMEOUT
    print(f"   ⏱️  Timeout: {timeout}s")

    validation_error = FlextCore.Constants.Errors.VALIDATION_ERROR
    print(f"   📋 Validation Error Code: {validation_error}")

    # Type aliases
    data: FlextCore.Types.Dict = {"key": "value"}
    print(f"   📦 Type alias: {type(data).__name__}")

    # ========================================
    # 4. DECORATORS
    # ========================================
    print("\n4. Decorators:")

    @FlextCore.Decorators.railway()
    def risky_operation(x: int) -> FlextCore.Result[int]:
        """Decorated operation with automatic railway handling."""
        if x <= 0:
            return FlextCore.Result[int].fail("Must be positive")  # Type inferred
        return FlextCore.Result[int].ok(x * 2)

    result = risky_operation(5)
    print(f"   ✅ @railway decorator: {result.value}")

    # Demonstrate decorator availability
    print("   📦 Available decorators:")
    print(f"      - railway: {FlextCore.Decorators.railway.__name__}")
    print(f"      - inject: {FlextCore.Decorators.inject.__name__}")
    print(f"      - log_operation: {FlextCore.Decorators.log_operation.__name__}")
    print(
        f"      - track_performance: {FlextCore.Decorators.track_performance.__name__}"
    )

    # ========================================
    # 5. VALIDATION UTILITIES
    # ========================================
    print("\n5. Validation Utilities:")

    # Cache management validation
    test_obj = {"_cache": {"key": "value"}}
    cache_result = FlextCore.Utilities.Validation.clear_all_caches(test_obj)
    print(f"   ✅ Cache clearing: {cache_result.is_success}")

    # ID generation utilities
    correlation_id = FlextCore.Utilities.Generators.generate_correlation_id()
    print(f"   ✅ Correlation ID: {correlation_id[:8]}...")

    # Short ID generation
    short_id = FlextCore.Utilities.Generators.generate_short_id()
    print(f"   ✅ Short ID: {short_id[:8]}...")

    # ========================================
    # 6. DOMAIN MODELS (DDD)
    # ========================================
    print("\n6. Domain Models (DDD Patterns):")

    # Entity - using base Entity with id only
    entity = FlextCore.Models.Entity(id="entity-123")
    print(f"   ✅ Entity: {entity.id}")

    # Value Object
    class Email(FlextCore.Models.Value):
        address: str

    email = Email(address="test@example.com")
    print(f"   ✅ Value Object: {email.address}")

    # ========================================
    # 7. EXCEPTIONS
    # ========================================
    print("\n7. Structured Exceptions:")

    try:
        error_msg = "Invalid input"
        raise FlextCore.Exceptions.ValidationError(
            error_msg,
            field="email",
            value="invalid",
            error_code=FlextCore.Constants.Errors.VALIDATION_ERROR,
        )
    except FlextCore.Exceptions.ValidationError as e:
        print(f"   ❌ ValidationError caught: {e.message}")
        print(f"      Field: {e.field}, Code: {e.error_code}")

    # ========================================
    # 8. EXTENDING FLEXT PATTERNS VIA FLEEXTBASE
    # ========================================
    print("\n8. Extending Patterns with FlextBase:")

    class DemoBase(FlextBase):
        """Demonstrate how domain libraries can extend FlextBase."""

        class Constants(FlextBase.Constants):
            class Demo:
                FEATURE_FLAG_ENABLED: bool = True

    domain_base = DemoBase(auto_register=False)
    domain_base.bind_context(example="00_single_import")

    # Fix: Access constants properly through the extended class
    demo_feature = DemoBase.Constants.Demo.FEATURE_FLAG_ENABLED
    domain_base.info(
        "DemoBase initialised",
        feature=demo_feature,
    )
    print("   ✅ FlextBase: domain extensions ready")
    print(
        "      Feature flag default:",
        demo_feature,
    )

    totals = domain_base.run_operation("demo_total", sum, [1, 2, 3])
    print(f"   🧮 run_operation helper: {totals.unwrap()}")

    helper = domain_base.ok("domain ready")
    print(f"   ✨ ok helper: {helper.unwrap()}")

    # ========================================
    # 9. SERVICE INFRASTRUCTURE
    # ========================================
    print("\n9. Service Infrastructure - Direct Component Access:")

    # Access infrastructure components directly via FlextCore
    config = FlextCore.Config()
    container = FlextCore.Container.get_global()
    logger = FlextCore.Logger("demo-service")
    bus = FlextCore.Bus()

    print("   ✅ Infrastructure components accessed:")
    print(f"      - Config: {type(config).__name__}")
    print(f"      - Container: {type(container).__name__}")
    print(f"      - Logger: {type(logger).__name__}")
    print(f"      - Bus: {type(bus).__name__}")

    print("\n" + "=" * 60)
    print("✨ SINGLE-IMPORT PATTERN COMPLETE!")
    print("🎯 All framework functionality via: from flext_core import FlextCore")
    print("🎯 Domain extension ready via: class MyBase(FlextBase)")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    demonstrate_single_import_pattern()

    print("\n📚 Next Steps:")
    print("   - See 01_basic_result.py for complete FlextCore.Result API")
    print("   - See 02_dependency_injection.py for DI patterns")
    print("   - See other examples for advanced patterns")


if __name__ == "__main__":
    main()
