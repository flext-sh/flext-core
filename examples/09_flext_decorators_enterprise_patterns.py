#!/usr/bin/env python3
"""09 - Enterprise Decorators: Modern FlextDecorators API Showcase.

Demonstrates the refactored FlextDecorators API with complete type safety.
Shows modern decorator composition patterns and FlextResult integration.

Key Patterns:
• Modern FlextDecorators API usage
• Type-safe decorator composition
• FlextResult integration patterns
• Enterprise-grade function enhancement
"""

from __future__ import annotations

import time

from shared_domain import SharedDomainFactory

from flext_core import FlextDecorators

# Constants to avoid magic numbers
MAX_AGE = 150
MIN_AGE = 0
SUCCESS_THRESHOLD = 0.4

# =============================================================================
# MODERN FLEXT DECORATORS SHOWCASE - New API
# =============================================================================


def demonstrate_cache_decorator() -> None:
    """Demonstrate modern cache decorator usage."""
    print("\n📝 Modern Cache Decorator:")

    # Create cache decorator using modern API
    cache_decorator = FlextDecorators.Performance.create_cache_decorator(max_size=100)

    def expensive_calculation(*args: object, **_kwargs: object) -> object:
        """Expensive calculation compatible with FlextCallable."""
        if args and isinstance(args[0], (int, float)):
            n = float(args[0])
            time.sleep(0.1)  # Simulate expensive work
            return n * n * n
        return 0

    # Apply cache decorator
    cached_calculation = cache_decorator(expensive_calculation)

    # Test cache functionality
    print("First call (cache miss):")
    start_time = time.time()
    result1 = cached_calculation(5)
    duration1 = time.time() - start_time
    print(f"Result: {result1}, Duration: {duration1:.3f}s")

    print("Second call (cache hit):")
    start_time = time.time()
    result2 = cached_calculation(5)
    duration2 = time.time() - start_time
    print(f"Result: {result2}, Duration: {duration2:.3f}s")

    print(f"Cache speedup: {duration1 / duration2:.1f}x faster")


def demonstrate_complete_decorator() -> None:
    """Demonstrate complete decorator composition."""
    print("\n📝 Complete Decorator Composition:")

    # Create complete decorator with multiple features
    complete_decorator = FlextDecorators.complete_decorator(
        cache_size=64,
        with_timing=True,
        with_logging=False,  # Simplified for clarity
    )

    def business_operation(*args: object, **_kwargs: object) -> object:
        """Business operation compatible with FlextCallable."""
        data = args[0] if args else {}
        if isinstance(data, dict):
            # Simulate complex processing
            time.sleep(0.05)  # Reduced for demo
            import random  # noqa: PLC0415

            if random.random() > SUCCESS_THRESHOLD:  # noqa: S311
                return {"status": "processed", "data": data, "timestamp": time.time()}
            msg = "Random processing failure"
            raise RuntimeError(msg)
        return {"status": "invalid_input"}

    # Apply complete decorator
    enhanced_operation = complete_decorator(business_operation)

    # Test the enhanced operation
    test_data = {"id": 123, "name": "Test Operation"}

    try:
        result = enhanced_operation(test_data)
        print(f"✅ Enhanced operation result: {result}")
    except Exception as e:
        print(f"❌ Enhanced operation failed: {e}")


def demonstrate_safe_result_decorator() -> None:
    """Demonstrate safe result decorator."""
    print("\n📝 Safe Result Decorator:")

    def risky_operation(*args: object, **_kwargs: object) -> object:
        """Risky operation that might fail."""
        if args and str(args[0]) == "fail":
            msg = "Intentional failure"
            raise ValueError(msg)
        return f"Success with {len(args)} arguments"

    # Apply safe result decorator
    safe_operation = FlextDecorators.safe_result(risky_operation)

    # Test success case
    result_success = safe_operation("success")
    print(f"Success result: {result_success}")

    # Test failure case
    result_failure = safe_operation("fail")
    print(f"Failure result: {result_failure}")


def demonstrate_user_creation_with_modern_decorators() -> None:
    """Demonstrate user creation with modern decorators."""
    print("\n📝 User Creation with Modern Decorators:")

    # Create a comprehensive decorator for user operations
    user_decorator = FlextDecorators.complete_decorator(
        cache_size=32, with_timing=True, with_logging=False
    )

    def create_user_generic(*args: object, **_kwargs: object) -> object:
        """Generic user creator compatible with FlextCallable."""
        if len(args) >= 3:
            try:
                name = str(args[0]) if args[0] is not None else ""
                email = str(args[1]) if args[1] is not None else ""
                age_val = args[2]

                # Safe age conversion
                if isinstance(age_val, (int, float)) or (
                    isinstance(age_val, str) and age_val.isdigit()
                ):
                    age = int(age_val)
                else:
                    msg = f"Invalid age: {age_val}"
                    raise ValueError(msg)

                # Basic validation
                if not name or not name.strip():
                    msg = "Name required"
                    raise ValueError(msg)
                if "@" not in email:
                    msg = "Valid email required"
                    raise ValueError(msg)
                if age < MIN_AGE or age > MAX_AGE:
                    msg = "Valid age required"
                    raise ValueError(msg)

                # Create user using SharedDomainFactory
                result = SharedDomainFactory.create_user(name, email, age)
                if result.success:
                    return result.value
                msg = f"User creation failed: {result.error}"
                raise ValueError(msg)

            except (ValueError, TypeError) as e:
                msg = f"Type conversion failed: {e}"
                raise ValueError(msg) from e
        msg = "Insufficient arguments"
        raise ValueError(msg)

    # Apply decorator
    enhanced_user_creator = user_decorator(create_user_generic)

    # Test user creation
    try:
        user_result = enhanced_user_creator(
            "Alice Modern", "alice.modern@example.com", 25
        )
        if hasattr(user_result, "name"):
            print(f"✅ User created: {user_result.name}")
        else:
            print(f"✅ User created: {user_result}")
    except Exception as e:
        print(f"❌ User creation failed: {e}")

    # Test validation failure
    try:
        invalid_result = enhanced_user_creator("", "invalid", -1)
        print(f"❌ Should have failed: {invalid_result}")
    except Exception as e:
        print(f"✅ Validation correctly failed: {e}")


def demonstrate_decorator_categories() -> None:
    """Demonstrate different decorator categories."""
    print("\n📝 Decorator Categories:")

    # Performance decorators
    print("🚀 Performance Category:")
    cache_dec = FlextDecorators.Performance.create_cache_decorator(max_size=50)
    print(f"  Cache decorator created: {cache_dec}")

    # Error handling decorators
    print("🛡️ Error Handling Category:")
    safe_dec = FlextDecorators.ErrorHandling.get_safe_decorator()
    print(f"  Safe decorator created: {safe_dec}")

    # Complete decorator composition
    print("🎯 Complete Decorator:")
    complete_dec = FlextDecorators.complete_decorator(
        cache_size=32, with_timing=True, with_logging=True
    )
    print(f"  Complete decorator created: {complete_dec}")


# =============================================================================
# DEMONSTRATIONS - Modern decorator usage
# =============================================================================


def main() -> None:
    """🎯 Example 09: Modern Enterprise Decorators."""
    print("=" * 70)
    print("🏢 EXAMPLE 09: MODERN ENTERPRISE DECORATORS")
    print("=" * 70)

    print("\n📚 Modern FlextDecorators Benefits:")
    print("  • Type-safe decorator composition")
    print("  • Automatic FlextResult integration")
    print("  • Performance optimization built-in")
    print("  • Clean categorical organization")

    print("\n🔍 MODERN API DEMONSTRATIONS")
    print("=" * 40)

    # Show the modern decorator patterns
    demonstrate_cache_decorator()
    demonstrate_complete_decorator()
    demonstrate_safe_result_decorator()
    demonstrate_user_creation_with_modern_decorators()
    demonstrate_decorator_categories()

    print("\n" + "=" * 70)
    print("✅ MODERN DECORATORS EXAMPLE COMPLETED!")
    print("=" * 70)

    print("\n🎓 Key Modern Improvements:")
    print("  • Categorical organization (Performance, ErrorHandling, etc.)")
    print("  • Type-safe FlextCallable protocol")
    print("  • Automatic error handling with FlextResult")
    print("  • Zero boilerplate decorator composition")
    print("  • Built-in performance optimization")


if __name__ == "__main__":
    main()
