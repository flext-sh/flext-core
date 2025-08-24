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

import contextlib
import random
import time

from shared_domain import SharedDomainFactory

from flext_core import FlextDecorators

# Constants to avoid magic numbers
MAX_AGE = 150
MIN_AGE = 0
SUCCESS_THRESHOLD = 0.4
MIN_USER_CREATION_ARGS = 3  # name, email, age

# =============================================================================
# MODERN FLEXT DECORATORS SHOWCASE - New API
# =============================================================================


def demonstrate_cache_decorator() -> None:
    """Demonstrate modern cache decorator usage."""
    # Create cache decorator using modern API
    performance_decorators = FlextDecorators.Performance
    cache_decorator = performance_decorators.create_cache_decorator(max_size=100)  # type: ignore[attr-defined]

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
    start_time = time.time()
    cached_calculation(5)
    time.time() - start_time

    start_time = time.time()
    cached_calculation(5)
    time.time() - start_time


def demonstrate_complete_decorator() -> None:
    """Demonstrate complete decorator composition."""
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
            if random.random() > SUCCESS_THRESHOLD:  # noqa: S311
                return {"status": "processed", "data": data, "timestamp": time.time()}
            msg = "Random processing failure"
            raise RuntimeError(msg)
        return {"status": "invalid_input"}

    # Apply complete decorator
    enhanced_operation = complete_decorator(business_operation)

    # Test the enhanced operation
    test_data = {"id": 123, "name": "Test Operation"}

    with contextlib.suppress(Exception):
        enhanced_operation(test_data)


def demonstrate_safe_result_decorator() -> None:
    """Demonstrate safe result decorator."""

    def risky_operation(*args: object, **_kwargs: object) -> object:
        """Risky operation that might fail."""
        if args and str(args[0]) == "fail":
            msg = "Intentional failure"
            raise ValueError(msg)
        return f"Success with {len(args)} arguments"

    # Apply safe result decorator
    safe_operation = FlextDecorators.safe_result(risky_operation)

    # Test success case
    safe_operation("success")

    # Test failure case
    safe_operation("fail")


def demonstrate_user_creation_with_modern_decorators() -> None:
    """Demonstrate user creation with modern decorators."""
    # Create a comprehensive decorator for user operations
    user_decorator = FlextDecorators.complete_decorator(
        cache_size=32, with_timing=True, with_logging=False
    )

    def create_user_generic(*args: object, **_kwargs: object) -> object:
        """Generic user creator compatible with FlextCallable."""
        if len(args) >= MIN_USER_CREATION_ARGS:
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
                    raise ValueError(msg)  # noqa: TRY301

                # Basic validation
                if not name or not name.strip():
                    msg = "Name required"
                    raise ValueError(msg)  # noqa: TRY301
                if "@" not in email:
                    msg = "Valid email required"
                    raise ValueError(msg)  # noqa: TRY301
                if age < MIN_AGE or age > MAX_AGE:
                    msg = "Valid age required"
                    raise ValueError(msg)  # noqa: TRY301

                # Create user using SharedDomainFactory
                result = SharedDomainFactory.create_user(name, email, age)
                if result.success:
                    return result.value
                msg = f"User creation failed: {result.error}"
                raise ValueError(msg)  # noqa: TRY301

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
            pass
    except Exception:  # noqa: S110
        pass

    # Test validation failure
    with contextlib.suppress(Exception):
        enhanced_user_creator("", "invalid", -1)


def demonstrate_decorator_categories() -> None:
    """Demonstrate different decorator categories."""
    # Performance decorators
    performance_decorators = FlextDecorators.Performance
    performance_decorators.create_cache_decorator(max_size=50)  # type: ignore[attr-defined]

    # Error handling decorators
    error_handling_decorators = FlextDecorators.ErrorHandling
    error_handling_decorators.create_safe_decorator()  # type: ignore[attr-defined]

    # Complete decorator composition
    FlextDecorators.complete_decorator(
        cache_size=32, with_timing=True, with_logging=True
    )


# =============================================================================
# DEMONSTRATIONS - Modern decorator usage
# =============================================================================


def main() -> None:
    """🎯 Example 09: Modern Enterprise Decorators."""
    # Show the modern decorator patterns
    demonstrate_cache_decorator()
    demonstrate_complete_decorator()
    demonstrate_safe_result_decorator()
    demonstrate_user_creation_with_modern_decorators()
    demonstrate_decorator_categories()


if __name__ == "__main__":
    main()
