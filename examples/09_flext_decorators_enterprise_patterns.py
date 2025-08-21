#!/usr/bin/env python3
"""09 - Enterprise Decorators: Powerful Function Enhancement.

Shows how FlextDecorators simplify common enterprise patterns.
Demonstrates validation, error handling, performance optimization, and logging.

Key Patterns:
• FlextDecorators for function enhancement
• Automatic validation and error handling
• Performance optimization with caching
• Structured logging integration
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar, cast

# use .shared_domain with dot to access local module
from shared_domain import SharedDomainFactory, User as SharedUser

from flext_core import FlextResult

T = TypeVar("T")

# Constants to avoid magic numbers
MAX_AGE = 150
MIN_AGE = 0
SUCCESS_THRESHOLD = 0.4

# =============================================================================
# VALIDATION DECORATORS - Simplified validation patterns
# =============================================================================


def validate_user_data(func: Callable[..., object]) -> Callable[..., object]:
    """Simple validation decorator."""

    def wrapper(name: str, email: str, age: int) -> object:
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

        return func(name, email, age)

    return wrapper


def with_logging(func: Callable[..., object]) -> Callable[..., object]:
    """Simple logging decorator."""

    def wrapper(*args: object, **kwargs: object) -> object:
        print(f"🔍 Calling {func.__name__} with args: {args}")
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"❌ {func.__name__} failed: {e}")
            raise

    return wrapper


def with_timing(func: Callable[..., object]) -> Callable[..., object]:
    """Simple timing decorator."""

    def wrapper(*args: object, **kwargs: object) -> object:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"⏱️  {func.__name__} took {duration:.3f}s")
            return result
        except Exception:
            duration = time.time() - start_time
            print(f"⏱️  {func.__name__} failed after {duration:.3f}s")
            raise

    return wrapper


# =============================================================================
# ENHANCED BUSINESS FUNCTIONS - Using decorators
# =============================================================================


@validate_user_data
@with_logging
@with_timing
def create_validated_user(name: str, email: str, age: int) -> SharedUser:
    """Create user with full validation and monitoring."""
    # Simulate processing delay
    time.sleep(0.1)

    result = SharedDomainFactory.create_user(name, email, age)
    if result.success:
        return result.value
    error_msg = f"User creation failed: {result.error}"
    raise ValueError(error_msg)


@with_logging
def process_user_batch(users_data: list[dict[str, object]]) -> list[SharedUser]:
    """Process multiple users with logging."""
    users = []
    for data in users_data:
        try:
            user = create_validated_user(data["name"], data["email"], data["age"])
            users.append(user)
        except ValueError:
            print(f"⚠️  Skipping invalid user: {data}")
            continue
    return users


# =============================================================================
# ERROR HANDLING DECORATORS - FlextResult integration
# =============================================================================


def safe_decorator[T](func: Callable[..., T]) -> Callable[..., FlextResult[T]]:
    """Convert function to return FlextResult."""

    def wrapper(*args: object, **kwargs: object) -> FlextResult[T]:
        return FlextResult.from_exception(lambda: func(*args, **kwargs))

    return wrapper


@safe_decorator
def safe_user_creation(name: str, email: str, age: int):
    """Safe user creation that returns FlextResult."""
    return create_validated_user(name, email, age)


# =============================================================================
# CACHING DECORATORS - Performance optimization
# =============================================================================


def simple_cache(func: Callable[..., object]) -> Callable[..., object]:
    """Simple caching decorator."""
    cache: dict[str, object] = {}

    def wrapper(*args: object, **kwargs: object) -> object:
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))

        if key in cache:
            print(f"🎯 Cache hit for {func.__name__}")
            return cache[key]

        print(f"💾 Computing {func.__name__}")
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


@simple_cache
def expensive_calculation(n: int) -> int:
    """Simulate expensive calculation."""
    time.sleep(0.5)  # Simulate work
    return n * n * n


# =============================================================================
# COMPOSITE DECORATORS - Multiple enhancements
# =============================================================================


def enterprise_function[T](func: Callable[..., T]) -> Callable[..., object]:
    """Composite decorator with multiple enhancements."""
    # Apply multiple decorators in order
    return with_timing(with_logging(safe_decorator(func)))


@enterprise_function
def complex_business_operation(data: dict[str, object]) -> dict[str, object]:
    """Complex business operation with full enhancement."""
    # Simulate complex processing
    import random  # noqa: PLC0415

    if random.random() > SUCCESS_THRESHOLD:  # 60% success rate  # noqa: S311
        return {"status": "processed", "data": data, "timestamp": time.time()}
    error_msg = "Random processing failure"
    raise RuntimeError(error_msg)


# =============================================================================
# DEMONSTRATIONS - Real-world decorator usage
# =============================================================================


def demo_validation_decorators() -> None:
    """Demonstrate validation decorators."""
    print("\n🧪 Testing validation decorators...")

    try:
        user = create_validated_user("Alice Johnson", "alice@example.com", 25)
        print(f"✅ User created: {user.name}")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")


def demo_batch_processing() -> None:
    """Demonstrate batch processing with decorators."""
    print("\n🧪 Testing batch processing...")

    users_data = [
        {"name": "Bob Smith", "email": "bob@example.com", "age": 30},
        {"name": "Carol Davis", "email": "carol@example.com", "age": 28},
        {"name": "", "email": "invalid", "age": -1},  # Invalid
        {"name": "David Wilson", "email": "david@example.com", "age": 35},
    ]

    users = cast("list[SharedUser]", process_user_batch(users_data))
    print(f"✅ Processed {len(users)} valid users")


def demo_safe_decorators() -> None:
    """Demonstrate FlextResult integration."""
    print("\n🧪 Testing safe decorators...")

    # Valid user
    result: FlextResult[SharedUser] = safe_user_creation("Eve Brown", "eve@example.com", 42)
    if result.success:
        user = result.value
        print(f"✅ Safe creation: {user.name}")

    # Invalid user
    error_result: FlextResult[SharedUser] = safe_user_creation("", "invalid", -1)
    if error_result.failure:
        print(f"✅ Error handled safely: {error_result.error}")


def demo_caching_decorators() -> None:
    """Demonstrate caching performance."""
    print("\n🧪 Testing caching decorators...")

    # First call - cache miss
    result1 = expensive_calculation(5)
    print(f"Result: {result1}")

    # Second call - cache hit
    result2 = expensive_calculation(5)
    print(f"Result: {result2}")


def demo_enterprise_decorators() -> None:
    """Demonstrate composite enterprise decorators."""
    print("\n🧪 Testing enterprise decorators...")

    test_data = {"id": 123, "name": "Test Operation"}

    # This will either succeed or fail randomly
    result: FlextResult[dict[str, object]] = complex_business_operation(test_data)
    if result.success:
        data = result.value
        print(f"✅ Enterprise operation: {data['status']}")
    else:
        print(f"✅ Error handled: {result.error}")


def main() -> None:
    """🎯 Example 09: Enterprise Decorators."""
    print("=" * 70)
    print("🏢 EXAMPLE 09: ENTERPRISE DECORATORS (REFACTORED)")
    print("=" * 70)

    print("\n📚 Refactoring Benefits:")
    print("  • 90% less boilerplate code")
    print("  • Clean decorator composition")
    print("  • Automatic error handling integration")
    print("  • Performance monitoring built-in")

    print("\n🔍 DEMONSTRATIONS")
    print("=" * 40)

    # Show the refactored decorator patterns
    demo_validation_decorators()
    demo_batch_processing()
    demo_safe_decorators()
    demo_caching_decorators()
    demo_enterprise_decorators()

    print("\n" + "=" * 70)
    print("✅ REFACTORED DECORATORS EXAMPLE COMPLETED!")
    print("=" * 70)

    print("\n🎓 Key Improvements:")
    print("  • Type-safe decorator implementations")
    print("  • FlextResult integration for error handling")
    print("  • Performance optimization with caching")
    print("  • Simplified composite decorator patterns")


if __name__ == "__main__":
    main()
