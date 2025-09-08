#!/usr/bin/env python3
"""Demonstration of the unified FlextTests testing utilities.

This example shows how to use the new unified FlextTests class that provides
a single entry point for all testing utilities in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio

from flext_tests import FlextTests


def demo_builders() -> None:
    """Demonstrate test builders."""
    print("\n=== Test Builders ===")

    # Create success and failure results
    success = FlextTests.Builders.success_result("test data")
    failure = FlextTests.Builders.failure_result("validation error")

    print(f"Success result: {success.success} - {success.value}")
    print(f"Failure result: {failure.success} - {failure.error}")

    # Build a test container
    container = FlextTests.Builders.test_container()
    services = container.list_services()
    if services.success:
        print(f"Container services: {services.value}")
    else:
        print("Container has no services yet")


def demo_domains() -> None:
    """Demonstrate domain factories."""
    print("\n=== Domain Factories ===")

    # Create user data
    user = FlextTests.Domains.UserData.create(name="John Doe", age=30)
    print(f"User: {user['name']}, Age: {user['age']}")

    # Create service data
    service = FlextTests.Domains.ServiceData.create()
    print(f"Service: {service['name']} v{service['version']}")

    # Create realistic data
    order = FlextTests.Domains.Realistic.order_data()
    print(f"Order ID: {order['order_id']}, Total: ${order['total']}")


def demo_factories() -> None:
    """Demonstrate test factories."""
    print("\n=== Test Factories ===")

    # Create test results
    success = FlextTests.Factories.success_result("data")
    print(f"Factory result: {success.success}")

    # Get edge cases
    unicode_strings = FlextTests.Factories.EdgeCases.unicode_strings()
    print(f"Unicode test strings: {unicode_strings[:3]}")

    # Create test hierarchy
    hierarchy = FlextTests.Factories.test_hierarchy()
    print(f"Test hierarchy keys: {list(hierarchy.keys())}")


async def demo_async() -> None:
    """Demonstrate async utilities."""
    print("\n=== Async Utilities ===")

    # Run with timeout
    async def slow_operation() -> str:
        await asyncio.sleep(0.1)
        return "completed"

    result = await FlextTests.Async.Utils.run_with_timeout(
        slow_operation(), timeout_seconds=1.0
    )
    print(f"Async operation result: {result}")

    # Run concurrent tasks
    async def task(n: int) -> str:
        await asyncio.sleep(0.01)
        return f"task_{n}"

    results = await FlextTests.Async.Utils.run_concurrently(task(1), task(2), task(3))
    print(f"Concurrent results: {results}")


def demo_utilities() -> None:
    """Demonstrate test utilities."""
    print("\n=== Test Utilities ===")

    # Create LDAP config
    ldap_config = FlextTests.Utilities.create_ldap_config()
    print(f"LDAP Config: {ldap_config['host']}:{ldap_config['port']}")

    # Create API response
    api_response = FlextTests.Utilities.create_api_response()
    print(
        f"API Response: {api_response['success']}, Message: {api_response['message']}"
    )


def main() -> None:
    """Run all demonstrations."""
    print("=" * 60)
    print("FLEXT Tests Unified Interface Demonstration")
    print("=" * 60)

    demo_builders()
    demo_domains()
    demo_factories()
    demo_utilities()

    # Run async demo

    asyncio.run(demo_async())

    print("\n" + "=" * 60)
    print("✅ All FlextTests modules demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
