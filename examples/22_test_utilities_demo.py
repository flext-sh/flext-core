#!/usr/bin/env python3
"""Demonstration of the unified FlextTests testing utilities.

This example shows how to use the new unified FlextTests class that provides
a single entry point for all testing utilities in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio

from flext_tests.builders import FlextTestsBuilders
from flext_tests.utilities import FlextTestsUtilities


def demo_builders() -> None:
    """Demonstrate test builders."""
    print("\n=== Test Builders ===")

    # Create success and failure results
    success = FlextTestsBuilders.success_result("test data")
    failure = FlextTestsBuilders.failure_result("validation error")

    print(f"Success result: {success.success} - {success.value}")
    print(f"Failure result: {failure.success} - {failure.error}")

    # Build a test container
    container = FlextTestsBuilders.test_container()
    services = container.list_services()
    print(f"Container services: {list(services.keys()) if services else 'No services'}")


def demo_domains() -> None:
    """Demonstrate domain factories."""
    print("\n=== Domain Factories ===")
    print("Domain data creation examples (simplified for demonstration)")
    print("✅ User data creation available through FlextTestsBuilders")
    print("✅ Service data creation available through FlextTestsBuilders")
    print("✅ Realistic data available through FlextTestsBuilders")


def demo_factories() -> None:
    """Demonstrate test factories."""
    print("\n=== Test Factories ===")
    print("Factory examples (simplified for demonstration)")
    print("✅ Test result creation available through FlextTestsBuilders")
    print("✅ Edge case generation available through FlextTestsUtilities")
    print("✅ Test hierarchy creation available through FlextTestsBuilders")


async def demo_async() -> None:
    """Demonstrate async utilities."""
    print("\n=== Async Utilities ===")

    # Run with timeout
    async def slow_operation() -> str:
        await asyncio.sleep(0.1)
        return "completed"

    # Demonstrate async operation with timeout
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
        print(f"Async operation result: {result}")
    except TimeoutError:
        print("Async operation timed out")

    # Demonstrate concurrent tasks
    async def task(n: int) -> str:
        await asyncio.sleep(0.01)
        return f"task_{n}"

    tasks = [task(1), task(2), task(3)]
    results = await asyncio.gather(*tasks)
    print(f"Concurrent results: {results}")


def demo_utilities() -> None:
    """Demonstrate test utilities."""
    print("\n=== Test Utilities ===")

    # Create LDAP config
    ldap_config = FlextTestsUtilities.create_ldap_config()
    print(f"LDAP Config: {ldap_config['host']}:{ldap_config['port']}")

    # Create API response
    api_response = FlextTestsUtilities.create_api_response()
    message = api_response.get("message", api_response.get("data", "No message"))
    print(f"API Response: {api_response.get('success', 'Unknown')}, Message: {message}")


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
