#!/usr/bin/env python3
"""Shared configurations and utilities for examples."""

from __future__ import annotations

from typing import Any

from flext_core import FlextResult


def run_example_demonstrations(demos: list[tuple[str, callable]]) -> None:
    """Run a list of demonstration functions."""
    print("\n🔍 DEMONSTRATIONS")
    print("=" * 40)

    for name, demo_func in demos:
        try:
            print(f"\n🧪 {name}...")
            demo_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")


def safe_run_demonstration(demo_func: callable) -> FlextResult[None]:
    """Safely run a demonstration function."""
    try:
        demo_func()
        return FlextResult[None].ok(None)
    except Exception as e:
        return FlextResult[None].fail(f"Demo failed: {e}")


class ExampleConfig:
    """Configuration for examples."""

    DEBUG = True
    TIMEOUT = 30
    MAX_RETRIES = 3
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 389


def get_test_data() -> dict[str, Any]:
    """Get test data for examples."""
    return {
        "users": [
            {"name": "Alice", "email": "alice@example.com", "age": 25},
            {"name": "Bob", "email": "bob@example.com", "age": 30},
            {"name": "Carol", "email": "carol@example.com", "age": 28},
        ],
        "config": {
            "host": ExampleConfig.DEFAULT_HOST,
            "port": ExampleConfig.DEFAULT_PORT,
            "timeout": ExampleConfig.TIMEOUT,
        },
    }


def format_example_output(title: str, content: str) -> str:
    """Format example output consistently."""
    separator = "=" * 70
    return f"{separator}\n{title}\n{separator}\n{content}\n{separator}"


__all__ = [
    "ExampleConfig",
    "format_example_output",
    "get_test_data",
    "run_example_demonstrations",
    "safe_run_demonstration",
]
