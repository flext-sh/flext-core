#!/usr/bin/env python3
"""Shared configurations and utilities for examples."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from flext_core import FlextConstants, FlextResult


def run_example_demonstrations(demos: list[tuple[str, Callable[[], None]]]) -> None:
    """Run a list of demonstration functions."""
    for _name, demo_func in demos:
        with contextlib.suppress(Exception):
            demo_func()


def safe_run_demonstration(demo_func: Callable[[], None]) -> FlextResult[None]:
    """Safely run a demonstration function."""
    try:
        demo_func()
        return FlextResult[None].ok(None)
    except Exception as e:
        return FlextResult[None].fail(f"Demo failed: {e}")


class ExampleConfig:
    """Configuration for examples."""

    DEBUG = True
    TIMEOUT = FlextConstants.Defaults.TIMEOUT
    MAX_RETRIES = FlextConstants.Defaults.MAX_RETRIES
    DEFAULT_HOST = FlextConstants.Platform.DEFAULT_HOST
    DEFAULT_PORT = FlextConstants.Network.LDAP_PORT


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
