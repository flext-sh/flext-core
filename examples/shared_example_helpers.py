"""Shared utilities for example demonstrations.

Common helper functions and utilities to eliminate duplication
across example files following DRY principles.
"""

from __future__ import annotations

from collections.abc import Callable


def run_example_demonstration(
    title: str,
    examples: list[tuple[str, Callable[[], None]]],
) -> None:
    """Run a standardized example demonstration with consistent formatting.

    Args:
      title: The main title for the demonstration
      examples: List of (title, function) tuples for each example

    """
    for _example_title, example_func in examples:
        example_func()

    title.split(" - ", maxsplit=1)[0] if " - " in title else title
