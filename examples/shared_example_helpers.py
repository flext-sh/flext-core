"""Shared Example Helpers for FLEXT Demonstrations.

Common utilities to eliminate duplication across example files, following
DRY principles and SOLID patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Callable


def run_example_demonstration(
    title: str,
    examples: list[tuple[str, Callable[[], None]]],
) -> None:
    """Run a standardized example demonstration with consistent formatting.

    Args:
        title: The main title for the demonstration
        examples: List of (title, function) tuples for each example
    """
    print("=" * 80)
    print(f"🏢 {title}")
    print("=" * 80)

    for i, (example_title, example_func) in enumerate(examples, 1):
        print(f"\n{'=' * 60}")
        print(f"📋 EXAMPLE {i}: {example_title}")
        print("=" * 60)
        example_func()

    demonstration_name = title.split(" - ")[0] if " - " in title else title
    print(f"\n{'=' * 80}")
    print(f"🎉 {demonstration_name} DEMONSTRATION COMPLETED")
    print("=" * 80)
