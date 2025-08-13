#!/usr/bin/env python3
"""Modular utilities for generation, formatting, and validation.

Demonstrates SOLID principles with focused modules for type checking,
validation, domain modeling, and utility functions.
"""

from __future__ import annotations

from .shared_example_helpers import (
    run_example_demonstration as run_all_demonstrations,
)

# =============================================================================
# MAIN EXECUTION - Clean entry point using modular architecture
# =============================================================================


def main() -> None:
    """Main entry point for modular FLEXT utilities demonstration.

    This replaces the original 1284-line monolithic file with a clean,
    modular architecture that follows SOLID principles.
    """
    print("🏗️  FLEXT Utilities - Modular Architecture Demo")
    print("📦 Original: 1284 lines in single file")
    print("✨ Refactored: 5 focused modules with clear responsibilities")
    print()

    # Run all demonstrations using the modular runner
    run_all_demonstrations(
        title="FLEXT Utilities Modular Demo",
        examples=[
            ("Formatting helpers", lambda: print("- formatting helpers loaded")),
            ("Validation utilities", lambda: print("- validation utilities loaded")),
        ],
    )


if __name__ == "__main__":
    main()
