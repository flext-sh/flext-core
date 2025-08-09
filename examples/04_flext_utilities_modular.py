#!/usr/bin/env python3
"""Modular utilities for generation, formatting, and validation.

Demonstrates SOLID principles with focused modules for type checking,
validation, domain modeling, and utility functions.
"""

from __future__ import annotations

# Import the modular demonstration runner
from utilities.demonstration_runner import run_all_demonstrations

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
    run_all_demonstrations()


if __name__ == "__main__":
    main()
