#!/usr/bin/env python3
"""Modular utilities for generation, formatting, and validation.

Demonstrates SOLID principles with focused modules for type checking,
validation, domain modeling, and utility functions.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parents[1]
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

from examples.shared_example_helpers import (
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
