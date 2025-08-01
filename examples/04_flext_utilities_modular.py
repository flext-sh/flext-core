#!/usr/bin/env python3
"""FLEXT Utilities - Modular Generation, Formatting, and Validation Example.

This is the modernized, modular version of the FLEXT utilities demonstration,
following SOLID principles with clear separation of concerns.

ARCHITECTURAL IMPROVEMENTS:
- Broke down 1284-line monolith into 5 focused modules 
- Implemented Single Responsibility Principle (SRP)
- Reduced cognitive complexity through helper classes
- Eliminated code duplication using shared utilities
- Enhanced maintainability and testability

Module Structure:
- utilities/formatting_helpers.py: Constants and basic utilities
- utilities/validation_utilities.py: Type checking and validation
- utilities/domain_models.py: Enhanced domain models with mixins
- utilities/complexity_helpers.py: SRP helper classes 
- utilities/demonstration_runner.py: Main demonstration orchestrator

This demonstrates enterprise-grade architecture patterns while maintaining
all original functionality in a much more maintainable structure.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
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