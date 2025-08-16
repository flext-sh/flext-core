"""Legacy compatibility module for FLEXT Core.

This module provides backward compatibility facades for APIs that have been
modernized or refactored. Use this for maintaining compatibility during
ecosystem transitions without duplicating implementation code.

All legacy imports should delegate to modern implementations in:
- models.py (for Pydantic BaseModel patterns)
- result.py (for FlextResult patterns)
- container.py (for dependency injection)
- Other modern modules

DO NOT implement new functionality here - only compatibility facades.
"""

from __future__ import annotations

# Import all modern implementations at top
from flext_core.config import FlextSettings
from flext_core.models import (
    FlextEntity,
    FlextFactory,
    FlextModel,
    FlextValue,
)
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidators

# =============================================================================
# LEGACY MODEL PATTERNS - Delegate to modern models.py
# =============================================================================

# Legacy model aliases - maintain backward compatibility
FlextDomainEntity = FlextEntity
FlextDomainValueObject = FlextValue
FlextBaseModel = FlextModel
FlextImmutableModel = FlextValue
FlextMutableModel = FlextEntity

# Legacy factory aliases
FlextEntityFactory = FlextFactory
FlextModelFactory = FlextFactory

# =============================================================================
# LEGACY CONFIGURATION PATTERNS - Delegate to modern config.py
# =============================================================================

# Legacy config aliases
FlextBaseSettings = FlextSettings
FlextConfiguration = FlextSettings

# =============================================================================
# LEGACY VALIDATION PATTERNS - Delegate to modern validation.py
# =============================================================================

# Legacy validator aliases
FlextBaseValidators = FlextValidators
FlextValidationUtils = FlextValidators

# =============================================================================
# LEGACY UTILITY PATTERNS - Delegate to modern utilities.py
# =============================================================================

# Legacy utility aliases
FlextBaseUtilities = FlextUtilities
FlextHelpers = FlextUtilities

# =============================================================================
# LEGACY EXPORTS - Maintain backward compatibility
# =============================================================================

__all__: list[str] = [
    "FlextBaseModel",
    "FlextBaseSettings",
    "FlextBaseUtilities",
    "FlextBaseValidators",
    "FlextConfiguration",
    "FlextDomainEntity",
    "FlextDomainValueObject",
    "FlextEntityFactory",
    "FlextHelpers",
    "FlextImmutableModel",
    "FlextModelFactory",
    "FlextMutableModel",
    "FlextValidationUtils",
]
