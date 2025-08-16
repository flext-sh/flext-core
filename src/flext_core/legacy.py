"""Legacy compatibility module for FLEXT Core.

This module provides backward compatibility facades for APIs that have been
modernized or refactored. Use this for maintaining compatibility during
ecosystem transitions without duplicating implementation code.

All legacy imports should delegate to modern implementations in:
- exceptions.py (for Pydantic-style error patterns)
- models.py (for Pydantic BaseModel patterns)
- result.py (for FlextResult patterns)
- container.py (for dependency injection)
- Other modern modules

DO NOT implement new functionality here - only compatibility facades.
Following user feedback: "use o legacy.py para o que foi removido de api como fachada"
"""

from __future__ import annotations

import warnings

from flext_core.config import FlextSettings
from flext_core.constants import FlextConstants
from flext_core.mixins import FlextCacheableMixin
from flext_core.models import (
    FlextEntity,
    FlextFactory,
    FlextModel,
    FlextValue,
)
from flext_core.observability import FlextConsoleLogger
from flext_core.schema_processing import FlextBaseConfigManager
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidators, flext_validate_non_empty_string


def _deprecation_warning(old_name: str, new_name: str) -> None:
    """Issue deprecation warning for legacy APIs."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# =============================================================================
# LEGACY MODEL PATTERNS - Delegate to modern models.py
# =============================================================================

# Legacy model aliases - maintain backward compatibility
FlextDomainEntity = FlextEntity
FlextDomainValueObject = FlextValue
FlextBaseModel = FlextModel
FlextImmutableModel = FlextValue
FlextMutableModel = FlextEntity

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
# LEGACY COMPATIBILITY EXPORTS - Delegate to modern implementations
# =============================================================================

# Legacy compatibility for test imports
ConsoleLogger = FlextConsoleLogger
BaseConfigManager = FlextBaseConfigManager
LegacyCompatibleCacheableMixin = FlextCacheableMixin
FlextValueObjectFactory = FlextFactory


# Legacy config defaults
class _BaseConfigDefaults:
    """Legacy compatibility for test config defaults."""

    TIMEOUT = FlextConstants.Defaults.TIMEOUT
    RETRIES = FlextConstants.Defaults.MAX_RETRIES
    PAGE_SIZE = FlextConstants.Defaults.PAGE_SIZE


# Legacy config validation
class _BaseConfigValidation:
    """Legacy compatibility for test config validation."""

    @staticmethod
    def validate_config(config: dict[str, object]) -> bool:
        """Legacy validation function."""
        return bool(config)


# Legacy performance config
class _PerformanceConfig:
    """Legacy compatibility for test performance config."""

    TIMEOUT = FlextConstants.Performance.SLOW_REQUEST_THRESHOLD
    BATCH_SIZE = FlextConstants.Performance.DEFAULT_BATCH_SIZE
    MAX_CONNECTIONS = FlextConstants.Performance.MAX_CONNECTIONS


# Legacy validation function alias
# Note: flext_validate_non_empty_string is already imported above

# =============================================================================
# LEGACY EXPORTS - Maintain backward compatibility
# =============================================================================

__all__: list[str] = [
    "BaseConfigManager",
    "ConsoleLogger",
    "FlextBaseModel",
    "FlextBaseSettings",
    "FlextBaseUtilities",
    "FlextBaseValidators",
    "FlextConfiguration",
    "FlextDomainEntity",
    "FlextDomainValueObject",
    "FlextFactory",
    "FlextHelpers",
    "FlextImmutableModel",
    "FlextMutableModel",
    "FlextValidationUtils",
    "FlextValueObjectFactory",
    "LegacyCompatibleCacheableMixin",
    "_BaseConfigDefaults",
    "_BaseConfigValidation",
    "_PerformanceConfig",
    "flext_validate_non_empty_string",
]
