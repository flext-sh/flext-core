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

import sys
import warnings

from packaging import version

from flext_core.__version__ import __version__
from flext_core.config import FlextSettings
from flext_core.constants import FlextConstants
from flext_core.mixins import (
    FlextCacheableMixin,
    FlextCommandMixin,
    FlextComparableMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
)
from flext_core.models import (
    FlextEntity,
    FlextFactory,
    FlextModel,
    FlextValue,
)
from flext_core.observability import (
    FlextConsoleLogger,
    FlextInMemoryMetrics,
    FlextMinimalObservability,
    FlextNoOpTracer,
    FlextSimpleAlerts,
)
from flext_core.result import FlextResult
from flext_core.schema_processing import (
    FlextBaseConfigManager,
    FlextBaseEntry,
    FlextBaseFileWriter,
    FlextBaseProcessor,
    FlextBaseSorter,
    FlextConfigAttributeValidator,
    FlextEntryType,
)
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

# =============================================================================
# LEGACY MIXIN ALIASES - Maintain test compatibility
# =============================================================================

# All legacy mixins delegate to modern mixins

# Legacy mixin aliases for test compatibility
LegacyCompatibleCommandMixin = FlextCommandMixin
LegacyCompatibleComparableMixin = FlextComparableMixin
LegacyCompatibleDataMixin = FlextValidatableMixin  # Closest match
LegacyCompatibleEntityMixin = FlextIdentifiableMixin  # Closest match
LegacyCompatibleFullMixin = FlextValidatableMixin  # Closest match
LegacyCompatibleIdentifiableMixin = FlextIdentifiableMixin
LegacyCompatibleLoggableMixin = FlextLoggableMixin
LegacyCompatibleSerializableMixin = FlextSerializableMixin
LegacyCompatibleServiceMixin = FlextLoggableMixin  # Closest match
LegacyCompatibleTimestampMixin = FlextTimestampMixin
LegacyCompatibleTimingMixin = FlextTimingMixin
LegacyCompatibleValidatableMixin = FlextValidatableMixin
LegacyCompatibleValueObjectMixin = FlextValidatableMixin  # Closest match

# =============================================================================
# LEGACY OBSERVABILITY ALIASES
# =============================================================================


# Legacy observability aliases
InMemoryMetrics = FlextInMemoryMetrics
MinimalObservability = FlextMinimalObservability
NoOpTracer = FlextNoOpTracer
SimpleAlerts = FlextSimpleAlerts

# =============================================================================
# LEGACY SCHEMA PROCESSING ALIASES
# =============================================================================


# Legacy schema processing aliases
BaseEntry = FlextBaseEntry
BaseFileWriter = FlextBaseFileWriter
BaseProcessor = FlextBaseProcessor
BaseSorter = FlextBaseSorter
ConfigAttributeValidator = FlextConfigAttributeValidator
EntryType = FlextEntryType

# =============================================================================
# LEGACY VERSION UTILITIES
# =============================================================================


def check_python_compatibility() -> bool:
    """Legacy function for Python compatibility check."""
    return sys.version_info >= (3, 13)


def compare_versions(v1: str, v2: str) -> int:
    """Legacy version comparison function."""
    ver1 = version.parse(v1)
    ver2 = version.parse(v2)
    if ver1 < ver2:
      return -1
    if ver1 > ver2:
      return 1
    return 0


def get_available_features() -> list[str]:
    """Legacy function to get available features."""
    return ["core", "validation", "container", "result"]


def get_version_info() -> dict[str, str]:
    """Legacy version info function."""
    return {"version": __version__, "python": "3.13+"}


def get_version_string() -> str:
    """Legacy version string function."""
    return __version__


def get_version_tuple() -> tuple[int, ...]:
    """Legacy version tuple function."""
    return tuple(int(x) for x in __version__.split(".") if x.isdigit())


def is_feature_available(feature: str) -> bool:
    """Legacy feature availability check."""
    return feature in get_available_features()


def validate_version_format(version_str: str) -> bool:
    """Legacy version format validation."""
    try:
      from packaging import version  # noqa: PLC0415

      version.parse(version_str)
      return True
    except Exception:
      return False


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
    def validate_config(config: dict[str, object]) -> FlextResult[bool]:
      """Legacy validation function."""
      if config:
          return FlextResult.ok(data=True)
      return FlextResult.fail("Configuration is empty")

    @staticmethod
    def validate_config_type(
      value: object,
      expected_type: type[object],
      key_name: str = "value",
    ) -> FlextResult[bool]:
      """Legacy type validation function."""
      try:
          if isinstance(value, expected_type):
              return FlextResult.ok(data=True)
          return FlextResult.fail(
              f"Configuration '{key_name}' must be {expected_type.__name__}, got {type(value).__name__}",
          )
      except Exception as e:
          return FlextResult.fail(f"Type validation error for '{key_name}': {e}")

    @staticmethod
    def validate_config_value(
      value: object,
      validator: object,
      error_message: str = "Validation failed",
    ) -> FlextResult[bool]:
      """Legacy value validation function."""
      try:
          if not callable(validator):
              return FlextResult.fail("Validator must be callable")

          if validator(value):
              return FlextResult.ok(data=True)
          return FlextResult.fail(error_message)
      except Exception as e:
          return FlextResult.fail(f"Validation error: {e}")

    @staticmethod
    def validate_config_range(
      value: float,
      min_value: float | None = None,
      max_value: float | None = None,
      key_name: str = "value",
    ) -> FlextResult[bool]:
      """Legacy range validation function."""
      try:
          if min_value is not None and value < min_value:
              return FlextResult.fail(
                  f"Configuration '{key_name}' must be >= {min_value}, got {value}",
              )
          if max_value is not None and value > max_value:
              return FlextResult.fail(
                  f"Configuration '{key_name}' must be <= {max_value}, got {value}",
              )
          return FlextResult.ok(data=True)
      except Exception as e:
          return FlextResult.fail(f"Range validation error for '{key_name}': {e}")


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
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "BaseSorter",
    "ConfigAttributeValidator",
    "ConsoleLogger",
    "EntryType",
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
    "InMemoryMetrics",
    "LegacyCompatibleCacheableMixin",
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "MinimalObservability",
    "NoOpTracer",
    "SimpleAlerts",
    "_BaseConfigDefaults",
    "_BaseConfigValidation",
    "_PerformanceConfig",
    "check_python_compatibility",
    "compare_versions",
    "flext_validate_non_empty_string",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
]
