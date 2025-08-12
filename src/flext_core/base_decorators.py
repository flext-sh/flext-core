"""Compatibility aliases for internal decorators used by legacy tests.

Exports stable symbols that mirror names previously used in tests while
preserving the new implementations under ``flext_core.decorators``.
"""

from flext_core.decorators import (
    FlextDecoratorFactory,
    FlextDecoratorUtils,
    FlextErrorHandlingDecorators,
    FlextFunctionalDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextValidationDecorators,
)
from flext_core.protocols import FlextDecoratedFunction

# Expose type alias for decorated functions
_DecoratedFunction = FlextDecoratedFunction

# Aliases for base decorator utilities and classes
_BaseDecoratorUtils = FlextDecoratorUtils
_BaseValidationDecorators = FlextValidationDecorators
_BaseErrorHandlingDecorators = FlextErrorHandlingDecorators
_BasePerformanceDecorators = FlextPerformanceDecorators
_BaseLoggingDecorators = FlextLoggingDecorators
_BaseFunctionalDecorators = FlextFunctionalDecorators
_BaseImmutabilityDecorators = FlextImmutabilityDecorators
_BaseDecoratorFactory = FlextDecoratorFactory

# Expose internal decorator functions for safe and validation decorators
_safe_call_decorator = FlextErrorHandlingDecorators.create_safe_decorator
_validate_input_decorator = FlextValidationDecorators.create_validation_decorator

# Define public API for this compatibility module (sorted for determinism)
__all__ = [
    "FlextFunctionalDecorators",
    "_BaseDecoratorFactory",
    "_BaseDecoratorUtils",
    "_BaseErrorHandlingDecorators",
    "_BaseFunctionalDecorators",
    "_BaseImmutabilityDecorators",
    "_BaseLoggingDecorators",
    "_BasePerformanceDecorators",
    "_BaseValidationDecorators",
    "_DecoratedFunction",
    "_safe_call_decorator",
    "_validate_input_decorator",
]
