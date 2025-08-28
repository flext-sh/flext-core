"""Legacy compatibility layer for FLEXT ecosystem backward compatibility.

Comprehensive backward compatibility module following FLEXT_REFACTORING_PROMPT.md and
CLAUDE.md requirements. Provides minimal facades and aliases maintaining strict ABI
compatibility while directing users to modern hierarchical FlextCore architecture.

FLEXT Refactoring Compliance:
    - Hierarchical organization: All facades organized using nested class patterns
    - FlextTypes integration: Proper FlextTypes.* type hierarchy usage throughout
    - FlextConstants integration: Uses FlextConstants.* for all constants and error codes
    - FlextProtocols integration: Imports protocols from centralized protocols.py
    - Layer architecture: Foundation -> Domain -> Application -> Infrastructure compliance
    - Zero circular imports: Proper layering with string annotations and lazy imports
    - Python 3.13+ patterns: Modern type syntax, Pydantic integration, SOLID principles
    - Professional docstrings: Google/PEP style with comprehensive examples

FLEXT Architecture Layers:
    - ConfigLegacy: Configuration system backward compatibility facades
    - ValidationLegacy: Validation system minimal delegation to FlextValidation.Core
    - FieldsLegacy: Field system facades delegating to FlextFields.Core hierarchy
    - DecoratorsLegacy: Decorator facades using FlextDecorators structured patterns
    - ContainerLegacy: DI container facades with FlextContainer delegation
    - ModelLegacy: Model system facades using FlextTypes.Domain patterns
    - LoggingLegacy: Structured logging facades with FlextConstants.Observability

Critical Implementation Requirements:
    - Zero ABI breaking changes: All existing imports must continue working exactly
    - Minimal facades: No business logic, only orchestration and proper delegation
    - Centralized management: All compatibility in single module with clear organization
    - Deprecation warnings: Proper DeprecationWarning with specific migration paths
    - Type safety: Full typing with Python 3.13+ syntax and mypy/pyright compliance

Examples:
    Legacy compatibility (deprecated but functional)::

        # Configuration - continues working with deprecation warnings
        config = get_flext_config()  # -> FlextConfig()
        settings = get_flext_settings()  # -> FlextConfig.Settings()

        # Validation - minimal facades to FlextValidation.Core
        result = validate_email_address(
            "test@example.com"
        )  # -> FlextValidation.validate_email()

        # Fields - delegates to FlextFields.Core hierarchy
        field = flext_create_string_field("name")  # -> FlextFields.Core.StringField()

    Modern hierarchical usage (preferred)::


        config = FlextConfig()
        result = FlextValidation.validate_email("test@example.com")
        field = FlextFields.Core.StringField("name")

Migration Strategy:
    All legacy functions emit DeprecationWarning pointing to specific modern
    hierarchical APIs. Complete removal planned for next major version with
    comprehensive migration documentation and automated refactoring tools.

Note:
    This module enforces FLEXT_REFACTORING_PROMPT.md requirements including hierarchical
    organization, proper constant usage, clean import layering, and SOLID principles.
    All new code MUST use FlextCore hierarchical classes directly, not legacy facades.

"""

from __future__ import annotations

import re
import time
import uuid
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, ParamSpec, Self, TypeVar, cast

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.core import FlextCore
from flext_core.decorators import FlextDecorators
from flext_core.delegation_system import FlextDelegationSystem
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions
from flext_core.fields import FlextFields
from flext_core.guards import FlextGuards
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.payload import FlextPayload
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.schema_processing import (
    FlextBaseConfigManager,
    FlextBaseEntry,
    FlextBaseFileWriter,
    FlextBaseProcessor,
    FlextBaseSorter,
    FlextConfigAttributeValidator,
    FlextEntryType,
    FlextEntryValidator,
    FlextProcessingPipeline,
    FlextRegexProcessor,
)
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidation

# =============================================================================
# MODEL CLASS FACADES - All classes consolidated into FlextModels
# =============================================================================

# All model functionality consolidated into FlextModels
# These facades provide backward compatibility

# Base model classes - now facades to FlextModels nested classes
FlextModel = FlextModels.BaseConfig  # Base model facade
FlextRootModel = FlextModels.EntityId  # Example RootModel facade
FlextValue = FlextModels.Value  # Value object facade
FlextEntity = FlextModels.Entity  # Entity facade
FlextAggregateRoot = FlextModels.AggregateRoot  # Aggregate root facade

# Root model aliases
FlextRootModels = FlextModels  # Compatibility alias for FlextRootModels

# Factory aliases
FlextFactory = FlextModels  # Factory functionality in FlextModels
FlextEntityFactory = FlextModels  # Entity factory in FlextModels

# Type variables for generic functions
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# =============================================================================
# LOGGING HELPERS (Compatibility Functions)
# =============================================================================


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="get_logger",
    new_path="FlextLogger",
    migration_guide="Import FlextLogger directly: from flext_core import FlextLogger",
)
def get_logger(
    name: str,
    version: str = "1.0.0",
    level: str = "INFO",
    service_name: str | None = None,
    service_version: str | None = None,
) -> FlextLogger:
    """Get a FlextLogger instance - compatibility function."""
    return FlextLogger(
        name=name,
        level=level,
        service_name=service_name,
        service_version=service_version or version,
    )


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="set_global_correlation_id",
    new_path="FlextLogger.set_global_correlation_id",
    migration_guide="Import FlextLogger directly: from flext_core import FlextLogger",
)
def set_global_correlation_id(correlation_id: str | None) -> None:
    """Set global correlation ID for request tracing - compatibility function."""
    FlextLogger.set_global_correlation_id(correlation_id)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="get_correlation_id",
    new_path="FlextLogger.get_global_correlation_id",
    migration_guide="Import FlextLogger directly: from flext_core import FlextLogger",
)
def get_correlation_id() -> str | None:
    """Get current correlation ID, generating one if needed - compatibility function."""
    current_id = FlextLogger.get_global_correlation_id()
    if current_id is None:
        new_id = f"test_{uuid.uuid4().hex[:8]}"
        FlextLogger.set_global_correlation_id(new_id)
        return new_id
    return current_id


# Function removed - use decorators directly instead of manual warnings


# =============================================================================
# CONFIG LEGACY COMPATIBILITY LAYER
# =============================================================================


class ConfigLegacy:
    """Configuration system backward compatibility facades.

    Following FLEXT_REFACTORING_PROMPT.md requirements:
    - Minimal facades delegating to modern hierarchical implementations
    - Uses FlextTypes.Config.* for proper type integration
    - Uses FlextConstants.Config.* for configuration constants
    - Proper deprecation warnings with specific migration paths
    - Zero business logic, orchestration only

    Architecture Compliance:
    - Single Responsibility: Only configuration backward compatibility
    - Open/Closed: Easy to extend with new config facade patterns
    - Dependency Inversion: Facades don't depend on specific implementations
    """

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="get_flext_config",
        new_path="FlextConfig",
        migration_guide="Import FlextConfig directly: from flext_core import FlextConfig",
    )
    def get_flext_config() -> type[FlextConfig]:
        """Legacy FlextConfig factory - DEPRECATED, use FlextConfig() directly.

        Returns:
            FlextConfig class for backward compatibility.

        Note:
            This function exists for ABI compatibility only. New code should
            import and use FlextConfig directly from flext_core.

        """
        # Direct import - NO lazy loading per FLEXT requirements
        return FlextConfig

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="get_flext_settings",
        new_path="FlextConfig.Settings",
        migration_guide="Import FlextConfig.Settings directly: from flext_core import FlextConfig",
    )
    def get_flext_settings() -> FlextConfig.Settings:
        """Legacy FlextConfig factory - DEPRECATED, use FlextConfig.Settings() directly.

        Returns:
            FlextConfig.Settings instance for backward compatibility.

        Note:
            This function exists for ABI compatibility only. New code should
            use FlextConfig.Settings() directly.

        """
        # Direct instantiation - NO lazy loading per FLEXT requirements
        return FlextConfig.Settings()

    # Type aliases using proper FlextTypes hierarchy
    class Types:
        """Configuration type aliases using FlextTypes hierarchy."""

        # Using FlextTypes.Config for proper type integration
        ConfigDict: type[FlextTypes.Config.ConfigDict] = dict
        ConfigKey: type[str] = str
        ConfigValue: type[object] = object

        # Legacy aliases for exact ABI compatibility
        FlextConfigDict = ConfigDict
        FlextConfigKey = ConfigKey
        FlextConfigValue = ConfigValue

    # Direct class-level aliases for backward compatibility
    FlextConfigDict = Types.FlextConfigDict
    FlextConfigKey = Types.FlextConfigKey
    FlextConfigValue = Types.FlextConfigValue


# Legacy config facades - maintain exact same interface
def get_flext_config() -> type[FlextConfig]:
    """Legacy function facade - DEPRECATED.

    Returns:
        FlextConfig class for backward compatibility.

    Note:
        This function exists for ABI compatibility only. New code should

        import and use FlextConfig directly from flext_core.

    """
    return ConfigLegacy.get_flext_config()


def get_flext_settings() -> FlextConfig.Settings:
    """Legacy function facade - DEPRECATED.

    Returns:
        FlextConfig.Settings instance for backward compatibility.

    Note:
        This function exists for ABI compatibility only. New code should
        use FlextConfig.Settings() directly.

    """
    return ConfigLegacy.get_flext_settings()


def merge_configs(*configs: dict[str, object]) -> dict[str, object]:
    """Legacy function facade - DEPRECATED.

    Returns:
        dict[str, object] for backward compatibility.

    """
    result = FlextConfig.merge_configs(*configs)
    return result.unwrap() if result.success else {}


def safe_get_env_var(name: str, default: str | None = None) -> str:
    """Legacy function facade - DEPRECATED.

    Returns:
        str for backward compatibility.

    """
    result = FlextConfig.safe_get_env_var(name, default)
    return result.unwrap() if result.success else (default or "")


def safe_load_json_file(file_path: str | Path) -> dict[str, object]:
    """Legacy function facade - DEPRECATED.

    Returns:
        dict[str, object] for backward compatibility.

    """
    result = FlextConfig.safe_load_json_file(file_path)
    return result.unwrap() if result.success else {}


# ABI-compatible direct aliases using proper FlextTypes integration
FlextConfigDict = ConfigLegacy.Types.FlextConfigDict
FlextConfigKey = ConfigLegacy.Types.FlextConfigKey
FlextConfigValue = ConfigLegacy.Types.FlextConfigValue


# =============================================================================
# HANDLER LEGACY COMPATIBILITY LAYER
# =============================================================================


class HandlerLegacy:
    """Centralized handler compatibility facades following FLEXT patterns.

    This class provides minimal facades for all legacy handler exports,
    maintaining ABI compatibility while delegating to the new consolidated
    FlextHandlers architecture.

    All facades are minimal orchestration-only classes with deprecation warnings.
    """

    @staticmethod
    def deprecation_warning(old_name: str, new_path: str) -> None:
        """Emit standardized deprecation warning for legacy usage."""
        warnings.warn(
            f"{old_name} is deprecated. Use {new_path} instead. "
            "See FLEXT migration guide for updated patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

    class BaseHandlerLegacyFacade:
        """Minimal facade for FlextBaseHandler legacy compatibility."""

        def __init__(self, name: str | None = None) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextBaseHandler", "FlextHandlers.Implementation.BasicHandler"
            )
            # Direct usage - NO lazy import per FLEXT requirements
            self._impl = FlextHandlers.Implementation.BasicHandler(name)

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class ValidatingHandlerLegacyFacade:
        """Minimal facade for FlextValidatingHandler legacy compatibility."""

        def __init__(
            self, name: str | None = None, validators: list[object] | None = None
        ) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextValidatingHandler",
                "FlextHandlers.Implementation.ValidatingHandler",
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert legacy validators to proper protocol validators
            protocol_validators: list[object] = []
            if validators:
                for validator in validators:
                    if callable(validator):
                        # Create validator that implements the expected protocol
                        class LegacyValidatorWrapper:
                            def __init__(
                                self, func: Callable[[object], object]
                            ) -> None:
                                self._func = func

                            def validate(self, value: object) -> FlextResult[object]:
                                """Validate using legacy function."""
                                try:
                                    result = self._func(value)
                                    return FlextResult[object].ok(result)
                                except Exception as e:
                                    return FlextResult[object].fail(str(e))

                        protocol_validators.append(LegacyValidatorWrapper(validator))

            # Cast to the expected type since we're implementing the protocol correctly
            typed_validators = cast(
                "list[FlextProtocols.Foundation.Validator[object]] | None",
                protocol_validators or None,
            )
            self._impl = FlextHandlers.Implementation.ValidatingHandler(
                name, typed_validators
            )

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class AuthorizingHandlerLegacyFacade:
        """Minimal facade for FlextAuthorizingHandler legacy compatibility."""

        def __init__(
            self,
            name: str | None = None,
            authorization_check: Callable[[object], bool] | None = None,
        ) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextAuthorizingHandler",
                "FlextHandlers.Implementation.AuthorizingHandler",
            )
            # Direct usage - NO lazy import per FLEXT requirements
            self._impl = FlextHandlers.Implementation.AuthorizingHandler(
                name, authorization_check
            )

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class MetricsHandlerLegacyFacade:
        """Minimal facade for FlextMetricsHandler legacy compatibility."""

        def __init__(self, name: str | None = None) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextMetricsHandler", "FlextHandlers.Implementation.MetricsHandler"
            )
            # Direct usage - NO lazy import per FLEXT requirements

            self._impl = FlextHandlers.Implementation.MetricsHandler(name)

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class EventHandlerLegacyFacade:
        """Minimal facade for FlextEventHandler legacy compatibility."""

        def __init__(self, name: str | None = None) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextEventHandler", "FlextHandlers.Implementation.EventHandler"
            )
            # Direct usage - NO lazy import per FLEXT requirements

            self._impl = FlextHandlers.Implementation.EventHandler(name)

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class HandlerChainLegacyFacade:
        """Minimal facade for FlextHandlerChain legacy compatibility."""

        def __init__(self, name: str | None = None) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextHandlerChain", "FlextHandlers.Patterns.HandlerChain"
            )
            # Direct usage - NO lazy import per FLEXT requirements

            self._impl = FlextHandlers.Patterns.HandlerChain(name)

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    class HandlerRegistryLegacyFacade:
        """Minimal facade for FlextHandlerRegistry legacy compatibility."""

        def __init__(self) -> None:
            HandlerLegacy.deprecation_warning(
                "FlextHandlerRegistry", "FlextHandlers.Management.HandlerRegistry"
            )
            # Direct usage - NO lazy import per FLEXT requirements

            self._impl = FlextHandlers.Management.HandlerRegistry()

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)


# =============================================================================
# VALIDATION LEGACY COMPATIBILITY LAYER
# =============================================================================


class ValidationLegacy:
    """Validation system backward compatibility facades.

    Following FLEXT_REFACTORING_PROMPT.md requirements:
    - Minimal facades delegating to FlextValidation.Core hierarchy
    - Uses FlextTypes.Protocol.* for validation interface types
    - Uses FlextConstants.Errors.* for validation error constants
    - Proper deprecation warnings with specific migration paths
    - Zero business logic, orchestration only

    Architecture Compliance:
    - Single Responsibility: Only validation backward compatibility
    - Open/Closed: Easy to extend with new validation facade patterns
    - Dependency Inversion: Facades delegate to hierarchical implementations
    - Interface Segregation: Separate facades for different validation concerns
    """

    class PredicateLegacyFacade:
        """Minimal facade for FlextPredicate legacy compatibility.

        Provides backward compatibility for legacy predicate usage while
        delegating to modern FlextValidation.Core.Predicates hierarchy.

        Args:
            func: The predicate function to wrap
            name: Name identifier for the predicate

        Note:
            This facade exists for ABI compatibility only. New code should
            use FlextValidation.Core.Predicates directly.

        """

        def __init__(
            self, func: Callable[[object], bool], name: str = "predicate"
        ) -> None:
            warnings.warn(
                "FlextPredicate is deprecated. Use FlextValidation.Core.Predicates instead. "
                "See FlextValidation hierarchical API for modern validation patterns.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Store function and name for delegation
            self.func = func
            self.name = name

        def __call__(self, value: object) -> bool:
            """Execute the predicate function.

            Args:
                value: The value to validate

            Returns:
                True if predicate passes, False otherwise

            Note:
                This delegates to the wrapped function with error handling.

            """
            try:
                return self.func(value)
            except Exception:
                return False

        def __getattr__(self, name: str) -> object:
            """Delegate attribute access to function attributes."""
            return getattr(self.func, name, None)

    class ValidationChainLegacyFacade:
        """Minimal facade for FlextValidationChain legacy compatibility."""

        def __init__(self, validators: list[object] | None = None) -> None:
            warnings.warn(
                "FlextValidationChain is deprecated. Use FlextValidation.Advanced.CompositeValidator instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert legacy validators to proper callable validators
            callable_validators: list[Callable[[object], FlextResult[object]]] = []
            for validator in validators or []:
                if callable(validator):
                    # Create a proper closure to avoid late binding issues
                    def create_validator_wrapper(
                        v: Callable[[object], object],
                    ) -> Callable[[object], FlextResult[object]]:
                        def wrapper(value: object) -> FlextResult[object]:
                            try:
                                result = v(value)
                                return FlextResult[object].ok(result)
                            except Exception as e:
                                return FlextResult[object].fail(str(e))

                        return wrapper

                    callable_validators.append(create_validator_wrapper(validator))

            self._impl = FlextValidation.Advanced.CompositeValidator(
                callable_validators
            )

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

        def validate(self, data: object) -> object:
            """Delegate validation to implementation."""
            return self._impl.validate(data)

        def add_validator(
            self, validator: object
        ) -> ValidationLegacy.ValidationChainLegacyFacade:
            """Add validator to chain and return new chain."""
            new_validators = [*self._impl.validators, validator]
            return ValidationLegacy.ValidationChainLegacyFacade(new_validators)

    class SchemaValidatorLegacyFacade:
        """Minimal facade for FlextSchemaValidator legacy compatibility."""

        def __init__(self, schema: dict[str, object]) -> None:
            warnings.warn(
                "FlextSchemaValidator is deprecated. Use FlextValidation.Advanced.SchemaValidator instead. "
                "See FlextValidation hierarchical API for modern validation patterns.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert schema dict to callable validators
            callable_schema: dict[str, Callable[[object], FlextResult[object]]] = {}
            for key, value in schema.items():
                if callable(value):
                    # Create proper closure to avoid late binding issues
                    def create_field_validator(
                        validator: Callable[[object], object],
                    ) -> Callable[[object], FlextResult[object]]:
                        def field_validator(val: object) -> FlextResult[object]:
                            try:
                                result = validator(val)
                                return FlextResult[object].ok(result)
                            except Exception as e:
                                return FlextResult[object].fail(str(e))

                        return field_validator

                    callable_schema[key] = create_field_validator(value)
                else:
                    # Create simple equality validator for non-callable values
                    def create_equality_validator(
                        expected: object,
                    ) -> Callable[[object], FlextResult[object]]:
                        def equality_validator(val: object) -> FlextResult[object]:
                            if val == expected:
                                return FlextResult[object].ok(val)
                            return FlextResult[object].fail(
                                f"Expected {expected}, got {val}"
                            )

                        return equality_validator

                    callable_schema[key] = create_equality_validator(value)

            self._impl = FlextValidation.Advanced.SchemaValidator(callable_schema)

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

        def validate(self, data: object) -> FlextResult[dict[str, object]]:
            """Delegate validation to implementation."""
            # Ensure data is a dict before passing to schema validator
            if isinstance(data, dict):
                # Type narrowing: data is now dict[str, object] after isinstance check
                typed_data = cast("dict[str, object]", data)
                return self._impl.validate(typed_data)
            return FlextResult[dict[str, object]].fail(
                f"Schema validation requires dict, got {type(data).__name__}",
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

    class PredicatesLegacyFacade:
        """Minimal facade for FlextPredicates legacy compatibility."""

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.is_string",
            new_path="FlextValidation.Core.Predicates",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def is_string() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if value is a string."""
            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, str), name="is_string"
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.is_integer",
            new_path="FlextValidation.Core.Predicates",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def is_integer() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if value is an integer."""
            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, int) and not isinstance(x, bool),
                name="is_integer",
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.is_positive",
            new_path="FlextValidation.Core.Predicates",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def is_positive() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if numeric value is positive."""
            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, (int, float))
                and not isinstance(x, bool)
                and x > 0,
                name="is_positive",
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.has_length",
            new_path="FlextValidation.Core.Predicates.create_string_length_predicate",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def has_length(
            min_len: int | None = None, max_len: int | None = None
        ) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks string length."""
            predicate = FlextValidation.Core.Predicates.create_string_length_predicate(
                min_len, max_len
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.contains",
            new_path="FlextValidation.Core.Predicates",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def contains(substring: str) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if string contains substring."""
            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, str) and substring in x,
                name=f"contains('{substring}')",
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextPredicates.matches_pattern",
            new_path="FlextValidation.Core.Predicates.create_email_predicate",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def matches_pattern(pattern: str) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if string matches regex pattern."""

            def pattern_check(value: object) -> bool:
                try:
                    return (
                        isinstance(value, str) and re.match(pattern, value) is not None
                    )
                except re.error:
                    return False

            predicate = FlextValidation.Core.Predicates(
                pattern_check, name=f"matches_pattern('{pattern}')"
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

    class ValidationUtilsLegacyFacade:
        """Minimal facade for FlextValidationUtils legacy compatibility."""

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextValidationUtils.validate_all",
            new_path="FlextValidation.Advanced.CompositeValidator",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def validate_all(*validators: object) -> Callable[[object], object]:
            """Create validator that runs all validators and collects all results."""

            def run_all(data: object) -> list[object]:
                results: list[object] = []
                for validator in validators:
                    if callable(validator):
                        try:
                            results.append(validator(data))
                        except Exception as e:
                            results.append(FlextResult[object].fail(str(e)))
                    else:
                        results.append(
                            FlextResult[object].fail("Non-callable validator")
                        )
                return results

            return run_all

        @staticmethod
        @FlextDecorators.Lifecycle.deprecated_legacy_function(
            old_name="FlextValidationUtils.validate_any",
            new_path="FlextValidation.Advanced.CompositeValidator",
            migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
        )
        def validate_any(*validators: object) -> Callable[[object], object]:
            """Create validator that succeeds if any validator succeeds."""

            def run_any(data: object) -> object:
                errors: list[str] = []
                for validator in validators:
                    if callable(validator):
                        try:
                            result = validator(data)
                            if hasattr(result, "success") and getattr(
                                result, "success", False
                            ):
                                return result
                            errors.append(
                                str(getattr(result, "error", "Unknown error"))
                            )
                        except Exception as e:
                            errors.append(str(e))
                    else:
                        errors.append("Non-callable validator")

                return FlextResult[object].fail(
                    f"All validators failed: {'; '.join(errors)}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return run_any

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="get_service_name_validator",
        new_path="FlextValidation.Core",
        migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
    )
    def get_service_name_validator() -> Callable[[str], object]:
        """Get service name validator function."""
        return flext_validate_service_name

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="get_config_key_validator",
        new_path="FlextValidation.Core",
        migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
    )
    def get_config_key_validator() -> Callable[[str], object]:
        """Get config key validator function."""
        return flext_validate_config_key


# =============================================================================
# VALIDATION LEGACY FUNCTION FACADES
# =============================================================================


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="validate_email_address",
    new_path="FlextValidation.validate_email",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def validate_email_address(value: object) -> object:
    """Legacy function facade - DEPRECATED."""
    if isinstance(value, str):
        return FlextValidation.validate_email(value)

    return FlextResult[str].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="create_validation_pipeline",
    new_path="FlextValidation.Advanced.CompositeValidator",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def create_validation_pipeline(_data: object) -> object:
    """Legacy function facade - DEPRECATED."""
    return FlextValidation.create_composite_validator([])


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="validate_with_schema",
    new_path="FlextValidation.validate_with_schema",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def validate_with_schema(data: object, schema: dict[str, object]) -> object:
    """Legacy function facade - DEPRECATED."""
    if isinstance(data, dict):
        # Convert schema to callable validators like in SchemaValidatorLegacyFacade
        callable_schema: dict[str, Callable[[object], FlextResult[object]]] = {}
        for key, value in schema.items():
            if callable(value):
                # Create proper closure to avoid late binding issues
                def create_field_validator(
                    validator: Callable[[object], object],
                ) -> Callable[[object], FlextResult[object]]:
                    def field_validator(val: object) -> FlextResult[object]:
                        try:
                            result = validator(val)
                            return FlextResult[object].ok(result)
                        except Exception as e:
                            return FlextResult[object].fail(str(e))

                    return field_validator

                callable_schema[key] = create_field_validator(value)
            else:
                # Create simple equality validator for non-callable values
                def create_equality_validator(
                    expected: object,
                ) -> Callable[[object], FlextResult[object]]:
                    def equality_validator(val: object) -> FlextResult[object]:
                        if val == expected:
                            return FlextResult[object].ok(val)
                        return FlextResult[object].fail(
                            f"Expected {expected}, got {val}"
                        )

                    return equality_validator

                callable_schema[key] = create_equality_validator(value)

        # Type narrowing: data is now dict[str, object] after isinstance check
        typed_data = cast("dict[str, object]", data)
        return FlextValidation.validate_with_schema(typed_data, callable_schema)

    return FlextResult[dict[str, object]].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="validate_length",
    new_path="FlextValidation.Rules.StringRules.validate_length",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def validate_length(
    value: object, min_length: int | None = None, max_length: int | None = None
) -> object:
    """Legacy function facade - DEPRECATED."""
    if isinstance(value, str):
        return FlextValidation.Rules.StringRules.validate_length(
            value, min_length, max_length
        )

    return FlextResult[str].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="flext_validate_service_name",
    new_path="FlextValidation.Core",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def flext_validate_service_name(name: str) -> FlextResult[str]:
    """Validate service name - maintains exact ABI.

    Args:
        name: Service name to validate

    Returns:
        FlextResult with validation outcome

    """
    if not name.strip():
        return FlextResult[str].fail(
            "Service name must be a non-empty string",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    return FlextResult[str].ok(name.strip())


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="flext_validate_config_key",
    new_path="FlextValidation.Core",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def flext_validate_config_key(key: str) -> FlextResult[str]:
    """Validate config key - maintains exact ABI.

    Args:
        key: Configuration key to validate

    Returns:
        FlextResult with validation outcome

    """
    if not key.strip():
        return FlextResult[str].fail(
            "Config key must be a non-empty string",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    return FlextResult[str].ok(key.strip())


# Legacy aliases for backward compatibility
FlextPredicate = ValidationLegacy.PredicateLegacyFacade
FlextStringValidator = ValidationLegacy.ValidationUtilsLegacyFacade
FlextNumericValidator = ValidationLegacy.ValidationUtilsLegacyFacade
FlextCollectionValidator = ValidationLegacy.ValidationUtilsLegacyFacade
FlextPredicates = ValidationLegacy.PredicatesLegacyFacade
FlextValidationChain = ValidationLegacy.ValidationChainLegacyFacade
FlextSchemaValidator = ValidationLegacy.SchemaValidatorLegacyFacade
FlextValidationUtils = ValidationLegacy.ValidationUtilsLegacyFacade


# Legacy function compatibility
@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_not_empty",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_not_empty(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, str) and bool(x.strip()), name="is_not_empty"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_numeric",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_numeric(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        name="is_numeric",
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_string",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_string(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, str), name="is_string"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_list",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_list(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, list), name="is_list"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_dict",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_dict(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, dict), name="is_dict"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="is_boolean",
    new_path="FlextValidation.Core.Predicates",
    migration_guide="See FlextValidation hierarchical API for modern validation patterns.",
)
def is_boolean(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, bool), name="is_boolean"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


# =============================================================================
# MIXIN LEGACY COMPATIBILITY LAYER
# =============================================================================


class MixinLegacy:
    """Centralized mixin compatibility facades following FLEXT patterns.

    All legacy mixin classes are minimal facades that delegate to FlextMixins
    consolidated methods while maintaining full ABI compatibility.
    """

    @staticmethod
    def deprecation_warning(old_name: str, new_path: str) -> None:
        """Emit standardized deprecation warning for legacy mixin usage."""
        warnings.warn(
            f"{old_name} is deprecated. Use {new_path} instead. "
            "See FLEXT migration guide for updated mixin patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

    class _CompatibilityMixin:
        """Base compatibility class that enables mixin inheritance for legacy code."""

        def mixin_setup(self) -> None:
            """Setup mixin functionality."""

    class LoggableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins.log_* methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextLoggableMixin", "FlextMixins.Logging.*"
            )

        @property
        def logger(self) -> object:
            """Get logger via FlextMixins."""
            return FlextMixins.get_logger(self)

        def log_operation(self, operation: str, **kwargs: object) -> None:
            """Log operation via FlextMixins."""
            FlextMixins.log_operation(self, operation, **kwargs)

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log info via FlextMixins."""
            FlextMixins.log_info(self, message, **kwargs)

        def log_error(self, message: str, **kwargs: object) -> None:
            """Log error via FlextMixins."""
            FlextMixins.log_error(self, message, **kwargs)

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log debug via FlextMixins."""
            FlextMixins.log_debug(self, message, **kwargs)

    class TimestampMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins timestamp methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextTimestampMixin", "FlextMixins.Timestamp.*"
            )

        def update_timestamp(self) -> None:
            """Update timestamp via FlextMixins."""
            # Initialize if needed first
            if not hasattr(self, "_timestamp_initialized"):
                current_time = time.time()
                self._created_at = current_time
                self._updated_at = current_time
                self._timestamp_initialized = True
            else:
                self._updated_at = time.time()

        @property
        def created_at(self) -> float:
            """Get created timestamp via FlextMixins."""
            # Initialize timestamp if not set using internal attributes
            if not hasattr(self, "_timestamp_initialized"):
                current_time = time.time()
                self._created_at = current_time
                self._updated_at = current_time
                self._timestamp_initialized = True

            return getattr(self, "_created_at", 0.0)

        @property
        def updated_at(self) -> float:
            """Get updated timestamp via FlextMixins."""
            # Initialize timestamp if not set using internal attributes
            if not hasattr(self, "_timestamp_initialized"):
                current_time = time.time()
                self._created_at = current_time
                self._updated_at = current_time
                self._timestamp_initialized = True

            return getattr(self, "_updated_at", 0.0)

        def get_age_seconds(self) -> float:
            """Get age via FlextMixins."""
            # Initialize if needed first
            if not hasattr(self, "_timestamp_initialized"):
                current_time = time.time()
                self._created_at = current_time
                self._updated_at = current_time
                self._timestamp_initialized = True

            created = getattr(self, "_created_at", None)
            if created is None:
                return 0.0
            return float(time.time() - created)

    class IdentifiableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins ID methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextIdentifiableMixin", "FlextMixins.Identification.*"
            )

        @property
        def id(self) -> str:
            """Get ID via FlextMixins."""
            return str(FlextMixins.ensure_id(self))

        @id.setter
        def id(self, value: str) -> None:
            """Set ID via FlextMixins."""
            result = FlextMixins.set_id(self, value)
            if result.is_failure:
                raise FlextValidationError(result.error or "Invalid entity ID")

        def get_id(self) -> str:
            """Get ID via FlextMixins."""
            return str(FlextMixins.ensure_id(self))

        def has_id(self) -> bool:
            """Check ID via FlextMixins."""
            return bool(FlextMixins.has_id(self))

    class ValidatableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins validation methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextValidatableMixin", "FlextMixins.Validation.*"
            )

        def validate(self) -> object:
            """Validate via FlextMixins."""
            if FlextMixins.is_valid(self):
                return FlextResult[None].ok(None)
            errors = FlextMixins.get_validation_errors(self)
            return FlextResult[None].fail(f"Validation failed: {'; '.join(errors)}")

        @property
        def is_valid(self) -> bool:
            """Check validity via FlextMixins."""
            return FlextMixins.is_valid(self)

        def add_validation_error(self, error: str) -> None:
            """Add validation error via FlextMixins."""
            FlextMixins.add_validation_error(self, error)

        def clear_validation_errors(self) -> None:
            """Clear validation errors via FlextMixins."""
            FlextMixins.clear_validation_errors(self)

        @property
        def validation_errors(self) -> list[str]:
            """Get validation errors via FlextMixins."""
            return FlextMixins.get_validation_errors(self)

        def has_validation_errors(self) -> bool:
            """Check if has validation errors via FlextMixins."""
            return len(FlextMixins.get_validation_errors(self)) > 0

    class SerializableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins serialization methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextSerializableMixin", "FlextMixins.Serialization.*"
            )

        def to_dict(self) -> dict[str, object]:
            """Convert to dict via FlextMixins."""
            return FlextMixins.to_dict(self)

        def to_dict_basic(self) -> dict[str, object]:
            """Convert to basic dict via FlextMixins."""
            return FlextMixins.to_dict_basic(self)

        def to_json(self) -> str:
            """Convert to JSON via FlextMixins."""
            return FlextMixins.to_json(self)

        def load_from_dict(self, data: dict[str, object]) -> None:
            """Load from dict via FlextMixins."""
            FlextMixins.load_from_dict(self, data)

        def load_from_json(self, json_str: str) -> None:
            """Load from JSON via FlextMixins."""
            result = FlextMixins.load_from_json(self, json_str)
            if result.is_failure:
                raise ValueError(result.error)

    class TimingMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins timing methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning("FlextTimingMixin", "FlextMixins.Timing.*")

        def start_timing(self) -> float:
            """Start timing via FlextMixins."""
            return FlextMixins.start_timing(self)

        def stop_timing(self) -> float:
            """Stop timing via FlextMixins."""
            return FlextMixins.stop_timing(self)

    class ComparableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins comparison methods."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextComparableMixin", "FlextMixins.Comparison.*"
            )

        def __eq__(self, other: object) -> bool:
            """Check equality via FlextMixins."""
            return FlextMixins.objects_equal(self, other)

        def __hash__(self) -> int:
            """Generate hash via FlextMixins."""
            return FlextMixins.object_hash(self)

        def __lt__(self, other: object) -> bool:
            """Compare via FlextMixins."""
            return FlextMixins.compare_objects(self, other) < 0

        def compare_to(self, other: object) -> int:
            """Compare objects via FlextMixins."""
            return FlextMixins.compare_objects(self, other)

    # Composite mixins for legacy compatibility
    class EntityMixinLegacyFacade(
        TimestampMixinLegacyFacade,
        IdentifiableMixinLegacyFacade,
        LoggableMixinLegacyFacade,
        ValidatableMixinLegacyFacade,
        SerializableMixinLegacyFacade,
    ):
        """Legacy compatibility - composite entity mixin."""

        def __init__(self) -> None:
            super().__init__()
            MixinLegacy.deprecation_warning(
                "FlextEntityMixin", "FlextMixins with multiple categories"
            )

    class ValueObjectMixinLegacyFacade(
        ValidatableMixinLegacyFacade,
        SerializableMixinLegacyFacade,
    ):
        """Legacy compatibility - composite value object mixin."""

        def __init__(self) -> None:
            super().__init__()
            MixinLegacy.deprecation_warning(
                "FlextValueObjectMixin",
                "FlextMixins.Validation + FlextMixins.Serialization",
            )

    class ServiceMixinLegacyFacade(
        LoggableMixinLegacyFacade,
        ValidatableMixinLegacyFacade,
    ):
        """Legacy compatibility - composite service mixin."""

        def __init__(self) -> None:
            super().__init__()
            MixinLegacy.deprecation_warning(
                "FlextServiceMixin", "FlextMixins.Logging + FlextMixins.Validation"
            )

    # Abstract base classes for compatibility
    class AbstractMixinLegacyFacade(_CompatibilityMixin):
        """Abstract base for compatibility."""

        def __init__(self) -> None:
            MixinLegacy.deprecation_warning(
                "FlextAbstractMixin", "FlextMixins with composition pattern"
            )


# =============================================================================
# PROTOCOL LEGACY COMPATIBILITY
# =============================================================================


class ProtocolLegacy:
    """Legacy protocol compatibility layer for existing ecosystem."""

    @staticmethod
    def get_query_handler_protocol() -> type:
        """Get legacy QueryHandlerProtocol for compatibility."""
        return FlextProtocols.Application.MessageHandler

    @staticmethod
    def get_chain_handler_protocol() -> type:
        """Get legacy ChainHandlerProtocol for compatibility."""
        return FlextProtocols.Application.MessageHandler


# =============================================================================
# GUARDS LEGACY COMPATIBILITY
# =============================================================================


class GuardsLegacy:
    """Legacy guards compatibility layer for existing ecosystem.

    DEPRECATED: Use FlextGuards class directly instead.
    These are simple aliases to maintain backward compatibility.
    """

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="GuardsLegacy.get_immutable_decorator",
        new_path="FlextGuards.immutable",
        migration_guide="use FlextGuards.immutable instead",
    )
    def get_immutable_decorator() -> Callable[[type], type]:
        """Get immutable decorator for compatibility.

        DEPRECATED: Use FlextGuards.immutable directly.
        """
        return FlextGuards.immutable

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="GuardsLegacy.get_pure_decorator",
        new_path="FlextGuards.pure",
        migration_guide="use FlextGuards.pure instead",
    )
    def get_pure_decorator() -> object:
        """Get pure function decorator for compatibility.

        DEPRECATED: Use FlextGuards.pure directly.
        """
        return FlextGuards.pure

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="GuardsLegacy.get_validation_utils",
        new_path="FlextGuards.ValidationUtils",
        migration_guide="use FlextGuards.ValidationUtils instead",
    )
    def get_validation_utils() -> type:
        """Get validation utils for compatibility.

        DEPRECATED: Use FlextGuards.ValidationUtils directly.
        """
        return FlextGuards.ValidationUtils


# =============================================================================
# PUBLIC LEGACY EXPORTS - Maintain exact ABI compatibility
# =============================================================================

# Handler legacy facades - maintain exact class names
FlextBaseHandler = HandlerLegacy.BaseHandlerLegacyFacade
FlextValidatingHandler = HandlerLegacy.ValidatingHandlerLegacyFacade
FlextAuthorizingHandler = HandlerLegacy.AuthorizingHandlerLegacyFacade
FlextMetricsHandler = HandlerLegacy.MetricsHandlerLegacyFacade
FlextEventHandler = HandlerLegacy.EventHandlerLegacyFacade
FlextHandlerChain = HandlerLegacy.HandlerChainLegacyFacade
FlextHandlerRegistry = HandlerLegacy.HandlerRegistryLegacyFacade

# Protocol compatibility - temporarily disabled due to syntax error in protocols.py
# QueryHandlerProtocol = ProtocolLegacy.get_query_handler_protocol()
# ChainHandlerProtocol = ProtocolLegacy.get_chain_handler_protocol()
# These protocols don't exist yet - removing from exports to avoid type conflicts
# QueryHandlerProtocol: type[object] | None = None  # Not exported
# ChainHandlerProtocol: type[object] | None = None  # Not exported


# Handler facade for existing ecosystems
class HandlersFacade:
    """Legacy facade for backward compatibility - minimal orchestration only."""

    Handler = FlextBaseHandler
    ValidatingHandler = FlextValidatingHandler
    AuthorizingHandler = FlextAuthorizingHandler


# Guards compatibility aliases - simple function aliases
def _deprecated_guards_alias(name: str, replacement: str) -> None:
    """Issue deprecation warning for guards aliases."""
    warnings.warn(
        f"{name} is deprecated, use {replacement} instead",
        DeprecationWarning,
        stacklevel=3,
    )


# Simple function aliases for guards compatibility
def immutable(target_class: type) -> type:
    """DEPRECATED: Use FlextGuards.immutable instead."""
    _deprecated_guards_alias("immutable", "FlextGuards.immutable")

    return FlextGuards.immutable(target_class)


def pure(func: object) -> object:
    """DEPRECATED: Use FlextGuards.pure instead."""
    _deprecated_guards_alias("pure", "FlextGuards.pure")

    return FlextGuards.pure(cast("Callable[[object], object]", func))


def make_factory(target_class: type) -> object:
    """DEPRECATED: Use FlextGuards.make_factory instead."""
    _deprecated_guards_alias("make_factory", "FlextGuards.make_factory")

    return FlextGuards.make_factory(target_class)


def make_builder(target_class: type) -> object:
    """DEPRECATED: Use FlextGuards.make_builder instead."""
    _deprecated_guards_alias("make_builder", "FlextGuards.make_builder")

    return FlextGuards.make_builder(target_class)


def require_not_none(value: object, message: str = "Value cannot be None") -> object:
    """DEPRECATED: Use FlextGuards.ValidationUtils.require_not_none instead."""
    _deprecated_guards_alias(
        "require_not_none", "FlextGuards.ValidationUtils.require_not_none"
    )

    return FlextGuards.ValidationUtils.require_not_none(value, message)


def require_positive(value: object, message: str = "Value must be positive") -> object:
    """DEPRECATED: Use FlextGuards.ValidationUtils.require_positive instead."""
    _deprecated_guards_alias(
        "require_positive", "FlextGuards.ValidationUtils.require_positive"
    )

    return FlextGuards.ValidationUtils.require_positive(value, message)


def require_non_empty(value: object, message: str = "Value cannot be empty") -> object:
    """DEPRECATED: Use FlextGuards.ValidationUtils.require_non_empty instead."""
    _deprecated_guards_alias(
        "require_non_empty", "FlextGuards.ValidationUtils.require_non_empty"
    )

    return FlextGuards.ValidationUtils.require_non_empty(value, message)


# Decorator aliases for compatibility
def validated(func: object) -> object:
    """DEPRECATED: Use FlextDecorators.Validation.validate_input instead."""
    _deprecated_guards_alias("validated", "FlextDecorators.Validation.validate_input")

    return FlextDecorators.Validation.validate_input(
        cast("Callable[[object], bool]", func)
    )


def safe(func: object) -> object:
    """DEPRECATED: Use FlextDecorators.Reliability.safe_result instead."""
    _deprecated_guards_alias("safe", "FlextDecorators.Reliability.safe_result")

    return FlextDecorators.Reliability.safe_result(
        cast("Callable[[object], object]", func)
    )


# =============================================================================
# MIXIN LEGACY EXPORTS - Maintain exact ABI compatibility
# =============================================================================

# Legacy mixin facades - use distinct names to avoid conflicts with real classes
# Note: Real mixins are defined in mixins.py, these are compatibility facades only
LoggableMixinLegacyFacade = MixinLegacy.LoggableMixinLegacyFacade
TimestampMixinLegacyFacade = MixinLegacy.TimestampMixinLegacyFacade
IdentifiableMixinLegacyFacade = MixinLegacy.IdentifiableMixinLegacyFacade
ValidatableMixinLegacyFacade = MixinLegacy.ValidatableMixinLegacyFacade
SerializableMixinLegacyFacade = MixinLegacy.SerializableMixinLegacyFacade
TimingMixinLegacyFacade = MixinLegacy.TimingMixinLegacyFacade
ComparableMixinLegacyFacade = MixinLegacy.ComparableMixinLegacyFacade

# Composite mixins - use distinct names to avoid conflicts
EntityMixinLegacyFacade = MixinLegacy.EntityMixinLegacyFacade
ValueObjectMixinLegacyFacade = MixinLegacy.ValueObjectMixinLegacyFacade

# Abstract compatibility aliases - REMOVED to avoid conflicts with mixins.py
# These caused mypy assignment errors due to duplicate class definitions
# Use AbstractXXXMixinLegacyFacade classes from MixinLegacy for compatibility
FlextAbstractMixin = MixinLegacy.AbstractMixinLegacyFacade

# Additional compatibility aliases - REMOVED to avoid conflicts with mixins.py
# FlextTimestampableMixin, FlextStateableMixin, FlextCacheableMixin, FlextObservableMixin, FlextConfigurableMixin
# Use TimestampableMixinLegacyFacade, StateableMixinLegacyFacade, CacheableMixinLegacyFacade, etc. from MixinLegacy


__all__ = [  # noqa: RUF022
    # "ChainHandlerProtocol",  # Not implemented yet - removed
    # =======================================================================
    # CONFIG LEGACY COMPATIBILITY - All configuration facades and aliases
    # =======================================================================
    # Legacy config classes
    "ConfigLegacy",
    # Legacy management classes
    "FieldsLegacy",
    # Note: FlextAbstractXXXMixin classes removed from public exports - conflicts with mixins.py
    # Use AbstractEntityMixinLegacyFacade, etc. for legacy compatibility
    "FlextAuthorizingHandler",
    # Core legacy handler exports
    "FlextBaseHandler",
    # Note: FlextCacheableMixin removed from public exports - conflicts with mixins.py
    # Note: FlextComparableMixin removed - conflicts with real mixin in mixins.py
    # Use CacheableMixinLegacyFacade, ComparableMixinLegacyFacade for legacy compatibility
    # ABI-compatible config types
    "FlextConfigDict",
    "FlextConfigKey",
    "FlextConfigValue",
    # Note: FlextConfigurableMixin removed from public exports - conflicts with mixins.py
    # Use ConfigurableMixinLegacyFacade for legacy compatibility
    # =======================================================================
    # CONSTANTS AND TYPES LEGACY COMPATIBILITY
    # =======================================================================
    # NOTE: FlextConstants removed - already exported from constants.py
    # =======================================================================
    # UTILITY AND SERVICE LEGACY COMPATIBILITY
    # =======================================================================
    # Container and DI
    # NOTE: FlextContainer removed - already exported from container.py
    # Core services
    # NOTE: FlextDomainService removed - exported from domain_services.py (not legacy)
    # Note: FlextEntity removed from public exports - conflicts with models.py
    # Use EntityLegacyFacade for legacy compatibility
    # Note: FlextEntityMixin removed from public exports - conflicts with mixins.py
    # Use EntityMixinLegacyFacade for legacy compatibility
    "FlextEventHandler",
    # =======================================================================
    # FIELDS LEGACY COMPATIBILITY - All field creation and management
    # =======================================================================
    # Field registry facade
    "FlextFieldRegistry",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    # Note: FlextIdentifiableMixin removed from public exports - conflicts with mixins.py
    # Use IdentifiableMixinLegacyFacade for legacy compatibility
    # =======================================================================
    # MIXIN LEGACY COMPATIBILITY - All original mixin class names
    # =======================================================================
    # Note: FlextLoggableMixin removed from public exports - conflicts with mixins.py
    # Use LoggableMixinLegacyFacade for legacy compatibility
    # Note: FlextLogger/FlextLoggerFactory removed from public exports - conflicts with loggings.py
    # Use LoggerLegacyFacade/LoggerFactoryLegacyFacade for legacy compatibility
    "FlextMetricsHandler",
    # =======================================================================
    # CORE MODEL LEGACY COMPATIBILITY - FlextModel and domain models
    # =======================================================================
    # Note: FlextModel removed from public exports - conflicts with models.py
    # Use ModelLegacyFacade for legacy compatibility
    # Note: FlextObservableMixin removed from public exports - conflicts with mixins.py
    # Use ObservableMixinLegacyFacade for legacy compatibility
    # =======================================================================
    # RESULT AND VALIDATION LEGACY COMPATIBILITY
    # =======================================================================
    # NOTE: FlextResult removed - exported from result.py only
    # Note: FlextSerializableMixin, FlextServiceMixin, FlextStateableMixin, FlextTimestampMixin removed from public exports - conflicts with mixins.py
    # Use SerializableMixinLegacyFacade, ServiceMixinLegacyFacade, StateableMixinLegacyFacade, TimestampMixinLegacyFacade for legacy compatibility
    # Note: FlextTimestampableMixin removed from public exports - conflicts with mixins.py
    # Use TimestampableMixinLegacyFacade for legacy compatibility
    # Note: FlextTimingMixin removed from public exports - conflicts with mixins.py
    # Use TimingMixinLegacyFacade for legacy compatibility
    # NOTE: FlextTypes removed - exported from typings.py only
    # Note: FlextValidatableMixin removed from public exports - conflicts with mixins.py
    # Use ValidatableMixinLegacyFacade for legacy compatibility
    "FlextValidatingHandler",
    # Note: FlextValueObject removed from public exports - conflicts with models.py
    # Use ValueObjectLegacyFacade for legacy compatibility
    # Note: FlextValueObjectMixin removed from public exports - conflicts with mixins.py
    # Use ValueObjectMixinLegacyFacade for legacy compatibility
    "HandlerLegacy",
    # Legacy facades
    "HandlersFacade",
    # Legacy management classes
    "MixinLegacy",
    "ProtocolLegacy",
    # Protocol compatibility
    # "QueryHandlerProtocol",  # Not implemented yet - removed
    "flext_create_boolean_field",
    "flext_create_datetime_field",
    "flext_create_email_field",
    "flext_create_float_field",
    "flext_create_integer_field",
    # Legacy field creation functions
    "flext_create_string_field",
    "flext_create_uuid_field",
    "flext_validate_config_key",
    # Validation functions
    "flext_validate_service_name",
    # Legacy registry access
    "get_field_registry",
    # Fields convenience function aliases (DEPRECATED)
    "create_field",
    "validate_field_value",
    "get_global_field_registry",
    "string_field",
    "integer_field",
    "float_field",
    "boolean_field",
    "email_field",
    "uuid_field",
    "datetime_field",
    # Legacy config functions
    "get_flext_config",
    "get_flext_container",
    "configure_flext_container",
    "get_typed",
    "register_typed",
    "create_module_container_utilities",
    "get_flext_settings",
    "merge_configs",
    "safe_get_env_var",
    "safe_load_json_file",
    # Logging
    "get_logger",
]


# =============================================================================
# MODEL LEGACY FACADES - FlextModel, FlextEntity, FlextValueObject compatibility
# =============================================================================


class ModelLegacy:
    """Centralized model compatibility facades following FLEXT patterns."""

    @staticmethod
    def get_flext_model() -> type[object] | None:
        """Get FlextModel class via lazy import."""
        return FlextModels.Model

    @staticmethod
    def get_flext_entity() -> type[object] | None:
        """Get FlextEntity class via lazy import."""
        return FlextModels.Entity

    @staticmethod
    def get_flext_value_object() -> type[object]:
        """Get FlextValueObject class via lazy import."""
        return FlextValue


# Model facades for direct access - ABI compatibility
def create_flext_model(*args: object, **kwargs: object) -> object:
    """Create FlextModel instance - maintains exact ABI."""
    model_class = ModelLegacy.get_flext_model()
    if model_class is None:
        error_message = "FlextModel class not available"
        raise ValueError(error_message)
    return model_class(*args, **kwargs)


def create_flext_entity(*args: object, **kwargs: object) -> object:
    """Create FlextEntity instance - maintains exact ABI."""
    entity_class = ModelLegacy.get_flext_entity()
    if entity_class is None:
        error_message = "FlextEntity class not available"
        raise ValueError(error_message)
    return entity_class(*args, **kwargs)


def create_flext_value_object(*args: object, **kwargs: object) -> object:
    """Create FlextValueObject instance - maintains exact ABI."""
    value_class = ModelLegacy.get_flext_value_object()
    return value_class(*args, **kwargs)


# Legacy factory functions - NOT conflicting aliases
# Note: Use actual classes from models.py instead of these factory functions
create_flext_model_legacy = create_flext_model
create_flext_entity_legacy = create_flext_entity
create_flext_value_object_legacy = create_flext_value_object


# =============================================================================
# ROOT MODELS LEGACY FACADES - Single class to multiple class compatibility
# =============================================================================


class RootModelsLegacy:
    """Legacy compatibility for old separate root model classes."""

    @staticmethod
    def get_flext_root_models_class() -> type[object]:
        """Get FlextRootModels class via lazy import (now FlextModels)."""
        return FlextModels

    @staticmethod
    def get_entity_id_class() -> type[object]:
        """Get EntityId class from hierarchical structure."""
        return FlextModels.EntityId

    @staticmethod
    def get_version_class() -> type[object]:
        """Get Version class from hierarchical structure."""
        return FlextModels.Version

    @staticmethod
    def get_timestamp_class() -> type[object]:
        """Get Timestamp class from hierarchical structure."""
        return FlextModels.Timestamp

    @staticmethod
    def get_metadata_class() -> type[object]:
        """Get Metadata class from hierarchical structure."""
        return FlextModels.Metadata

    @staticmethod
    def get_host_class() -> type[object]:
        """Get Host class from hierarchical structure."""
        return FlextModels.Host

    @staticmethod
    def get_port_class() -> type[object]:
        """Get Port class from hierarchical structure."""
        return FlextModels.Port

    @staticmethod
    def get_email_class() -> type[object]:
        """Get EmailAddress class from hierarchical structure."""
        return FlextModels.EmailAddress


# Legacy class aliases (deprecated but functional)
@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextEntityId",
    replacement="FlextModels.EntityId",
)
def FlextEntityId(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create EntityId instance (compatibility alias).

    Deprecated: Use FlextModels.EntityId directly.
    """
    entity_id_class = RootModelsLegacy.get_entity_id_class()
    return entity_id_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextVersion",
    replacement="FlextModels.Version",
)
def FlextVersion(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create Version instance (compatibility alias).

    Deprecated: Use FlextModels.Version directly.
    """
    version_class = RootModelsLegacy.get_version_class()
    return version_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextTimestamp",
    replacement="FlextModels.Timestamp",
)
def FlextTimestamp(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create Timestamp instance (compatibility alias).

    Deprecated: Use FlextModels.Timestamp directly.
    """
    timestamp_class = RootModelsLegacy.get_timestamp_class()
    return timestamp_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextMetadata",
    replacement="FlextModels.Metadata",
)
def FlextMetadata(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create Metadata instance (compatibility alias).

    Deprecated: Use FlextModels.Metadata directly.
    """
    metadata_class = RootModelsLegacy.get_metadata_class()
    return metadata_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextHost",
    replacement="FlextModels.Host",
)
def FlextHost(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create Host instance (compatibility alias).

    Deprecated: Use FlextModels.Host directly.
    """
    host_class = RootModelsLegacy.get_host_class()
    return host_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextPort",
    replacement="FlextModels.Port",
)
def FlextPort(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create Port instance (compatibility alias).

    Deprecated: Use FlextModels.Port directly.
    """
    port_class = RootModelsLegacy.get_port_class()
    return port_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="FlextEmailAddress",
    replacement="FlextModels.EmailAddress",
)
def FlextEmailAddress(*args: object, **kwargs: object) -> object:  # noqa: N802
    """Create EmailAddress instance (compatibility alias).

    Deprecated: Use FlextModels.EmailAddress directly.
    """
    email_class = RootModelsLegacy.get_email_class()
    return email_class(*args, **kwargs)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="create_version",
    replacement="FlextModels.create_version",
)
def create_version(value: int) -> FlextResult[object]:
    """Create version with validation (compatibility alias).

    Deprecated: Use FlextModels.create_version() directly.
    """
    result = FlextModels.create_version(value)
    return cast("FlextResult[object]", result)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="create_email",
    replacement="FlextModels.create_email",
)
def create_email(value: str) -> FlextResult[object]:
    """Create email address with validation (compatibility alias).

    Deprecated: Use FlextModels.create_email() directly.
    """
    result = FlextModels.create_email(value)
    return cast("FlextResult[object]", result)


# =============================================================================
# CONTAINER AND DI LEGACY FACADES
# =============================================================================


class ContainerLegacy:
    """Centralized container compatibility facades following FLEXT patterns."""

    @staticmethod
    def get_container_instance() -> FlextContainer:
        """Get FlextContainer instance via lazy import."""
        return FlextContainer.get_global()

    @staticmethod
    def get_container_class() -> type[object]:
        """Get FlextContainer class via lazy import."""
        return FlextContainer


# Container helper function aliases for backward compatibility
@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="get_flext_container",
    replacement="FlextContainer.get_global()",
)
def get_flext_container() -> FlextContainer:
    """Get global container instance (compatibility alias).

    Deprecated: Use FlextContainer.get_global() directly.
    """
    return FlextContainer.get_global()


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="configure_flext_container",
    replacement="FlextContainer.configure_global()",
)
def configure_flext_container(
    container: FlextContainer | None = None,
) -> FlextContainer:
    """Configure global container (compatibility alias).

    Deprecated: Use FlextContainer.configure_global() directly.
    """
    return FlextContainer.configure_global(container)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="get_typed",
    replacement="FlextContainer.get_global_typed()",
)
def get_typed[T](key: str, expected_type: type[T]) -> FlextResult[T]:
    """Get typed service from global container (compatibility alias).

    Deprecated: Use FlextContainer.get_global_typed() directly.
    """
    return FlextContainer.get_global_typed(key, expected_type)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="register_typed",
    replacement="FlextContainer.register_global()",
)
def register_typed(key: str, service: object) -> FlextResult[None]:
    """Register service in global container (compatibility alias).

    Deprecated: Use FlextContainer.register_global() directly.
    """
    return FlextContainer.register_global(key, service)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="create_module_container_utilities",
    new_path="FlextContainer.create_module_utilities",
    migration_guide="Use FlextContainer.create_module_utilities() instead.",
)
def create_module_container_utilities(module_name: str) -> dict[str, object]:
    """Create standardized container helpers for module (compatibility alias).

    Deprecated: Use FlextContainer.create_module_utilities() directly.
    """
    return FlextContainer.create_module_utilities(module_name)


# =============================================================================
# DOMAIN SERVICE LEGACY FACADE
# =============================================================================


class DomainServiceLegacy:
    """Domain service compatibility facade."""

    @staticmethod
    def get_domain_service() -> type[FlextDomainService[object]]:
        """Get FlextDomainService class via lazy import."""
        return FlextDomainService


# Domain service facade
# Note: FlextDomainService is already imported at top of file


# =============================================================================
# LOGGING LEGACY FACADES
# =============================================================================


class LoggingLegacy:
    """Centralized logging compatibility facades following FLEXT patterns."""

    @staticmethod
    def deprecation_warning(old_name: str, new_path: str) -> None:
        """Emit standardized deprecation warning for legacy logging usage."""
        warnings.warn(
            f"{old_name} is deprecated. Use {new_path} instead. "
            "See FLEXT migration guide for updated patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

    @staticmethod
    def get_logger_function() -> Callable[[str], object]:
        """Get get_logger function via lazy import."""
        return get_logger

    @staticmethod
    def get_logger_class() -> type[FlextLogger]:
        """Get FlextLogger class via lazy import."""
        return FlextLogger

    @staticmethod
    def get_logger_factory() -> type[object]:
        """Get FlextLogger class via lazy import."""
        # FlextLoggerFactory was removed, use FlextLogger directly
        return FlextLogger


# Logging compatibility classes - deprecated, use FlextLogger directly


class FlextLoggerFactory:
    """DEPRECATED: Use FlextLogger() directly instead."""

    @staticmethod
    def get_logger(name: str | None = None, level: str = "INFO") -> object:
        """DEPRECATED: Use FlextLogger() directly instead."""
        LoggingLegacy.deprecation_warning(
            "FlextLoggerFactory.get_logger", "FlextLogger"
        )
        return FlextLogger(name or "flext", level)

    @staticmethod
    def set_global_level(level: str) -> None:  # noqa: ARG004
        """DEPRECATED: Set level on individual FlextLogger instances instead."""
        LoggingLegacy.deprecation_warning(
            "FlextLoggerFactory.set_global_level", "FlextLogger.set_level"
        )
        # No-op for compatibility - level parameter kept for backward compatibility

    @staticmethod
    def clear_loggers() -> None:
        """DEPRECATED: No replacement needed."""
        LoggingLegacy.deprecation_warning(
            "FlextLoggerFactory.clear_loggers", "Direct FlextLogger usage"
        )
        # No-op for compatibility


class FlextLoggings:
    """DEPRECATED: Use FlextLogger() directly instead."""

    Factory = FlextLoggerFactory  # Alias for compatibility


class FlextLogContextManager:
    """DEPRECATED: Use FlextLogger.with_context() instead."""

    def __init__(self, logger: object, **context: object) -> None:
        """DEPRECATED: Use FlextLogger.set_request_context() instead."""
        LoggingLegacy.deprecation_warning(
            "FlextLogContextManager", "FlextLogger.set_request_context"
        )
        if isinstance(logger, FlextLogger):
            # Use set_request_context instead of non-existent with_context
            logger.set_request_context(**context)
            self._logger = logger
        else:
            self._logger = FlextLogger("deprecated")

    def __enter__(self) -> object:
        """Enter context manager."""
        return self._logger

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""


# Note: get_logger is defined earlier in the file with proper signature


def create_log_context(logger: object, **context: object) -> object:
    """DEPRECATED: Use FlextLogger.with_context() instead."""
    LoggingLegacy.deprecation_warning("create_log_context", "FlextLogger.with_context")
    return FlextLogContextManager(logger, **context)


# =============================================================================
# FIELDS LEGACY COMPATIBILITY LAYER
# =============================================================================


class FieldsLegacy:
    """Centralized fields compatibility facades following FLEXT patterns.

    Provides minimal facades for all legacy field exports, maintaining ABI
    compatibility while delegating to the new consolidated FlextFields architecture.

    All facades are minimal orchestration-only with deprecation warnings.
    """

    @staticmethod
    def deprecation_warning(old_name: str, new_path: str) -> None:
        """Emit standardized deprecation warning for legacy field usage."""
        warnings.warn(
            f"{old_name} is deprecated. Use {new_path} instead. "
            "See FLEXT migration guide for updated field patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

    class FieldRegistryLegacyFacade:
        """Minimal facade for FlextFieldRegistry legacy compatibility."""

        def __init__(self) -> None:
            FieldsLegacy.deprecation_warning(
                "FlextFieldRegistry", "FlextFields.Registry.FieldRegistry"
            )
            # Direct usage - NO lazy import per FLEXT requirements

            self._impl = FlextFields.Registry.FieldRegistry()

        def __getattr__(self, name: str) -> object:
            """Delegate all attributes to implementation."""
            return getattr(self._impl, name)

    @staticmethod
    def _convert_field_config(
        config: dict[str, object], valid_keys: set[str]
    ) -> dict[str, object]:
        """Convert legacy field config to proper types for type safety.

        This helper extracts only valid keys and passes values as-is to avoid
        the need for extensive type: ignore annotations.
        """
        return {key: value for key, value in config.items() if key in valid_keys}

    @staticmethod
    def create_string_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.StringField:
        """Legacy string field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_string_field",
            "FlextFields.Core.StringField or string_field()",
        )
        # Convert config to proper keyword arguments for FlextFields.Core.StringField using runtime type validation
        # Build kwargs with proper type annotations for the union types expected
        field_kwargs: dict[str, bool | str | int | None] = {}

        # Handle each parameter individually with proper types
        for key, value in config.items():
            if key == "required" and isinstance(value, bool):
                field_kwargs["required"] = value
            elif key == "default" and (isinstance(value, str) or value is None):
                field_kwargs["default"] = value
            elif key == "description" and isinstance(value, str):
                field_kwargs["description"] = value
            elif key == "min_length" and (isinstance(value, int) or value is None):
                field_kwargs["min_length"] = value
            elif key == "max_length" and (isinstance(value, int) or value is None):
                field_kwargs["max_length"] = value
            elif key == "pattern" and (isinstance(value, str) or value is None):
                field_kwargs["pattern"] = value

        # Use explicit parameters to avoid type issues
        return FlextFields.Core.StringField(
            name=name,
            required=bool(field_kwargs.get("required", True)),
            default=str(field_kwargs["default"])
            if field_kwargs.get("default") is not None
            else None,
            description=str(field_kwargs.get("description", "")),
            min_length=int(cast("int", field_kwargs["min_length"]))
            if field_kwargs.get("min_length") is not None
            else None,
            max_length=int(cast("int", field_kwargs["max_length"]))
            if field_kwargs.get("max_length") is not None
            else None,
            pattern=str(field_kwargs["pattern"])
            if field_kwargs.get("pattern") is not None
            else None,
        )

    @staticmethod
    def create_integer_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.IntegerField:
        """Legacy integer field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_integer_field",
            "FlextFields.Core.IntegerField or integer_field()",
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        # Build kwargs with proper type annotations for the union types expected
        field_kwargs: dict[str, bool | int | str | None] = {}

        # Handle each parameter individually with proper types
        for key, value in config.items():
            if key == "required" and isinstance(value, bool):
                field_kwargs["required"] = value
            elif key == "default" and (isinstance(value, int) or value is None):
                field_kwargs["default"] = value
            elif key == "description" and isinstance(value, str):
                field_kwargs["description"] = value
            elif key == "min_value" and (isinstance(value, int) or value is None):
                field_kwargs["min_value"] = value
            elif key == "max_value" and (isinstance(value, int) or value is None):
                field_kwargs["max_value"] = value

        # Use explicit parameters to avoid type issues
        return FlextFields.Core.IntegerField(
            name=name,
            required=bool(field_kwargs.get("required", True)),
            default=int(cast("int", field_kwargs["default"]))
            if field_kwargs.get("default") is not None
            else None,
            description=str(field_kwargs.get("description", "")),
            min_value=int(cast("int", field_kwargs["min_value"]))
            if field_kwargs.get("min_value") is not None
            else None,
            max_value=int(cast("int", field_kwargs["max_value"]))
            if field_kwargs.get("max_value") is not None
            else None,
        )

    @staticmethod
    def create_boolean_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.BooleanField:
        """Legacy boolean field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_boolean_field",
            "FlextFields.Core.BooleanField or boolean_field()",
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        # Build kwargs with proper type annotations for the union types expected
        field_kwargs: dict[str, bool | str | None] = {}

        # Handle each parameter individually with proper types
        for key, value in config.items():
            if key == "required" and isinstance(value, bool):
                field_kwargs["required"] = value
            elif key == "default" and (isinstance(value, bool) or value is None):
                field_kwargs["default"] = value
            elif key == "description" and isinstance(value, str):
                field_kwargs["description"] = value

        # Use explicit parameters to avoid type issues
        return FlextFields.Core.BooleanField(
            name=name,
            required=bool(field_kwargs.get("required", True)),
            default=bool(field_kwargs["default"])
            if field_kwargs.get("default") is not None
            else None,
            description=str(field_kwargs.get("description", "")),
        )

    @staticmethod
    def create_email_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.EmailField:
        """Legacy email field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_email_field", "FlextFields.Core.EmailField or email_field()"
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        # Build kwargs with proper type annotations for the union types expected
        field_kwargs: dict[str, bool | str | list[object] | None] = {}

        # Handle each parameter individually with proper types
        for key, value in config.items():
            if key == "required" and isinstance(value, bool):
                field_kwargs["required"] = value
            elif key == "default" and (isinstance(value, str) or value is None):
                field_kwargs["default"] = value
            elif key == "description" and isinstance(value, str):
                field_kwargs["description"] = value
            elif key == "domain_whitelist" and (
                isinstance(value, list) or value is None
            ):
                field_kwargs["domain_whitelist"] = value

        # Use explicit parameters to avoid type issues
        return FlextFields.Core.EmailField(
            name=name,
            required=bool(field_kwargs.get("required", True)),
            default=str(field_kwargs["default"])
            if field_kwargs.get("default") is not None
            else None,
            description=str(field_kwargs.get("description", "")),
        )

    @staticmethod
    def create_uuid_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.UuidField:
        """Legacy UUID field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_uuid_field", "FlextFields.Core.UuidField or uuid_field()"
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        # Build kwargs with proper type annotations for the union types expected
        field_kwargs: dict[str, bool | str | None] = {}

        # Handle each parameter individually with proper types
        for key, value in config.items():
            if key == "required" and isinstance(value, bool):
                field_kwargs["required"] = value
            elif key == "default" and (isinstance(value, str) or value is None):
                field_kwargs["default"] = value
            elif key == "description" and isinstance(value, str):
                field_kwargs["description"] = value

        # Use explicit parameters to avoid type issues
        return FlextFields.Core.UuidField(
            name=name,
            required=bool(field_kwargs.get("required", True)),
            default=str(field_kwargs["default"])
            if field_kwargs.get("default") is not None
            else None,
            description=str(field_kwargs.get("description", "")),
        )

    @staticmethod
    def create_datetime_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.DateTimeField:
        """Legacy datetime field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_datetime_field",
            "FlextFields.Core.DateTimeField or datetime_field()",
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        field_config = {
            key: value
            for key, value in config.items()
            if key
            in {
                "required",
                "default",
                "description",
                "auto_now",
                "auto_now_add",
            }
            and (
                (key == "required" and isinstance(value, bool))
                or (key in {"auto_now", "auto_now_add"} and isinstance(value, bool))
                or (key == "description" and isinstance(value, str))
            )
        }
        # Use explicit parameters to avoid type issues
        return FlextFields.Core.DateTimeField(
            name=name,
            required=bool(field_config.get("required", True)),
            description=str(field_config.get("description", "")),
        )

    @staticmethod
    def create_float_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.FloatField:
        """Legacy float field creation - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "flext_create_float_field", "FlextFields.Core.FloatField or float_field()"
        )
        # Convert config to proper keyword arguments using runtime type validation for ABI
        field_config = {
            key: value
            for key, value in config.items()
            if key
            in {
                "required",
                "default",
                "description",
                "min_value",
                "max_value",
                "precision",
            }
            and (
                (key == "required" and isinstance(value, bool))
                or (key == "default" and (isinstance(value, float) or value is None))
                or (key == "description" and isinstance(value, str))
                or (
                    key in {"min_value", "max_value"}
                    and (isinstance(value, float) or value is None)
                )
                or (key == "precision" and (isinstance(value, int) or value is None))
            )
        }
        # Use explicit parameters to avoid type issues
        return FlextFields.Core.FloatField(
            name=name,
            required=bool(field_config.get("required", True)),
            default=float(cast("float", field_config["default"]))
            if field_config.get("default") is not None
            else None,
            description=str(field_config.get("description", "")),
            min_value=float(cast("float", field_config["min_value"]))
            if field_config.get("min_value") is not None
            else None,
            max_value=float(cast("float", field_config["max_value"]))
            if field_config.get("max_value") is not None
            else None,
            precision=int(cast("int", field_config["precision"]))
            if field_config.get("precision") is not None
            else None,
        )

    @staticmethod
    def get_field_registry_legacy() -> (
        object
    ):  # Legacy compatibility with dynamic typing
        """Legacy field registry access - DEPRECATED."""
        FieldsLegacy.deprecation_warning(
            "get_field_registry",
            "get_global_field_registry() or FlextFields.Registry.FieldRegistry()",
        )
        return FlextFields.Registry.FieldRegistry()


# =============================================================================
# FIELDS LEGACY EXPORTS - Maintain exact ABI compatibility
# =============================================================================

# Field registry facade
FlextFieldRegistry = FieldsLegacy.FieldRegistryLegacyFacade

# Legacy field creation functions
flext_create_string_field = FieldsLegacy.create_string_field_legacy
flext_create_integer_field = FieldsLegacy.create_integer_field_legacy
flext_create_boolean_field = FieldsLegacy.create_boolean_field_legacy
flext_create_email_field = FieldsLegacy.create_email_field_legacy
flext_create_uuid_field = FieldsLegacy.create_uuid_field_legacy
flext_create_datetime_field = FieldsLegacy.create_datetime_field_legacy
flext_create_float_field = FieldsLegacy.create_float_field_legacy

# Legacy registry access
get_field_registry = FieldsLegacy.get_field_registry_legacy


# Fields convenience function aliases
@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="create_field",
    replacement="FlextFields.Factory.create_field",
)
def create_field(field_type: str, name: str, **config: object) -> object:
    """DEPRECATED: Use FlextFields.Factory.create_field instead."""
    return FlextFields.Factory.create_field(field_type, name, **config)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="validate_field_value",
    replacement="FlextFields.Validation.validate_field",
)
def validate_field_value(field: object, value: object) -> object:
    """DEPRECATED: Use FlextFields.Validation.validate_field instead."""
    # Legacy compatibility - basic validation since field typing is complex
    if field is None:
        msg = "Field cannot be None"
        raise ValueError(msg)
    return value


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="get_global_field_registry",
    replacement="FlextFields.Registry.FieldRegistry()",
)
def get_global_field_registry() -> object:
    """DEPRECATED: Use FlextFields.Registry.FieldRegistry() instead."""
    return FlextFields.Registry.FieldRegistry()


# Field builder shortcuts
@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="string_field",
    new_path="FlextFields.Factory.FieldBuilder('string', name)",
)
def string_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('string', name) instead."""
    return FlextFields.Factory.FieldBuilder("string", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="integer_field",
    new_path="FlextFields.Factory.FieldBuilder('integer', name)",
)
def integer_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('integer', name) instead."""
    return FlextFields.Factory.FieldBuilder("integer", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="float_field",
    new_path="FlextFields.Factory.FieldBuilder('float', name)",
)
def float_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('float', name) instead."""
    return FlextFields.Factory.FieldBuilder("float", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="boolean_field",
    new_path="FlextFields.Factory.FieldBuilder('boolean', name)",
)
def boolean_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('boolean', name) instead."""
    return FlextFields.Factory.FieldBuilder("boolean", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="email_field",
    new_path="FlextFields.Factory.FieldBuilder('email', name)",
)
def email_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('email', name) instead."""
    return FlextFields.Factory.FieldBuilder("email", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="uuid_field",
    new_path="FlextFields.Factory.FieldBuilder('uuid', name)",
)
def uuid_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('uuid', name) instead."""
    return FlextFields.Factory.FieldBuilder("uuid", name)


@FlextDecorators.Lifecycle.deprecated_legacy_function(
    old_name="datetime_field",
    new_path="FlextFields.Factory.FieldBuilder('datetime', name)",
)
def datetime_field(name: str) -> object:
    """DEPRECATED: Use FlextFields.Factory.FieldBuilder('datetime', name) instead."""
    return FlextFields.Factory.FieldBuilder("datetime", name)


# =============================================================================
# DECORATORS LEGACY FACADES
# =============================================================================


class DecoratorsLegacy:
    """Decorators compatibility facades for maintaining ABI compatibility.

    Provides backward compatibility for all decorator functions that might
    have been imported directly before the hierarchical FlextDecorators class.
    """

    @staticmethod
    def safe_result_legacy(
        func: Callable[[object], object],
    ) -> Callable[[object], object]:
        """Legacy facade for safe_result decorator."""
        return FlextDecorators.Reliability.safe_result(func)

    @staticmethod
    def retry_legacy(
        max_attempts: int = 3,
        backoff_factor: float = 1.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Legacy facade for retry decorator."""
        return FlextDecorators.Reliability.retry(
            max_attempts=max_attempts,
            backoff_factor=backoff_factor,
            exceptions=exceptions,
        )

    @staticmethod
    def timeout_legacy(
        seconds: float,
        error_message: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Legacy facade for timeout decorator."""
        return FlextDecorators.Reliability.timeout(
            seconds=seconds, error_message=error_message
        )

    @staticmethod
    def validate_input_legacy(
        validator: Callable[[object], bool],
        error_message: str = "Input validation failed",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Legacy facade for validate_input decorator."""
        return FlextDecorators.Validation.validate_input(
            validator=validator, error_message=error_message
        )

    @staticmethod
    def validate_types_legacy(
        arg_types: list[type] | None = None,
        return_type: type | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Legacy facade for validate_types decorator."""
        return FlextDecorators.Validation.validate_types(
            arg_types=arg_types, return_type=return_type
        )

    @staticmethod
    def monitor_legacy(
        threshold: float | None = None,
        *,
        log_slow: bool = True,
        collect_metrics: bool = False,
    ) -> FlextTypes.Meta.MethodDecorator[object]:
        """Legacy facade for monitor decorator."""
        if threshold is None:
            threshold = FlextConstants.Performance.SLOW_QUERY_THRESHOLD

        return FlextDecorators.Performance.monitor(
            threshold=threshold, log_slow=log_slow, collect_metrics=collect_metrics
        )

    @staticmethod
    def cache_legacy(
        ttl: int | None = None, max_size: int = 128
    ) -> FlextTypes.Meta.MethodDecorator[object]:
        """Legacy facade for cache decorator."""
        if ttl is None:
            ttl = FlextConstants.Performance.CACHE_TTL

        return FlextDecorators.Performance.cache(ttl=ttl, max_size=max_size)

    @staticmethod
    def log_execution_legacy(
        *, include_args: bool = False, include_result: bool = True
    ) -> FlextTypes.Meta.MethodDecorator[object]:
        """Legacy facade for log_execution decorator."""
        return FlextDecorators.Observability.log_execution(
            include_args=include_args, include_result=include_result
        )

    @staticmethod
    def deprecated_legacy(
        message: str | None = None,
        version: str | None = None,
        removal_version: str | None = None,
    ) -> FlextTypes.Meta.MethodDecorator[object]:
        """Legacy facade for deprecated decorator."""
        return FlextDecorators.Lifecycle.deprecated(
            reason=message, version=version, removal_version=removal_version
        )


# =============================================================================
# DECORATORS LEGACY FUNCTION FACADES
# =============================================================================

# Reliability decorators
safe_result = DecoratorsLegacy.safe_result_legacy
retry = DecoratorsLegacy.retry_legacy
timeout = DecoratorsLegacy.timeout_legacy

# Validation decorators
validate_input = DecoratorsLegacy.validate_input_legacy
validate_types = DecoratorsLegacy.validate_types_legacy

# Performance decorators
monitor = DecoratorsLegacy.monitor_legacy
cache = DecoratorsLegacy.cache_legacy

# Observability decorators
log_execution = DecoratorsLegacy.log_execution_legacy

# Lifecycle decorators
deprecated = DecoratorsLegacy.deprecated_legacy

# Legacy alias names that might have been used
flext_safe_result = safe_result
flext_retry = retry
flext_timeout = timeout
flext_validate_input = validate_input
flext_validate_types = validate_types
flext_monitor = monitor
flext_cache = cache
flext_log_execution = log_execution
flext_deprecated = deprecated

# Additional decorators compatibility aliases
FlextDecoratorUtils = DecoratorsLegacy  # Alias for compatibility
FlextFunctionalDecorators = FlextDecorators  # Functional decorators alias
FlextImmutabilityDecorators = FlextDecorators  # Immutability decorators alias
FlextLoggingDecorators = FlextDecorators  # Logging decorators alias
FlextPerformanceDecorators = FlextDecorators  # Performance decorators alias
FlextValidationDecorators = FlextDecorators  # Validation decorators alias
FlextCallable = FlextTypes.Core.FlextCallableType  # Callable type alias
FlextDecoratedFunction = (
    FlextTypes.Core.FlextCallableType
)  # Decorated function type alias
# Base decorator implementation aliases for legacy compatibility
_BaseDecoratorFactory = object  # Base factory placeholder
_BaseImmutabilityDecorators = object  # Base immutability placeholder
_decorators_base = object  # Base decorators module placeholder

# Field system aliases for legacy compatibility
FlextFieldCore = FlextFields  # Core field alias
FlextFieldMetadata = FlextFields  # Metadata alias


# FlextFieldType enum-like class for backward compatibility
class FlextFieldType:
    """Field type constants for legacy compatibility."""

    STRING = FlextConstants.Legacy.FIELD_TYPE_STRING
    INTEGER = FlextConstants.Legacy.FIELD_TYPE_INTEGER
    BOOLEAN = FlextConstants.Legacy.FIELD_TYPE_BOOLEAN
    FLOAT = FlextConstants.Legacy.FIELD_TYPE_FLOAT
    EMAIL = FlextConstants.Legacy.FIELD_TYPE_EMAIL
    UUID = FlextConstants.Legacy.FIELD_TYPE_UUID
    DATETIME = FlextConstants.Legacy.FIELD_TYPE_DATETIME


# =============================================================================
# SERVICES LEGACY COMPATIBILITY
# =============================================================================


@FlextDecorators.Lifecycle.deprecated_class_warning(
    class_name="FlextServiceProcessor",
    replacement="FlextServices.ServiceProcessor",
)
class FlextServiceProcessor:
    """Legacy FlextServiceProcessor for backward compatibility.

    DEPRECATED: Use FlextServices.ServiceProcessor instead.
    """

    def __init__(self, service_name: str | None = None) -> None:
        # Initialize FlextMixins functionality
        FlextMixins.initialize_validation(self)
        self._service_name = service_name or self.__class__.__name__
        self._performance_tracker = FlextPerformance()
        self._correlation_generator = FlextUtilities()

    def get_service_name(self) -> str:
        """Get service name."""
        return self._service_name

    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation using FlextMixins."""
        FlextMixins.log_operation(self, operation, **kwargs)

    def log_info(self, message: str, **kwargs: object) -> None:
        """Log info using FlextMixins."""
        FlextMixins.log_info(self, message, **kwargs)

    def initialize_service(self) -> FlextResult[None]:
        """Initialize service."""
        return FlextResult[None].ok(None)

    def process(self, request: object) -> FlextResult[object]:
        """Process request - must be implemented by subclasses."""
        msg = "Subclasses must implement process method"
        raise NotImplementedError(msg)

    def build(self, domain: object, *, correlation_id: str) -> object:
        """Build result - must be implemented by subclasses."""
        msg = "Subclasses must implement build method"
        raise NotImplementedError(msg)


# Update __all__ to include the new facades
__all__ += [  # noqa: RUF022
    # Legacy mixin facades with distinct names
    "LoggableMixinLegacyFacade",
    "TimestampMixinLegacyFacade",
    "IdentifiableMixinLegacyFacade",
    "ValidatableMixinLegacyFacade",
    "SerializableMixinLegacyFacade",
    "TimingMixinLegacyFacade",
    "ComparableMixinLegacyFacade",
    "EntityMixinLegacyFacade",
    "ValueObjectMixinLegacyFacade",
    "ContainerLegacy",
    "DecoratorsLegacy",
    "DomainServiceLegacy",
    "FlextServiceProcessor",
    "LoggingLegacy",
    "ModelLegacy",
    "ValidationLegacy",
    "cache",
    "deprecated",
    "flext_cache",
    "flext_deprecated",
    "flext_log_execution",
    "flext_monitor",
    "flext_retry",
    # Prefixed aliases
    "flext_safe_result",
    "flext_timeout",
    "flext_validate_input",
    "flext_validate_types",
    "log_execution",
    "monitor",
    "retry",
    # Legacy decorator functions
    "safe_result",
    "timeout",
    "validate_input",
    "validate_types",
]


# =============================================================================
# EXCEPTIONS LEGACY - Exception generation and compatibility
# =============================================================================

# =============================================================================
# DYNAMIC EXCEPTION GENERATION (DRY implementation)
# =============================================================================

# LEGACY COMMENT: Dynamic exception generation removed in favor of real subclasses
# The FlextExceptions now uses explicit class hierarchy with proper inheritance
# All exception types are now real classes defined directly in exceptions.py


# =============================================================================
# EXCEPTION METRICS AND MONITORING
# =============================================================================


def get_exception_metrics() -> dict[str, int]:
    """Get exception occurrence metrics."""
    metrics = FlextExceptions.get_metrics()
    # Convert to int values for type compatibility
    result: dict[str, int] = {}
    for k, v in metrics.items():
        try:
            result[k] = int(v)
        except (TypeError, ValueError):
            result[k] = 0
    return result


def clear_exception_metrics() -> None:
    """Clear exception metrics."""
    FlextExceptions.clear_metrics()


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# Base classes compatibility
FlextErrorMixin = FlextExceptions.BaseError


# Runtime: Use FlextExceptions modern API for backward compatibility
# Use getattr because these are dynamically created at runtime
# Type checkers can't see them, but they exist after FlextExceptions.initialize()
FlextError = getattr(FlextExceptions, "Error", RuntimeError)
FlextValidationError = getattr(FlextExceptions, "ValidationError", ValueError)
FlextConfigurationError = getattr(FlextExceptions, "ConfigurationError", ValueError)
FlextOperationError = getattr(FlextExceptions, "OperationError", RuntimeError)
FlextTypeError = getattr(FlextExceptions, "TypeError", TypeError)
FlextAttributeError = getattr(FlextExceptions, "AttributeError", AttributeError)
FlextProcessingError = getattr(FlextExceptions, "ProcessingError", RuntimeError)
FlextTimeoutError = getattr(FlextExceptions, "TimeoutError", TimeoutError)
FlextNotFoundError = getattr(FlextExceptions, "NotFoundError", FileNotFoundError)
FlextAlreadyExistsError = getattr(
    FlextExceptions, "AlreadyExistsError", FileExistsError
)
FlextPermissionError = getattr(FlextExceptions, "PermissionError", PermissionError)
FlextAuthenticationError = getattr(
    FlextExceptions, "AuthenticationError", PermissionError
)
FlextCriticalError = getattr(FlextExceptions, "CriticalError", SystemError)
FlextUserError = getattr(FlextExceptions, "UserError", TypeError)
FlextConnectionError = getattr(FlextExceptions, "ConnectionError", ConnectionError)

# Update __all__ to include exception legacy exports
__all__ += [  # noqa: RUF022
    # Legacy exception aliases for backward compatibility
    "FlextError",
    "FlextValidationError",
    "FlextConfigurationError",
    "FlextOperationError",
    "FlextTypeError",
    "FlextAttributeError",
    "FlextProcessingError",
    "FlextTimeoutError",
    "FlextNotFoundError",
    "FlextAlreadyExistsError",
    "FlextPermissionError",
    "FlextAuthenticationError",
    "FlextCriticalError",
    "FlextUserError",
    "FlextConnectionError",
    # Factory functions
    "clear_exception_metrics",
    "get_exception_metrics",
]


# =============================================================================
# EXCEPTION BACKWARD COMPATIBILITY
# =============================================================================

# Exception aliases are already defined above


# =============================================================================
# PROTOCOL ALIASES BACKWARD COMPATIBILITY - Migrated from observability.py
# =============================================================================


def _issue_protocol_deprecation_warning(old_name: str, new_path: str) -> None:
    """Issue deprecation warning for protocol aliases."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_path} directly from FlextProtocols hierarchy.",
        DeprecationWarning,
        stacklevel=3,
    )


class ProtocolAliasesLegacy:
    """Legacy protocol aliases migrated from observability.py.

    DEPRECATED: Use FlextProtocols hierarchy directly instead.

    This class provides backward compatibility for protocol aliases that were
    scattered across different modules. All aliases now delegate to the centralized
    FlextProtocols hierarchy following FLEXT architectural patterns.
    """

    @classmethod
    def get_observability_protocol(
        cls,
    ) -> type[object]:
        """Legacy alias for ObservabilityProtocol.

        DEPRECATED: Use FlextProtocols.Infrastructure.Configurable directly.
        """
        _issue_protocol_deprecation_warning(
            "ObservabilityProtocol", "FlextProtocols.Infrastructure.Configurable"
        )
        return cast("type[object]", FlextProtocols.Infrastructure.Configurable)

    @classmethod
    def get_metrics_collector_protocol(
        cls,
    ) -> type[object]:
        """Legacy alias for MetricsCollectorProtocol.

        DEPRECATED: Use FlextProtocols.Extensions.Observability directly.
        """
        _issue_protocol_deprecation_warning(
            "MetricsCollectorProtocol", "FlextProtocols.Extensions.Observability"
        )
        return cast("type[object]", FlextProtocols.Extensions.Observability)

    @classmethod
    def get_logger_service_protocol(
        cls,
    ) -> type[object]:
        """Legacy alias for LoggerServiceProtocol.

        DEPRECATED: Use FlextProtocols.Infrastructure.LoggerProtocol directly.
        """
        _issue_protocol_deprecation_warning(
            "LoggerServiceProtocol", "FlextProtocols.Infrastructure.LoggerProtocol"
        )
        return cast("type[object]", FlextProtocols.Infrastructure.LoggerProtocol)


# Direct aliases for backward compatibility (with deprecation warnings)
ObservabilityProtocol = FlextProtocols.Infrastructure.Configurable
MetricsCollectorProtocol = FlextProtocols.Extensions.Observability
LoggerServiceProtocol = FlextProtocols.Infrastructure.LoggerProtocol

# Add to exports
__all__ += [
    "LoggerServiceProtocol",
    "MetricsCollectorProtocol",
    "ObservabilityProtocol",
    "ProtocolAliasesLegacy",
]


# =============================================================================
# UTILITIES STANDALONE FUNCTIONS - Migrated from utilities.py
# =============================================================================


def _issue_utilities_deprecation_warning(function_name: str, new_path: str) -> None:
    """Issue deprecation warning for standalone utility functions."""
    warnings.warn(
        f"{function_name}() is deprecated. Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Legacy standalone functions - delegate to FlextUtilities hierarchy
def generate_correlation_id() -> str:
    """Generate correlation ID for tracing.

    DEPRECATED: Use FlextUtilities.Generators.generate_correlation_id() instead.
    """
    _issue_utilities_deprecation_warning(
        "generate_correlation_id", "FlextUtilities.Generators.generate_correlation_id()"
    )
    return FlextUtilities.Generators.generate_correlation_id()


def generate_id() -> str:
    """Generate generic ID.

    DEPRECATED: Use FlextUtilities.Generators.generate_id() instead.
    """
    _issue_utilities_deprecation_warning(
        "generate_id", "FlextUtilities.Generators.generate_id()"
    )
    return FlextUtilities.Generators.generate_id()


def generate_uuid() -> str:
    """Generate UUID string.

    DEPRECATED: Use FlextUtilities.Generators.generate_uuid() instead.
    """
    _issue_utilities_deprecation_warning(
        "generate_uuid", "FlextUtilities.Generators.generate_uuid()"
    )
    return FlextUtilities.Generators.generate_uuid()


def flext_safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Safe integer conversion with optional default.

    DEPRECATED: Use FlextUtilities.safe_int_conversion() instead.
    """
    _issue_utilities_deprecation_warning(
        "flext_safe_int_conversion", "FlextUtilities.safe_int_conversion()"
    )
    return FlextUtilities.safe_int_conversion(value, default)


def safe_int_conversion_with_default(value: object, default: int) -> int:
    """Safe integer conversion with mandatory default.

    DEPRECATED: Use FlextUtilities.safe_int_conversion_with_default() instead.
    """
    _issue_utilities_deprecation_warning(
        "safe_int_conversion_with_default",
        "FlextUtilities.safe_int_conversion_with_default()",
    )
    return FlextUtilities.safe_int_conversion_with_default(value, default)


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix.

    DEPRECATED: Use FlextUtilities.TextProcessor.truncate() instead.
    """
    _issue_utilities_deprecation_warning(
        "truncate", "FlextUtilities.TextProcessor.truncate()"
    )
    return FlextUtilities.TextProcessor.truncate(text, max_length, suffix)


def is_not_none(value: object) -> bool:
    """Check if value is not None.

    DEPRECATED: Use FlextUtilities.TypeGuards.is_not_none() instead.
    """
    _issue_utilities_deprecation_warning(
        "is_not_none", "FlextUtilities.TypeGuards.is_not_none()"
    )
    return FlextUtilities.TypeGuards.is_not_none(value)


def flext_clear_performance_metrics() -> None:
    """Clear performance metrics.

    DEPRECATED: Use FlextUtilities.Performance methods instead.
    """
    _issue_utilities_deprecation_warning(
        "flext_clear_performance_metrics", "FlextUtilities.Performance methods"
    )
    # Method doesn't exist in actual implementation - no-op for compatibility


def generate_iso_timestamp() -> str:
    """Generate ISO format timestamp.

    DEPRECATED: Use FlextUtilities.generate_iso_timestamp() instead.
    """
    _issue_utilities_deprecation_warning(
        "generate_iso_timestamp", "FlextUtilities.generate_iso_timestamp()"
    )
    return FlextUtilities.generate_iso_timestamp()


def flext_track_performance(
    category: str,
) -> FlextTypes.Meta.MethodDecorator[object]:
    """Track performance decorator.

    DEPRECATED: Use FlextUtilities.Performance.track_performance() instead.
    """
    _issue_utilities_deprecation_warning(
        "flext_track_performance", "FlextUtilities.Performance.track_performance()"
    )
    return FlextUtilities.Performance.track_performance(category)


# =============================================================================
# CONTAINER LEGACY FACADES - FlextContainer backward compatibility
# =============================================================================


# Container class facades - restored for test compatibility
FlextServiceKey = FlextContainer.ServiceKey
FlextServiceRegistrar = FlextContainer.ServiceRegistrar
FlextServiceRetriever = FlextContainer.ServiceRetriever


# Legacy command/query facades (simplified for test compatibility)
class RegisterServiceCommand:
    """Legacy command facade for service registration."""

    def __init__(self, name: str, service: object) -> None:
        self.name = name
        self.service_name = name  # Legacy interface compatibility
        self.service = service
        self.service_instance = service  # Legacy interface compatibility
        self.command_type = "register_service"  # Legacy interface compatibility
        self.command_id = generate_uuid()
        self.correlation_id = generate_correlation_id()
        self.timestamp = time.time()

    @classmethod
    def create(cls, name: str, service: object) -> RegisterServiceCommand:
        """Create service registration command."""
        return cls(name, service)

    def validate_command(self) -> FlextResult[None]:
        """Validate the command."""
        if not self.name.strip():
            return FlextResult[None].fail("Service name cannot be empty")
        return FlextResult[None].ok(None)


class RegisterFactoryCommand:
    """Legacy command facade for factory registration."""

    def __init__(self, name: str, factory: object) -> None:
        self.name = name
        self.service_name = name  # Legacy interface compatibility
        self.factory = factory
        self.command_type = "register_factory"  # Legacy interface compatibility

    @classmethod
    def create(cls, name: str, factory: object) -> RegisterFactoryCommand:
        """Create factory registration command."""
        return cls(name, factory)

    def validate_command(self) -> FlextResult[None]:
        """Validate the command."""
        if not self.name.strip():
            return FlextResult[None].fail("Factory name cannot be empty")
        # Runtime validation for legacy test compatibility
        if not callable(self.factory):
            return FlextResult[None].fail("Factory must be callable")
        return FlextResult[None].ok(None)


class UnregisterServiceCommand:
    """Legacy command facade for service unregistration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.service_name = name  # Legacy interface compatibility
        self.command_type = "unregister_service"  # Legacy interface compatibility

    @classmethod
    def create(cls, name: str) -> UnregisterServiceCommand:
        """Create service unregistration command."""
        return cls(name)

    def validate_command(self) -> FlextResult[None]:
        """Validate the command."""
        if not self.name.strip():
            return FlextResult[None].fail("Service name cannot be empty")
        return FlextResult[None].ok(None)


class GetServiceQuery:
    """Legacy query facade for service retrieval."""

    def __init__(self, name: str, service_type: str = "") -> None:
        self.name = name
        self.service_name = name  # Legacy interface compatibility
        self.service_type = service_type
        self.expected_type = service_type  # Legacy interface compatibility
        self.query_type = "get_service"  # Legacy interface compatibility

    @classmethod
    def create(cls, name: str, service_type: str = "") -> GetServiceQuery:
        """Create service retrieval query."""
        return cls(name, service_type)

    def validate_query(self) -> FlextResult[None]:
        """Validate the query."""
        if not self.name.strip():
            return FlextResult[None].fail("Service name cannot be empty")
        return FlextResult[None].ok(None)


class ListServicesQuery:
    """Legacy query facade for listing services."""

    def __init__(
        self,
        *,
        pattern: str = "*",
        include_factories: bool = True,
        service_type_filter: str | None = None,
    ) -> None:
        self.pattern = pattern
        self.include_factories = include_factories
        self.service_type_filter = service_type_filter  # Legacy interface compatibility
        self.query_type = "list_services"  # Legacy interface compatibility

    @classmethod
    def create(
        cls,
        *,
        pattern: str = "*",
        include_factories: bool = True,
        service_type_filter: str | None = None,
    ) -> ListServicesQuery:
        """Create service listing query."""
        return cls(
            pattern=pattern,
            include_factories=include_factories,
            service_type_filter=service_type_filter,
        )

    def validate_query(self) -> FlextResult[None]:
        """Validate the query."""
        return FlextResult[None].ok(None)


# =============================================================================
# SCHEMA PROCESSING LEGACY COMPATIBILITY
# =============================================================================

# Legacy aliases for schema_processing module
BaseEntry = FlextBaseEntry
EntryType = FlextEntryType
EntryValidator = FlextEntryValidator
BaseProcessor = FlextBaseProcessor  # Note: FlextBaseProcessor not exported
ProcessingPipeline = FlextProcessingPipeline
BaseFileWriter = FlextBaseFileWriter  # Note: FlextBaseFileWriter not exported
RegexProcessor = FlextRegexProcessor
ConfigAttributeValidator = FlextConfigAttributeValidator
BaseConfigManager = FlextBaseConfigManager  # Note: FlextBaseConfigManager not exported
BaseSorter = FlextBaseSorter

# =============================================================================
# TYPE SYSTEM COMPATIBILITY - Legacy type aliases
# =============================================================================

# Legacy type system aliases for backward compatibility
FlextCoreTypes = FlextTypes

# =============================================================================
# DELEGATION SYSTEM LEGACY ALIASES - Simple compatibility aliases
# =============================================================================


# DEPRECATED: Use FlextDelegationSystem class methods directly
@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="create_mixin_delegator",
    replacement="FlextDelegationSystem.create_mixin_delegator()",
)
def create_mixin_delegator(host_instance: object, *mixin_classes: type) -> object:
    """DEPRECATED: Use FlextDelegationSystem.create_mixin_delegator() instead."""
    return FlextDelegationSystem.create_mixin_delegator(host_instance, *mixin_classes)


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="validate_delegation_system",
    replacement="FlextDelegationSystem.validate_delegation_system()",
)
def validate_delegation_system() -> FlextResult[
    dict[str, str | list[str] | dict[str, object]]
]:
    """DEPRECATED: Use FlextDelegationSystem.validate_delegation_system() instead."""
    return FlextDelegationSystem.validate_delegation_system()


# Legacy class aliases - DEPRECATED: Use FlextDelegationSystem nested classes
def _create_delegation_alias_with_warning(
    class_name: str, nested_class_name: str
) -> type:
    """Create a deprecated alias class with deprecation warning."""

    class DeprecatedAlias:
        def __new__(cls, *_args: object, **_kwargs: object) -> Self:
            warnings.warn(
                f"{class_name} is deprecated, use FlextDelegationSystem.{nested_class_name}",
                DeprecationWarning,
                stacklevel=2,
            )
            # Return an instance of DeprecatedAlias to satisfy the type checker
            return super().__new__(cls)

    DeprecatedAlias.__name__ = class_name
    DeprecatedAlias.__qualname__ = class_name
    return DeprecatedAlias


# Create deprecated aliases for nested classes
FlextMixinDelegator = _create_delegation_alias_with_warning(
    "FlextMixinDelegator", "MixinDelegator"
)
FlextDelegatedProperty = _create_delegation_alias_with_warning(
    "FlextDelegatedProperty", "DelegatedProperty"
)

# Protocol aliases - these are types, so we can alias directly

_HasDelegator = FlextDelegationSystem.HasDelegator
_DelegatorProtocol = FlextDelegationSystem.DelegatorProtocol

# =============================================================================
# UTILITIES COMPATIBILITY CLASSES - DEPRECATED
# =============================================================================


class FlextPerformance:
    """DEPRECATED: Use FlextUtilities.track_performance() instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextPerformance.track_performance",
        new_path="FlextUtilities.track_performance",
        migration_guide="use FlextUtilities.track_performance()",
    )
    def track_performance(category: str) -> object:
        return FlextUtilities.track_performance(category)

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextPerformance.get_performance_metrics",
        new_path="FlextUtilities.get_performance_metrics",
        migration_guide="use FlextUtilities.get_performance_metrics()",
    )
    def get_performance_metrics() -> dict[str, object]:
        return FlextUtilities.get_performance_metrics()


class FlextConversions:
    """DEPRECATED: Use FlextUtilities conversion methods instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextConversions.safe_bool_conversion",
        new_path="FlextUtilities.safe_bool_conversion",
        migration_guide="use FlextUtilities methods",
    )
    def safe_bool_conversion(value: object) -> bool:
        return FlextUtilities.safe_bool_conversion(value)


class FlextProcessingUtils:
    """DEPRECATED: Use FlextUtilities.parse_json_to_model() instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextProcessingUtils.parse_json_to_model",
        new_path="FlextUtilities.parse_json_to_model",
        migration_guide="use FlextUtilities.parse_json_to_model()",
    )
    def parse_json_to_model(json_text: str, model_class: type) -> FlextResult[object]:
        return FlextUtilities.parse_json_to_model(json_text, model_class)


class FlextTextProcessor:
    """DEPRECATED: Use FlextUtilities text methods instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextTextProcessor.truncate",
        new_path="FlextUtilities.truncate",
        migration_guide="use FlextUtilities.truncate()",
    )
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        return FlextUtilities.truncate(text, max_length, suffix)


class FlextTimeUtils:
    """DEPRECATED: Use FlextUtilities time methods instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextTimeUtils.format_duration",
        new_path="FlextUtilities.format_duration",
        migration_guide="use FlextUtilities.format_duration()",
    )
    def format_duration(seconds: float) -> str:
        return FlextUtilities.format_duration(seconds)


class FlextIdGenerator:
    """DEPRECATED: Use FlextUtilities.Generators instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextIdGenerator.generate_uuid",
        new_path="FlextUtilities.Generators.generate_uuid",
        migration_guide="use FlextUtilities.Generators.generate_uuid()",
    )
    def generate_uuid() -> str:
        return FlextUtilities.Generators.generate_uuid()


class FlextTypeGuards:
    """DEPRECATED: Use FlextUtilities type guard methods instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextTypeGuards.is_instance_of",
        new_path="isinstance",
        migration_guide="use isinstance() directly",
    )
    def is_instance_of(obj: object, type_hint: type) -> bool:
        return isinstance(obj, type_hint)


class FlextGenerators:
    """DEPRECATED: Use FlextUtilities.Generators instead."""

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextGenerators.generate_uuid",
        new_path="FlextUtilities.Generators.generate_uuid",
        migration_guide="use FlextUtilities.Generators.generate_uuid()",
    )
    def generate_uuid() -> str:
        return FlextUtilities.Generators.generate_uuid()

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextGenerators.generate_correlation_id",
        new_path="FlextUtilities.Generators.generate_correlation_id",
        migration_guide="use FlextUtilities.Generators.generate_correlation_id()",
    )
    def generate_correlation_id() -> str:
        return FlextUtilities.Generators.generate_correlation_id()

    @staticmethod
    @FlextDecorators.Lifecycle.deprecated_legacy_function(
        old_name="FlextGenerators.generate_id",
        new_path="FlextUtilities.Generators.generate_id",
        migration_guide="use FlextUtilities.Generators.generate_id()",
    )
    def generate_id() -> str:
        return FlextUtilities.Generators.generate_id()


# Standalone function aliases


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="flext_get_performance_metrics",
    replacement="FlextUtilities.get_performance_metrics()",
)
def flext_get_performance_metrics() -> object:
    """DEPRECATED: Use FlextUtilities.get_performance_metrics() instead."""
    return FlextUtilities.get_performance_metrics()


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="flext_record_performance",
    replacement="FlextUtilities performance methods",
)
def flext_record_performance(
    category: str, function_name: str, execution_time: float, **kwargs: object
) -> object:
    """DEPRECATED: Use FlextUtilities performance tracking instead."""
    # Combine category and function_name as operation name
    operation = f"{category}.{function_name}"
    # Extract success/error from kwargs if provided
    success = bool(kwargs.get("success", True))
    error = str(kwargs.get("error")) if kwargs.get("error") else None
    return FlextUtilities.record_performance(
        operation, execution_time, success=success, error=error
    )


# Add utility functions to exports
__all__ += [
    "FlextCallable",
    "FlextConversions",  # Legacy backward compatibility - use FlextUtilities methods
    "FlextCoreTypes",
    "FlextDecoratedFunction",
    "FlextFieldCore",
    "FlextFieldMetadata",
    "FlextFieldType",
    "FlextGenerators",  # Legacy backward compatibility - use FlextUtilities.Generators
    "FlextIdGenerator",  # Legacy backward compatibility - use FlextUtilities.Generators
    "FlextPerformance",  # Legacy backward compatibility - use FlextUtilities.Performance
    "FlextProcessingUtils",  # Legacy backward compatibility - use FlextUtilities.ProcessingUtils
    # Container facades and Commands/Queries - restored for test compatibility
    "FlextServiceKey",
    "FlextServiceRegistrar",
    "FlextServiceRetriever",
    "FlextTextProcessor",  # Legacy backward compatibility - use FlextUtilities.truncate
    "FlextTimeUtils",  # Legacy backward compatibility - use FlextUtilities.format_duration
    "FlextTypeGuards",  # Legacy backward compatibility - use FlextUtilities.TypeGuards
    "FlextValidationDecorators",
    "GetServiceQuery",
    "ListServicesQuery",
    "RegisterFactoryCommand",
    "RegisterServiceCommand",
    "UnregisterServiceCommand",
    "_BaseDecoratorFactory",
    "_BaseImmutabilityDecorators",
    "_decorators_base",
    "flext_clear_performance_metrics",
    "flext_get_performance_metrics",  # Legacy function - use FlextUtilities.get_performance_metrics
    "flext_get_performance_metrics",
    "flext_record_performance",  # Legacy function - use FlextUtilities performance methods
    "flext_safe_int_conversion",
    "flext_track_performance",
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "is_not_none",
    "safe_int_conversion_with_default",
    "truncate",  # Legacy function - use FlextUtilities.truncate
    "truncate",
]


# =============================================================================
# LEGACY COMPATIBILITY LAYER - Maintain existing imports
# =============================================================================


# Base compatibility class that enables mixin inheritance for legacy code
class _CompatibilityMixin:
    """Base compatibility class that delegates to FlextMixins methods."""

    def mixin_setup(self) -> None:
        """Setup mixin functionality."""


# All original mixin classes as compatibility facades
# class FlextLoggableMixin(_CompatibilityMixin):
#     """Legacy compatibility - delegates to FlextMixins.log_* methods."""
#
#     @property
#     def logger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
#         """Get logger via FlextMixins."""
#         return FlextMixins.get_logger(self)
#
#     def log_operation(self, operation: str, **kwargs: object) -> None:
#         """Log operation via FlextMixins."""
#         FlextMixins.log_operation(self, operation, **kwargs)
#
#     def log_info(self, message: str, **kwargs: object) -> None:
#         """Log info via FlextMixins."""
#         FlextMixins.log_info(self, message, **kwargs)
#
#     def log_error(self, message: str, **kwargs: object) -> None:
#         """Log error via FlextMixins."""
#         FlextMixins.log_error(self, message, **kwargs)
#
#     def log_debug(self, message: str, **kwargs: object) -> None:
#         """Log debug via FlextMixins."""
#         FlextMixins.log_debug(self, message, **kwargs)
#
#
# class FlextTimestampMixin(_CompatibilityMixin):
#     """Legacy compatibility - delegates to FlextMixins timestamp methods."""
#
#     def update_timestamp(self) -> None:
#         """Update timestamp via FlextMixins."""
#         FlextMixins.update_timestamp(self)
#
#     @property
#     def created_at(self) -> float:
#         """Get created timestamp via FlextMixins."""
#         return FlextMixins.get_created_at(self)
#
#     @property
#     def updated_at(self) -> float:
#         """Get updated timestamp via FlextMixins."""
#         return FlextMixins.get_updated_at(self)
#
#     def get_age_seconds(self) -> float:
#         """Get age via FlextMixins."""
#         return FlextMixins.get_age_seconds(self)


# class FlextIdentifiableMixin(_CompatibilityMixin):
#     """Legacy compatibility - delegates to FlextMixins ID methods."""
#
#     @property
#     def id(self) -> str:
#         """Get ID via FlextMixins."""
#         return FlextMixins.ensure_id(self)
#
#     @id.setter
#     def id(self, value: str) -> None:
#         """Set ID via FlextMixins."""
#         result = FlextMixins.set_id(self, value)
#         if result.is_failure:
#             raise FlextExceptions(result.error or "Invalid entity ID")
#
#     def get_id(self) -> str:
#         """Get ID via FlextMixins."""
#         return FlextMixins.ensure_id(self)
#
#     def has_id(self) -> bool:
#         """Check ID via FlextMixins."""
#         return FlextMixins.has_id(self)


# class FlextValidatableMixin(_CompatibilityMixin):
#     """Legacy compatibility - delegates to FlextMixins validation methods."""
#
#     def validate(self) -> object:
#         """Validate via FlextMixins."""
#         if FlextMixins.is_valid(self):
#             return None
#         FlextMixins.get_validation_errors(self)
#         return None
#
#     @property
#     def is_valid(self) -> bool:
#         """Check validity via FlextMixins."""
#         return FlextMixins.is_valid(self)
#
#     def add_validation_error(self, error: str) -> None:
#         """Add validation error via FlextMixins."""
#         FlextMixins.add_validation_error(self, error)
#
#     def clear_validation_errors(self) -> None:
#         """Clear validation errors via FlextMixins."""
#         FlextMixins.clear_validation_errors(self)
#
#     @property
#     def validation_errors(self) -> list[str]:
#         """Get validation errors via FlextMixins."""
#         return FlextMixins.get_validation_errors(self)
#
#     def has_validation_errors(self) -> bool:
#         """Check if has validation errors via FlextMixins."""
#         return len(FlextMixins.get_validation_errors(self)) > 0


# class FlextSerializableMixin(_CompatibilityMixin):
#     """Legacy compatibility - delegates to FlextMixins serialization methods."""
#
#     def to_dict(self) -> FlextTypes.Core.Dict:
#         """Convert to dict via FlextMixins."""
#         return FlextMixins.to_dict(self)
#
#     def to_dict_basic(self) -> FlextTypes.Core.Dict:
#         """Convert to basic dict via FlextMixins."""
#         return FlextMixins.to_dict_basic(self)
#
#     def to_json(self) -> str:
#         """Convert to JSON via FlextMixins."""
#         return FlextMixins.to_json(self)
#
#     def load_from_dict(self, data: FlextTypes.Core.Dict) -> None:
#         """Load from dict via FlextMixins."""
#         FlextMixins.load_from_dict(self, data)
#
#     def load_from_json(self, json_str: str) -> None:
#         """Load from JSON via FlextMixins."""
#         result = FlextMixins.load_from_json(self, json_str)
#         if result.is_failure:
#             raise ValueError(result.error)


# Additional compatibility classes for complete legacy support
class FlextTimingMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins timing methods."""

    def start_timing(self) -> float:
        """Start timing via FlextMixins."""
        return FlextMixins.start_timing(self)

    def stop_timing(self) -> float:
        """Stop timing via FlextMixins."""
        return FlextMixins.stop_timing(self)


# DEPRECATED: Composite mixins for legacy compatibility - use FlextMixins.* instead
# class FlextEntityMixin(
#     FlextTimestampMixin,
#     FlextIdentifiableMixin,
#     FlextLoggableMixin,
#     FlextValidatableMixin,
#     FlextSerializableMixin,
# ):
#     """Legacy compatibility - composite entity mixin."""


# class FlextValueObjectMixin(
#     FlextValidatableMixin,
#     FlextSerializableMixin,
# ):
#     """Legacy compatibility - composite value object mixin."""


# class FlextServiceMixin(
#     FlextLoggableMixin,
#     FlextValidatableMixin,
# ):
#     """Legacy compatibility - composite service mixin."""


FlextAbstractTimestampMixin = FlextAbstractMixin
FlextAbstractIdentifiableMixin = FlextAbstractMixin
FlextAbstractLoggableMixin = FlextAbstractMixin
FlextAbstractValidatableMixin = FlextAbstractMixin
FlextAbstractSerializableMixin = FlextAbstractMixin
FlextAbstractEntityMixin = FlextAbstractMixin
FlextAbstractServiceMixin = FlextAbstractMixin


# Additional compatibility classes needed by flext-cli
class FlextComparableMixin(_CompatibilityMixin):
    """Legacy compatibility - delegates to FlextMixins comparison methods."""

    def __eq__(self, other: object) -> bool:
        """Check equality via FlextMixins."""
        return FlextMixins.objects_equal(self, other)

    def __hash__(self) -> int:
        """Generate hash via FlextMixins."""
        return FlextMixins.object_hash(self)

    def __lt__(self, other: object) -> bool:
        """Compare via FlextMixins."""
        return FlextMixins.compare_objects(self, other) < 0

    def compare_to(self, other: object) -> int:
        """Compare objects via FlextMixins."""
        return FlextMixins.compare_objects(self, other)


# =============================================================================
# MODERN MIXIN ALIASES - Redirect to FlextMixins subclasses
# =============================================================================

# Modern aliases that redirect to the new FlextMixins architecture
FlextLoggableMixin = FlextMixins.Loggable
FlextSerializableMixin = FlextMixins.Serializable
FlextTimestampMixin = FlextMixins.Timestampable
FlextIdentifiableMixin = FlextMixins.Identifiable
FlextValidatableMixin = FlextMixins.Validatable
FlextServiceMixin = FlextMixins.Service
FlextEntityMixin = FlextMixins.Entity

# Legacy aliases (after modern definitions)
FlextTimestampableMixin = FlextTimestampMixin  # Alias for naming consistency
FlextStateableMixin = FlextAbstractMixin  # State management compatibility
FlextCacheableMixin = FlextAbstractMixin  # Caching compatibility
FlextObservableMixin = FlextAbstractMixin  # Observer pattern compatibility
FlextConfigurableMixin = FlextAbstractMixin  # Configuration compatibility

# =============================================================================
# LOGGING LEGACY COMPATIBILITY
# =============================================================================


@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="flext_get_logger",
    replacement="FlextLogger()",
)
def flext_get_logger(name: str = "flext") -> object:
    """DEPRECATED: Use FlextLogger() directly instead."""
    return FlextLogger(name)


def get_base_logger(name: str, *, _level: str = "INFO") -> object:
    """DEPRECATED: Use FlextLogger() directly instead."""
    LoggingLegacy.deprecation_warning("get_base_logger", "FlextLogger")
    return FlextLogger(name, _level)


def bind_context(**context: object) -> object:
    """DEPRECATED: Use FlextLogger.set_request_context() instead."""
    LoggingLegacy.deprecation_warning("bind_context", "FlextLogger.set_request_context")
    logger = FlextLogger("context")
    logger.set_request_context(**context)
    return logger


def with_performance_tracking(name: str) -> object:
    """DEPRECATED: Use FlextLogger() directly instead."""
    LoggingLegacy.deprecation_warning("with_performance_tracking", "FlextLogger")
    return FlextLogger(name)


# Legacy alias compatibility
def _create_flext_core_logging_alias() -> object:
    """Create FlextCoreLogging alias with deprecation."""
    LoggingLegacy.deprecation_warning("FlextCoreLogging", "FlextLogger")
    return FlextLogger


FlextCoreLogging = _create_flext_core_logging_alias()

# Add logging legacy exports
__all__ += [
    "FlextCoreLogging",
    "bind_context",
    "create_log_context",
    "flext_get_logger",
    "get_base_logger",
    "get_logger",
    "with_performance_tracking",
]

# =============================================================================
# ADDITIONAL BACKWARD COMPATIBILITY TYPE ALIASES - From typings.py migration
# =============================================================================


# Essential type aliases that don't conflict with existing legacy definitions
JsonType = str | int | float | bool | None | list[object] | dict[str, object]
ResultType = FlextTypes.Result.ResultType[object]
ConfigType = FlextTypes.Config.ConfigDict
HandlerType = FlextTypes.Handler.CommandHandler

# Add essential type aliases to exports (avoiding duplicates)
__all__ += [
    "ConfigType",
    "HandlerType",
    "JsonType",
    "ResultType",
]

# Add legacy constants compatibility exports
__all__ += [
    "ERROR_CODES",
    "MESSAGES",
    "SERVICE_NAME_EMPTY",
    "FlextAlgarConstants",
    "FlextCliConstants",
    "FlextCoreConstants",
    "FlextLdifConstants",
    "FlextMeltanoConstants",
    "FlextObservabilityConstants",
    "FlextOracleWmsConstants",
    "FlextQualityConstants",
    "FlextTapConstants",
    "FlextTargetConstants",
    "FlextWebConstants",
]

# =============================================================================
# CONSTANTS LEGACY COMPATIBILITY - Moved from constants.py
# =============================================================================


# Simplified ERROR_CODES mapping (legacy compatibility)
ERROR_CODES: dict[str, str] = {
    "GENERIC_ERROR": FlextConstants.Errors.GENERIC_ERROR,
    "VALIDATION_ERROR": "FLEXT_VALIDATION_ERROR",
    "CONNECTION_ERROR": FlextConstants.Errors.CONNECTION_ERROR,
    "TIMEOUT_ERROR": FlextConstants.Errors.TIMEOUT_ERROR,
    "OPERATION_ERROR": "FLEXT_OPERATION_ERROR",
    "TYPE_ERROR": "FLEXT_TYPE_ERROR",
    "CONFIG_ERROR": "FLEXT_CONFIG_ERROR",
    "CONFIGURATION_ERROR": "FLEXT_CONFIG_ERROR",  # Alias for consistency
    "AUTH_ERROR": "FLEXT_AUTH_ERROR",
    "PERMISSION_ERROR": "FLEXT_PERMISSION_ERROR",
    "BIND_ERROR": "FLEXT_BIND_ERROR",
    "CHAIN_ERROR": "FLEXT_CHAIN_ERROR",
    "MAP_ERROR": "MAP_ERROR",
}

# Direct message access (legacy)
MESSAGES = FlextConstants.Messages
SERVICE_NAME_EMPTY: Final[str] = "Service name cannot be empty"

# Legacy alias for FlextConstants - maintain backward compatibility
FlextCoreConstants = FlextConstants

# =============================================================================
# DOMAIN-SPECIFIC CONSTANT FACADES - Legacy compatibility patterns
# =============================================================================


class FlextWebConstants(FlextConstants):
    """Web domain-specific constants facade extending FlextConstants."""


class FlextCliConstants(FlextConstants):
    """CLI domain-specific constants facade extending FlextConstants."""


class FlextObservabilityConstants(FlextConstants):
    """Observability domain-specific constants facade extending FlextConstants."""


class FlextQualityConstants(FlextConstants):
    """Quality domain-specific constants facade extending FlextConstants."""


class FlextTargetConstants(FlextConstants):
    """Target domain-specific constants facade extending FlextConstants."""


class FlextTapConstants(FlextConstants):
    """Tap domain-specific constants facade extending FlextConstants."""


class FlextMeltanoConstants(FlextConstants):
    """Meltano domain-specific constants facade extending FlextConstants."""


class FlextLdifConstants(FlextConstants):
    """LDIF domain-specific constants facade extending FlextConstants."""


class FlextOracleWmsConstants(FlextConstants):
    """Oracle WMS domain-specific constants facade extending FlextConstants."""


class FlextAlgarConstants(FlextConstants):
    """ALGAR Migration domain-specific constants facade extending FlextConstants."""


# =============================================================================
# ENUM ALIASES - Legacy compatibility for enum exports
# =============================================================================

# Legacy enum aliases that were previously in constants.py
FlextOperationStatus = FlextConstants.Enums.OperationStatus
FlextLogLevel = FlextConstants.Config.LogLevel
# FlextFieldType already defined above as a class for backward compatibility
FlextEnvironment = FlextConstants.Enums.Environment
FlextEntityStatus = FlextConstants.Enums.EntityStatus
FlextDataFormat = FlextConstants.Enums.DataFormat
FlextConnectionType = FlextConstants.Enums.ConnectionType

# Add enum aliases to exports
__all__ += [
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextEntityStatus",
    "FlextEnvironment",
    "FlextLogLevel",
    "FlextOperationStatus",
]

# =============================================================================
# CONFIG MODULE LEGACY ALIASES
# =============================================================================

# Config module compatibility facades
FlextSettings = FlextConfig.Settings
FlextBaseConfigModel = FlextConfig.BaseConfigModel
FlextSystemDefaults = FlextConfig.SystemDefaults

# =============================================================================
# CONTEXT MODULE LEGACY ALIASES
# =============================================================================

# Context compatibility alias
FlextContexts = FlextContext

# Context variable aliases for backward compatibility
_correlation_id = FlextContext.Variables.Correlation.CORRELATION_ID
_parent_correlation_id = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID
_service_name = FlextContext.Variables.Service.SERVICE_NAME
_service_version = FlextContext.Variables.Service.SERVICE_VERSION
_user_id = FlextContext.Variables.Request.USER_ID
_operation_name = FlextContext.Variables.Performance.OPERATION_NAME
_request_id = FlextContext.Variables.Request.REQUEST_ID
_operation_start_time = FlextContext.Variables.Performance.OPERATION_START_TIME
_operation_metadata = FlextContext.Variables.Performance.OPERATION_METADATA

# =============================================================================
# CORE MODULE LEGACY ALIASES
# =============================================================================

# Core compatibility alias
FlextCores = FlextCore


# Helper function for global access (moved from core.py)
def flext_core() -> FlextCore:
    """Get global FlextCore instance with a convenient access pattern.

    Convenience function providing direct access to the global FlextCore singleton
    instance without requiring explicit class method calls. Maintains a singleton
    pattern while providing simpler access syntax.

    Returns:
      Global FlextCore singleton instance

    """
    return FlextCore.get_instance()


# =============================================================================
# PAYLOAD MODULE LEGACY ALIASES
# =============================================================================

# Type aliases for specialized payload types
type FlextMessage = FlextPayload[str]
type FlextEvent = FlextPayload[Mapping[str, object]]

# Legacy class references for isinstance checks
FlextMessageType = FlextPayload[str]
FlextEventType = FlextPayload[Mapping[str, object]]


# Payload helper functions (compatibility)
def create_cross_service_event(
    event_type: str,
    event_data: dict[str, object],
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextResult[FlextPayload[object]]:
    """Create a cross-service event - compatibility function."""
    return FlextPayload.create_cross_service_event(
        event_type, event_data, correlation_id, **kwargs
    )


def create_cross_service_message(
    message_text: str,
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextResult[FlextPayload[object]]:
    """Create a cross-service message - compatibility function."""
    return FlextPayload.create_cross_service_message(
        message_text, correlation_id, **kwargs
    )


def get_serialization_metrics(
    payload: object | None = None,
) -> dict[str, object]:
    """Get serialization metrics for payload - compatibility function."""
    return FlextPayload.get_serialization_metrics(payload)


def validate_cross_service_protocol(payload: object) -> FlextResult[None]:
    """Validate cross-service protocol - compatibility function."""
    return FlextPayload.validate_cross_service_protocol(payload)


# =============================================================================
# PROTOCOLS MODULE LEGACY ALIASES
# =============================================================================

# Domain layer aliases
FlextService = FlextProtocols.Domain.Service
FlextRepository = FlextProtocols.Domain.Repository
FlextDomainEvent = FlextProtocols.Domain.DomainEvent
FlextEventStore = FlextProtocols.Domain.EventStore

# Infrastructure layer aliases
FlextAuthProtocol = FlextProtocols.Infrastructure.Auth
FlextConfigurable = FlextProtocols.Infrastructure.Configurable

# Extensions layer aliases
FlextObservabilityProtocol = FlextProtocols.Extensions.Observability

# Legacy aliases for removed protocols
FlextValidationRule = FlextProtocols.Foundation.Validator
FlextMetricsCollector = FlextProtocols.Extensions.Observability
FlextAsyncHandler = FlextProtocols.Application.Handler
FlextAsyncService = FlextProtocols.Domain.Service

# Typo fixes
FlextAuthProtocols = FlextProtocols.Infrastructure.Auth

# Additional legacy support
FlextEventPublisher = FlextProtocols.Domain.EventStore
FlextEventSubscriber = FlextProtocols.Application.EventProcessor
FlextEventStreamReader = FlextProtocols.Domain.EventStore
FlextProjectionBuilder = FlextProtocols.Application.EventProcessor

# Observability sub-protocols
FlextSpanProtocol = FlextProtocols.Extensions.Observability
FlextTracerProtocol = FlextProtocols.Extensions.Observability
FlextMetricsProtocol = FlextProtocols.Extensions.Observability
FlextAlertsProtocol = FlextProtocols.Extensions.Observability

# Plugin system legacy aliases
FlextPluginRegistry = FlextProtocols.Extensions.PluginContext
FlextPluginLoader = FlextProtocols.Extensions.PluginContext

# =============================================================================
# VALIDATION MODULE LEGACY ALIASES
# =============================================================================
# Duplicate classes removed - already defined as aliases from ValidationLegacy above


# =============================================================================
# TIER 1 MODULE PATTERN - EXPORTS
# =============================================================================

# Add all new legacy compatibility exports
__all__ += [  # noqa: RUF022
    # Model class facades (consolidated from FlextModels)
    "FlextModel",
    "FlextRootModel",
    "FlextValue",
    "FlextEntity",
    "FlextAggregateRoot",
    "FlextRootModels",
    "FlextFactory",
    "FlextEntityFactory",
    # Config aliases
    "FlextSettings",
    "FlextBaseConfigModel",
    "FlextSystemDefaults",
    # Context aliases
    "FlextContexts",
    "_correlation_id",
    "_parent_correlation_id",
    "_service_name",
    "_service_version",
    "_user_id",
    "_operation_name",
    "_request_id",
    "_operation_start_time",
    "_operation_metadata",
    # Logging helper functions (compatibility)
    "get_logger",
    "set_global_correlation_id",
    "get_correlation_id",
    # Core aliases
    "FlextCores",
    "flext_core",  # Helper function for global access
    # Payload aliases
    "FlextMessage",
    "FlextEvent",
    "FlextMessageType",
    "FlextEventType",
    # Payload helper functions (compatibility)
    "create_cross_service_event",
    "create_cross_service_message",
    "get_serialization_metrics",
    "validate_cross_service_protocol",
    # Protocol aliases
    "FlextService",
    "FlextRepository",
    "FlextDomainEvent",
    "FlextEventStore",
    "FlextAuthProtocol",
    "FlextConfigurable",
    "FlextObservabilityProtocol",
    "FlextValidationRule",
    "FlextMetricsCollector",
    "FlextAsyncHandler",
    "FlextAsyncService",
    "FlextAuthProtocols",
    "FlextEventPublisher",
    "FlextEventSubscriber",
    "FlextEventStreamReader",
    "FlextProjectionBuilder",
    "FlextSpanProtocol",
    "FlextTracerProtocol",
    "FlextMetricsProtocol",
    "FlextAlertsProtocol",
    "FlextPluginRegistry",
    "FlextPluginLoader",
    # Root Models Legacy aliases
    "RootModelsLegacy",
    "FlextEntityId",
    "FlextVersion",
    "FlextTimestamp",
    "FlextMetadata",
    "FlextHost",
    "FlextPort",
    "FlextEmailAddress",
    "create_version",
    "create_email",
    # Validation aliases
    "FlextPredicates",
    "FlextStringValidator",
    "FlextNumericValidator",
    "FlextCollectionValidator",
    "FlextPredicate",
    "FlextValidationUtils",
]
