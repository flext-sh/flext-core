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
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer, get_flext_container
from flext_core.decorators import FlextDecorators
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions
from flext_core.fields import FlextFields, get_global_field_registry
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger, FlextLoggerFactory, get_logger
from flext_core.mixins import FlextMixins, FlextServiceMixin
from flext_core.models import FlextModels, FlextValue
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
from flext_core.utilities import (
    FlextGenerators,
    FlextPerformance,
    FlextProcessingUtils,
    FlextTypeGuards,
    FlextUtilities,
)
from flext_core.validation import FlextValidation

# Type variables for generic functions
P = ParamSpec("P")
R = TypeVar("R")


def _emit_validation_deprecation_warning(old_name: str, new_path: str) -> None:
    """Global function for emitting validation deprecation warnings.

    Args:
        old_name: The deprecated API name
        new_path: The modern hierarchical API path

    Note:
        This is a global function to avoid ordering issues with class methods.

    """
    warnings.warn(
        f"{old_name} is deprecated. Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# =============================================================================
# CONFIG LEGACY COMPATIBILITY LAYER
# =============================================================================


class ConfigLegacy:
    """Configuration system backward compatibility facades.

    Following FLEXT_REFACTORING_PROMPT.md requirements:
    - Minimal facades delegating to modern hierarchical implementations
    - Uses FlextTypes.Config.* for proper type integration
    - Uses FlextConstants.Configuration.* for configuration constants
    - Proper deprecation warnings with specific migration paths
    - Zero business logic, orchestration only

    Architecture Compliance:
    - Single Responsibility: Only configuration backward compatibility
    - Open/Closed: Easy to extend with new config facade patterns
    - Dependency Inversion: Facades don't depend on specific implementations
    """

    @staticmethod
    def _emit_deprecation_warning(old_api: str, new_api: str) -> None:
        """Emit standardized deprecation warning for configuration APIs."""
        warnings.warn(
            f"{old_api} is deprecated. Use {new_api} instead. "
            f"See FlextConfig hierarchical API for modern configuration patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

    @staticmethod
    def get_flext_config() -> type[FlextConfig]:
        """Legacy FlextConfig factory - DEPRECATED, use FlextConfig() directly.

        Returns:
            FlextConfig class for backward compatibility.

        Note:
            This function exists for ABI compatibility only. New code should
            import and use FlextConfig directly from flext_core.config.

        """
        ConfigLegacy._emit_deprecation_warning(
            "get_flext_config()", "from flext_core.config import FlextConfig"
        )
        # Direct import - NO lazy loading per FLEXT requirements
        return FlextConfig

    @staticmethod
    def get_flext_settings() -> FlextConfig.Settings:
        """Legacy FlextSettings factory - DEPRECATED, use FlextConfig.Settings() directly.

        Returns:
            FlextConfig.Settings instance for backward compatibility.

        Note:
            This function exists for ABI compatibility only. New code should
            use FlextConfig.Settings() directly.

        """
        ConfigLegacy._emit_deprecation_warning(
            "get_flext_settings()", "FlextConfig.Settings()"
        )
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
        import and use FlextConfig directly from flext_core.config.

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
    def _deprecation_warning(old_name: str, new_path: str) -> None:
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
            HandlerLegacy._deprecation_warning(
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
            HandlerLegacy._deprecation_warning(
                "FlextValidatingHandler",
                "FlextHandlers.Implementation.ValidatingHandler",
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert legacy validators to proper protocol validators
            protocol_validators = []
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
                protocol_validators if protocol_validators else None,
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
            HandlerLegacy._deprecation_warning(
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
            HandlerLegacy._deprecation_warning(
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
            HandlerLegacy._deprecation_warning(
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
            HandlerLegacy._deprecation_warning(
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
            HandlerLegacy._deprecation_warning(
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

    @staticmethod
    def _emit_deprecation_warning(old_api: str, new_api: str) -> None:
        """Emit standardized deprecation warning for validation APIs."""
        warnings.warn(
            f"{old_api} is deprecated. Use {new_api} instead. "
            f"See FlextValidation hierarchical API for modern validation patterns.",
            DeprecationWarning,
            stacklevel=3,
        )

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
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicate", "FlextValidation.Core.Predicates"
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
            _emit_validation_deprecation_warning(
                "FlextValidationChain", "FlextValidation.Advanced.CompositeValidator"
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert legacy validators to proper callable validators
            callable_validators = []
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
            ValidationLegacy._emit_deprecation_warning(
                "FlextSchemaValidator", "FlextValidation.Advanced.SchemaValidator"
            )
            # Direct usage - NO lazy import per FLEXT requirements
            # Convert schema dict to callable validators
            callable_schema = {}
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

        def validate(self, data: object) -> object:
            """Delegate validation to implementation."""
            # Ensure data is a dict before passing to schema validator
            if isinstance(data, dict):
                return self._impl.validate(data)
            return FlextResult[object].fail(
                f"Schema validation requires dict, got {type(data).__name__}",
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

    class PredicatesLegacyFacade:
        """Minimal facade for FlextPredicates legacy compatibility."""

        @staticmethod
        def is_string() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if value is a string."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.is_string", "FlextValidation.Core.Predicates"
            )

            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, str), name="is_string"
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        def is_integer() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if value is an integer."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.is_integer", "FlextValidation.Core.Predicates"
            )

            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, int) and not isinstance(x, bool),
                name="is_integer",
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        def is_positive() -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if numeric value is positive."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.is_positive", "FlextValidation.Core.Predicates"
            )

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
        def has_length(
            min_len: int | None = None, max_len: int | None = None
        ) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks string length."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.has_length",
                "FlextValidation.Core.Predicates.create_string_length_predicate",
            )

            predicate = FlextValidation.Core.Predicates.create_string_length_predicate(
                min_len, max_len
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        def contains(substring: str) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if string contains substring."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.contains", "FlextValidation.Core.Predicates"
            )

            predicate = FlextValidation.Core.Predicates(
                lambda x: isinstance(x, str) and substring in x,
                name=f"contains('{substring}')",
            )
            return ValidationLegacy.PredicateLegacyFacade(
                predicate.func, predicate.name
            )

        @staticmethod
        def matches_pattern(pattern: str) -> ValidationLegacy.PredicateLegacyFacade:
            """Create predicate that checks if string matches regex pattern."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextPredicates.matches_pattern",
                "FlextValidation.Core.Predicates.create_email_predicate",
            )

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
        def validate_all(*validators: object) -> Callable[[object], object]:
            """Create validator that runs all validators and collects all results."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextValidationUtils.validate_all",
                "FlextValidation.Advanced.CompositeValidator",
            )

            def run_all(data: object) -> list[object]:
                results = []
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
        def validate_any(*validators: object) -> Callable[[object], object]:
            """Create validator that succeeds if any validator succeeds."""
            ValidationLegacy._emit_deprecation_warning(
                "FlextValidationUtils.validate_any",
                "FlextValidation.Advanced.CompositeValidator",
            )

            def run_any(data: object) -> object:
                errors = []
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
    def get_service_name_validator() -> Callable[[str], object]:
        """Get service name validator function."""
        ValidationLegacy._emit_deprecation_warning(
            "get_service_name_validator", "FlextValidation.Core"
        )
        return flext_validate_service_name

    @staticmethod
    def get_config_key_validator() -> Callable[[str], object]:
        """Get config key validator function."""
        ValidationLegacy._emit_deprecation_warning(
            "get_config_key_validator", "FlextValidation.Core"
        )
        return flext_validate_config_key


# =============================================================================
# VALIDATION LEGACY FUNCTION FACADES
# =============================================================================


def validate_email_address(value: object) -> object:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "validate_email_address", "FlextValidation.validate_email"
    )

    if isinstance(value, str):
        return FlextValidation.validate_email(value)

    return FlextResult[str].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


def create_validation_pipeline(_data: object) -> object:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "create_validation_pipeline", "FlextValidation.Advanced.CompositeValidator"
    )

    return FlextValidation.create_composite_validator([])


def validate_with_schema(data: object, schema: dict[str, object]) -> object:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "validate_with_schema", "FlextValidation.validate_with_schema"
    )

    if isinstance(data, dict):
        # Convert schema to callable validators like in SchemaValidatorLegacyFacade
        callable_schema = {}
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

        return FlextValidation.validate_with_schema(data, callable_schema)

    return FlextResult[dict[str, object]].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


def validate_length(
    value: object, min_length: int | None = None, max_length: int | None = None
) -> object:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "validate_length", "FlextValidation.Rules.StringRules.validate_length"
    )

    if isinstance(value, str):
        return FlextValidation.Rules.StringRules.validate_length(
            value, min_length, max_length
        )

    return FlextResult[str].fail(
        FlextConstants.Messages.TYPE_MISMATCH,
        error_code=FlextConstants.Errors.TYPE_ERROR,
    )


def flext_validate_service_name(name: str) -> FlextResult[str]:
    """Validate service name - maintains exact ABI.

    Args:
        name: Service name to validate

    Returns:
        FlextResult with validation outcome

    """
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "flext_validate_service_name", "FlextValidation.Core"
    )

    if not isinstance(name, str) or not name.strip():
        return FlextResult[str].fail(
            "Service name must be a non-empty string",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    return FlextResult[str].ok(name.strip())


def flext_validate_config_key(key: str) -> FlextResult[str]:
    """Validate config key - maintains exact ABI.

    Args:
        key: Configuration key to validate

    Returns:
        FlextResult with validation outcome

    """
    ValidationLegacy._emit_deprecation_warning(
        "flext_validate_config_key", "FlextValidation.Core"
    )

    if not isinstance(key, str) or not key.strip():
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
def is_not_empty(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "is_not_empty", "FlextValidation.Core.Predicates"
    )
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, str) and bool(x.strip()), name="is_not_empty"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


def is_numeric(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "is_numeric", "FlextValidation.Core.Predicates"
    )
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        name="is_numeric",
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


def is_string(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "is_string", "FlextValidation.Core.Predicates"
    )
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, str), name="is_string"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


def is_list(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "is_list", "FlextValidation.Core.Predicates"
    )
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, list), name="is_list"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


def is_dict(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(
        "is_dict", "FlextValidation.Core.Predicates"
    )
    predicate = FlextValidation.Core.Predicates(
        lambda x: isinstance(x, dict), name="is_dict"
    )
    return ValidationLegacy.PredicateLegacyFacade(predicate.func, predicate.name)


def is_boolean(_value: object) -> ValidationLegacy.PredicateLegacyFacade:
    """Legacy function facade - DEPRECATED."""
    ValidationLegacy._emit_deprecation_warning(  # pyright: ignore[reportPrivateUsage]
        "is_boolean", "FlextValidation.Core.Predicates"
    )
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
    def _deprecation_warning(old_name: str, new_path: str) -> None:
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
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning("FlextTimingMixin", "FlextMixins.Timing.*")

        def start_timing(self) -> float:
            """Start timing via FlextMixins."""
            return FlextMixins.start_timing(self)

        def stop_timing(self) -> float:
            """Stop timing via FlextMixins."""
            return FlextMixins.stop_timing(self)

    class ComparableMixinLegacyFacade(_CompatibilityMixin):
        """Legacy compatibility - delegates to FlextMixins comparison methods."""

        def __init__(self) -> None:
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
                "FlextEntityMixin", "FlextMixins with multiple categories"
            )

    class ValueObjectMixinLegacyFacade(
        ValidatableMixinLegacyFacade,
        SerializableMixinLegacyFacade,
    ):
        """Legacy compatibility - composite value object mixin."""

        def __init__(self) -> None:
            super().__init__()
            MixinLegacy._deprecation_warning(
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
            MixinLegacy._deprecation_warning(
                "FlextServiceMixin", "FlextMixins.Logging + FlextMixins.Validation"
            )

    # Abstract base classes for compatibility
    class AbstractMixinLegacyFacade(_CompatibilityMixin):
        """Abstract base for compatibility."""

        def __init__(self) -> None:
            MixinLegacy._deprecation_warning(
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


__all__ = [
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
    # Legacy config functions
    "get_flext_config",
    "get_flext_container",
    "get_flext_settings",
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


# Legacy aliases for backward compatibility
FlextModel = create_flext_model
FlextEntity = create_flext_entity
FlextValueObject = create_flext_value_object


# =============================================================================
# CONTAINER AND DI LEGACY FACADES
# =============================================================================


class ContainerLegacy:
    """Centralized container compatibility facades following FLEXT patterns."""

    @staticmethod
    def get_container_instance() -> FlextContainer:
        """Get FlextContainer instance via lazy import."""
        return get_flext_container()

    @staticmethod
    def get_container_class() -> type[object]:
        """Get FlextContainer class via lazy import."""
        return FlextContainer


# Container facades for direct access
# Note: FlextContainer is already imported at top of file


# Note: get_flext_container is already imported at top of file


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
    def get_logger_function() -> Callable[[str], object]:
        """Get get_logger function via lazy import."""
        return get_logger

    @staticmethod
    def get_logger_class() -> type[FlextLogger]:
        """Get FlextLogger class via lazy import."""
        return FlextLogger

    @staticmethod
    def get_logger_factory() -> type[object]:
        """Get FlextLoggerFactory class via lazy import."""
        # FlextLoggers.Factory doesn't exist, use FlextLoggerFactory directly
        return FlextLoggerFactory


# Logging facades for direct access - maintain exact function signature
# Note: get_logger is already imported at top of file

# Legacy compatibility - direct assignment with type compatibility
# Legacy compatibility - use imports from top level to avoid F811 redefinition


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
    def _deprecation_warning(old_name: str, new_path: str) -> None:
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
            FieldsLegacy._deprecation_warning(
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
        FieldsLegacy._deprecation_warning(
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

        # Use type: ignore for legacy compatibility - runtime validates types
        return FlextFields.Core.StringField(name, **field_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def create_integer_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.IntegerField:
        """Legacy integer field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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

        # Type ignore for kwargs expansion since we've validated all types above
        return FlextFields.Core.IntegerField(name, **field_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def create_boolean_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.BooleanField:
        """Legacy boolean field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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

        # Type ignore for kwargs expansion since we've validated all types above
        return FlextFields.Core.BooleanField(name, **field_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def create_email_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.EmailField:
        """Legacy email field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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

        # Type ignore for kwargs expansion since we've validated all types above
        return FlextFields.Core.EmailField(name, **field_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def create_uuid_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.UuidField:
        """Legacy UUID field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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

        # Type ignore for kwargs expansion since we've validated all types above
        return FlextFields.Core.UuidField(name, **field_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def create_datetime_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.DateTimeField:
        """Legacy datetime field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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
        return FlextFields.Core.DateTimeField(name, **field_config)  # type: ignore[arg-type]

    @staticmethod
    def create_float_field_legacy(
        name: str, **config: object
    ) -> FlextFields.Core.FloatField:
        """Legacy float field creation - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
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
        return FlextFields.Core.FloatField(name, **field_config)  # type: ignore[arg-type]

    @staticmethod
    def get_field_registry_legacy() -> FlextFields.Registry.FieldRegistry:
        """Legacy field registry access - DEPRECATED."""
        FieldsLegacy._deprecation_warning(
            "get_field_registry",
            "get_global_field_registry() or FlextFields.Registry.FieldRegistry()",
        )
        return get_global_field_registry()


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

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"
    EMAIL = "email"
    UUID = "uuid"
    DATETIME = "datetime"


# =============================================================================
# SERVICES LEGACY COMPATIBILITY
# =============================================================================


class FlextServiceProcessor(FlextServiceMixin):
    """Legacy FlextServiceProcessor for backward compatibility.

    DEPRECATED: Use FlextServices.ServiceProcessor instead.
    """

    def __init__(self, service_name: str | None = None) -> None:
        warnings.warn(
            "FlextServiceProcessor is deprecated. Use FlextServices.ServiceProcessor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self._service_name = service_name or self.__class__.__name__
        self._performance_tracker = FlextPerformance()
        self._correlation_generator = FlextUtilities()

    def get_service_name(self) -> str:
        """Get service name."""
        return self._service_name

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

# Generate dynamic exception classes using the factory pattern
for spec in FlextExceptions.EXCEPTION_SPECS:
    name, base_exception, default_code, doc, fields = spec
    # Create the exception class using the factory
    generated_class = FlextExceptions.Base.create_exception_type(
        name=name,
        base_exception=base_exception,
        default_code=default_code,
        doc=doc,
        fields=fields,
    )
    # Add to FlextExceptions namespace
    setattr(FlextExceptions, name, generated_class)


# =============================================================================
# EXCEPTION METRICS AND MONITORING
# =============================================================================


def get_exception_metrics() -> dict[str, int]:
    """Get exception occurrence metrics."""
    metrics = FlextExceptions.get_metrics()
    # Convert to int values for type compatibility
    return {k: int(v) if isinstance(v, (int, float)) else 0 for k, v in metrics.items()}


def clear_exception_metrics() -> None:
    """Clear exception metrics."""
    FlextExceptions.clear_metrics()


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# Error codes compatibility - facade for FlextErrorCodes
class FlextErrorCodes:
    """COMPATIBILITY FACADE: Use FlextErrorCodes instead.

    This class provides backward compatibility for existing code.
    All attributes delegate to FlextErrorCodes.

    DEPRECATED: Use FlextErrorCodes.[CODE] instead of FlextErrorCodes.[CODE]
    """

    GENERIC_ERROR = FlextExceptions.ErrorCodes.GENERIC_ERROR
    VALIDATION_ERROR = FlextExceptions.ErrorCodes.VALIDATION_ERROR
    CONFIGURATION_ERROR = FlextExceptions.ErrorCodes.CONFIGURATION_ERROR
    CONNECTION_ERROR = FlextExceptions.ErrorCodes.CONNECTION_ERROR
    AUTHENTICATION_ERROR = FlextExceptions.ErrorCodes.AUTHENTICATION_ERROR
    PERMISSION_ERROR = FlextExceptions.ErrorCodes.PERMISSION_ERROR
    NOT_FOUND = FlextExceptions.ErrorCodes.NOT_FOUND
    ALREADY_EXISTS = FlextExceptions.ErrorCodes.ALREADY_EXISTS
    TIMEOUT_ERROR = FlextExceptions.ErrorCodes.TIMEOUT_ERROR
    PROCESSING_ERROR = FlextExceptions.ErrorCodes.PROCESSING_ERROR
    CRITICAL_ERROR = FlextExceptions.ErrorCodes.CRITICAL_ERROR
    OPERATION_ERROR = FlextExceptions.ErrorCodes.OPERATION_ERROR
    UNWRAP_ERROR = FlextExceptions.ErrorCodes.UNWRAP_ERROR
    BUSINESS_ERROR = FlextExceptions.ErrorCodes.BUSINESS_ERROR
    INFRASTRUCTURE_ERROR = FlextExceptions.ErrorCodes.INFRASTRUCTURE_ERROR
    TYPE_ERROR = FlextExceptions.ErrorCodes.TYPE_ERROR


# Base classes compatibility
FlextErrorMixin = FlextExceptions.Base.FlextErrorMixin


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


def flext_get_performance_metrics() -> FlextTypes.Core.PerformanceMetrics:
    """Get performance metrics.

    DEPRECATED: Use FlextUtilities.Performance methods instead.
    """
    _issue_utilities_deprecation_warning(
        "flext_get_performance_metrics", "FlextUtilities.Performance methods"
    )
    # Return empty dict for compatibility since method doesn't exist in implementation
    return {}


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


# Container class facades for legacy test compatibility
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

    def __init__(self, name: str, factory: Callable[[], object]) -> None:
        self.name = name
        self.service_name = name  # Legacy interface compatibility
        self.factory = factory
        self.command_type = "register_factory"  # Legacy interface compatibility

    @classmethod
    def create(cls, name: str, factory: Callable[[], object]) -> RegisterFactoryCommand:
        """Create factory registration command."""
        return cls(name, factory)

    def validate_command(self) -> FlextResult[None]:
        """Validate the command."""
        if not self.name.strip():
            return FlextResult[None].fail("Factory name cannot be empty")
        # factory is guaranteed to be callable by typing, no need to check
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

# Add utility functions to exports
__all__ += [
    "FlextCallable",
    "FlextCoreTypes",
    "FlextDecoratedFunction",
    "FlextFieldCore",
    "FlextFieldMetadata",
    "FlextFieldType",
    "FlextGenerators",  # Legacy backward compatibility - use FlextUtilities.Generators
    "FlextPerformance",  # Legacy backward compatibility - use FlextUtilities.Performance
    "FlextProcessingUtils",  # Legacy backward compatibility - use FlextUtilities.ProcessingUtils
    "FlextServiceKey",
    "FlextServiceRegistrar",
    "FlextServiceRetriever",
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
    "flext_get_performance_metrics",
    "flext_safe_int_conversion",
    "flext_track_performance",
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "is_not_none",
    "safe_int_conversion_with_default",
    "truncate",
]
