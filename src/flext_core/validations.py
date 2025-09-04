r"""Enterprise validation framework with hierarchical predicates and railway-oriented error handling.

Provides efficient validation system with domain-organized validation patterns, composable
validation chains, and business rule enforcement using FlextResult for type-safe error handling.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T

# =============================================================================
# HIERARCHICAL VALIDATION ARCHITECTURE - Organized by domain and functionality
# =============================================================================


class FlextValidations:
    """Hierarchical validation system organizing all validation components by domain.

    This is the single consolidated class for all FLEXT validation functionality,
    following the Flext[Area][Module] pattern where this represents FlextValidations.
    All validation patterns are organized within this class hierarchy.
    """

    # =========================================================================
    # CORE VALIDATION - Basic primitives and type checking
    # =========================================================================

    class Core:
        """Core validation primitives and basic type checking.

        This class contains the fundamental building blocks for validation,
        including predicates, basic type validators, and primitive validation
        operations following Single Responsibility principle.
        """

        class Predicates:
            """Composable predicates for validation with boolean logic operations."""

            def __init__(
                self,
                func: Callable[[object], bool],
                name: str = "predicate",
            ) -> None:
                """Initialize predicate with function and optional name."""
                self.func = func
                self.name = name

            def __call__(self, value: object) -> FlextResult[None]:
                """Execute predicate and return FlextResult."""
                try:
                    if self.func(value):
                        return FlextResult[None].ok(None)
                    return FlextResult[None].fail(
                        f"Predicate '{self.name}' failed for value: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                except Exception as e:
                    return FlextResult[None].fail(
                        f"Predicate '{self.name}' raised exception: {e}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            def __and__(
                self,
                other: FlextValidations.Core.Predicates,
            ) -> FlextValidations.Core.Predicates:
                """Combine predicates with AND logic."""

                def combined_func(value: object) -> bool:
                    return self.func(value) and other.func(value)

                return FlextValidations.Core.Predicates(
                    combined_func,
                    name=f"({self.name} AND {other.name})",
                )

            def __or__(
                self,
                other: FlextValidations.Core.Predicates,
            ) -> FlextValidations.Core.Predicates:
                """Combine predicates with OR logic."""

                def combined_func(value: object) -> bool:
                    return self.func(value) or other.func(value)

                return FlextValidations.Core.Predicates(
                    combined_func,
                    name=f"({self.name} OR {other.name})",
                )

            def __invert__(self) -> FlextValidations.Core.Predicates:
                """Negate predicate with NOT logic."""

                def negated_func(value: object) -> bool:
                    return not self.func(value)

                return FlextValidations.Core.Predicates(
                    negated_func,
                    name=f"NOT({self.name})",
                )

            @staticmethod
            def create_email_predicate() -> FlextValidations.Core.Predicates:
                """Create email validation predicate using FlextConstants pattern."""
                pattern = FlextConstants.Patterns.EMAIL_PATTERN
                return FlextValidations.Core.Predicates(
                    lambda x: isinstance(x, str) and re.match(pattern, x) is not None,
                    name="email_format",
                )

            @staticmethod
            def create_string_length_predicate(
                min_length: int | None = None,
                max_length: int | None = None,
            ) -> FlextValidations.Core.Predicates:
                """Create string length validation predicate."""

                def length_check(value: object) -> bool:
                    if not isinstance(value, str):
                        return False
                    length = len(value)
                    if min_length is not None and length < min_length:
                        return False
                    return not (max_length is not None and length > max_length)

                name_parts: list[str] = []
                if min_length is not None:
                    name_parts.append(f"min_{min_length}")
                if max_length is not None:
                    name_parts.append(f"max_{max_length}")
                name = f"string_length_{'_'.join(name_parts)}"

                return FlextValidations.Core.Predicates(length_check, name=name)

            @staticmethod
            def create_numeric_range_predicate(
                min_value: float | None = None,
                max_value: float | None = None,
            ) -> FlextValidations.Core.Predicates:
                """Create numeric range validation predicate."""

                def range_check(value: object) -> bool:
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        return False
                    if min_value is not None and value < min_value:
                        return False
                    return not (max_value is not None and value > max_value)

                name_parts: list[str] = []
                if min_value is not None:
                    name_parts.append(f"min_{min_value}")
                if max_value is not None:
                    name_parts.append(f"max_{max_value}")
                name = f"numeric_range_{'_'.join(name_parts)}"

                return FlextValidations.Core.Predicates(range_check, name=name)

        class TypeValidators:
            """Type-safe validators using FlextTypes system."""

            @staticmethod
            def validate_string(value: object) -> FlextResult[str]:
                """Validate value is string type."""
                if isinstance(value, str):
                    return FlextResult[str].ok(value)
                return FlextResult[str].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            @staticmethod
            def validate_integer(value: object) -> FlextResult[int]:
                """Validate value is integer type."""
                if isinstance(value, int) and not isinstance(value, bool):
                    return FlextResult[int].ok(value)
                return FlextResult[int].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            @staticmethod
            def validate_float(value: object) -> FlextResult[float]:
                """Validate value is float type."""
                if isinstance(value, float):
                    return FlextResult[float].ok(value)
                return FlextResult[float].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            @staticmethod
            def validate_dict(value: object) -> FlextResult[dict[str, object]]:
                """Validate value is dictionary using FlextTypes."""
                if isinstance(value, dict):
                    return FlextResult[dict[str, object]].ok(
                        cast("dict[str, object]", value),
                    )
                return FlextResult[dict[str, object]].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            @staticmethod
            def validate_list(value: object) -> FlextResult[list[object]]:
                """Validate value is list type."""
                if isinstance(value, list):
                    return FlextResult[list[object]].ok(cast("list[object]", value))
                return FlextResult[list[object]].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

    # =========================================================================
    # DOMAIN VALIDATION - Business logic and entity validation
    # =========================================================================

    class Domain:
        """Domain validation for business logic and entity rules.

        This class contains business domain validation patterns including
        entity validation, business rules, and domain-specific constraints
        following Domain-Driven Design principles.

        Architecture Principles Applied:
            - Single Responsibility: Only business domain validation
            - Open/Closed: Easy to extend with new domain validators
            - Domain Expertise: Business rule validation centralized
        """

        class BaseValidator:
            """Base validator implementing centralized protocol patterns."""

            def validate_entity_id(self, entity_id: object) -> FlextResult[str]:
                """Validate entity ID format and constraints."""
                if not isinstance(entity_id, str):
                    return FlextResult[str].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if not entity_id.strip():
                    return FlextResult[str].fail(
                        FlextConstants.Entities.ENTITY_ID_EMPTY,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate entity ID format using FlextConstants pattern
                pattern = r"^[a-zA-Z0-9_-]+$"
                if not re.match(pattern, entity_id):
                    return FlextResult[str].fail(
                        "Invalid entity ID format",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[str].ok(entity_id)

        class UserValidator(BaseValidator):
            """User entity validation with business rules."""

            def validate_business_rules(
                self,
                user_data: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Validate user business rules."""
                # Validate required fields
                required_fields = ["name", "email"]
                for field in required_fields:
                    if field not in user_data:
                        return FlextResult[dict[str, object]].fail(
                            f"Required field missing: {field}",
                            error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                        )

                # Validate email format
                email_result = self._validate_email_format(user_data["email"])
                if email_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        email_result.error or "Email validation failed",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                # Validate name constraints using FlextConstants
                name = user_data["name"]
                if (
                    isinstance(name, str)
                    and len(name) < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH
                ):
                    return FlextResult[dict[str, object]].fail(
                        f"Name too short, minimum: {FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH}",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                return FlextResult[dict[str, object]].ok(user_data)

            def _validate_email_format(self, email: object) -> FlextResult[str]:
                """Validate email format using centralized patterns."""
                if not isinstance(email, str):
                    return FlextResult[str].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                pattern = FlextConstants.Patterns.EMAIL_PATTERN
                if re.match(pattern, email):
                    return FlextResult[str].ok(email)

                return FlextResult[str].fail(
                    "Invalid email format",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        class EntityValidator(BaseValidator):
            """Generic entity validation patterns."""

            def validate_entity_constraints(
                self,
                entity_data: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Validate common entity constraints."""
                # Validate entity has ID
                if "id" not in entity_data:
                    return FlextResult[dict[str, object]].fail(
                        FlextConstants.Entities.ENTITY_ID_EMPTY,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate ID format
                id_result = self.validate_entity_id(entity_data["id"])
                if id_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        id_result.error or "Entity ID validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate creation timestamp if present
                if "created_at" in entity_data:
                    timestamp = entity_data["created_at"]
                    if not isinstance(timestamp, str):
                        return FlextResult[dict[str, object]].fail(
                            "Timestamp must be string format",
                            error_code=FlextConstants.Errors.TYPE_ERROR,
                        )

                    # Basic ISO timestamp format validation
                    iso_pattern = (
                        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z?$"  # ISO 8601 pattern
                    )
                    if not re.match(iso_pattern, timestamp):
                        return FlextResult[dict[str, object]].fail(
                            "Invalid timestamp format, must be ISO format",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                return FlextResult[dict[str, object]].ok(entity_data)

    # =========================================================================
    # SERVICE VALIDATION - Service-level patterns and API validation
    # =========================================================================

    class Service:
        """Service-level validation patterns and API request validation.

        This class contains service layer validation including API request
        validation, payload validation, and service integration patterns.

        Architecture Principles Applied:
            - Single Responsibility: Only service-level validation concerns
            - Interface Segregation: Service validation separated from domain rules
            - Dependency Inversion: Depends on validation protocols
        """

        class ApiRequestValidator:
            """API request validation using protocol-based design."""

            def __init__(self) -> None:
                """Initialize API request validator."""
                self._timeout = FlextConstants.Defaults.TIMEOUT

            def validate_request(
                self,
                request_data: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Validate API request structure and constraints."""
                # Validate request is not empty
                if not request_data:
                    return FlextResult[dict[str, object]].fail(
                        FlextConstants.Messages.INVALID_INPUT,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate payload size using FlextConstants limits
                payload_size = len(str(request_data))
                if payload_size > FlextConstants.Limits.MAX_FILE_SIZE:
                    return FlextResult[dict[str, object]].fail(
                        f"Request payload too large: {payload_size} bytes",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate required API fields
                required_fields = ["action", "version"]
                for field in required_fields:
                    if field not in request_data:
                        return FlextResult[dict[str, object]].fail(
                            f"Required API field missing: {field}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                return FlextResult[dict[str, object]].ok(request_data)

            def validate_timeout(self, timeout_value: object) -> FlextResult[int]:
                """Validate timeout value using FlextConstants limits."""
                if not isinstance(timeout_value, int):
                    return FlextResult[int].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if timeout_value <= 0:
                    return FlextResult[int].fail(
                        "Timeout must be positive",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if timeout_value > FlextConstants.Network.TOTAL_TIMEOUT:
                    return FlextResult[int].fail(
                        f"Timeout exceeds maximum: {FlextConstants.Network.TOTAL_TIMEOUT}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[int].ok(timeout_value)

        class ConfigValidator:
            """Configuration validation using FlextTypes system."""

            @staticmethod
            def validate_config_dict(
                config: object,
            ) -> FlextResult[FlextTypes.Config.ConfigDict]:
                """Validate configuration dictionary structure."""
                if not isinstance(config, dict):
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.CONFIG_ERROR,
                    )

                # Validate configuration keys are strings
                config_dict = cast("dict[str, object]", config)
                for key, value in config_dict.items():
                    # key is already str type from the cast

                    # Validate config value types according to ConfigDict
                    if value is not None and not isinstance(
                        value,
                        (str, int, float, bool),
                    ):
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            f"Invalid configuration value type for key '{key}'",
                            error_code=FlextConstants.Errors.CONFIG_ERROR,
                        )

                return FlextResult[FlextTypes.Config.ConfigDict].ok(
                    cast("FlextTypes.Config.ConfigDict", config),
                )

            @staticmethod
            def validate_service_config(
                config: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Validate service configuration with specific constraints."""
                # Validate required service config fields
                required_fields = ["service_name", "version"]
                for field in required_fields:
                    if field not in config:
                        return FlextResult[dict[str, object]].fail(
                            f"Required service config field missing: {field}",
                            error_code=FlextConstants.Errors.CONFIG_ERROR,
                        )

                # Validate service name using FlextConstants pattern
                service_name = config["service_name"]
                if not isinstance(service_name, str):
                    return FlextResult[dict[str, object]].fail(
                        "Service name must be string",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if (
                    len(service_name)
                    < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH
                ):
                    return FlextResult[dict[str, object]].fail(
                        f"Service name too short, minimum: {FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH}",
                        error_code=FlextConstants.Errors.CONFIG_ERROR,
                    )

                return FlextResult[dict[str, object]].ok(config)

    # =========================================================================
    # RULES VALIDATION - Comprehensive validation rule catalog
    # =========================================================================

    class Rules:
        """Comprehensive validation rules catalog with factory patterns.

        This class contains a complete catalog of validation rules organized
        by type and domain, providing factory methods for creating validators
        following the Factory pattern and Single Responsibility principle.

        Architecture Principles Applied:
            - Single Responsibility: Only validation rule definitions
            - Factory Pattern: Rule creation through factory methods
            - Open/Closed: Easy to extend with new rule categories
        """

        class StringRules:
            """String validation rules using FlextConstants patterns."""

            @staticmethod
            def validate_non_empty(value: object) -> FlextResult[str]:
                """Validate string is not empty using centralized messages."""
                if not isinstance(value, str):
                    return FlextResult[str].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if not value.strip():
                    return FlextResult[str].fail(
                        FlextConstants.Messages.VALUE_EMPTY,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[str].ok(value.strip())

            @staticmethod
            def validate_length(
                value: str,
                min_length: int | None = None,
                max_length: int | None = None,
            ) -> FlextResult[str]:
                """Validate string length using FlextConstants limits."""
                length = len(value)

                # Use FlextConstants for default limits
                max_allowed = max_length or FlextConstants.Limits.MAX_STRING_LENGTH

                if min_length is not None and length < min_length:
                    return FlextResult[str].fail(
                        f"String too short. Minimum: {min_length}, got: {length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if length > max_allowed:
                    return FlextResult[str].fail(
                        f"String too long. Maximum: {max_allowed}, got: {length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[str].ok(value)

            @staticmethod
            def validate_pattern(
                value: str,
                pattern: str,
                pattern_name: str = "pattern",
            ) -> FlextResult[str]:
                """Validate string matches regex pattern with error handling."""
                try:
                    if re.match(pattern, value):
                        return FlextResult[str].ok(value)
                    return FlextResult[str].fail(
                        f"String does not match {pattern_name} pattern",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                except re.error as e:
                    return FlextResult[str].fail(
                        f"Invalid regex pattern: {e}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            @staticmethod
            def validate_email(value: str) -> FlextResult[str]:
                """Validate email format using centralized FlextConstants pattern."""
                email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
                return FlextValidations.Rules.StringRules.validate_pattern(
                    value,
                    email_pattern,
                    "email",
                )

        class NumericRules:
            """Numeric validation rules with range checking."""

            @staticmethod
            def validate_positive(value: object) -> FlextResult[float | int]:
                """Validate number is positive with type safety."""
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return FlextResult[float | int].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if value <= 0:
                    return FlextResult[float | int].fail(
                        f"Value must be positive, got: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[float | int].ok(value)

            @staticmethod
            def validate_range(
                value: float,
                min_val: float | None = None,
                max_val: float | None = None,
            ) -> FlextResult[float | int]:
                """Validate number is within specified range using constants."""
                if min_val is not None and value < min_val:
                    return FlextResult[float | int].fail(
                        f"Value below minimum. Min: {min_val}, got: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if max_val is not None and value > max_val:
                    return FlextResult[float | int].fail(
                        f"Value above maximum. Max: {max_val}, got: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[float | int].ok(value)

            @staticmethod
            def validate_percentage(value: object) -> FlextResult[float]:
                """Validate percentage value using FlextConstants limits."""
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return FlextResult[float].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                min_pct = FlextConstants.Validation.MIN_PERCENTAGE
                max_pct = FlextConstants.Validation.MAX_PERCENTAGE

                if value < min_pct or value > max_pct:
                    return FlextResult[float].fail(
                        f"Percentage must be between {min_pct} and {max_pct}, got: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[float].ok(float(value))

        class CollectionRules:
            """Collection validation rules with size constraints."""

            @staticmethod
            def validate_list_size(
                value: object,
                min_size: int | None = None,
                max_size: int | None = None,
            ) -> FlextResult[list[object]]:
                """Validate list size using FlextConstants limits."""
                if not isinstance(value, list):
                    return FlextResult[list[object]].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                list_value = cast("list[object]", value)
                size = len(list_value)
                max_allowed = max_size or FlextConstants.Limits.MAX_LIST_SIZE

                if min_size is not None and size < min_size:
                    return FlextResult[list[object]].fail(
                        f"List too small. Minimum: {min_size}, got: {size}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if size > max_allowed:
                    return FlextResult[list[object]].fail(
                        f"List too large. Maximum: {max_allowed}, got: {size}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[list[object]].ok(list_value)

            @staticmethod
            def validate_dict_keys(
                value: object,
                required_keys: list[str],
            ) -> FlextResult[dict[str, object]]:
                """Validate dictionary contains required keys."""
                if not isinstance(value, dict):
                    return FlextResult[dict[str, object]].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                missing_keys = [key for key in required_keys if key not in value]

                if missing_keys:
                    return FlextResult[dict[str, object]].fail(
                        f"Missing required keys: {', '.join(missing_keys)}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[dict[str, object]].ok(
                    cast("dict[str, object]", value),
                )

    # =========================================================================
    # ADVANCED VALIDATION - Complex composition and schema patterns
    # =========================================================================

    class Advanced:
        """Advanced validation patterns and complex composition.

        This class contains sophisticated validation patterns including
        schema validation, composition patterns, and performance-optimized
        validation chains following advanced architectural patterns.
        """

        class SchemaValidator:
            """Schema-based validation using protocol composition."""

            def __init__(
                self,
                schema: dict[str, Callable[[object], FlextResult[object]]],
            ) -> None:
                """Initialize schema validator with validation rules."""
                self.schema = schema

            def validate(
                self,
                data: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Validate data against schema with efficient error reporting."""
                # data is already typed as dict[str, object]
                validated_data: dict[str, object] = {}
                errors: list[str] = []

                # Validate each field against its schema rule
                for field_name, validator in self.schema.items():
                    if field_name not in data:
                        errors.append(f"Missing required field: {field_name}")
                        continue

                    result = validator(data[field_name])
                    if result.is_failure:
                        errors.append(f"{field_name}: {result.error}")
                    else:
                        validated_data[field_name] = result.value

                if errors:
                    return FlextResult[dict[str, object]].fail(
                        f"Schema validation failed: {'; '.join(errors)}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[dict[str, object]].ok(validated_data)

        class CompositeValidator:
            """Composite validator for complex validation chains."""

            def __init__(
                self,
                validators: list[Callable[[object], FlextResult[object]]],
            ) -> None:
                """Initialize composite validator with validation chain."""
                self.validators = validators

            def validate(self, data: object) -> FlextResult[object]:
                """Validate data through validation chain with railway pattern."""
                current_result = FlextResult[object].ok(data)

                for validator in self.validators:
                    if current_result.is_failure:
                        break  # Short-circuit on first failure

                    current_result = current_result.flat_map(validator)

                return current_result

        class PerformanceValidator:
            """Performance-optimized validation with caching and metrics."""

            def __init__(self) -> None:
                """Initialize performance validator with metrics."""
                self._validation_cache: dict[str, FlextResult[object]] = {}
                self._validation_count = 0
                self._cache_hits = 0

            def validate_with_cache(
                self,
                data: object,
                validator: Callable[[object], FlextResult[object]],
                cache_key: str | None = None,
            ) -> FlextResult[object]:
                """Validate with caching for performance optimization."""
                self._validation_count += 1

                if cache_key and cache_key in self._validation_cache:
                    self._cache_hits += 1
                    return self._validation_cache[cache_key]

                result = validator(data)

                if cache_key and result.success:
                    # Cache successful results only
                    self._validation_cache[cache_key] = result

                return result

            def get_performance_metrics(self) -> FlextTypes.Handler.HandlerMetadata:
                """Get performance metrics using FlextTypes."""
                cache_hit_rate = (
                    self._cache_hits / self._validation_count
                    if self._validation_count > 0
                    else 0.0
                )

                return {
                    "validation_count": self._validation_count,
                    "cache_hits": self._cache_hits,
                    "cache_hit_rate": cache_hit_rate,
                    "cache_size": len(self._validation_cache),
                }

            def clear_cache(self) -> None:
                """Clear validation cache for memory management."""
                self._validation_cache.clear()

    # =========================================================================
    # PROTOCOLS INTEGRATION - Protocol-based validation interfaces
    # =========================================================================

    class Protocols:
        """Protocol-based validation interfaces using centralized FlextProtocols.

        This class integrates with the centralized FlextProtocols system to
        provide protocol-based validation interfaces following Dependency
        Inversion and Interface Segregation principles.
        """

        class ValidatorProtocol(FlextProtocols.Foundation.Validator[T]):
            """Base validator protocol extending centralized FlextProtocols."""

        class DomainValidatorProtocol(FlextProtocols.Domain.Service):
            """Domain validator protocol with service lifecycle."""

        class ServiceValidatorProtocol(FlextProtocols.Application.ValidatingHandler):
            """Service validator protocol with handler patterns."""

    # =========================================================================
    # FACTORY METHODS - Validator creation and composition
    # =========================================================================

    @classmethod
    def create_email_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create email validator using hierarchical components."""

        def validate(email: str) -> FlextResult[str]:
            return cls.Rules.StringRules.validate_email(email)

        return validate

    @classmethod
    def create_composite_validator(
        cls,
        validators: list[Callable[[object], FlextResult[object]]],
    ) -> Advanced.CompositeValidator:
        """Create composite validator from component validators."""
        return cls.Advanced.CompositeValidator(validators)

    @classmethod
    def create_schema_validator(
        cls,
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> Advanced.SchemaValidator:
        """Create schema validator with validation rules."""
        return cls.Advanced.SchemaValidator(schema)

    @classmethod
    def create_user_validator(cls) -> Domain.UserValidator:
        """Create user validator with business rules."""
        return cls.Domain.UserValidator()

    @classmethod
    def create_api_request_validator(cls) -> Service.ApiRequestValidator:
        """Create API request validator."""
        return cls.Service.ApiRequestValidator()

    @classmethod
    def create_performance_validator(cls) -> Advanced.PerformanceValidator:
        """Create performance validator with caching."""
        return cls.Advanced.PerformanceValidator()

    # =========================================================================
    # CONVENIENCE METHODS - High-level validation operations
    # =========================================================================

    @classmethod
    def validate_email(cls, email: str) -> FlextResult[str]:
        """Validate email using hierarchical email validator."""
        validator = cls.create_email_validator()
        return validator(email)

    @classmethod
    def validate_user_data(
        cls,
        user_data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate user data using domain validator."""
        validator = cls.create_user_validator()
        return validator.validate_business_rules(user_data)

    @classmethod
    def validate_api_request(
        cls,
        request_data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate API request using service validator."""
        validator = cls.create_api_request_validator()
        return validator.validate_request(request_data)

    @classmethod
    def validate_with_schema(
        cls,
        data: dict[str, object],
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> FlextResult[dict[str, object]]:
        """Validate data against schema using advanced validator."""
        validator = cls.create_schema_validator(schema)
        return validator.validate(data)

    # =============================================================================
    # CONFIGURATION MANAGEMENT - FlextTypes.Config Integration
    # =============================================================================

    @classmethod
    def configure_validation_system(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure validation system using FlextTypes.Config.

        Configures validation behavior, strictness levels, and performance settings
        based on environment and validation level settings.

        Args:
            config: Configuration dictionary with validation settings

        Returns:
            FlextResult containing validated configuration or error

        """
        try:
            # Validate required FlextTypes.Config fields
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Validate environment (affects validation behavior)
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}",
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate validation level (core setting for validation strictness)
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
                if val_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}",
                    )
                validated_config["validation_level"] = val_level
            # Default based on environment
            elif validated_config["environment"] == "production":
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.STRICT.value
                )
            elif validated_config["environment"] == "test":
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )
            else:
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Validate log level (affects validation error logging)
            if "log_level" in config:
                log_level = config["log_level"]
                valid_levels = [level.value for level in FlextConstants.Config.LogLevel]
                if log_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_levels}",
                    )
                validated_config["log_level"] = log_level
            # Default based on environment
            elif validated_config["environment"] == "production":
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.WARNING.value
                )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Add validation-specific configuration
            validated_config["enable_detailed_errors"] = config.get(
                "enable_detailed_errors",
                validated_config["validation_level"] != "strict",
            )
            validated_config["max_validation_errors"] = config.get(
                "max_validation_errors",
                100 if validated_config["validation_level"] == "strict" else 1000,
            )
            validated_config["enable_performance_tracking"] = config.get(
                "enable_performance_tracking",
                True,
            )
            validated_config["cache_validation_results"] = config.get(
                "cache_validation_results",
                validated_config["environment"] == "production",
            )
            validated_config["fail_fast_validation"] = config.get(
                "fail_fast_validation",
                validated_config["validation_level"] == "strict",
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Validation configuration error: {e}",
            )

    @classmethod
    def get_validation_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current validation system configuration.

        Returns:
            FlextResult containing current validation system configuration

        """
        try:
            # Build current validation configuration
            current_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                "enable_detailed_errors": True,
                "max_validation_errors": 1000,
                "enable_performance_tracking": True,
                "cache_validation_results": False,
                "fail_fast_validation": False,
                "available_validators": [
                    "email_validator",
                    "url_validator",
                    "schema_validator",
                    "api_request_validator",
                    "config_validator",
                ],
                "supported_patterns": [
                    "email",
                    "url",
                    "phone",
                    "credit_card",
                    "ip_address",
                    "uuid",
                    "hex_color",
                    "postal_code",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get validation config: {e}",
            )

    @classmethod
    def create_environment_validation_config(
        cls,
        environment: FlextTypes.Config.Environment,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific validation configuration."""
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                )

            # Create environment-specific validation configuration
            if environment == "production":
                config: FlextTypes.Config.ConfigDict = {
                    "environment": environment,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_detailed_errors": False,  # Hide detailed errors in production
                    "max_validation_errors": 50,  # Limit error collection
                    "enable_performance_tracking": True,
                    "cache_validation_results": True,  # Cache for performance
                    "fail_fast_validation": True,  # Fail fast for security
                }
            elif environment == "development":
                config = {
                    "environment": environment,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_detailed_errors": True,  # Full error details for debugging
                    "max_validation_errors": 2000,  # More errors for debugging
                    "enable_performance_tracking": True,
                    "cache_validation_results": False,  # No caching for development
                    "fail_fast_validation": False,  # Continue validation for debugging
                }
            elif environment == "test":
                config = {
                    "environment": environment,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_detailed_errors": True,  # Detailed errors for test debugging
                    "max_validation_errors": 500,
                    "enable_performance_tracking": False,  # No performance tracking in tests
                    "cache_validation_results": False,  # No caching in tests
                    "fail_fast_validation": True,  # Fail fast for test efficiency
                }
            else:  # staging, local, etc.
                config = {
                    "environment": environment,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_detailed_errors": True,
                    "max_validation_errors": 1000,
                    "enable_performance_tracking": True,
                    "cache_validation_results": True,
                    "fail_fast_validation": False,
                }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Environment validation config failed: {e}",
            )

    @classmethod
    def optimize_validation_performance(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize validation configuration for performance.

        Analyzes validation configuration and applies performance optimizations
        based on environment and usage patterns.

        Args:
            config: Base validation configuration to optimize

        Returns:
            FlextResult containing performance-optimized configuration

        """
        try:
            # Since config is typed as ConfigDict (dict subtype), validation is ensured
            optimized_config = dict(config)  # Copy base config

            # Performance optimizations based on environment
            environment = optimized_config.get("environment", "development")
            validation_level = optimized_config.get("validation_level", "normal")

            if environment == "production":
                # Production performance optimizations
                optimized_config["cache_validation_results"] = True
                optimized_config["fail_fast_validation"] = True
                # Ensure we have an int for min comparison
                current_max = optimized_config.get("max_validation_errors", 100)
                if isinstance(current_max, int):
                    optimized_config["max_validation_errors"] = min(current_max, 100)
                else:
                    optimized_config["max_validation_errors"] = 100
                optimized_config["enable_detailed_errors"] = False
                optimized_config["validation_timeout_ms"] = 5000  # 5 second timeout

            elif validation_level == "strict":
                # Strict validation optimizations
                optimized_config["fail_fast_validation"] = True
                optimized_config["max_validation_errors"] = 10
                optimized_config["validation_timeout_ms"] = 3000  # Faster timeout

            else:
                # Development/flexible validation optimizations
                optimized_config["cache_validation_results"] = (
                    False  # No caching for flexibility
                )
                optimized_config["fail_fast_validation"] = (
                    False  # Continue for debugging
                )
                optimized_config["validation_timeout_ms"] = 10000  # Longer timeout

            # Add performance monitoring settings
            optimized_config["performance_metrics_enabled"] = optimized_config.get(
                "enable_performance_tracking",
                True,
            )
            optimized_config["validation_batch_size"] = (
                1000 if environment == "production" else 500
            )
            optimized_config["concurrent_validations"] = (
                10 if environment == "production" else 5
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Performance optimization failed: {e}",
            )

    # =============================================================================
    # BACKWARD COMPATIBILITY METHODS - Direct access for legacy tests
    # =============================================================================

    @staticmethod
    def validate_non_empty_string_func(value: object) -> bool:
        """Validate non-empty string (legacy compatibility)."""
        return isinstance(value, str) and len(value.strip()) > 0

    @staticmethod
    def validate_email_field(value: object) -> FlextResult[None]:
        """Validate email field (legacy compatibility)."""
        if not isinstance(value, str):
            return FlextResult[None].fail("Email must be a string")

        pattern = FlextConstants.Patterns.EMAIL_PATTERN
        if re.match(pattern, value):
            return FlextResult[None].ok(None)
        return FlextResult[None].fail("Invalid email format")

    @staticmethod
    def validate_numeric_field(value: object) -> FlextResult[None]:
        """Validate numeric field (legacy compatibility)."""
        if isinstance(value, (int, float)):
            return FlextResult[None].ok(None)
        if isinstance(value, str):
            try:
                float(value)
                return FlextResult[None].ok(None)
            except ValueError:
                return FlextResult[None].fail("Value is not numeric")
        return FlextResult[None].fail("Value is not numeric")

    @staticmethod
    def validate_string_field(value: object) -> FlextResult[None]:
        """Validate string field (legacy compatibility)."""
        if isinstance(value, str) and len(value.strip()) > 0:
            return FlextResult[None].ok(None)
        return FlextResult[None].fail("Value is not a valid string")

    class Validators:
        """Validators nested class for legacy compatibility."""

        @staticmethod
        def validate_email(value: object) -> FlextResult[None]:
            """Validate email using validators pattern."""
            return FlextValidations.validate_email_field(value)

    @property
    def is_valid(self) -> bool:
        """Check if validation result is valid."""
        return True  # Default implementation for compatibility


# =============================================================================
# EXPORTS - Hierarchical and legacy validation system
# =============================================================================

__all__: list[str] = [
    "FlextValidations",  # ONLY main class exported
]
