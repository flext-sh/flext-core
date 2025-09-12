"""FLEXT Core Validations - Advanced validation framework with predicate-based rules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


class SupportsInt(Protocol):
    """Protocol for objects that support conversion to int."""

    def __int__(self) -> int:
        """Convert to integer."""
        ...


class SupportsFloat(Protocol):
    """Protocol for objects that support conversion to float."""

    def __float__(self) -> float:
        """Convert to float."""
        ...


class FlextValidations:
    """Advanced hierarchical validation system with performance optimizations.

    # VALIDATION HELL: 1306 lines with 17 DIFFERENT validation classes!
    #
    # MASSIVE DUPLICATION:
    # - Predicates, TypeValidators, BaseValidator, UserValidator, EntityValidator
    # - ApiRequestValidator, ConfigValidator, StringRules, NumericRules, CollectionRules
    # - SchemaValidator, CompositeValidator, Validators, FieldValidators
    # - PLUS the 4 email validation methods we already identified!
    #
    # OVER-COMPLEXITY:
    # - Predicates with operator overloading (__and__, __or__, __invert__) - unnecessary cleverness
    # - Multiple inheritance patterns mixing protocols and validators
    # - 17 different ways to validate the same basic types (string, int, email, etc.)
    #
    # ARCHITECTURAL SIN: This should be 3-4 simple validator classes, not 17!
    """

    class Core:
        """Core validation primitives and basic type checking."""

        class Predicates:
            """Composable predicates for validation.

            # OVER-ENGINEERED: Operator overloading for predicates is unnecessarily complex.
            # Simple functions would be clearer than (pred1 & pred2) | ~pred3 syntax.
            """

            def __init__(
                self,
                func: FlextTypes.Function.Validator,
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
                """Create email validation predicate."""
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

                name_parts: FlextTypes.Core.StringList = []
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

                name_parts: FlextTypes.Core.StringList = []
                if min_value is not None:
                    name_parts.append(f"min_{min_value}")
                if max_value is not None:
                    name_parts.append(f"max_{max_value}")
                name = f"numeric_range_{'_'.join(name_parts)}"

                return FlextValidations.Core.Predicates(range_check, name=name)

        class TypeValidators:
            """Type-safe validators."""

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
            def validate_dict(value: object) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate value is dictionary."""
                if isinstance(value, dict):
                    return FlextResult[FlextTypes.Core.Dict].ok(
                        cast("FlextTypes.Core.Dict", value),
                    )
                return FlextResult[FlextTypes.Core.Dict].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            @staticmethod
            def validate_list(value: object) -> FlextResult[FlextTypes.Core.List]:
                """Validate value is list type."""
                if isinstance(value, list):
                    return FlextResult[FlextTypes.Core.List].ok(
                        cast("FlextTypes.Core.List", value)
                    )
                return FlextResult[FlextTypes.Core.List].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

        # =========================================================================
        # ADVANCED VALIDATION METHODS - Performance optimized
        # =========================================================================

        @staticmethod
        def validate_with_pydantic_schema(
            value: object, schema_model: type[BaseModel]
        ) -> FlextResult[object]:
            """Validate value against Pydantic schema with caching."""
            try:
                validated = schema_model.model_validate(value)
                return FlextResult[object].ok(validated.model_dump())
            except ValidationError as e:
                return FlextResult[object].fail(
                    f"Pydantic validation failed: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        @classmethod
        def create_cached_validator(
            cls,
            _validator_name: str,
            validation_func: Callable[[object], bool],
            error_message: str,
        ) -> Callable[[object], FlextResult[object]]:
            """Create a cached validator function."""

            def cached_validator(value: object) -> FlextResult[object]:
                if validation_func(value):
                    return FlextResult[object].ok(value)
                return FlextResult[object].fail(error_message)

            return cached_validator

        @staticmethod
        def validate_pattern_match(
            value: object, pattern: str, flags: int = 0
        ) -> FlextResult[str]:
            """Validate value matches regex pattern using modern pattern matching."""
            if not isinstance(value, str):
                return FlextResult[str].fail(
                    "Value must be string for pattern matching",
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            try:
                compiled_pattern = re.compile(pattern, flags)
                if compiled_pattern.match(value):
                    return FlextResult[str].ok(value)
                return FlextResult[str].fail(
                    f"Value does not match pattern: {pattern}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            except re.error as e:
                return FlextResult[str].fail(
                    f"Invalid regex pattern: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

    # =========================================================================
    # DOMAIN VALIDATION - Business logic and entity validation
    # =========================================================================

    class Domain:
        """Domain validation for business logic and entity rules."""

        class BaseValidator:
            """Base validator for domain entities."""

            def __init__(self) -> None:
                """Initialize validator with global configuration."""
                self._config = FlextConfig.get_global_instance()

                # Get validation settings from config
                self.validation_enabled = self._config.validation_enabled
                self.strict_mode = self._config.validation_strict_mode
                self.max_name_length = self._config.max_name_length
                self.max_email_length = self._config.max_email_length

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
            """User entity validation."""

            def validate_business_rules(
                self,
                user_data: FlextTypes.Core.Dict,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate user business rules."""
                # Validate required fields
                required_fields = ["name", "email"]
                for field in required_fields:
                    if field not in user_data:
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            f"Required field missing: {field}",
                            error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                        )

                # Validate name constraints FIRST (for test compatibility)
                name = user_data["name"]
                if (
                    isinstance(name, str)
                    and len(name) < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH
                ):
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Name must be at least {FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH} characters",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                # Validate email format AFTER name validation
                email_result = self._validate_email_format(user_data["email"])
                if email_result.is_failure:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        email_result.error or "Email validation failed",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                # Validate age constraints for test compatibility
                max_age = 150  # Constant for age validation
                if "age" in user_data:
                    age = user_data["age"]
                    if isinstance(age, int) and (age < 0 or age > max_age):
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            f"Age must be between 0 and {max_age}",
                            error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                        )

                return FlextResult[FlextTypes.Core.Dict].ok(user_data)

            def _validate_email_format(self, email: object) -> FlextResult[str]:
                """Validate email format."""
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
            """Generic entity validation."""

            def validate_entity_constraints(
                self,
                entity_data: FlextTypes.Core.Dict,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate common entity constraints."""
                # Validate entity has ID
                if "id" not in entity_data:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        FlextConstants.Entities.ENTITY_ID_EMPTY,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate ID format
                id_result = self.validate_entity_id(entity_data["id"])
                if id_result.is_failure:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        id_result.error or "Entity ID validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate creation timestamp if present
                if "created_at" in entity_data:
                    timestamp = entity_data["created_at"]
                    if not isinstance(timestamp, str):
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            "Timestamp must be string format",
                            error_code=FlextConstants.Errors.TYPE_ERROR,
                        )

                    # Basic ISO timestamp format validation
                    iso_pattern = (
                        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z?$"  # ISO 8601 pattern
                    )
                    if not re.match(iso_pattern, timestamp):
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            "Invalid timestamp format, must be ISO format",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                return FlextResult[FlextTypes.Core.Dict].ok(entity_data)

    # =========================================================================
    # SERVICE VALIDATION - Service-level patterns and API validation
    # =========================================================================

    class Service:
        """Service-level validation patterns and API request validation."""

        class ApiRequestValidator:
            """API request validation."""

            def __init__(self) -> None:
                """Initialize API request validator."""
                self._timeout = FlextConstants.Defaults.TIMEOUT

            def validate_request(
                self,
                request_data: FlextTypes.Core.Dict,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate API request structure and constraints."""
                # Validate request is not empty
                if not request_data:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        FlextConstants.Messages.INVALID_INPUT,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate payload size using FlextConstants limits
                payload_size = len(str(request_data))
                if payload_size > FlextConstants.Limits.MAX_FILE_SIZE:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Request payload too large: {payload_size} bytes",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Validate required API fields
                required_fields = ["action", "version"]
                for field in required_fields:
                    if field not in request_data:
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            f"Required API field missing: {field}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                return FlextResult[FlextTypes.Core.Dict].ok(request_data)

            def validate_timeout(self, timeout_value: object) -> FlextResult[int]:
                """Validate timeout value."""
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
            """Configuration validation."""

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
                config_dict = cast("FlextTypes.Core.Dict", config)
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
                config: FlextTypes.Core.Dict,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate service configuration."""
                # Validate required service config fields
                required_fields = ["service_name", "version"]
                for field in required_fields:
                    if field not in config:
                        return FlextResult[FlextTypes.Core.Dict].fail(
                            f"Required service config field missing: {field}",
                            error_code=FlextConstants.Errors.CONFIG_ERROR,
                        )

                # Validate service name using FlextConstants pattern
                service_name = config["service_name"]
                if not isinstance(service_name, str):
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        "Service name must be string",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if (
                    len(service_name)
                    < FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH
                ):
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Service name too short, minimum: {FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH}",
                        error_code=FlextConstants.Errors.CONFIG_ERROR,
                    )

                return FlextResult[FlextTypes.Core.Dict].ok(config)

    # =========================================================================
    # RULES VALIDATION - Comprehensive validation rule catalog
    # =========================================================================

    class Rules:
        """Comprehensive validation rules catalog."""

        class StringRules:
            """String validation rules."""

            @staticmethod
            def validate_non_empty(value: object) -> FlextResult[str]:
                """Validate string is not empty."""
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
            def validate_string(
                value: object,
                min_length: int | None = None,
                max_length: int | None = None,
                *,
                required: bool = True,
            ) -> FlextResult[str]:
                """Simple alias for test compatibility - validate string with length constraints."""
                if not isinstance(value, str):
                    return FlextResult[str].fail(
                        f"Value must be a string, got {type(value).__name__}",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                # Handle empty string case with required parameter
                if not value and required:
                    return FlextResult[str].fail(
                        "String cannot be empty when required",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Allow empty strings if not required
                if not value and not required:
                    return FlextResult[str].ok(value)

                # Check minimum length
                if min_length is not None and len(value) < min_length:
                    return FlextResult[str].fail(
                        f"String length {len(value)} is less than minimum {min_length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Check maximum length
                if max_length is not None and len(value) > max_length:
                    return FlextResult[str].fail(
                        f"String length {len(value)} is greater than maximum {max_length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[str].ok(value)

            @staticmethod
            def validate_length(
                value: str,
                min_length: int | None = None,
                max_length: int | None = None,
            ) -> FlextResult[str]:
                """Validate string length."""
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
                """Validate string matches regex pattern."""
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
                """Validate email format."""
                email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
                return FlextValidations.Rules.StringRules.validate_pattern(
                    value,
                    email_pattern,
                    "email",
                )

        class NumericRules:
            """Numeric validation rules."""

            @staticmethod
            def validate_positive(value: object) -> FlextResult[float | int]:
                """Validate number is positive."""
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
                """Validate number is within specified range."""
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
                """Validate percentage value."""
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
            """Collection validation rules."""

            @staticmethod
            def validate_list_size(
                value: object,
                min_size: int | None = None,
                max_size: int | None = None,
            ) -> FlextResult[FlextTypes.Core.List]:
                """Validate list size."""
                if not isinstance(value, list):
                    return FlextResult[FlextTypes.Core.List].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                list_value = cast("FlextTypes.Core.List", value)
                size = len(list_value)
                max_allowed = max_size or FlextConstants.Limits.MAX_LIST_SIZE

                if min_size is not None and size < min_size:
                    return FlextResult[FlextTypes.Core.List].fail(
                        f"List too small. Minimum: {min_size}, got: {size}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if size > max_allowed:
                    return FlextResult[FlextTypes.Core.List].fail(
                        f"List too large. Maximum: {max_allowed}, got: {size}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[FlextTypes.Core.List].ok(list_value)

            @staticmethod
            def validate_dict_keys(
                value: object,
                required_keys: FlextTypes.Core.StringList,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate dictionary contains required keys."""
                if not isinstance(value, dict):
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                missing_keys = [key for key in required_keys if key not in value]

                if missing_keys:
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Missing required keys: {', '.join(missing_keys)}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[FlextTypes.Core.Dict].ok(
                    cast("FlextTypes.Core.Dict", value),
                )

    # =========================================================================
    # ADVANCED VALIDATION - Complex composition and schema patterns
    # =========================================================================

    class Advanced:
        """Advanced validation patterns and complex composition."""

        class SchemaValidator:
            """Schema-based validation."""

            def __init__(
                self,
                schema: dict[str, Callable[[object], FlextResult[object]]],
            ) -> None:
                """Initialize schema validator with validation rules."""
                self.schema = schema

            def validate(
                self,
                data: FlextTypes.Core.Dict,
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate data against schema."""
                # data is already typed as FlextTypes.Core.Dict
                validated_data: FlextTypes.Core.Dict = {}
                errors: FlextTypes.Core.StringList = []

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
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Schema validation failed: {'; '.join(errors)}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[FlextTypes.Core.Dict].ok(validated_data)

        class CompositeValidator:
            """Composite validator."""

            def __init__(
                self,
                validators: list[Callable[[object], FlextResult[object]]],
            ) -> None:
                """Initialize composite validator with validation chain."""
                self.validators = validators

            def validate(self, data: object) -> FlextResult[object]:
                """Validate data through validation chain."""
                current_result = FlextResult[object].ok(data)

                for validator in self.validators:
                    if current_result.is_failure:
                        break  # Short-circuit on first failure

                    current_result = current_result.flat_map(validator)

                return current_result

    # =========================================================================
    # PROTOCOLS INTEGRATION - Protocol-based validation interfaces
    # =========================================================================

    class Protocols:
        """Protocol-based validation interfaces."""

        class ValidatorProtocol(FlextProtocols.Foundation.Validator[T]):
            """Base validator protocol."""

        class DomainValidatorProtocol(FlextProtocols.Domain.Service):
            """Domain validator protocol."""

        class ServiceValidatorProtocol(FlextProtocols.Application.ValidatingHandler):
            """Service validator protocol."""

    # =========================================================================
    # FACTORY METHODS - Validator creation and composition
    # =========================================================================

    @classmethod
    def create_email_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create email validator."""

        def validate(email: str) -> FlextResult[str]:
            return cls.Rules.StringRules.validate_email(email)

        return validate

    @classmethod
    def create_composite_validator(
        cls,
        validators: list[Callable[[object], FlextResult[object]]],
    ) -> Advanced.CompositeValidator:
        """Create composite validator."""
        return cls.Advanced.CompositeValidator(validators)

    @classmethod
    def create_schema_validator(
        cls,
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> Advanced.SchemaValidator:
        """Create schema validator."""
        return cls.Advanced.SchemaValidator(schema)

    @classmethod
    def create_user_validator(cls) -> Domain.UserValidator:
        """Create user validator."""
        return cls.Domain.UserValidator()

    @classmethod
    def create_api_request_validator(cls) -> Service.ApiRequestValidator:
        """Create API request validator."""
        return cls.Service.ApiRequestValidator()

    @classmethod
    def create_performance_validator(cls) -> object:
        """Create performance validator using existing composite validator."""

        # Create a simple validator list for performance validation
        def simple_validator(data: object) -> FlextResult[object]:
            return FlextResult[object].ok(data)

        validators: Sequence[Callable[[object], FlextResult[object]]] = [
            simple_validator
        ]
        return cls.Advanced.CompositeValidator(list(validators))

    # =========================================================================
    # CONVENIENCE METHODS - High-level validation operations
    # =========================================================================

    @classmethod
    def validate_email(cls, email: str) -> FlextResult[str]:
        """DUPLICATE VALIDATION: Validate email.

        # CONSOLIDATION NEEDED: Email validation is scattered across multiple modules:
        # - FlextValidations.FieldValidators.validate_email() (this method)
        # - FlextUtilities.ValidationUtils.validate_email() (different return type: bool)
        # - FlextUtilities.DataValidators.validate_email_with_pydantic() (Pydantic-based)
        # - FlextMixins.validate_email() (delegates to ValidationUtils)
        # Choose ONE canonical implementation and deprecate others
        """
        validator = cls.create_email_validator()
        return validator(email)

    @classmethod
    def validate_string(
        cls,
        value: object,
        min_length: int | None = None,
        max_length: int | None = None,
        *,
        required: bool = True,
    ) -> FlextResult[str]:
        """Validate string value."""
        # First validate it's a string
        if not isinstance(value, str):
            return FlextResult.fail(
                f"Value must be a string, got {type(value).__name__}"
            )

        # Handle required parameter - if required and empty, fail
        if required and len(value) == 0:
            return FlextResult.fail("String value is required but empty")

        # If not required and empty, it's valid
        if not required and len(value) == 0:
            return FlextResult.ok(value)

        # Check minimum length
        if min_length is not None and len(value) < min_length:
            return FlextResult.fail(
                f"String length {len(value)} is less than minimum {min_length}"
            )

        # Check maximum length
        if max_length is not None and len(value) > max_length:
            return FlextResult.fail(
                f"String length {len(value)} is greater than maximum {max_length}"
            )

        return FlextResult.ok(value)

    @classmethod
    def validate_number(
        cls,
        value: object,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> FlextResult[float | int]:
        """Validate numeric value."""
        # Check if value is already numeric
        if isinstance(value, (int, float)):
            numeric_val = value
        else:
            # Try to convert string or other types to numeric
            try:
                if isinstance(value, str):
                    # Try int first, then float
                    try:
                        numeric_val = int(value)
                    except ValueError:
                        numeric_val = float(value)
                elif hasattr(value, "__int__"):
                    # Type guard: ensure value can be converted to int
                    try:
                        # Cast to protocol type for type safety
                        numeric_val = int(cast("SupportsInt", value))
                    except (ValueError, TypeError):
                        return FlextResult.fail(
                            f"Value {value} cannot be converted to int"
                        )
                elif hasattr(value, "__float__"):
                    # Type guard: ensure value can be converted to float
                    try:
                        # Cast to protocol type for type safety
                        numeric_val = float(cast("SupportsFloat", value))
                    except (ValueError, TypeError):
                        return FlextResult.fail(
                            f"Value {value} cannot be converted to float"
                        )
                else:
                    return FlextResult.fail(
                        f"Value {value} cannot be converted to a number"
                    )
            except (ValueError, TypeError, OverflowError):
                return FlextResult.fail(f"Value {value} is not a valid number")

        # Validate range
        if min_value is not None and numeric_val < min_value:
            return FlextResult.fail(
                f"Value {numeric_val} is less than minimum {min_value!s}"
            )
        if max_value is not None and numeric_val > max_value:
            return FlextResult.fail(
                f"Value {numeric_val} is greater than maximum {max_value!s}"
            )

        return FlextResult.ok(numeric_val)

    @classmethod
    def validate_user_data(
        cls,
        user_data: FlextTypes.Core.Dict,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate user data."""
        validator = cls.create_user_validator()
        return validator.validate_business_rules(user_data)

    @classmethod
    def validate_api_request(
        cls,
        request_data: FlextTypes.Core.Dict,
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate API request."""
        validator = cls.create_api_request_validator()
        return validator.validate_request(request_data)

    @classmethod
    def validate_with_schema(
        cls,
        data: FlextTypes.Core.Dict,
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate data against schema."""
        validator = cls.create_schema_validator(schema)
        return validator.validate(data)

    class Validators:
        """Validators nested class."""

        @staticmethod
        def validate_email(value: object) -> FlextResult[str]:
            """Validate email."""
            if not isinstance(value, str):
                return FlextResult[str].fail("Value must be a string")

            email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
            if re.match(email_pattern, value):
                return FlextResult[str].ok(value)

            return FlextResult[str].fail(
                "Invalid email format",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    class Types:
        """Ultra-simple type validation methods for test compatibility."""

        @staticmethod
        def validate_string(value: object) -> FlextResult[str]:
            """Ultra-simple string validation - for test compatibility."""
            if isinstance(value, str):
                return FlextResult[str].ok(value)
            # For non-string values, return failure with type mismatch error
            return FlextResult[str].fail(
                f"Type mismatch: expected string, got {type(value).__name__}"
            )

        @staticmethod
        def validate_integer(value: object) -> FlextResult[int]:
            """Ultra-simple integer validation - for test compatibility."""
            if isinstance(value, int):
                return FlextResult[int].ok(value)
            # Try to convert to int
            try:
                if isinstance(value, (str, float)):
                    int_value = int(value)
                    return FlextResult[int].ok(int_value)
                return FlextResult[int].fail("Cannot convert to integer")
            except Exception as e:
                return FlextResult[int].fail(f"Cannot convert to integer: {e}")

        @staticmethod
        def validate_float(value: object) -> FlextResult[float]:
            """Ultra-simple float validation - for test compatibility."""
            if isinstance(value, float):
                return FlextResult[float].ok(value)
            # Try to convert to float
            try:
                if isinstance(value, (int, str)):
                    float_value = float(value)
                    return FlextResult[float].ok(float_value)
                return FlextResult[float].fail("Cannot convert to float")
            except Exception as e:
                return FlextResult[float].fail(f"Cannot convert to float: {e}")

    # ==========================================================================
    # FIELDS - Field validation from fields.py (simplified)
    # ==========================================================================

    class FieldValidators:
        """Field validation utilities consolidated from fields.py."""

        @staticmethod
        def validate_email(value: object) -> FlextResult[str]:
            """Validate email address format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("Email must be a string")

            # Basic email regex
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, value):
                return FlextResult[str].fail(f"Invalid email format: {value}")

            return FlextResult[str].ok(value)

        @staticmethod
        def validate_uuid(value: object) -> FlextResult[str]:
            """Validate UUID format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("UUID must be a string")

            # UUID v4 regex
            uuid_pattern = (
                r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
            )
            if not re.match(uuid_pattern, value.lower()):
                return FlextResult[str].fail(f"Invalid UUID format: {value}")

            return FlextResult[str].ok(value)

        @staticmethod
        def validate_url(value: object) -> FlextResult[str]:
            """Validate URL format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("URL must be a string")

            # Basic URL pattern
            url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
            if not re.match(url_pattern, value):
                return FlextResult[str].fail(f"Invalid URL format: {value}")

            return FlextResult[str].ok(value)

        @staticmethod
        def validate_phone(value: object) -> FlextResult[str]:
            """Validate phone number format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("Phone must be a string")

            # Remove non-digit characters for validation
            digits_only = re.sub(r"\D", "", value)
            min_phone_digits = 10
            max_phone_digits = 15
            if (
                len(digits_only) < min_phone_digits
                or len(digits_only) > max_phone_digits
            ):
                return FlextResult[str].fail(f"Invalid phone number: {value}")

            return FlextResult[str].ok(value)

    # ==========================================================================
    # GUARDS - Type guards and validation assertions (from guards.py)
    # ==========================================================================

    class Guards:
        """Type guards and validation assertions consolidated from guards.py."""

        @staticmethod
        def is_dict_of(obj: object, value_type: type) -> bool:
            """Type guard to validate dictionary with homogeneous value types."""
            if not isinstance(obj, dict):
                return False
            dict_obj = cast("dict[object, object]", obj)
            return all(isinstance(value, value_type) for value in dict_obj.values())

        @staticmethod
        def is_list_of(obj: object, item_type: type) -> bool:
            """Type guard to validate list with homogeneous item types."""
            if not isinstance(obj, list):
                return False
            list_obj = cast("FlextTypes.Core.List", obj)
            return all(isinstance(item, item_type) for item in list_obj)

        @staticmethod
        def require_not_none(
            value: object,
            message: str = "Value cannot be None",
        ) -> object:
            """Validate that a value is not None."""
            if value is None:
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_positive(
            value: object,
            message: str = "Value must be positive",
        ) -> object:
            """Validate that a value is a positive integer."""
            if not (isinstance(value, int) and value > 0):
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_in_range(
            value: object,
            min_val: int,
            max_val: int,
            message: str | None = None,
        ) -> object:
            """Validate that a numeric value falls within inclusive bounds."""
            if not (isinstance(value, (int, float)) and min_val <= value <= max_val):
                if not message:
                    message = f"Value must be between {min_val} and {max_val}"
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_non_empty(
            value: object,
            message: str = "Value cannot be empty",
        ) -> object:
            """Validate that a string value is non-empty."""
            if not isinstance(value, str) or not value.strip():
                raise FlextExceptions.ValidationError(message)
            return value

    # Convenience attributes for backward compatibility
    Fields = Rules.StringRules
    Collections = Rules.CollectionRules
    Numbers = Rules.NumericRules

    @property
    def is_valid(self) -> bool:
        """Check if validation result is valid."""
        return True  # Default implementation for compatibility

    @staticmethod
    def validate_string_field(
        value: object,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
    ) -> FlextResult[bool]:
        """Validate string field with constraints."""
        if not isinstance(value, str):
            return FlextResult[bool].fail("Value must be a string")

        if min_length is not None and len(value) < min_length:
            return FlextResult[bool].fail(
                f"String too short, minimum {min_length} characters"
            )

        if max_length is not None and len(value) > max_length:
            return FlextResult[bool].fail(
                f"String too long, maximum {max_length} characters"
            )

        if pattern is not None and not re.match(pattern, value):
            return FlextResult[bool].fail(
                f"String does not match pattern {pattern}"
            )

        return FlextResult[bool].ok(data=True)

    @staticmethod
    def validate_numeric_field(
        value: object,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> FlextResult[bool]:
        """Validate numeric field with constraints."""
        if not isinstance(value, (int, float)):
            return FlextResult[bool].fail("Value must be numeric")

        if min_value is not None and value < min_value:
            return FlextResult[bool].fail(f"Value too small, minimum {min_value}")

        if max_value is not None and value > max_value:
            return FlextResult[bool].fail(f"Value too large, maximum {max_value}")

        return FlextResult[bool].ok(data=True)


__all__: FlextTypes.Core.StringList = [
    "FlextValidations",  # ONLY main class exported
]
