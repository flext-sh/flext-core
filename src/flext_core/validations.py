"""FLEXT Core Validations - SOLID validation framework.

Optimized to eliminate duplication and follow SOLID principles.
Uses existing flext-core utilities extensively.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Callable
from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


@runtime_checkable
class SupportsInt(Protocol):
    """Protocol for objects that support conversion to int."""

    def __int__(self) -> int:
        """Convert to integer."""
        ...


@runtime_checkable
class SupportsFloat(Protocol):
    """Protocol for objects that support conversion to float."""

    def __float__(self) -> float:
        """Convert to float."""
        ...


class FlextValidations:
    """SOLID validation framework - Single Responsibility Principle.

    Each validation class has ONE responsibility:
    - TypeValidators: Basic type validation
    - FieldValidators: Field-specific validation (email, phone, etc.)
    - BusinessValidators: Domain-specific business rules
    - SchemaValidators: Complex schema validation
    """

    # Constants for validation rules
    MIN_PASSWORD_LENGTH = 8
    MIN_CREDIT_CARD_LENGTH = 13
    MAX_CREDIT_CARD_LENGTH = 19

    class TypeValidators:
        """Single Responsibility: Basic type validation only."""

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
            """Validate value is integer type or can be converted to integer."""
            if isinstance(value, int) and not isinstance(value, bool):
                return FlextResult[int].ok(value)
            # Try to convert string to int
            if isinstance(value, str):
                try:
                    int_value = int(value)
                    return FlextResult[int].ok(int_value)
                except ValueError:
                    return FlextResult[int].fail(
                        f"Cannot convert '{value}' to integer",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            # Try to convert using __int__ protocol
            if hasattr(value, "__int__") and not isinstance(value, bool):
                try:
                    # Type narrowing: value has __int__ method, so it's SupportsInt
                    if isinstance(value, SupportsInt):
                        int_value = int(value)
                        return FlextResult[int].ok(int_value)
                    # Fallback for objects with __int__ but not SupportsInt
                    # Use getattr to safely access the method
                    int_method = getattr(value, "__int__", None)
                    if int_method is not None:
                        int_value = int_method()
                        return FlextResult[int].ok(int_value)
                except (ValueError, TypeError):
                    return FlextResult[int].fail(
                        f"Cannot convert '{value}' to integer",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            return FlextResult[int].fail(
                FlextConstants.Messages.TYPE_MISMATCH,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

        @staticmethod
        def validate_float(value: object) -> FlextResult[float]:
            """Validate value is float type or can be converted to float."""
            if isinstance(value, float):
                return FlextResult[float].ok(value)
            # Also accept int as float
            if isinstance(value, int) and not isinstance(value, bool):
                return FlextResult[float].ok(float(value))
            # Try to convert string to float
            if isinstance(value, str):
                try:
                    float_value = float(value)
                    return FlextResult[float].ok(float_value)
                except ValueError:
                    return FlextResult[float].fail(
                        f"Cannot convert '{value}' to float",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            # Try to convert using __float__ protocol
            if hasattr(value, "__float__") and not isinstance(value, bool):
                try:
                    # Type narrowing: value has __float__ method, so it's SupportsFloat
                    if isinstance(value, SupportsFloat):
                        float_value = float(value)
                        return FlextResult[float].ok(float_value)
                    # Fallback for objects with __float__ but not SupportsFloat
                    # Use getattr to safely access the method
                    float_method = getattr(value, "__float__", None)
                    if float_method is not None:
                        float_value = float_method()
                        return FlextResult[float].ok(float_value)
                except (ValueError, TypeError):
                    return FlextResult[float].fail(
                        f"Cannot convert '{value}' to float",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            return FlextResult[float].fail(
                FlextConstants.Messages.TYPE_MISMATCH,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

        @staticmethod
        def validate_dict(value: object) -> FlextResult[FlextTypes.Core.Dict]:
            """Validate value is dictionary."""
            if isinstance(value, dict):
                return FlextResult[FlextTypes.Core.Dict].ok(value)
            return FlextResult[FlextTypes.Core.Dict].fail(
                FlextConstants.Messages.TYPE_MISMATCH,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

        @staticmethod
        def validate_list(value: object) -> FlextResult[FlextTypes.Core.List]:
            """Validate value is list type."""
            if isinstance(value, list):
                return FlextResult[FlextTypes.Core.List].ok(value)
            return FlextResult[FlextTypes.Core.List].fail(
                FlextConstants.Messages.TYPE_MISMATCH,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

    class FieldValidators:
        """Single Responsibility: Field-specific validation patterns."""

        @staticmethod
        def validate_email(value: object) -> FlextResult[str]:
            """Validate email address format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("Email must be a string")

            # Use FlextConstants pattern
            email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
            if re.match(email_pattern, value):
                return FlextResult[str].ok(value)
            return FlextResult[str].fail(f"Invalid email format: {value}")

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

        @staticmethod
        def validate_url(value: object) -> FlextResult[str]:
            """Validate URL format."""
            if not isinstance(value, str):
                return FlextResult[str].fail("URL must be a string")

            # Basic URL pattern
            url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
            if re.match(url_pattern, value):
                return FlextResult[str].ok(value)
            return FlextResult[str].fail(f"Invalid URL format: {value}")

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

    class BusinessValidators:
        """Single Responsibility: Domain-specific business rules."""

        @staticmethod
        def validate_string_field(
            value: object,
            min_length: int | None = None,
            max_length: int | None = None,
            pattern: str | None = None,
        ) -> FlextResult[str]:
            """Validate string field with constraints."""
            if not isinstance(value, str):
                return FlextResult[str].fail("Value must be a string")

            if min_length is not None and len(value) < min_length:
                return FlextResult[str].fail(
                    f"String too short, minimum {min_length} characters"
                )

            if max_length is not None and len(value) > max_length:
                return FlextResult[str].fail(
                    f"String too long, maximum {max_length} characters"
                )

            if pattern is not None and not re.match(pattern, value):
                return FlextResult[str].fail(f"String does not match pattern {pattern}")

            return FlextResult[str].ok(value)

        @staticmethod
        def validate_numeric_field(
            value: object,
            min_value: float | None = None,
            max_value: float | None = None,
        ) -> FlextResult[float | int]:
            """Validate numeric field with constraints."""
            numeric_value: float | int | None = None

            # Try to convert string to number if possible
            if isinstance(value, str):
                try:
                    # Try int first, then float
                    numeric_value = float(value) if "." in value else int(value)
                except ValueError:
                    return FlextResult[float | int].fail("Value must be numeric")
            elif isinstance(value, (int, float)):
                numeric_value = value
            elif isinstance(value, Decimal):
                numeric_value = float(value)
            else:
                # Try to convert objects that support numeric conversion
                try:
                    if hasattr(value, "__float__") and not isinstance(value, bool):
                        if isinstance(value, SupportsFloat):
                            numeric_value = float(value)
                        else:
                            # Use getattr to safely access the method
                            float_method = getattr(value, "__float__", None)
                            if float_method is not None:
                                numeric_value = float_method()
                    elif hasattr(value, "__int__") and not isinstance(value, bool):
                        if isinstance(value, SupportsInt):
                            numeric_value = int(value)
                        else:
                            # Use getattr to safely access the method
                            int_method = getattr(value, "__int__", None)
                            if int_method is not None:
                                numeric_value = int_method()
                    else:
                        return FlextResult[float | int].fail(
                            "Value cannot be converted to a number"
                        )
                except (ValueError, TypeError):
                    return FlextResult[float | int].fail(
                        "Value cannot be converted to a number"
                    )

            # Ensure numeric_value is actually numeric
            if not isinstance(numeric_value, (int, float)):
                return FlextResult[float | int].fail(
                    "Value could not be converted to a number"
                )

            if min_value is not None and numeric_value < min_value:
                return FlextResult[float | int].fail(
                    f"Value too small, minimum {min_value}"
                )

            if max_value is not None and numeric_value > max_value:
                return FlextResult[float | int].fail(
                    f"Value too large, maximum {max_value}"
                )

            return FlextResult[float | int].ok(numeric_value)

        @staticmethod
        def validate_range(
            value: float,
            min_val: float,
            max_val: float,
        ) -> FlextResult[float]:
            """Validate value is within range."""
            if value < min_val or value > max_val:
                return FlextResult[float].fail(
                    f"Value {value} out of range [{min_val}, {max_val}]"
                )
            return FlextResult[float].ok(value)

        @staticmethod
        def validate_password_strength(password: str) -> FlextResult[str]:
            """Validate password strength."""
            if len(password) < FlextValidations.MIN_PASSWORD_LENGTH:
                return FlextResult[str].fail(
                    f"Password must be at least {FlextValidations.MIN_PASSWORD_LENGTH} characters"
                )
            if not any(c.isupper() for c in password):
                return FlextResult[str].fail("Password must contain uppercase letter")
            if not any(c.islower() for c in password):
                return FlextResult[str].fail("Password must contain lowercase letter")
            if not any(c.isdigit() for c in password):
                return FlextResult[str].fail("Password must contain digit")
            return FlextResult[str].ok(password)

        @staticmethod
        def validate_credit_card(card: str) -> FlextResult[str]:
            """Validate credit card format."""
            # Remove spaces and dashes
            clean_card = re.sub(r"[\s-]", "", card)
            if not clean_card.isdigit():
                return FlextResult[str].fail("Credit card must contain only digits")
            if (
                len(clean_card) < FlextValidations.MIN_CREDIT_CARD_LENGTH
                or len(clean_card) > FlextValidations.MAX_CREDIT_CARD_LENGTH
            ):
                return FlextResult[str].fail("Invalid credit card length")
            return FlextResult[str].ok(card)

        @staticmethod
        def validate_ipv4(ip: str) -> FlextResult[str]:
            """Validate IPv4 address."""
            try:
                ipaddress.IPv4Address(ip)
                return FlextResult[str].ok(ip)
            except ipaddress.AddressValueError:
                return FlextResult[str].fail(f"Invalid IPv4 address: {ip}")

        @staticmethod
        def validate_date(date_str: str) -> FlextResult[str]:
            """Validate date format."""
            # Simple date validation - ISO format
            iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
            if re.match(iso_pattern, date_str):
                return FlextResult[str].ok(date_str)
            return FlextResult[str].fail(f"Invalid date format: {date_str}")

        @staticmethod
        def validate_json(json_str: str) -> FlextResult[dict[str, object]]:
            """Validate JSON string."""
            try:
                parsed = json.loads(json_str)
                return FlextResult[dict[str, object]].ok(parsed)
            except json.JSONDecodeError as e:
                return FlextResult[dict[str, object]].fail(f"Invalid JSON: {e}")

    class Guards:
        """Single Responsibility: Type guards and validation utilities."""

        @staticmethod
        def require_not_none(
            value: object, message: str = "Value cannot be None"
        ) -> FlextResult[object]:
            """Require value to not be None."""
            if value is None:
                return FlextResult[object].fail(message)
            return FlextResult[object].ok(value)

        @staticmethod
        def require_positive(
            value: object, message: str = "Value must be positive"
        ) -> FlextResult[object]:
            """Require value to be positive."""
            if not isinstance(value, (int, float)) or value <= 0:
                return FlextResult[object].fail(message)
            return FlextResult[object].ok(value)

        @staticmethod
        def require_in_range(
            value: object,
            min_val: float,
            max_val: float,
            message: str = "Value out of range",
        ) -> FlextResult[object]:
            """Require value to be within range."""
            if (
                not isinstance(value, (int, float))
                or value < min_val
                or value > max_val
            ):
                return FlextResult[object].fail(message)
            return FlextResult[object].ok(value)

        @staticmethod
        def require_non_empty(
            value: object, message: str = "Value cannot be empty"
        ) -> FlextResult[object]:
            """Require value to be non-empty."""
            if isinstance(value, str) and len(value.strip()) == 0:
                return FlextResult[object].fail(message)
            if isinstance(value, (list, dict)) and len(value) == 0:
                return FlextResult[object].fail(message)
            return FlextResult[object].ok(value)

        @staticmethod
        def is_dict_of(value: object, value_type: type) -> bool:
            """Check if value is dict with values of specified type."""
            if not isinstance(value, dict):
                return False
            return all(isinstance(v, value_type) for v in value.values())

        @staticmethod
        def is_list_of(value: object, item_type: type) -> bool:
            """Check if value is list with items of specified type."""
            if not isinstance(value, list):
                return False
            return all(isinstance(item, item_type) for item in value)

    class SchemaValidators:
        """Single Responsibility: Complex schema validation."""

        @staticmethod
        def validate_with_pydantic_schema(
            value: object, schema_model: type[BaseModel]
        ) -> FlextResult[object]:
            """Validate using Pydantic schema."""
            try:
                validated = schema_model.model_validate(value)
                return FlextResult[object].ok(validated)
            except ValidationError as e:
                return FlextResult[object].fail(f"Schema validation failed: {e}")

        @staticmethod
        def validate_schema(
            data: FlextTypes.Core.Dict,
            schema: dict[str, Callable[[object], FlextResult[object]]],
        ) -> FlextResult[FlextTypes.Core.Dict]:
            """Validate data against schema."""
            validated_data: FlextTypes.Core.Dict = {}
            errors: list[str] = []

            for field_name, validator in schema.items():
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

    # ==========================================================================
    # CONVENIENCE METHODS - Delegation to appropriate validators
    # ==========================================================================

    @classmethod
    def validate_email(cls, email: str) -> FlextResult[str]:
        """Validate email format."""
        return cls.FieldValidators.validate_email(email)

    @classmethod
    def validate_phone(cls, phone: str, locale: str = "US") -> FlextResult[str]:
        """Validate phone number format."""
        # Basic phone validation - locale parameter reserved for future enhancement
        result = cls.FieldValidators.validate_phone(phone)
        if result.is_success and locale != "US":
            # For non-US locales, add locale-specific validation if needed
            pass  # Currently using basic validation for all locales
        return result

    @classmethod
    def validate_url(cls, url: str) -> FlextResult[str]:
        """Validate URL format."""
        return cls.FieldValidators.validate_url(url)

    @classmethod
    def validate_uuid(cls, uuid_str: str) -> FlextResult[str]:
        """Validate UUID format."""
        return cls.FieldValidators.validate_uuid(uuid_str)

    @classmethod
    def validate_integer(cls, value: object) -> FlextResult[int]:
        """Validate integer value."""
        return cls.TypeValidators.validate_integer(value)

    @classmethod
    def validate_number(cls, value: object) -> FlextResult[float | int]:
        """Validate numeric value (int or float), with string conversion."""
        # Try to convert string to number if possible
        if isinstance(value, str):
            try:
                # Try int first, then float
                converted_value = float(value) if "." in value else int(value)
                return FlextResult[float | int].ok(converted_value)
            except ValueError:
                return FlextResult[float | int].fail("Value must be numeric")

        return cls.BusinessValidators.validate_numeric_field(value)

    @classmethod
    def validate_range(
        cls, value: float, min_val: float, max_val: float
    ) -> FlextResult[float]:
        """Validate value is within range."""
        return cls.BusinessValidators.validate_range(value, min_val, max_val)

    @classmethod
    def validate_length(
        cls, value: str, min_length: int, max_length: int
    ) -> FlextResult[str]:
        """Validate string length."""
        return cls.BusinessValidators.validate_string_field(
            value, min_length, max_length
        )

    @classmethod
    def validate_string_field(cls, value: object) -> FlextResult[str]:
        """Validate string field - convenience method."""
        return cls.BusinessValidators.validate_string_field(value)

    @classmethod
    def validate_date(cls, date_str: str) -> FlextResult[str]:
        """Validate date format."""
        return cls.BusinessValidators.validate_date(date_str)

    @classmethod
    def validate_json(cls, json_str: str) -> FlextResult[dict[str, object]]:
        """Validate JSON string."""
        return cls.BusinessValidators.validate_json(json_str)

    @classmethod
    def validate_password_strength(cls, password: str) -> FlextResult[str]:
        """Validate password strength."""
        return cls.BusinessValidators.validate_password_strength(password)

    @classmethod
    def validate_credit_card(cls, card: str) -> FlextResult[str]:
        """Validate credit card format."""
        return cls.BusinessValidators.validate_credit_card(card)

    @classmethod
    def validate_ipv4(cls, ip: str) -> FlextResult[str]:
        """Validate IPv4 address."""
        return cls.BusinessValidators.validate_ipv4(ip)

    # ==========================================================================
    # FACTORY METHODS - Validator creation and composition
    # ==========================================================================

    @classmethod
    def create_email_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create email validator using centralized implementation."""

        def validate(email: str) -> FlextResult[str]:
            return cls.FieldValidators.validate_email(email)

        return validate

    @classmethod
    def create_phone_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create phone validator using centralized implementation."""

        def validate(phone: str) -> FlextResult[str]:
            return cls.FieldValidators.validate_phone(phone)

        return validate

    @classmethod
    def create_url_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create URL validator using centralized implementation."""

        def validate(url: str) -> FlextResult[str]:
            return cls.FieldValidators.validate_url(url)

        return validate

    @classmethod
    def create_uuid_validator(cls) -> Callable[[str], FlextResult[str]]:
        """Create UUID validator using centralized implementation."""

        def validate(uuid_str: str) -> FlextResult[str]:
            return cls.FieldValidators.validate_uuid(uuid_str)

        return validate

    @classmethod
    def create_composite_validator(
        cls,
        validators: list[Callable[[object], FlextResult[object]]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create composite validator."""

        def validate(value: object) -> FlextResult[object]:
            for validator in validators:
                result = validator(value)
                if result.is_failure:
                    return result
            return FlextResult[object].ok(value)

        return validate

    @classmethod
    def create_schema_validator(
        cls,
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> Callable[[FlextTypes.Core.Dict], FlextResult[FlextTypes.Core.Dict]]:
        """Create schema validator."""

        def validate(data: FlextTypes.Core.Dict) -> FlextResult[FlextTypes.Core.Dict]:
            return cls.SchemaValidators.validate_schema(data, schema)

        return validate

    @classmethod
    def create_cached_validator(
        cls,
        validator: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create cached validator for performance optimization."""
        cache: dict[object, FlextResult[object]] = {}

        def validate(value: object) -> FlextResult[object]:
            if value in cache:
                return cache[value]
            result = validator(value)
            cache[value] = result
            return result

        return validate

    @classmethod
    def create_user_validator(
        cls,
    ) -> Callable[[FlextTypes.Core.Dict], FlextResult[FlextTypes.Core.Dict]]:
        """Create user validator - test compatibility."""

        def validate(
            user_data: FlextTypes.Core.Dict,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            return cls.validate_user_data(user_data)

        return validate

    # ==========================================================================
    # LEGACY COMPATIBILITY - Simple aliases for test compatibility
    # ==========================================================================

    # Simple aliases for backward compatibility
    Types = TypeValidators
    Fields = FieldValidators
    Rules = BusinessValidators
    Advanced = SchemaValidators
    Numbers = BusinessValidators
    Validators = FieldValidators

    # Legacy compatibility classes
    class Core:
        """Legacy Core class for test compatibility."""

        # Placeholder attributes for dynamic assignment
        TypeValidators: type
        Collections: type
        Domain: type
        Predicates: type

    class Service:
        """Legacy Service class for test compatibility."""

        class ApiRequestValidator:
            """API Request validator for compatibility."""

            def validate_request(
                self, request_data: FlextTypes.Core.Dict
            ) -> FlextResult[FlextTypes.Core.Dict]:
                """Validate API request."""
                return FlextValidations.validate_api_request(request_data)

    class Protocols:
        """Legacy Protocols class for test compatibility."""

    # Additional convenience methods for test compatibility

    @classmethod
    def validate_numeric_field(cls, value: object) -> FlextResult[bool]:
        """Validate numeric field - test compatibility."""
        if isinstance(value, (int, float)):
            return FlextResult[bool].ok(data=True)
        return FlextResult[bool].fail("Value must be numeric")

    @classmethod
    def validate_user_data(
        cls, user_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate user data - test compatibility."""
        # Simple validation - check required fields
        required_fields = ["name", "email"]
        for field in required_fields:
            if field not in user_data:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"Missing required field: {field}"
                )
        return FlextResult[FlextTypes.Core.Dict].ok(user_data)

    @classmethod
    def validate_api_request(
        cls, request_data: FlextTypes.Core.Dict
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate API request - test compatibility."""
        # Simple validation - check required fields
        required_fields = ["method", "path"]
        for field in required_fields:
            if field not in request_data:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"Missing required field: {field}"
                )
        return FlextResult[FlextTypes.Core.Dict].ok(request_data)

    @staticmethod
    def is_valid(value: object) -> bool:
        """Simple validation utility - test compatibility."""
        return value is not None

    @staticmethod
    def is_non_empty_string(value: object) -> bool:
        """Check if value is non-empty string - test compatibility."""
        if not isinstance(value, str):
            return False
        return FlextUtilities.Validation.is_non_empty_string(value)

    @staticmethod
    def validate_email_field(value: object) -> bool:
        """Simple boolean email validation - test compatibility."""
        if not isinstance(value, str):
            return False
        # Simple email validation - just check for @ and basic structure
        return "@" in value and "." in value.split("@")[-1] if "@" in value else False

    @classmethod
    def validate_with_schema(
        cls,
        data: FlextTypes.Core.Dict,
        schema: dict[str, Callable[[object], FlextResult[object]]],
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Validate data against schema - test compatibility."""
        return cls.SchemaValidators.validate_schema(data, schema)

    @classmethod
    def validate_string(cls, value: object) -> FlextResult[str]:
        """Validate string value - test compatibility."""
        return cls.TypeValidators.validate_string(value)

    @staticmethod
    def validate_non_empty_string_func(value: object) -> bool:
        """Simple boolean validation for non-empty strings - test compatibility."""
        return isinstance(value, str) and len(value.strip()) > 0


# Add legacy compatibility after class definition
FlextValidations.Core.TypeValidators = FlextValidations.TypeValidators
FlextValidations.Core.Collections = FlextValidations.BusinessValidators
FlextValidations.Core.Domain = FlextValidations.BusinessValidators


# Create the Predicates class that tests expect
class Predicates:
    """Predicate wrapper for functional validation."""

    def __init__(self, func: Callable[[object], bool], name: str = "predicate") -> None:
        """Initialize predicate with validation function."""
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


FlextValidations.Core.Predicates = Predicates

__all__: FlextTypes.Core.StringList = [
    "FlextValidations",  # ONLY main class exported
]
