"""Consolidated utilities for FLEXT ecosystem with railway composition patterns.

Eliminates cross-module duplication by providing centralized validation,
transformation, and processing utilities using FlextResult monadic operators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import inspect
import json
import math
import operator
import os
import pathlib
import re
import secrets
import string
import threading
import time
import typing
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import (
    ClassVar,
    Protocol,
    cast,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.typings import Dict, T, U


@runtime_checkable
class HasModelDump(Protocol):
    """Protocol for objects that have model_dump method.

    Supports Pydantic's model_dump signature with optional mode parameter.
    """

    def model_dump(self, mode: str = "python") -> dict[str, object]:
        """Dump the model to a dictionary.

        Args:
            mode: Serialization mode ('python' or 'json')

        Returns:
            Dictionary representation of the model

        """
        ...


# All constants now use FlextConstants - no local constants needed
# Using FlextConstants.Utilities for utility-specific constants
# Using FlextConstants.Security for security-related constants
# Using FlextConstants.Network for network-related constants
# Using FlextConstants.Reliability for retry-related constants


class FlextUtilities:
    """Comprehensive utility functions for FLEXT ecosystem operations.

    FlextUtilities provides centralized validation, transformation, and
    processing utilities using FlextResult railway patterns. Includes
    18+ nested utility classes covering validation, data transformation,
    type conversion, reliability patterns, and text processing for all
    32+ dependent FLEXT projects.

    **Function**: Centralized utility functions with railway patterns
        - Validation helpers (18 nested classes with 100+ validators)
        - Data transformation utilities (JSON, dict, list operations)
        - Processing utilities (batch, parallel, sequential)
        - Type conversions with proper error handling
        - Type guards and runtime type checking
        - Text processing and sanitization
        - ID and correlation ID generation
        - Cache management utilities (CQRS, general purpose)
        - Reliability patterns (retry, timeout, circuit breaker)
        - Message validation for CQRS handlers
        - Function composition utilities
        - String manipulation and formatting

    **Uses**: Core infrastructure with minimal dependencies
        - FlextResult[T] for all operation results (railway pattern)
        - FlextConstants for validation limits and defaults
        - FlextConfig for configuration integration
        - FlextLogger for operation logging
        - FlextExceptions for structured error handling
        - re module for pattern matching and validation
        - typing module for type checks and guards
        - json module for serialization operations
        - uuid module for ID generation
        - secrets module for secure token generation
        - threading for thread-safe operations
        - contextvars for context management

    **How to use**: Utility operations with FlextResult
        ```python
        from flext_core import FlextUtilities, FlextResult

        # Example 1: String validation with railway pattern
        result = FlextUtilities.Validation.validate_string_not_none(
            user_input, field_name="username"
        )
        if result.is_success:
            validated_name = result.unwrap()

        # Example 2: Email validation with pattern matching
        email_result = FlextUtilities.Validation.validate_email("user@example.com")

        # Example 3: Data transformation with error handling
        json_result = FlextUtilities.Transformation.to_json({"key": "value"})

        # Example 4: Retry pattern for reliability
        result = FlextUtilities.Reliability.retry(
            operation=lambda: call_external_api(), max_attempts=3, delay=1.0
        )

        # Example 5: Type conversion with validation
        int_result = FlextUtilities.TypeConversions.to_int("42", field_name="user_age")

        # Example 6: Generate correlation ID for tracking
        correlation_id = FlextUtilities.Correlation.generate_id()

        # Example 7: Text processing and sanitization
        sanitized = FlextUtilities.TextProcessor.sanitize_input(user_text)

        # Example 8: Cache operations for performance
        cache_result = FlextUtilities.Cache.get("cache_key")
        FlextUtilities.Cache.set(
            "cache_key", value, ttl=FlextConstants.Defaults.CACHE_TTL
        )
        ```

    **TODO**: Enhanced utility features for 1.0.0+ releases
        - [ ] Add more validation patterns (credit cards, phone, etc.)
        - [ ] Implement performance optimization for hot paths
        - [ ] Add utility variants for concurrent operations
        - [ ] Enhance caching with distributed cache support
        - [ ] Add stream processing utilities for large data
        - [ ] Implement data sanitization for security
        - [ ] Add more type guards for complex types
        - [ ] Support custom validation rule composition
        - [ ] Implement utility function memoization
        - [ ] Add data anonymization utilities for GDPR

    Attributes:
        Validation: Validation utilities (18 nested classes).
        Transformation: Data transformation helpers.
        Processing: Processing and batch utilities.
        Utilities: General purpose utilities.
        Cache: Cache management utilities.
        CqrsCache: CQRS-specific caching.
        Generators: ID and data generation.
        Correlation: Correlation ID utilities.
        TextProcessor: Text processing and sanitization.
        TypeConversions: Type conversion utilities.
        Reliability: Retry and circuit breaker patterns.
        TypeGuards: Runtime type checking.
        TypeChecker: Type validation utilities.
        MessageValidator: Message validation for handlers.
        Composition: Function composition utilities.

    Note:
        All utilities return FlextResult for consistency. Use
        FlextConstants for validation limits. Utilities are
        stateless and thread-safe. Validation logic should
        primarily use FlextModels.Validation for domain rules.

    Warning:
        Some validation methods marked as audit violations should
        be moved to FlextModels.Validation for centralized domain
        validation. Cache utilities are in-memory only - use
        distributed cache for production. Retry patterns may
        increase latency - configure appropriately.

    Example:
        Complete utility usage workflow:

        >>> result = FlextUtilities.Validation.validate_email("test@example.com")
        >>> print(result.is_success)
        True
        >>> correlation_id = FlextUtilities.Correlation.generate_id()
        >>> print(len(correlation_id))
        36

    See Also:
        FlextResult: For railway-oriented error handling.
        FlextConstants: For validation limits and defaults.
        FlextModels: For domain model validation patterns.

    """

    MIN_TOKEN_LENGTH = FlextConstants.Security.MIN_PASSWORD_LENGTH

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID."""
        return FlextUtilities.Generators.generate_id()

    @staticmethod
    def safe_json_parse(json_string: str) -> FlextResult[dict[str, object]]:
        """Parse JSON string with error handling (convenience method).

        Args:
            json_string: JSON string to parse

        Returns:
            FlextResult[dict]: Parsed JSON data or error

        """
        return FlextUtilities.Transformation.safe_json_parse(json_string)

    @staticmethod
    def safe_json_stringify(data: dict[str, object]) -> FlextResult[str]:
        """Stringify data to JSON with error handling (convenience method).

        Args:
            data: Dictionary to convert to JSON

        Returns:
            FlextResult[str]: JSON string or error

        """
        return FlextUtilities.Transformation.safe_json_stringify(data)

    @staticmethod
    def validate_data(
        data: dict[str, object],
        required_fields: list[str] | dict[str, type | tuple[type, ...]] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Validate dictionary data (convenience method).

        Args:
            data: Dictionary to validate
            required_fields: Optional list of field names or dict mapping fields to types

        Returns:
            FlextResult[dict]: Validated data or error

        """
        return FlextUtilities.Validation.validate_data(data, required_fields)

    class Validation:
        """Unified validation patterns using railway composition.

        🚨 AUDIT VIOLATION: This entire Validation class violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Validation logic should be centralized in FlextConfig and FlextModels ONLY
        ❌ ARCHITECTURAL VIOLATION: Inline validation scattered across utilities instead of centralized

        🔧 REQUIRED ACTION:
        - Move ALL validation logic to FlextConfig.Validation for configuration validation
        - Move ALL validation logic to FlextModels.Validation for domain validation
        - Remove this entire Validation class from utilities.py
        - Keep only transformation, processing, and reliability patterns in utilities

        📍 SHOULD BE USED INSTEAD:
        - FlextConfig.Validation for configuration validation
        - FlextModels.Validation for domain model validation
        - FlextModels.Field validators for Pydantic model validation
        """

        @staticmethod
        def validate_string_not_none(
            value: str | None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate that string is not None.

            🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
            ❌ CRITICAL ISSUE: String validation should be in FlextModels.Validation, not utilities
            ❌ INLINE VALIDATION: This is inline validation that should be centralized

            🔧 REQUIRED ACTION: Move to FlextModels.Validation.validate_string_not_none()
            📍 SHOULD BE USED INSTEAD: FlextModels.Field(validator=validate_not_none)

            Returns:
                FlextResult[str]: Success with validated string or failure with error message

            """
            # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextModels.Validation
            if value is None:
                return FlextResult[str].fail(f"{field_name} cannot be None")
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string_not_empty(
            value: str,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate that string is not empty after stripping.

            🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
            ❌ CRITICAL ISSUE: String validation should be in FlextModels.Validation, not utilities
            ❌ INLINE VALIDATION: This is inline validation that should be centralized

            🔧 REQUIRED ACTION: Move to FlextModels.Validation.validate_string_not_empty()
            📍 SHOULD BE USED INSTEAD: FlextModels.Field(validator=validate_not_empty)

            Returns:
                FlextResult[str]: Success with validated string or failure with error message

            """
            # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextModels.Validation
            stripped = value.strip()
            if not stripped:
                return FlextResult[str].fail(
                    f"{field_name} cannot be empty or whitespace only",
                )
            return FlextResult[str].ok(stripped)

        @staticmethod
        def validate_string_length(
            value: str,
            min_length: int = 1,
            max_length: int | None = None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate string length constraints."""
            length = len(value)
            if length < min_length:
                return FlextResult[str].fail(
                    f"{field_name} must be at least {min_length} characters, got {length}",
                )
            if max_length is not None and length > max_length:
                return FlextResult[str].fail(
                    f"{field_name} must be at most {max_length} characters, got {length}",
                )
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string_pattern(
            value: str,
            pattern: str | None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate string against regex pattern."""
            if pattern is None:
                return FlextResult[str].ok(value)

            # First validate the pattern itself using FlextResult composition
            pattern_validation = FlextUtilities.Processing.validate_regex_pattern(
                pattern,
            )
            if pattern_validation.is_failure:
                return FlextResult[str].fail(
                    f"Invalid pattern for {field_name}: {pattern_validation.error}",
                )

            # Then validate the value against the validated pattern
            compiled_pattern = pattern_validation.unwrap()
            if not compiled_pattern.match(value):
                return FlextResult[str].fail(
                    f"{field_name} does not match required pattern",
                )
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string(
            value: str | None,
            min_length: int = 1,
            max_length: int | None = None,
            pattern: str | None = None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Comprehensive string validation using railway composition."""
            # Use explicit function calls instead of lambdas to avoid type inference issues
            not_none_result = FlextUtilities.Validation.validate_string_not_none(
                value,
                field_name,
            )
            if not_none_result.is_failure:
                return not_none_result

            not_empty_result = FlextUtilities.Validation.validate_string_not_empty(
                not_none_result.unwrap(),
                field_name,
            )
            if not_empty_result.is_failure:
                return not_empty_result

            length_result = FlextUtilities.Validation.validate_string_length(
                not_empty_result.unwrap(),
                min_length,
                max_length,
                field_name,
            )
            if length_result.is_failure:
                return length_result

            if pattern:
                return FlextUtilities.Validation.validate_string_pattern(
                    length_result.unwrap(),
                    pattern,
                    field_name,
                )
            return FlextResult[str].ok(length_result.unwrap())

        @staticmethod
        def validate_email(email: str) -> FlextResult[str]:
            """Validate email format using railway composition.

            🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
            ❌ CRITICAL ISSUE: Email validation should be in FlextModels.Validation, not utilities
            ❌ INLINE VALIDATION: This is inline validation that should be centralized

            🔧 REQUIRED ACTION: Move to FlextModels.Validation.validate_email()
            📍 SHOULD BE USED INSTEAD: FlextModels.EmailAddress field type
            """
            # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextModels.Validation
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return FlextUtilities.Validation.validate_string(
                email,
                min_length=5,
                max_length=254,
                pattern=email_pattern,
                field_name="email",
            )

        @staticmethod
        def validate_url(url: str) -> FlextResult[str]:
            """Validate URL format using railway composition."""
            url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
            return FlextUtilities.Validation.validate_string(
                url,
                min_length=10,
                pattern=url_pattern,
                field_name="URL",
            )

        @staticmethod
        def validate_port(port: int | str) -> FlextResult[int]:
            """Validate network port number using explicit FlextResult patterns."""
            # First validate the input type and convert to integer explicitly
            port_conversion = FlextUtilities.Processing.convert_to_integer(port)
            if port_conversion.is_failure:
                return FlextResult[int].fail(
                    f"Port must be a valid integer, got {port}: {port_conversion.error}",
                )

            port_int = port_conversion.unwrap()

            # Then validate the port range
            if not (1 <= port_int <= FlextConstants.Network.MAX_PORT):
                return FlextResult[int].fail(
                    f"Port must be between 1 and {FlextConstants.Network.MAX_PORT}, got {port_int}",
                )
            return FlextResult[int].ok(port_int)

        @staticmethod
        def validate_environment_value(
            value: str,
            allowed_environments: list[str],
        ) -> FlextResult[str]:
            """Validate environment value against allowed list."""
            string_result = FlextUtilities.Validation.validate_string(
                value,
                min_length=1,
                field_name="environment",
            )
            if string_result.is_failure:
                return string_result

            env = string_result.unwrap()
            if env in allowed_environments:
                return FlextResult[str].ok(env)
            return FlextResult[str].fail(
                f"Environment must be one of {allowed_environments}, got '{env}'",
            )

        @staticmethod
        def validate_log_level(level: str) -> FlextResult[str]:
            """Validate log level value."""
            allowed_levels = list(FlextConstants.Logging.VALID_LEVELS)
            return FlextUtilities.Validation.validate_environment_value(
                level.upper(),
                allowed_levels,
            )

        @staticmethod
        def validate_security_token(token: str) -> FlextResult[str]:
            """Validate security token format and strength."""
            return FlextUtilities.Validation.validate_string(
                token,
                min_length=FlextConstants.Security.MIN_PASSWORD_LENGTH,
                field_name="security token",
            )

        @staticmethod
        def validate_connection_string(conn_str: str) -> FlextResult[str]:
            """Validate database connection string format."""
            return FlextUtilities.Validation.validate_string(
                conn_str,
                min_length=10,
                field_name="connection string",
            )

        @staticmethod
        def validate_directory_path(path: str) -> FlextResult[str]:
            """Validate directory path format."""
            # Check for null bytes and other illegal characters
            if "\x00" in path:
                return FlextResult[str].fail("directory path cannot contain null bytes")

            string_result = FlextUtilities.Validation.validate_string(
                path,
                min_length=1,
                field_name="directory path",
            )
            if string_result.is_failure:
                return string_result

            return FlextResult[str].ok(os.path.normpath(string_result.unwrap()))

        @staticmethod
        def validate_file_path(path: str) -> FlextResult[str]:
            """Validate file path format."""
            string_result = FlextUtilities.Validation.validate_string(
                path,
                min_length=1,
                field_name="file path",
            )
            if string_result.is_failure:
                return string_result

            return FlextResult[str].ok(os.path.normpath(string_result.unwrap()))

        @staticmethod
        def validate_existing_file_path(path: str) -> FlextResult[str]:
            """Validate that file path exists on filesystem."""
            file_path_result: FlextResult[str] = (
                FlextUtilities.Validation.validate_file_path(path)
            )
            if file_path_result.is_failure:
                return file_path_result

            p = file_path_result.unwrap()
            if pathlib.Path(p).is_file():
                return FlextResult[str].ok(p)
            return FlextResult[str].fail(f"file does not exist: {p}")

        @staticmethod
        def validate_timeout_seconds(timeout: float) -> FlextResult[float]:
            """Validate timeout value in seconds using explicit FlextResult patterns."""
            # First convert to float using explicit validation
            float_conversion = FlextUtilities.Processing.convert_to_float(timeout)
            if float_conversion.is_failure:
                return FlextResult[float].fail(
                    f"Timeout must be a valid number, got {timeout}: {float_conversion.error}",
                )

            timeout_float = float_conversion.unwrap()

            # Then validate the timeout constraints
            if timeout_float <= FlextConstants.Core.INITIAL_TIME:
                return FlextResult[float].fail(
                    f"Timeout must be positive, got {timeout_float}",
                )
            if timeout_float > FlextConstants.Utilities.MAX_TIMEOUT_SECONDS:
                return FlextResult[float].fail(
                    f"Timeout too large (max {FlextConstants.Utilities.MAX_TIMEOUT_SECONDS}s), got {timeout_float}",
                )
            return FlextResult[float].ok(timeout_float)

        @staticmethod
        def validate_retry_count(retries: int) -> FlextResult[int]:
            """Validate retry count value."""
            try:
                if retries < FlextConstants.Core.ZERO:
                    return FlextResult[int].fail(
                        f"Retry count cannot be negative, got {retries}",
                    )
                if retries > FlextConstants.Reliability.MAX_RETRY_ATTEMPTS:
                    return FlextResult[int].fail(
                        f"Retry count too high (max {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}), got {retries}",
                    )
                return FlextResult[int].ok(retries)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"Retry count must be a valid integer, got {retries}",
                )

        @staticmethod
        def validate_positive_integer(
            value: int,
            field_name: str = "value",
        ) -> FlextResult[int]:
            """Validate that value is a positive integer."""
            try:
                if value <= FlextConstants.Core.ZERO:
                    return FlextResult[int].fail(
                        f"{field_name} must be positive, got {value}",
                    )
                return FlextResult[int].ok(value)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"{field_name} must be a valid integer, got {value}",
                )

        @staticmethod
        def validate_non_negative_integer(
            value: int,
            field_name: str = "value",
        ) -> FlextResult[int]:
            """Validate that value is a non-negative integer."""
            try:
                if value < FlextConstants.Core.ZERO:
                    return FlextResult[int].fail(
                        f"{field_name} cannot be negative, got {value}",
                    )
                return FlextResult[int].ok(value)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"{field_name} must be a valid integer, got {value}",
                )

        @staticmethod
        def validate_host(host: str) -> FlextResult[str]:
            """Validate host name or IP address."""
            return FlextUtilities.Validation.validate_string(
                host,
                min_length=1,
                field_name="host",
            )

        @staticmethod
        def validate_http_status(status_code: int) -> FlextResult[int]:
            """Validate HTTP status code range."""
            try:
                min_http_status = FlextConstants.Platform.MIN_HTTP_STATUS_RANGE
                max_http_status = FlextConstants.Platform.MAX_HTTP_STATUS_RANGE
                if not (min_http_status <= status_code <= max_http_status):
                    return FlextResult[int].fail(
                        f"HTTP status code must be between {FlextConstants.Platform.MIN_HTTP_STATUS_RANGE} and {FlextConstants.Platform.MAX_HTTP_STATUS_RANGE}, got {status_code}",
                    )
                return FlextResult[int].ok(status_code)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"HTTP status code must be a valid integer, got {status_code}",
                )

        @staticmethod
        def is_non_empty_string(value: str) -> bool:
            """Check if string is non-empty after stripping."""
            return bool(value.strip())

        @staticmethod
        def validate_pipeline[TValidate](
            value: TValidate,
            validators: list[Callable[[TValidate], FlextResult[None]]],
        ) -> FlextResult[TValidate]:
            """Comprehensive validation pipeline using advanced railway patterns.

            Args:
                value: Value to validate
                validators: List of validation functions to apply

            Returns:
                Original value if all validations pass, accumulated errors otherwise

            """
            return FlextResult.validate_all(value, *validators)

        @staticmethod
        def validate_with_context[TContext](
            value: TContext,
            context_name: str,
            validator: Callable[[TContext], FlextResult[None]],
        ) -> FlextResult[TContext]:
            """Validate with enhanced error context using railway patterns.

            Args:
                value: Value to validate
                context_name: Context name for error messages
                validator: Validation function

            Returns:
                Value if validation passes, contextual error otherwise

            """
            validation_result: FlextResult[None] = validator(value)
            if validation_result.is_failure:
                return FlextResult[TContext].fail(
                    f"{context_name}: {validation_result.error}",
                )
            return FlextResult[TContext].ok(value)

        @staticmethod
        def validate_email_address(email: str) -> FlextResult[str]:
            """Enhanced email validation matching FlextModels.EmailAddress pattern."""
            if not email:
                return FlextResult[str].fail("Email cannot be empty")

            # Basic format validation - must have @ and domain part
            if "@" not in email:
                return FlextResult[str].fail("Invalid email format: missing @")

            parts = email.split("@", 1)
            if (
                len(parts) != FlextConstants.Validation.EMAIL_PARTS_COUNT
                or not parts[0]
                or not parts[1]
            ):
                return FlextResult[str].fail(f"Invalid email format: {email}")

            domain = parts[1]
            if "." not in domain:
                return FlextResult[str].fail("Invalid email format: missing domain dot")

            # Length validation using FlextConstants
            if len(email) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                return FlextResult[str].fail(
                    f"Email too long (max {FlextConstants.Validation.MAX_EMAIL_LENGTH} chars)",
                )

            return FlextResult[str].ok(email.lower())

        @staticmethod
        def validate_hostname(hostname: str) -> FlextResult[str]:
            """Validate hostname format matching FlextModels.Host pattern."""
            # Trim whitespace first
            hostname = hostname.strip()

            # Check if empty after trimming
            if not hostname:
                return FlextResult[str].fail("Hostname cannot be empty")

            # Basic hostname validation
            if len(hostname) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                return FlextResult[str].fail("Hostname too long")

            if not all(c.isalnum() or c in ".-" for c in hostname):
                return FlextResult[str].fail("Invalid hostname characters")

            # Check for consecutive dots or dashes
            if ".." in hostname or "--" in hostname:
                return FlextResult[str].fail(
                    "Invalid hostname format: consecutive dots or dashes",
                )

            # Check that hostname doesn't start or end with dot or dash
            if hostname.startswith((".", "-")) or hostname.endswith((".", "-")):
                return FlextResult[str].fail(
                    "Invalid hostname format: cannot start or end with dot or dash",
                )

            return FlextResult[str].ok(hostname.lower())

        @staticmethod
        def validate_entity_id(entity_id: str) -> FlextResult[str]:
            """Validate entity ID format matching FlextModels.EntityId pattern."""
            # Trim whitespace first
            entity_id = entity_id.strip()

            # Check if empty after trimming
            if not entity_id:
                return FlextResult[str].fail("Entity ID cannot be empty")

            # Check minimum length
            if len(entity_id) < FlextConstants.Validation.MIN_NAME_LENGTH:
                return FlextResult[str].fail(
                    f"Entity ID too short (min {FlextConstants.Validation.MIN_NAME_LENGTH} chars)",
                )

            # Allow UUIDs, alphanumeric with dashes/underscores
            if not re.match(r"^[a-zA-Z0-9_-]+$", entity_id):
                return FlextResult[str].fail("Invalid entity ID format")

            return FlextResult[str].ok(entity_id)

        @staticmethod
        def validate_phone_number(phone: str) -> FlextResult[str]:
            """Validate phone number with minimum digit requirement."""
            if not phone:
                return FlextResult[str].fail("Phone number cannot be empty")

            # Extract digits only for length validation
            digits_only = "".join(c for c in phone if c.isdigit())

            if len(digits_only) < FlextConstants.Validation.MIN_PHONE_DIGITS:
                return FlextResult[str].fail(
                    f"Phone number must have at least {FlextConstants.Validation.MIN_PHONE_DIGITS} digits",
                )

            return FlextResult[str].ok(phone)

        @staticmethod
        def validate_name_length(name: str) -> FlextResult[str]:
            """Validate name length using FlextConstants."""
            if not name:
                return FlextResult[str].fail("Name cannot be empty")

            if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
                return FlextResult[str].fail(
                    f"Name too short (min {FlextConstants.Validation.MIN_NAME_LENGTH} chars)",
                )

            if len(name) > FlextConstants.Validation.MAX_NAME_LENGTH:
                return FlextResult[str].fail(
                    f"Name too long (max {FlextConstants.Validation.MAX_NAME_LENGTH} chars)",
                )

            return FlextResult[str].ok(name.strip())

        @staticmethod
        def validate_bcrypt_rounds(rounds: int) -> FlextResult[int]:
            """Validate BCrypt rounds using FlextConstants."""
            if rounds < FlextConstants.Security.MIN_BCRYPT_ROUNDS:
                return FlextResult[int].fail(
                    f"BCrypt rounds too low (min {FlextConstants.Security.MIN_BCRYPT_ROUNDS})",
                )

            if rounds > FlextConstants.Security.MAX_BCRYPT_ROUNDS:
                return FlextResult[int].fail(
                    f"BCrypt rounds too high (max {FlextConstants.Security.MAX_BCRYPT_ROUNDS})",
                )

            return FlextResult[int].ok(rounds)

        @staticmethod
        def validate_data(
            data: dict[str, object],
            required_fields: list[str]
            | dict[str, type | tuple[type, ...]]
            | None = None,
        ) -> FlextResult[dict[str, object]]:
            """Validate dictionary data with optional required fields and type checking.

            Args:
                data: Dictionary to validate
                required_fields: Optional list of field names or dict mapping fields to types

            Returns:
                FlextResult[dict]: Validated data or error

            """
            if not isinstance(data, dict):
                return FlextResult[dict].fail("Data must be a dictionary")

            if required_fields:
                if isinstance(required_fields, list):
                    # Simple presence check
                    missing_fields = [
                        field for field in required_fields if field not in data
                    ]
                    if missing_fields:
                        return FlextResult[dict].fail(
                            f"Missing required fields: {', '.join(missing_fields)}"
                        )
                elif isinstance(required_fields, dict):
                    # Type validation
                    for field, expected_type in required_fields.items():
                        if field not in data:
                            return FlextResult[dict].fail(
                                f"Missing required field: {field}"
                            )
                        if not isinstance(data[field], expected_type):
                            return FlextResult[dict].fail(
                                f"Field '{field}' has incorrect type: expected {expected_type}, got {type(data[field])}"
                            )

            return FlextResult[dict].ok(data)

    class Transformation:
        """Data transformation utilities using railway composition."""

        @staticmethod
        def normalize_string(value: str) -> FlextResult[str]:
            """Normalize string by stripping whitespace and title casing."""
            return FlextResult[str].ok(value.strip().title())

        @staticmethod
        def sanitize_filename(filename: str) -> FlextResult[str]:
            """Sanitize filename by removing/replacing invalid characters."""
            validation_result = FlextUtilities.Validation.validate_string(
                filename,
                min_length=1,
                field_name="filename",
            )
            if validation_result.is_failure:
                return validation_result

            name: str = validation_result.unwrap()
            sanitized_name = re.sub(r'[<>: "/\\|?*]', "_", name)
            limited_name = sanitized_name[: FlextConstants.Validation.MAX_EMAIL_LENGTH]

            return FlextResult[str].ok(limited_name)

        @staticmethod
        def parse_comma_separated(value: str) -> FlextResult[list[str]]:
            """Parse comma-separated string into list."""
            validation_result = FlextUtilities.Validation.validate_string(
                value,
                min_length=1,
                field_name="comma-separated value",
            )
            if validation_result.is_failure:
                return FlextResult[list[str]].fail(
                    validation_result.error or "Validation failed",
                )

            v: str = validation_result.unwrap()
            items = [item.strip() for item in v.split(",") if item.strip()]
            return FlextResult[list[str]].ok(items)

        @staticmethod
        def format_error_message(
            error: str,
            context: str | None = None,
        ) -> FlextResult[str]:
            """Format error message with optional context."""
            validation_result = FlextUtilities.Validation.validate_string(
                error,
                min_length=1,
                field_name="error message",
            )
            if validation_result.is_failure:
                return validation_result

            msg: str = validation_result.unwrap()
            formatted_msg = f"{context}: {msg}" if context else msg
            return FlextResult[str].ok(formatted_msg)

        @staticmethod
        def safe_json_parse(json_string: str) -> FlextResult[dict]:
            """Parse JSON string with error handling using railway pattern.

            Args:
                json_string: JSON string to parse

            Returns:
                FlextResult[dict]: Parsed JSON data or error

            """
            if not json_string:
                return FlextResult[dict].fail("JSON string cannot be empty")

            try:
                data = json.loads(json_string)
                if not isinstance(data, dict):
                    return FlextResult[dict].fail("Parsed JSON is not a dictionary")
                return FlextResult[dict].ok(data)
            except json.JSONDecodeError as e:
                return FlextResult[dict].fail(f"JSON parse error: {e}")
            except Exception as e:
                return FlextResult[dict].fail(f"Unexpected error parsing JSON: {e}")

        @staticmethod
        def safe_json_stringify(data: dict) -> FlextResult[str]:
            """Stringify data to JSON with error handling using railway pattern.

            Args:
                data: Dictionary to convert to JSON

            Returns:
                FlextResult[str]: JSON string or error

            """
            if not isinstance(data, dict):
                return FlextResult[str].fail("Data must be a dictionary")

            try:
                json_string = json.dumps(data)
                return FlextResult[str].ok(json_string)
            except TypeError as e:
                return FlextResult[str].fail(f"JSON stringify error: {e}")
            except Exception as e:
                return FlextResult[str].fail(f"Unexpected error stringifying JSON: {e}")

    class Processing:
        """Processing utilities with reliability patterns."""

        @staticmethod
        def _get_circuit_breaker_state(operation_id: str) -> dict[str, object]:
            """Get circuit breaker state for operation."""
            # This is a simplified implementation
            # In a real implementation, this would track circuit breaker states
            return {
                "operation_id": operation_id,
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "next_retry_time": None,
            }

        @staticmethod
        def retry_operation[T](
            operation: Callable[[], FlextResult[T]],
            max_retries: int = 3,
            delay_seconds: float = 1.0,
        ) -> FlextResult[T]:
            """Retry operation with exponential backoff using advanced railway pattern."""
            # Validate parameters first
            retry_validation = FlextUtilities.Validation.validate_retry_count(
                max_retries,
            )
            if retry_validation.is_failure:
                error_msg = retry_validation.error or "Invalid retry count"
                return FlextResult[T].fail(error_msg)

            delay_validation = FlextUtilities.Validation.validate_timeout_seconds(
                delay_seconds,
            )
            if delay_validation.is_failure:
                error_msg = delay_validation.error or "Invalid delay seconds"
                return FlextResult[T].fail(error_msg)

            # Implement retry logic with exponential backoff
            result: FlextResult[T] = FlextResult[T].fail("No attempts made")

            for attempt in range(max_retries + 1):
                result = operation()
                if result.is_success:
                    return result

                if attempt < max_retries:  # Don't sleep after the last attempt
                    time.sleep(delay_seconds * (2**attempt))  # Exponential backoff

            return result  # Return the last failure result

        @staticmethod
        def timeout_operation[T](
            operation: Callable[[], FlextResult[T]],
            timeout_seconds: float = FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS,
        ) -> FlextResult[T]:
            """Execute operation with timeout using advanced railway pattern."""
            # Validate timeout first
            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                timeout_seconds,
            )
            if timeout_validation.is_failure:
                return FlextResult[T].fail(
                    timeout_validation.error or "Invalid timeout",
                )

            # Use the reliability pattern instead of FlextResult.with_timeout
            return FlextUtilities.Reliability.with_timeout(operation, timeout_seconds)

        @classmethod
        def circuit_breaker[T](
            cls,
            operation: Callable[[], FlextResult[T]],
            failure_threshold: int = FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD,
            recovery_timeout: float = FlextConstants.Reliability.DEFAULT_RECOVERY_TIMEOUT,
        ) -> FlextResult[T]:
            """Circuit breaker pattern implementation with config integration."""
            # Check if circuit breaker is enabled in configuration
            config = FlextConfig.get_global_instance()
            if not config.enable_circuit_breaker:
                # Circuit breaker disabled - execute operation directly
                try:
                    return operation()
                except Exception as e:
                    return FlextResult[T].fail(f"Operation failed: {e}")

            threshold_validation = FlextUtilities.Validation.validate_retry_count(
                failure_threshold,
            )
            if threshold_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid failure threshold: {threshold_validation.error}",
                )

            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                recovery_timeout,
            )
            if timeout_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid recovery timeout: {timeout_validation.error}",
                )

            # Get or create circuit breaker state for this operation
            operation_id = f"{operation.__name__ if hasattr(operation, '__name__') else 'anonymous'}_{id(operation)}"
            state = cls._get_circuit_breaker_state(operation_id)

            # Check current circuit state with proper type casting
            current_time = time.time()
            # Extract circuit state with type checking
            circuit_state_raw = state["circuit_state"]
            circuit_state = (
                str(circuit_state_raw) if circuit_state_raw is not None else "CLOSED"
            )

            # Extract last failure time with type checking
            last_failure_time_raw = state["last_failure_time"]
            if isinstance(last_failure_time_raw, (int, float)):
                last_failure_time = float(last_failure_time_raw)
            else:
                last_failure_time = 0.0

            # Handle OPEN state (circuit is open, failures exceeded threshold)
            if circuit_state == "OPEN":
                if current_time - last_failure_time >= recovery_timeout:
                    # Transition to HALF_OPEN state
                    state["circuit_state"] = "HALF_OPEN"
                    state["failure_count"] = 0
                else:
                    # Circuit still open - reject immediately
                    return FlextResult[T].fail(
                        f"Circuit breaker OPEN - threshold {failure_threshold} exceeded. "
                        f"Next retry in {recovery_timeout - (current_time - last_failure_time):.1f}s",
                    )

            # Execute operation (CLOSED or HALF_OPEN state)
            try:
                result: FlextResult[T] = operation()

                if result.is_success:
                    # Operation succeeded - reset failure count and close circuit
                    state["failure_count"] = 0
                    state["circuit_state"] = "CLOSED"
                    state["last_success_time"] = current_time
                    return result
                # Operation failed - increment failure count
                current_failure_count_raw = state["failure_count"]
                current_failure_count = (
                    int(current_failure_count_raw)
                    if isinstance(current_failure_count_raw, (int, float))
                    else 0
                )
                state["failure_count"] = current_failure_count + 1
                state["last_failure_time"] = current_time

                new_failure_count = current_failure_count + 1
                if new_failure_count >= failure_threshold:
                    # Open circuit - failures exceeded threshold
                    state["circuit_state"] = "OPEN"
                    return FlextResult[T].fail(
                        f"Circuit breaker OPENED - failure threshold {failure_threshold} exceeded. "
                        f"Error: {result.error}",
                    )

                return result

            except Exception as e:
                # Exception during operation - treat as failure
                current_failure_count_raw = state["failure_count"]
                current_failure_count = (
                    int(current_failure_count_raw)
                    if isinstance(current_failure_count_raw, (int, float))
                    else 0
                )
                state["failure_count"] = current_failure_count + 1
                state["last_failure_time"] = current_time

                new_failure_count = current_failure_count + 1
                if new_failure_count >= failure_threshold:
                    state["circuit_state"] = "OPEN"
                    return FlextResult[T].fail(
                        f"Circuit breaker OPENED - failure threshold {failure_threshold} exceeded. "
                        f"Exception: {e}",
                    )

                return FlextResult[T].fail(f"Circuit breaker operation failed: {e}")

        # Circuit breaker state management (class-level state)
        _circuit_breaker_states: ClassVar[dict[str, dict[str, object]]] = {}
        _circuit_breaker_lock: ClassVar[threading.Lock] = threading.Lock()

        @classmethod
        def get_circuit_breaker_state(cls, operation_id: str) -> dict[str, object]:
            """Get or create circuit breaker state for an operation."""
            with cls._circuit_breaker_lock:
                if operation_id not in cls._circuit_breaker_states:
                    cls._circuit_breaker_states[operation_id] = {
                        "circuit_state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                        "failure_count": 0,
                        "last_failure_time": 0.0,
                        "last_success_time": time.time(),
                    }
                return cls._circuit_breaker_states[operation_id]

        @staticmethod
        def validate_regex_pattern(pattern: str) -> FlextResult[re.Pattern[str]]:
            """Validate and compile a regex pattern using explicit FlextResult handling.

            This replaces try/except patterns with explicit FlextResult error handling
            following the CLAUDE.md architectural standards.

            Args:
                pattern: Regular expression pattern to validate and compile

            Returns:
                FlextResult containing compiled pattern or validation error

            """
            if not pattern:
                return FlextResult[re.Pattern[str]].fail("Pattern cannot be empty")

            # Type annotation guarantees pattern is str, isinstance check unnecessary

            # Check for basic pattern validity before compilation
            if len(pattern) > FlextConstants.Utilities.MAX_REGEX_PATTERN_LENGTH:
                return FlextResult[re.Pattern[str]].fail(
                    "Pattern too long (max 1000 characters)",
                )

            # Use explicit error checking instead of try/except
            # Compile pattern and check for errors using direct validation
            try:
                compiled_pattern = re.compile(pattern)
                return FlextResult[re.Pattern[str]].ok(compiled_pattern)
            except re.error as e:
                # This try/except is acceptable for interfacing with external libraries
                # that don't provide non-exception APIs for validation
                return FlextResult[re.Pattern[str]].fail(f"Invalid regex pattern: {e}")

        @staticmethod
        def convert_to_integer(value: int | str) -> FlextResult[int]:
            """Convert value to integer using explicit FlextResult handling.

            This replaces try/except patterns with explicit validation following
            the CLAUDE.md architectural standards.

            Args:
                value: Value to convert to integer (int or string)

            Returns:
                FlextResult containing converted integer or conversion error

            """
            if isinstance(value, int):
                return FlextResult[int].ok(value)

            # value is str at this point due to type annotation int | str
            # Type checking already ensures this

            # Validate string before conversion
            cleaned_value = value.strip()
            if not cleaned_value:
                return FlextResult[int].fail("Cannot convert empty string to integer")

            # Check for obvious non-numeric patterns
            if not cleaned_value.lstrip("-+").isdigit():
                return FlextResult[int].fail(
                    f"String '{value}' does not represent a valid integer",
                )

            # Use minimal try/except only for interfacing with built-in int()
            # which doesn't provide non-exception validation API
            try:
                converted_int = int(cleaned_value)
                return FlextResult[int].ok(converted_int)
            except (ValueError, OverflowError) as e:
                # This try/except is acceptable for interfacing with built-in functions
                # that don't provide non-exception APIs for validation
                return FlextResult[int].fail(
                    f"Cannot convert '{value}' to integer: {e}",
                )

        @staticmethod
        def convert_to_float(value: float | str) -> FlextResult[float]:
            """Convert value to float using explicit FlextResult handling.

            This replaces try/except patterns with explicit validation following
            the CLAUDE.md architectural standards.

            Args:
                value: Value to convert to float (float, int, or string)

            Returns:
                FlextResult containing converted float or conversion error

            """
            if isinstance(value, (float, int)):
                return FlextResult[float].ok(float(value))

            # value is str at this point due to type annotation
            # Type checking already ensures this

            # Validate string before conversion
            cleaned_value = value.strip()
            if not cleaned_value:
                return FlextResult[float].fail("Cannot convert empty string to float")

            # Check for obvious non-numeric patterns (basic validation)
            if cleaned_value.lower() in {"inf", "+inf", "-inf", "nan"}:
                return FlextResult[float].fail(
                    f"Special float values not allowed: {value}",
                )

            # Use minimal try/except only for interfacing with built-in float()
            # which doesn't provide non-exception validation API
            try:
                converted_float = float(cleaned_value)
                # Check for infinity and NaN after conversion
                if not math.isfinite(converted_float):
                    return FlextResult[float].fail(
                        f"Infinite or NaN values not allowed: {value}",
                    )
                return FlextResult[float].ok(converted_float)
            except (ValueError, OverflowError) as e:
                # This try/except is acceptable for interfacing with built-in functions
                # that don't provide non-exception APIs for validation
                return FlextResult[float].fail(
                    f"Cannot convert '{value}' to float: {e}",
                )

    class Utilities:
        """General utility functions using railway composition."""

        @staticmethod
        def safe_cast[T](
            value: object,
            target_type: type[T],
            field_name: str = "value",
        ) -> FlextResult[T]:
            """Safely cast value to target type."""
            try:
                if isinstance(value, target_type):
                    return FlextResult[T].ok(value)

                # For object type, return value as-is
                if target_type is object:
                    # Type safety: object type can hold any value
                    return FlextResult[T].ok(cast("T", value))

                # For basic types, try to construct with value
                if target_type is str:
                    converted_str = str(value)
                    return FlextResult[T].ok(cast("T", converted_str))
                if target_type is int:
                    converted_int = int(str(value))
                    return FlextResult[T].ok(cast("T", converted_int))
                if target_type is float:
                    converted_float = float(str(value))
                    return FlextResult[T].ok(cast("T", converted_float))
                if target_type is bool:
                    converted_bool = bool(value)
                    return FlextResult[T].ok(cast("T", converted_bool))

                # For other types, try to cast directly
                try:
                    # For other types with constructors, try calling them
                    # Skip object type as it doesn't accept arguments
                    if callable(target_type) and target_type is not object:
                        # Use proper type handling for constructor call
                        try:
                            converted_value = cast(
                                "Callable[[object], T]",
                                target_type,
                            )(value)
                            return FlextResult[T].ok(converted_value)
                        except (TypeError, ValueError):
                            # If constructor fails with args, try no-arg constructor
                            if callable(target_type) and target_type is not type:
                                converted_value = target_type()
                                return FlextResult[T].ok(converted_value)
                            raise
                    # If no constructor or object type, return the value with proper type annotation
                    return FlextResult[T].ok(cast("T", value))
                except (TypeError, ValueError):
                    # If constructor fails, return the value with explicit type annotation
                    return FlextResult[T].ok(cast("T", value))

            except (ValueError, TypeError) as e:
                type_name = getattr(target_type, "__name__", str(target_type))
                return FlextResult[T].fail(
                    f"Cannot cast {field_name} to {type_name}: {e}",
                )

        @staticmethod
        def merge_dictionaries(
            *dicts: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Merge multiple dictionaries with conflict detection."""
            # Merge with conflict detection
            result: dict[str, object] = {}
            conflicts: list[str] = []

            for d in dicts:
                for key, value in d.items():
                    if key in result and result[key] != value:
                        conflicts.append(f"Key '{key}' has conflicting values")
                    result[key] = value

            if conflicts:
                return FlextResult[dict[str, object]].fail(
                    f"Dictionary merge conflicts: {'; '.join(conflicts)}",
                )

            return FlextResult[dict[str, object]].ok(result)

        @staticmethod
        def deep_get(
            data: dict[str, object],
            path: str,
            default: object = None,
        ) -> FlextResult[object]:
            """Safely get nested dictionary value using dot notation."""
            try:
                # Handle empty path - return the entire data
                if not path:
                    return FlextResult[object].ok(data)

                keys = path.split(".")
                current: object = data  # Start with the data object
                for key in keys:
                    if isinstance(current, dict):
                        # Type narrow: current is now dict[str, object] due to isinstance check
                        dict_current = current
                        if key in dict_current:
                            current_value = dict_current[key]
                            current = current_value
                        else:
                            return FlextResult[object].ok(default)
                    else:
                        return FlextResult[object].ok(default)
                return FlextResult[object].ok(current)
            except Exception as e:
                return FlextResult[object].fail(f"Error accessing path '{path}': {e}")

        @staticmethod
        def ensure_list(value: object) -> FlextResult[list[object]]:
            """Ensure value is a list, wrapping single values."""
            if isinstance(value, list):
                return FlextResult[list[object]].ok(cast("list[object]", value))
            if value is None:
                return FlextResult[list[object]].ok([])
            return FlextResult[list[object]].ok([value])

        @staticmethod
        def filter_none_values(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Remove None values from dictionary."""
            return FlextResult[dict[str, object]].ok({
                k: v for k, v in data.items() if v is not None
            })

        @staticmethod
        def batch_process(
            items: list[T],
            processor: Callable[[T], FlextResult[U]],
            batch_size: int = 100,
            *,
            fail_fast: bool = False,
        ) -> FlextResult[list[U]]:
            """Process items in batches with advanced railway pattern error handling.

            Args:
                items: Items to process
                processor: Function to process each item
                batch_size: Number of items per batch
                fail_fast: If True, stop on first error; if False, accumulate all errors

            """
            # Validate batch size
            if batch_size <= FlextConstants.Core.ZERO:
                return FlextResult[list[U]].fail("Batch size must be positive")

            return FlextUtilities.Utilities.process_batches_railway(
                items,
                processor,
                batch_size,
                fail_fast=fail_fast,
            )

        @staticmethod
        def process_batches_railway(
            items: list[T],
            processor: Callable[[T], FlextResult[U]],
            batch_size: int,
            *,
            fail_fast: bool,
        ) -> FlextResult[list[U]]:
            """Internal method for railway-based batch processing."""
            if not items:
                return FlextResult[list[U]].ok([])

            # Create batches
            batches = [
                items[i : i + batch_size] for i in range(0, len(items), batch_size)
            ]

            # Process each batch using railway patterns
            if fail_fast:
                # Use sequence for early termination
                def _identity(x: list[FlextResult[U]]) -> list[FlextResult[U]]:
                    return x

                batch_results: list[FlextResult[list[U]]] = [
                    typing.cast(
                        "FlextResult[list[U]]",
                        getattr(FlextResult, "sequence", _identity)([
                            processor(item) for item in batch
                        ]),
                    )
                    for batch in batches
                ]

                def flatten_results(nested_results: list[list[U]]) -> list[U]:
                    return [
                        item for batch_result in nested_results for item in batch_result
                    ]

                def _identity2(
                    x: list[FlextResult[list[U]]],
                ) -> list[FlextResult[list[U]]]:
                    return x

                # Fix: The sequence returns FlextResult[list[list[U]]] so we need the correct type annotation
                sequence_result: FlextResult[list[list[U]]] = cast(
                    "FlextResult[list[list[U]]]",
                    getattr(FlextResult, "sequence", _identity2)(batch_results),
                )
                return sequence_result.map(flatten_results)
            # Use accumulate_errors for collecting all errors
            all_results: list[FlextResult[U]] = [
                processor(item) for batch in batches for item in batch
            ]
            return FlextResult.accumulate_errors(*all_results)

    class Cache:
        """Cache management utilities for FlextMixins and other components.

        Extended with CQRS cache functionality for command/query result caching.
        """

        @staticmethod
        def clear_object_cache(obj: object) -> FlextResult[None]:
            """Clear cache for object if it has cache-related attributes.

            Args:
                obj: Object to clear cache for

            Returns:
                FlextResult indicating success or failure

            """
            try:
                # Common cache attribute names to check and clear
                cache_attributes = [
                    "_cache",
                    "__cache__",
                    "cache",
                    "_cached_data",
                    "_memoized",
                ]

                cleared_count = 0
                for attr_name in cache_attributes:
                    if hasattr(obj, attr_name):
                        cache_attr = getattr(obj, attr_name, None)
                        if cache_attr is not None:
                            # Clear dict-like caches
                            if hasattr(cache_attr, "clear") and callable(
                                cache_attr.clear,
                            ):
                                cache_attr.clear()
                                cleared_count += 1
                            # Reset to None for simple cached values
                            else:
                                setattr(obj, attr_name, None)
                                cleared_count += 1

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Failed to clear cache: {e}")

        @staticmethod
        def has_cache_attributes(obj: object) -> bool:
            """Check if object has any cache-related attributes.

            Args:
                obj: Object to check for cache attributes

            Returns:
                True if object has cache attributes, False otherwise

            """
            cache_attributes = [
                "_cache",
                "__cache__",
                "cache",
                "_cached_data",
                "_memoized",
            ]

            return any(hasattr(obj, attr) for attr in cache_attributes)

        @staticmethod
        def sort_key(value: object) -> str:
            """Return a deterministic string for ordering normalized cache components."""
            return json.dumps(value, sort_keys=True, default=str)

        @staticmethod
        def normalize_component(value: object) -> object:
            """Normalize arbitrary objects into cache-friendly deterministic structures."""
            if value is None or isinstance(value, (bool, int, float, str)):
                return value

            if isinstance(value, bytes):
                return ("bytes", value.hex())

            if isinstance(value, HasModelDump):
                try:
                    dumped: dict[str, object] = value.model_dump()
                except TypeError:
                    dumped = {}
                return ("pydantic", FlextUtilities.Cache.normalize_component(dumped))

            if is_dataclass(value):
                # Ensure we have a dataclass instance, not a class
                if isinstance(value, type):
                    return ("dataclass_class", str(value))
                return (
                    "dataclass",
                    FlextUtilities.Cache.normalize_component(asdict(value)),
                )

            if isinstance(value, Mapping):
                # Return sorted dict for cache-friendly deterministic ordering
                mapping_value = cast("Mapping[object, object]", value)
                sorted_items = sorted(
                    mapping_value.items(),
                    key=lambda x: FlextUtilities.Cache.sort_key(x[0]),
                )
                return {
                    FlextUtilities.Cache.normalize_component(
                        k,
                    ): FlextUtilities.Cache.normalize_component(v)
                    for k, v in sorted_items
                }

            if isinstance(value, (list, tuple)):
                sequence_value = cast("Sequence[object]", value)
                sequence_items = [
                    FlextUtilities.Cache.normalize_component(item)
                    for item in sequence_value
                ]
                return ("sequence", tuple(sequence_items))

            if isinstance(value, set):
                set_value = cast("set[object]", value)
                set_items = [
                    FlextUtilities.Cache.normalize_component(item) for item in set_value
                ]

                # Sort by cache sort key
                set_items.sort(key=FlextUtilities.Cache.sort_key)

                normalized_set = tuple(set_items)
                return ("set", normalized_set)

            try:
                # Cast to proper type for type checker
                value_vars_dict: dict[str, object] = cast(
                    "Dict",
                    vars(value),
                )
            except TypeError:
                return ("repr", repr(value))

            normalized_vars = tuple(
                (key, FlextUtilities.Cache.normalize_component(val))
                for key, val in sorted(
                    value_vars_dict.items(),
                    key=operator.itemgetter(0),
                )
            )
            return ("vars", normalized_vars)

        @staticmethod
        def generate_cache_key(command: object, command_type: type[object]) -> str:
            """Generate a deterministic cache key for the command.

            Args:
                command: The command/query object
                command_type: The type of the command

            Returns:
                str: Deterministic cache key

            """
            try:
                # For Pydantic models, use model_dump with sorted keys
                if isinstance(command, HasModelDump):
                    data = command.model_dump(mode="python")
                    # Sort keys recursively for deterministic ordering
                    sorted_data = FlextUtilities.Cache.sort_dict_keys(data)
                    return f"{command_type.__name__}_{hash(str(sorted_data))}"

                # For dataclasses, use asdict with sorted keys
                if (
                    hasattr(command, "__dataclass_fields__")
                    and is_dataclass(command)
                    and not isinstance(command, type)
                ):
                    dataclass_data = asdict(command)
                    dataclass_sorted_data = FlextUtilities.Cache.sort_dict_keys(
                        dataclass_data,
                    )
                    return f"{command_type.__name__}_{hash(str(dataclass_sorted_data))}"

                # For dictionaries, sort keys
                if isinstance(command, dict):
                    dict_sorted_data = FlextUtilities.Cache.sort_dict_keys(
                        cast("Dict", command),
                    )
                    return f"{command_type.__name__}_{hash(str(dict_sorted_data))}"

                # For other objects, use string representation
                command_str = str(command) if command is not None else "None"
                command_hash = hash(command_str)
                return f"{command_type.__name__}_{command_hash}"

            except Exception:
                # Fallback to string representation if anything fails
                command_str_fallback = str(command) if command is not None else "None"
                try:
                    command_hash_fallback = hash(command_str_fallback)
                    return f"{command_type.__name__}_{command_hash_fallback}"
                except TypeError:
                    # If hash fails, use a deterministic fallback
                    return f"{command_type.__name__}_{abs(hash(command_str_fallback.encode('utf-8')))}"

        @staticmethod
        def sort_dict_keys(obj: object) -> object:
            """Recursively sort dictionary keys for deterministic ordering.

            Args:
                obj: Object to sort (dict, list, or other)

            Returns:
                Object with sorted keys

            """
            if isinstance(obj, dict):
                dict_obj: dict[str, object] = cast("Dict", obj)
                sorted_items: list[tuple[object, object]] = sorted(
                    dict_obj.items(),
                    key=lambda x: str(x[0]),
                )
                return {
                    str(k): FlextUtilities.Cache.sort_dict_keys(v)
                    for k, v in sorted_items
                }
            if isinstance(obj, list):
                obj_list: list[object] = cast("list[object]", obj)
                return [FlextUtilities.Cache.sort_dict_keys(item) for item in obj_list]
            if isinstance(obj, tuple):
                obj_tuple: tuple[object, ...] = cast("tuple[object, ...]", obj)
                return tuple(
                    FlextUtilities.Cache.sort_dict_keys(item) for item in obj_tuple
                )
            return obj

    class CqrsCache:
        """CQRS-specific cache manager for command/query result caching."""

        def __init__(
            self,
            max_size: int = FlextConstants.Performance.DEFAULT_CACHE_SIZE,
        ) -> None:
            """Initialize CQRS cache manager.

            Args:
                max_size: Maximum number of cached results

            """
            self._cache: OrderedDict[str, FlextResult[object]] = OrderedDict()
            self._max_size = max_size

        def get(self, key: str) -> FlextResult[object] | None:
            """Get cached result by key.

            Args:
                key: Cache key

            Returns:
                Cached result or None if not found

            """
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

        def put(self, key: str, result: FlextResult[object]) -> None:
            """Store result in cache.

            Args:
                key: Cache key
                result: Result to cache

            """
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        def clear(self) -> None:
            """Clear all cached results."""
            self._cache.clear()

        def size(self) -> int:
            """Get current cache size.

            Returns:
                Number of cached items

            """
            return len(self._cache)

    class Generators:
        """ID and data generation utilities."""

        @staticmethod
        def generate_id() -> str:
            """Generate a unique ID using UUID4."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_uuid() -> str:
            """Generate a UUID string."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_timestamp() -> str:
            """Generate ISO format timestamp."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO format timestamp."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate a correlation ID for tracking."""
            return f"corr_{str(uuid.uuid4())[:8]}"

        @staticmethod
        def generate_short_id(length: int = 8) -> str:
            """Generate a short random ID."""
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(length))

        @staticmethod
        def generate_entity_id() -> str:
            """Generate a unique entity ID for domain entities.

            Returns:
                A unique entity identifier suitable for domain entities

            """
            return str(uuid.uuid4())

        @staticmethod
        def create_module_utilities(module_name: str) -> FlextResult[object]:
            """Create utilities for a specific module.

            Args:
                module_name: Name of the module to create utilities for

            Returns:
                FlextResult containing module utilities or error

            """
            if not module_name:
                return FlextResult[object].fail(
                    "Module name must be a non-empty string",
                )

            # For now, return a simple utilities object
            # This can be expanded with actual module-specific functionality
            utilities = type(
                f"{module_name}_utilities",
                (),
                {
                    "module_name": module_name,
                    "logger": lambda: f"Logger for {module_name}",
                    "config": lambda: f"Config for {module_name}",
                },
            )()

            return FlextResult[object].ok(utilities)

        @staticmethod
        def generate_correlation_id_with_context(context: str) -> str:
            """Generate a correlation ID with context prefix."""
            return f"{context}_{str(uuid.uuid4())[:8]}"

        @staticmethod
        def generate_batch_id(batch_size: int) -> str:
            """Generate a batch ID with size information."""
            return f"batch_{batch_size}_{str(uuid.uuid4())[:8]}"

        @staticmethod
        def generate_transaction_id() -> str:
            """Generate a transaction ID for distributed transactions."""
            return f"txn_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_saga_id() -> str:
            """Generate a saga ID for distributed transaction patterns."""
            return f"saga_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_event_id() -> str:
            """Generate an event ID for domain events."""
            return f"evt_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_command_id() -> str:
            """Generate a command ID for CQRS patterns."""
            return f"cmd_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_query_id() -> str:
            """Generate a query ID for CQRS patterns."""
            return f"qry_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_aggregate_id(aggregate_type: str) -> str:
            """Generate an aggregate ID with type prefix."""
            return f"{aggregate_type}_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_entity_version() -> int:
            """Generate an entity version number."""
            return int(datetime.now(UTC).timestamp() * 1000) % 1000000

    class Correlation:
        """Distributed tracing and correlation ID management."""

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate a correlation ID for tracking."""
            return FlextUtilities.Generators.generate_correlation_id()

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO format timestamp."""
            return FlextUtilities.Generators.generate_iso_timestamp()

        @staticmethod
        def generate_command_id() -> str:
            """Generate a command ID for CQRS patterns."""
            return FlextUtilities.Generators.generate_command_id()

        @staticmethod
        def generate_query_id() -> str:
            """Generate a query ID for CQRS patterns."""
            return FlextUtilities.Generators.generate_query_id()

    class TextProcessor:
        """Text processing utilities using railway composition."""

        @staticmethod
        def clean_text(text: str) -> FlextResult[str]:
            """Clean text by removing extra whitespace and control characters."""
            # Remove control characters except tab and newline
            cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
            # Normalize whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            return FlextResult[str].ok(cleaned)

        @staticmethod
        def truncate_text(
            text: str,
            max_length: int = 100,
            suffix: str = "...",
        ) -> FlextResult[str]:
            """Truncate text to maximum length with suffix."""
            if len(text) <= max_length:
                return FlextResult[str].ok(text)

            truncated = text[: max_length - len(suffix)] + suffix
            return FlextResult[str].ok(truncated)

        @staticmethod
        def safe_string(
            text: str,
            default: str = FlextConstants.Performance.DEFAULT_EMPTY_STRING,
        ) -> str:
            """Convert text to safe string, handling None and empty values.

            Args:
                text: Text to make safe
                default: Default value if text is None or empty

            Returns:
                Safe string value

            """
            if not text:
                return default
            return text.strip()

    # Note: This class handles type conversions (str->bool, str->int), while "Conversion" handles table formatting
    class TypeConversions:
        """Type conversion utilities using railway composition."""

        @staticmethod
        def to_bool(*, value: str | bool | int | None) -> FlextResult[bool]:
            """Convert value to boolean using railway composition."""
            if isinstance(value, bool):
                return FlextResult[bool].ok(value)

            if isinstance(value, str):
                lower_value = value.lower().strip()
                if lower_value in {"true", "1", "yes", "on", "enabled"}:
                    return FlextResult[bool].ok(True)
                if lower_value in {"false", "0", "no", "off", "disabled", ""}:
                    return FlextResult[bool].ok(False)
                return FlextResult[bool].fail(f"Cannot convert '{value}' to boolean")

            if isinstance(value, int):
                return FlextResult[bool].ok(bool(value))

            # value is None case
            return FlextResult[bool].ok(False)

        @staticmethod
        def to_int(value: str | float | None) -> FlextResult[int]:
            """Convert value to integer using railway composition."""
            if value is None:
                return FlextResult[int].fail("Cannot convert None to integer")

            try:
                if isinstance(value, int):
                    return FlextResult[int].ok(value)
                return FlextResult[int].ok(int(value))
            except (ValueError, TypeError) as e:
                return FlextResult[int].fail(f"Integer conversion failed: {e}")

    class Reliability:
        """Reliability patterns for resilient operations."""

        @staticmethod
        def retry_with_backoff(
            operation: Callable[[], FlextResult[T]],
            max_retries: int = 3,
            initial_delay: float = 1.0,
            backoff_factor: float = 2.0,
        ) -> FlextResult[T]:
            """Enhanced retry with exponential backoff using railway patterns."""
            # Simple implementation that tries the operation multiple times
            last_error = "Operation failed"

            for attempt in range(max_retries):
                try:
                    result: FlextResult[T] = operation()
                    if result.is_success:
                        return result
                    last_error = result.error or f"Attempt {attempt + 1} failed"

                    # Add delay before next attempt (exponential backoff)
                    if attempt < max_retries - 1:  # Don't wait after last attempt
                        time.sleep(initial_delay * (backoff_factor**attempt))

                except Exception as e:
                    last_error = f"Exception in attempt {attempt + 1}: {e}"

            return FlextResult[T].fail(
                f"All {max_retries} attempts failed. Last error: {last_error}",
            )

        @staticmethod
        def with_timeout[TTimeout](
            operation: Callable[[], FlextResult[TTimeout]],
            timeout_seconds: float,
        ) -> FlextResult[TTimeout]:
            """Execute operation with timeout using railway patterns."""
            if timeout_seconds <= FlextConstants.Core.INITIAL_TIME:
                return FlextResult[TTimeout].fail("Timeout must be positive")

            # Use proper typing for containers
            result_container: list[FlextResult[TTimeout] | None] = [None]
            exception_container: list[Exception | None] = [None]

            def run_operation() -> None:
                try:
                    result_container[0] = operation()
                except Exception as e:
                    exception_container[0] = e

            # Copy current context to the new thread
            context = contextvars.copy_context()
            thread = threading.Thread(target=context.run, args=(run_operation,))
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Thread is still running, timeout occurred
                return FlextResult[TTimeout].fail(
                    f"Operation timed out after {timeout_seconds} seconds",
                )

            if exception_container[0]:
                return FlextResult[TTimeout].fail(
                    f"Operation failed with exception: {exception_container[0]}",
                )

            if result_container[0] is None:
                return FlextResult[TTimeout].fail(
                    "Operation completed but returned no result",
                )

            result = result_container[0]
            if result is None:
                return FlextResult[TTimeout].fail(
                    "Operation completed but returned no result",
                )
            return result

        @staticmethod
        def circuit_breaker[TCircuit](
            operation: Callable[[], FlextResult[TCircuit]],
            failure_threshold: int = FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD,
            recovery_timeout: float = FlextConstants.Reliability.DEFAULT_RECOVERY_TIMEOUT,
        ) -> FlextResult[TCircuit]:
            """Circuit breaker pattern using railway composition with config integration."""
            # Check if circuit breaker is enabled in configuration
            config = FlextConfig.get_global_instance()
            if not config.enable_circuit_breaker:
                # Circuit breaker disabled - execute operation directly
                try:
                    return operation()
                except Exception as e:
                    return FlextResult[TCircuit].fail(f"Operation failed: {e}")

            threshold_validation = FlextUtilities.Validation.validate_retry_count(
                failure_threshold,
            )
            if threshold_validation.is_failure:
                return FlextResult[TCircuit].fail(
                    f"Invalid failure threshold: {threshold_validation.error}",
                )

            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                recovery_timeout,
            )
            if timeout_validation.is_failure:
                return FlextResult[TCircuit].fail(
                    f"Invalid recovery timeout: {timeout_validation.error}",
                )

            # Get or create circuit breaker state for this operation
            operation_id = f"{operation.__name__ if hasattr(operation, '__name__') else 'anonymous'}_{id(operation)}"
            state = FlextUtilities.Processing.get_circuit_breaker_state(operation_id)

            # Check current circuit state with proper type casting
            current_time = time.time()
            # Extract circuit state with type checking
            circuit_state_raw = state["circuit_state"]
            circuit_state = (
                str(circuit_state_raw) if circuit_state_raw is not None else "CLOSED"
            )

            # Extract last failure time with type checking
            last_failure_time_raw = state["last_failure_time"]
            if isinstance(last_failure_time_raw, (int, float)):
                last_failure_time = float(last_failure_time_raw)
            else:
                last_failure_time = 0.0

            # Handle OPEN state (circuit is open, failures exceeded threshold)
            if circuit_state == "OPEN":
                if current_time - last_failure_time >= recovery_timeout:
                    # Transition to HALF_OPEN state
                    state["circuit_state"] = "HALF_OPEN"
                    state["failure_count"] = 0
                else:
                    # Circuit still open - reject immediately
                    return FlextResult[TCircuit].fail(
                        f"Circuit breaker OPEN - threshold {failure_threshold} exceeded. "
                        f"Next retry in {recovery_timeout - (current_time - last_failure_time):.1f}s",
                    )

            # Execute operation (CLOSED or HALF_OPEN state)
            try:
                result: FlextResult[TCircuit] = operation()

                if result.is_success:
                    # Operation succeeded - reset failure count and close circuit
                    state["failure_count"] = 0
                    state["circuit_state"] = "CLOSED"
                    state["last_success_time"] = current_time
                    return result
                # Operation failed - increment failure count
                current_failure_count_raw = state["failure_count"]
                current_failure_count = (
                    int(current_failure_count_raw)
                    if isinstance(current_failure_count_raw, (int, float))
                    else 0
                )
                state["failure_count"] = current_failure_count + 1
                state["last_failure_time"] = current_time

                new_failure_count = current_failure_count + 1
                if new_failure_count >= failure_threshold:
                    # Open circuit - failures exceeded threshold
                    state["circuit_state"] = "OPEN"
                    return FlextResult[TCircuit].fail(
                        f"Circuit breaker OPENED - failure threshold {failure_threshold} exceeded. "
                        f"Error: {result.error}",
                    )

                return result

            except Exception as e:
                # Exception during operation - treat as failure
                current_failure_count_raw = state["failure_count"]
                current_failure_count = (
                    int(current_failure_count_raw)
                    if isinstance(current_failure_count_raw, (int, float))
                    else 0
                )
                state["failure_count"] = current_failure_count + 1
                state["last_failure_time"] = current_time

                new_failure_count = current_failure_count + 1
                if new_failure_count >= failure_threshold:
                    state["circuit_state"] = "OPEN"
                    return FlextResult[TCircuit].fail(
                        f"Circuit breaker OPENED - failure threshold {failure_threshold} exceeded. "
                        f"Exception: {e}",
                    )

                return FlextResult[TCircuit].fail(
                    f"Circuit breaker operation failed: {e}",
                )

        @staticmethod
        def execute_with_retry[TRetry](
            operation: Callable[[], FlextResult[TRetry]],
            max_attempts: int = 3,
        ) -> FlextResult[TRetry]:
            """Execute operation with retry logic - no fallbacks, explicit error handling."""
            if max_attempts <= FlextConstants.Core.ZERO:
                return FlextResult[TRetry].fail(
                    "Invalid max_attempts: must be positive",
                )

            last_error = "No attempts made"
            for attempt in range(max_attempts):
                result = operation()
                if result.is_success:
                    return result
                last_error = result.error or f"Attempt {attempt + 1} failed"

            return FlextResult[TRetry].fail(
                f"All {max_attempts} attempts failed. Last error: {last_error}",
            )

    # Note: This class handles table/display formatting, while "TypeConversions" handles type conversions
    class TableConversion:
        """Data conversion utilities for table formatting and display."""

        @staticmethod
        def normalize_data_for_table(
            data: list[object],
        ) -> FlextResult[list[dict[str, object]]]:
            """Normalize data for table display."""
            if all(isinstance(item, dict) for item in data):
                return FlextResult[list[dict[str, object]]].ok(
                    cast("list[dict[str, object]]", data),
                )
            return FlextResult[list[dict[str, object]]].fail(
                "Data must be list of dictionaries",
            )

    class TypeGuards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is a non-empty string (excluding whitespace-only strings)."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is a non-empty dictionary."""
            return isinstance(value, dict) and len(cast("Dict", value)) > 0

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is a non-empty list."""
            return isinstance(value, list) and len(cast("list[object]", value)) > 0

    class TypeChecker:
        """Handler type checking utilities for FlextHandlers complexity reduction.

        Extracts type introspection and compatibility logic from FlextHandlers
        to simplify handler initialization and provide reusable type checking.
        """

        @classmethod
        def compute_accepted_message_types(
            cls,
            handler_class: type,
        ) -> tuple[object, ...]:
            """Compute message types accepted by a handler using cached introspection.

            Args:
                handler_class: Handler class to analyze

            Returns:
                Tuple of accepted message types

            """
            message_types: list[object] = []
            message_types.extend(cls._extract_generic_message_types(handler_class))

            if not message_types:
                explicit_type = cls._extract_message_type_from_handle(handler_class)
                if explicit_type is not None:
                    message_types.append(explicit_type)

            return tuple(message_types)

        @classmethod
        def _extract_generic_message_types(cls, handler_class: type) -> list[object]:
            """Extract message types from generic base annotations.

            Args:
                handler_class: Handler class to analyze

            Returns:
                List of message types from generic annotations

            """
            message_types: list[object] = []
            for base in getattr(handler_class, "__orig_bases__", ()) or ():
                origin = get_origin(base)
                # Check by name to avoid circular import
                if origin and origin.__name__ == "FlextHandlers":
                    args = getattr(base, "__args__", ())
                    if args:
                        message_types.append(args[0])
            return message_types

        @classmethod
        def _extract_message_type_from_handle(
            cls,
            handler_class: type,
        ) -> object | None:
            """Extract message type from handle method annotations when generics are absent.

            Args:
                handler_class: Handler class to analyze

            Returns:
                Message type from handle method or None

            """
            handle_method = getattr(handler_class, "handle", None)
            if handle_method is None:
                return None

            try:
                signature = inspect.signature(handle_method)
            except (TypeError, ValueError):
                return None

            try:
                type_hints = get_type_hints(
                    handle_method,
                    globalns=getattr(handle_method, "__globals__", {}),
                    localns=dict(vars(handler_class)),
                )
            except (NameError, AttributeError, TypeError):
                type_hints = {}

            for name, parameter in signature.parameters.items():
                if name == "self":
                    continue

                if name in type_hints:
                    # Cast the object type hint to object for return type compatibility
                    return cast("object", type_hints[name])

                annotation = parameter.annotation
                if annotation is not inspect.Signature.empty:
                    return cast("object", annotation)

                break

            return None

        @classmethod
        def can_handle_message_type(
            cls,
            accepted_types: tuple[object, ...],
            message_type: object,
        ) -> bool:
            """Check if handler can process this message type.

            Args:
                accepted_types: Types accepted by handler
                message_type: Type to check

            Returns:
                True if handler can process this message type

            """
            if not accepted_types:
                return False

            for expected_type in accepted_types:
                if cls._evaluate_type_compatibility(expected_type, message_type):
                    return True
            return False

        @classmethod
        def _evaluate_type_compatibility(
            cls,
            expected_type: object,
            message_type: object,
        ) -> bool:
            """Evaluate compatibility between expected and actual message types.

            Args:
                expected_type: Expected message type
                message_type: Actual message type

            Returns:
                True if types are compatible

            """
            # object type should be compatible with everything
            if expected_type is object:
                return True

            # Any type should be compatible with everything
            if (
                hasattr(expected_type, "__name__")
                and getattr(expected_type, "__name__", "") == "Any"
            ):
                return True

            origin_type = get_origin(expected_type) or expected_type
            message_origin = get_origin(message_type) or message_type

            if isinstance(message_type, type) or hasattr(message_type, "__origin__"):
                return cls._handle_type_or_origin_check(
                    expected_type,
                    message_type,
                    origin_type,
                    message_origin,
                )
            return cls._handle_instance_check(message_type, origin_type)

        @classmethod
        def _handle_type_or_origin_check(
            cls,
            expected_type: object,
            message_type: object,
            origin_type: object,
            message_origin: object,
        ) -> bool:
            """Handle type checking for types or objects with __origin__.

            Args:
                expected_type: Expected type
                message_type: Message type
                origin_type: Origin of expected type
                message_origin: Origin of message type

            Returns:
                True if types are compatible

            """
            try:
                if hasattr(message_type, "__origin__"):
                    return message_origin == origin_type
                if isinstance(message_type, type) and isinstance(origin_type, type):
                    return issubclass(message_type, origin_type)
                return message_type == expected_type
            except TypeError:
                return message_type == expected_type

        @classmethod
        def _handle_instance_check(
            cls,
            message_type: object,
            origin_type: object,
        ) -> bool:
            """Handle instance checking for non-type objects.

            Args:
                message_type: Message type to check
                origin_type: Origin type to check against

            Returns:
                True if instance check passes

            """
            try:
                if isinstance(origin_type, type):
                    return isinstance(message_type, origin_type)
                return True
            except TypeError:
                return True

    class MessageValidator:
        """Message validation utilities for FlextHandlers complexity reduction.

        🚨 AUDIT VIOLATION: This entire MessageValidator class violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Message validation should be centralized in FlextModels.Validation, not utilities
        ❌ ARCHITECTURAL VIOLATION: Inline validation scattered across utilities instead of centralized

        🔧 REQUIRED ACTION:
        - Move ALL message validation logic to FlextModels.Validation
        - Remove this entire MessageValidator class from utilities.py
        - Use FlextModels validation patterns for message validation

        📍 SHOULD BE USED INSTEAD:
        - FlextModels.Validation for message validation
        - FlextModels.Field validators for Pydantic model validation
        - FlextModels.Command/Query validation patterns

        Extracts message validation and serialization logic from FlextHandlers
        to simplify handler validation and provide reusable validation patterns.
        """

        _SERIALIZABLE_MESSAGE_EXPECTATION = (
            "dict, str, int, float, bool, dataclass, attrs class, or object exposing "
            "model_dump/dict/as_dict/__slots__ representations"
        )

        @classmethod
        def validate_command(cls, command: object) -> FlextResult[None]:
            """Validate command using enhanced Pydantic 2 validation and FlextExceptions.

            🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
            ❌ CRITICAL ISSUE: Command validation should be in FlextModels.Validation, not utilities
            ❌ INLINE VALIDATION: This is inline validation that should be centralized

            🔧 REQUIRED ACTION: Move to FlextModels.Validation.validate_command()
            📍 SHOULD BE USED INSTEAD: FlextModels.Command validation patterns
            """
            # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextModels.Validation
            return cls.validate_message(
                command,
                operation="command",
            )

        @classmethod
        def validate_query(cls, query: object) -> FlextResult[None]:
            """Validate query using enhanced Pydantic 2 validation and FlextExceptions."""
            return cls.validate_message(
                query,
                operation="query",
            )

        @classmethod
        def validate_message(
            cls,
            message: object,
            *,
            operation: str,
            revalidate_pydantic_messages: bool = False,
        ) -> FlextResult[None]:
            """Validate a message for the given operation.

            Args:
                message: The message object to validate
                operation: The operation name for context
                revalidate_pydantic_messages: Whether to revalidate Pydantic models

            Returns:
                FlextResult[None]: Success if valid, failure with error details if invalid

            """
            # Check for custom validation methods first (for both Pydantic and non-Pydantic models)
            validation_method_name = f"validate_{operation}"
            if hasattr(message, validation_method_name):
                validation_method = getattr(message, validation_method_name)
                if callable(validation_method):
                    try:
                        # Check if it's a custom validation method (callable without parameters)
                        # and returns a FlextResult (not a Pydantic field validator)
                        sig = inspect.signature(validation_method)
                        if (
                            len(sig.parameters) == 0
                        ):  # No parameters = custom validation method
                            validation_result_obj = validation_method()
                            # Type narrow to FlextResult
                            if isinstance(validation_result_obj, FlextResult):
                                validation_result: FlextResult[object] = cast(
                                    "FlextResult[object]",
                                    validation_result_obj,
                                )
                                if validation_result.is_failure:
                                    return FlextResult[None].fail(
                                        validation_result.error
                                        or f"{operation} validation failed",
                                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                                    )
                    except Exception as e:
                        # If calling without parameters fails, it's likely a Pydantic field validator
                        # Skip custom validation in this case - this is expected behavior
                        # Log at debug level since this is expected for Pydantic field validators
                        logger = FlextLogger(__name__)
                        logger.debug(
                            "Skipping validation method %s: %s",
                            validation_method_name,
                            e,
                        )

            # If message is a Pydantic model, assume it is already validated unless
            # the explicit revalidation flag requests an additional check
            if isinstance(message, BaseModel):
                if not revalidate_pydantic_messages:
                    return FlextResult[None].ok(None)

                try:
                    message.__class__.model_validate(message.model_dump(mode="python"))
                    return FlextResult[None].ok(None)
                except Exception as e:
                    validation_error = FlextExceptions.ValidationError(
                        f"Pydantic revalidation failed: {e}",
                        field="pydantic_model",
                        value=str(message)[:100]
                        if hasattr(message, "__str__")
                        else "unknown",
                        validation_details={
                            "pydantic_exception": str(e),
                            "model_class": message.__class__.__name__,
                            "revalidated": True,
                        },
                        context={
                            "operation": operation,
                            "message_type": type(message).__name__,
                            "validation_type": "pydantic_revalidation",
                            "revalidate_pydantic_messages": revalidate_pydantic_messages,
                        },
                        correlation_id=f"pydantic_validation_{int(time.time() * 1000)}",
                    )

                    return FlextResult[None].fail(
                        str(validation_error),
                        error_code=validation_error.error_code,
                        error_data={"exception_context": validation_error.context},
                    )

            # For non-Pydantic objects, ensure a serializable representation can be constructed
            try:
                cls.build_serializable_message_payload(message, operation=operation)
            except Exception as exc:
                if isinstance(exc, FlextExceptions.TypeError):
                    return FlextResult[None].fail(
                        str(exc),
                        error_code=exc.error_code,
                        error_data={"exception_context": exc.context},
                    )

                fallback_error = FlextExceptions.TypeError(
                    f"Invalid message type for {operation}: {type(message).__name__}",
                    expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                    actual_type=type(message).__name__,
                    context={
                        "operation": operation,
                        "message_type": type(message).__name__,
                        "validation_type": "serializable_check",
                        "original_exception": str(exc),
                    },
                    correlation_id=f"type_validation_{int(time.time() * 1000)}",
                )

                return FlextResult[None].fail(
                    str(fallback_error),
                    error_code=fallback_error.error_code,
                    error_data={"exception_context": fallback_error.context},
                )

            return FlextResult[None].ok(None)

        @classmethod
        def build_serializable_message_payload(
            cls,
            message: object,
            *,
            operation: str | None = None,
        ) -> object:
            """Build a serializable representation for message validation heuristics."""
            operation_name = operation or "message"
            context_operation = operation or "unknown"

            if isinstance(message, (dict, str, int, float, bool)):
                return cast("dict[str, object] | str | int | float | bool", message)

            if message is None:
                msg = f"Invalid message type for {operation_name}: NoneType"
                raise FlextExceptions.TypeError(
                    msg,
                    expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                    actual_type="NoneType",
                    context={
                        "operation": context_operation,
                        "message_type": type(None),
                        "validation_type": "serializable_check",
                    },
                    correlation_id=f"message_serialization_{int(time.time() * 1000)}",
                )

            if isinstance(message, BaseModel):
                return message.model_dump()

            if is_dataclass(message) and not isinstance(message, type):
                # is_dataclass() confirms this is a dataclass instance, so we can safely call asdict
                return asdict(message)

            # Handle attrs classes with proper type annotation
            attrs_fields = getattr(message, "__attrs_attrs__", None)
            if (
                attrs_fields is not None
                and not isinstance(message, type)
                and hasattr(message, "__attrs_attrs__")
                and hasattr(message, "__class__")
            ):
                # This is an attrs instance, convert to dict manually
                result: dict[str, object] = {}
                for attr_field in attrs_fields:
                    field_name = attr_field.name
                    if hasattr(message, field_name):
                        result[field_name] = getattr(message, field_name)
                return result

            # Try common serialization methods
            logger = FlextLogger(__name__)
            for method_name in ("model_dump", "dict", "as_dict"):
                method = getattr(message, method_name, None)
                if callable(method):
                    try:
                        result_data = method()
                        # Type narrow to dict and return immediately if valid
                        if isinstance(result_data, dict):
                            return cast("Dict", result_data)
                    except Exception as e:
                        # Log the exception for debugging purposes as required by S112
                        logger.debug(
                            f"Serialization method '{method_name}' failed for {type(message).__name__}: {e}",
                            method_name=method_name,
                            message_type=type(message).__name__,
                            error=str(e),
                        )
                        continue

            # Handle __slots__
            slots = getattr(message, "__slots__", None)
            if slots:
                if isinstance(slots, str):
                    slot_names: tuple[str, ...] = (slots,)
                elif isinstance(slots, (list, tuple)):
                    slot_names = tuple(cast("list[str] | tuple[str, ...]", slots))
                else:
                    msg = f"Invalid __slots__ type for {operation_name}: {type(slots).__name__}"
                    raise FlextExceptions.TypeError(
                        msg,
                        expected_type="str, list, or tuple",
                        actual_type=type(slots).__name__,
                        context={
                            "operation": context_operation,
                            "message_type": type(message).__name__,
                            "validation_type": "serializable_check",
                            "__slots__": repr(slots),
                        },
                        correlation_id=f"message_serialization_{int(time.time() * 1000)}",
                    )

                def get_slot_value(slot_name: str) -> object:
                    return getattr(message, slot_name)

                return {
                    slot_name: get_slot_value(slot_name)
                    for slot_name in slot_names
                    if hasattr(message, slot_name)
                }

            if hasattr(message, "__dict__"):
                return vars(message)

            msg = f"Invalid message type for {operation_name}: {type(message).__name__}"
            raise FlextExceptions.TypeError(
                msg,
                expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                actual_type=type(message).__name__,
                context={
                    "operation": context_operation,
                    "message_type": type(message).__name__,
                    "validation_type": "serializable_check",
                },
                correlation_id=f"message_serialization_{int(time.time() * 1000)}",
            )

    class Composition:
        """Advanced composition patterns using railway-oriented programming.

        Provides utilities for composing complex operations using FlextResult
        monadic patterns and railway-oriented programming principles.
        """

        @staticmethod
        def compose_pipeline[T](
            initial_value: T,
            functions: list[Callable[[T], FlextResult[T]]],
        ) -> FlextResult[T]:
            """Compose and execute a pipeline of functions using railway patterns.

            Args:
                initial_value: Initial value to pass through the pipeline
                functions: List of functions to compose in sequence

            Returns:
                FlextResult[T]: Result of executing the pipeline

            """
            current_value = initial_value
            for func in functions:
                result = func(current_value)
                if result.is_failure:
                    return result
                current_value = result.unwrap()
            return FlextResult[T].ok(current_value)

        @staticmethod
        def compose_parallel[T, U](
            *functions: Callable[[T], FlextResult[U]],
        ) -> Callable[[T], FlextResult[list[U]]]:
            """Compose functions to run in parallel using railway patterns.

            Args:
                *functions: Functions to run in parallel

            Returns:
                Callable[[T], FlextResult[list[U]]]: Parallel composition function

            Example:
                ```python
                parallel = FlextUtilities.Composition.compose_parallel(
                    validate_user, validate_permissions, validate_data
                )
                result = parallel(input_data)
                ```

            """

            def parallel_composed(value: T) -> FlextResult[list[U]]:
                results = []
                for func in functions:
                    result = func(value)
                    if result.is_failure:
                        return FlextResult[list[U]].fail(
                            f"Parallel execution failed: {result.error}",
                            error_code="PARALLEL_EXECUTION_FAILED",
                            error_data={
                                "failed_function": func.__name__,
                                "error": result.error,
                            },
                        )
                    results.append(result.unwrap())
                return FlextResult[list[U]].ok(results)

            return parallel_composed

        @staticmethod
        def compose_conditional[T](
            condition: Callable[[T], bool],
            true_func: Callable[[T], FlextResult[T]],
            false_func: Callable[[T], FlextResult[T]],
        ) -> Callable[[T], FlextResult[T]]:
            """Compose conditional execution using railway patterns.

            Args:
                condition: Condition function to evaluate
                true_func: Function to execute if condition is true
                false_func: Function to execute if condition is false

            Returns:
                Callable[[T], FlextResult[T]]: Conditional composition function

            Example:
                ```python
                conditional = FlextUtilities.Composition.compose_conditional(
                    lambda x: x.amount > 1000,
                    lambda x: process_large_amount(x),
                    lambda x: process_small_amount(x),
                )
                result = conditional(transaction_data)
                ```

            """

            def conditional_composed(value: T) -> FlextResult[T]:
                if condition(value):
                    return true_func(value)
                return false_func(value)

            return conditional_composed

        @staticmethod
        def compose_retry[T](
            func: Callable[[T], FlextResult[T]],
            max_retries: int = 3,
            retry_delay: float = 1.0,
        ) -> Callable[[T], FlextResult[T]]:
            """Compose retry logic using railway patterns.

            Args:
                func: Function to retry
                max_retries: Maximum number of retries
                retry_delay: Delay between retries in seconds

            Returns:
                Callable[[T], FlextResult[T]]: Retry composition function

            Example:
                ```python
                retry_func = FlextUtilities.Composition.compose_retry(
                    unreliable_function, max_retries=5, retry_delay=2.0
                )
                result = retry_func(input_data)
                ```

            """

            def retry_composed(value: T) -> FlextResult[T]:
                last_error = None
                for attempt in range(max_retries + 1):
                    result = func(value)
                    if result.is_success:
                        return result
                    last_error = result.error
                    if attempt < max_retries:
                        time.sleep(retry_delay)

                return FlextResult[T].fail(
                    f"Function failed after {max_retries} retries: {last_error}",
                    error_code="RETRY_EXHAUSTED",
                    error_data={"max_retries": max_retries, "last_error": last_error},
                )

            return retry_composed

        @staticmethod
        def compose_circuit_breaker[T](
            func: Callable[[T], FlextResult[T]],
            failure_threshold: int = FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout: float = FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_RECOVERY,
        ) -> Callable[[T], FlextResult[T]]:
            """Compose circuit breaker logic using railway patterns.

            Args:
                func: Function to protect with circuit breaker
                failure_threshold: Number of failures before opening circuit
                recovery_timeout: Time to wait before attempting recovery

            Returns:
                Callable[[T], FlextResult[T]]: Circuit breaker composition function

            Example:
                ```python
                protected_func = FlextUtilities.Composition.compose_circuit_breaker(
                    external_service_call, failure_threshold=3, recovery_timeout=30.0
                )
                result = protected_func(input_data)
                ```

            """
            circuit_state = {"failures": 0, "last_failure": 0.0, "is_open": False}

            def circuit_breaker_composed(value: T) -> FlextResult[T]:
                current_time = time.time()

                # Check if circuit is open and if we should attempt recovery
                if circuit_state["is_open"]:
                    if current_time - circuit_state["last_failure"] < recovery_timeout:
                        return FlextResult[T].fail(
                            "Circuit breaker is open",
                            error_code="CIRCUIT_BREAKER_OPEN",
                            error_data={"recovery_timeout": recovery_timeout},
                        )
                    # Attempt recovery
                    circuit_state["is_open"] = False
                    circuit_state["failures"] = 0

                # Execute function
                result = func(value)

                if result.is_success:
                    # Reset failure count on success
                    circuit_state["failures"] = 0
                    return result
                # Increment failure count
                circuit_state["failures"] += 1
                circuit_state["last_failure"] = current_time

                # Open circuit if threshold reached
                if circuit_state["failures"] >= failure_threshold:
                    circuit_state["is_open"] = True

                return result

            return circuit_breaker_composed
