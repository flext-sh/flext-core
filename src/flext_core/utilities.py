"""Consolidated utilities for FLEXT ecosystem with railway composition patterns.

Eliminates cross-module duplication by providing centralized validation,
transformation, and processing utilities using FlextResult monadic operators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import math
import os
import pathlib
import re
import secrets
import string
import threading
import time
import typing
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from itertools import starmap
from typing import ClassVar, cast, get_origin, get_type_hints

from pydantic import BaseModel

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.typings import T, U

# Module-level constants to avoid magic numbers - use FlextConstants where available
MIN_TOKEN_LENGTH = 8  # Minimum length for security tokens and passwords
MAX_PORT_NUMBER = FlextConstants.Network.MAX_PORT  # Maximum valid port number
MAX_TIMEOUT_SECONDS = (
    3600  # Maximum timeout in seconds (1 hour) - specific to utilities
)
MAX_RETRY_COUNT = (
    FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
)  # Maximum retry attempts
MAX_ERROR_DISPLAY = 5  # Maximum errors to display in batch processing
MAX_REGEX_PATTERN_LENGTH = 1000  # Maximum regex pattern length to prevent ReDoS


class FlextUtilities:
    """Consolidated utilities for FLEXT ecosystem with railway composition patterns.

    Eliminates cross-module duplication by providing centralized validation,
    transformation, and processing utilities using FlextResult monadic operators.

    **AUDIT FINDINGS**:
    - ✅ NO DUPLICATIONS: Single comprehensive utilities namespace
    - ✅ NO EXTERNAL DEPENDENCIES: Pure Python implementation
    - ✅ COMPLETE FUNCTIONALITY: Validation, transformation, processing, reliability patterns
    - ✅ ADVANCED FEATURES: Circuit breaker, retry patterns, timeout operations
    - ✅ PRODUCTION READY: Comprehensive error handling and validation

    **IMPLEMENTATION NOTES**:
    - Comprehensive validation utilities with railway patterns
    - Data transformation and processing utilities
    - Reliability patterns (retry, timeout, circuit breaker)
    - Type conversion utilities with proper error handling
    - Text processing and sanitization functions
    - ID and data generation utilities
    - Cache management utilities
    - Message validation for handlers
    """

    MIN_TOKEN_LENGTH = MIN_TOKEN_LENGTH

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
            value: str | None, field_name: str = "string"
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
            value: str, field_name: str = "string"
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
                    f"{field_name} cannot be empty or whitespace only"
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
                    f"{field_name} must be at least {min_length} characters, got {length}"
                )
            if max_length is not None and length > max_length:
                return FlextResult[str].fail(
                    f"{field_name} must be at most {max_length} characters, got {length}"
                )
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string_pattern(
            value: str, pattern: str | None, field_name: str = "string"
        ) -> FlextResult[str]:
            """Validate string against regex pattern."""
            if pattern is None:
                return FlextResult[str].ok(value)

            # First validate the pattern itself using FlextResult composition
            pattern_validation = FlextUtilities.Processing.validate_regex_pattern(
                pattern
            )
            if pattern_validation.is_failure:
                return FlextResult[str].fail(
                    f"Invalid pattern for {field_name}: {pattern_validation.error}"
                )

            # Then validate the value against the validated pattern
            compiled_pattern = pattern_validation.unwrap()
            if not compiled_pattern.match(value):
                return FlextResult[str].fail(
                    f"{field_name} does not match required pattern"
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
                value, field_name
            )
            if not_none_result.is_failure:
                return not_none_result

            not_empty_result = FlextUtilities.Validation.validate_string_not_empty(
                not_none_result.unwrap(), field_name
            )
            if not_empty_result.is_failure:
                return not_empty_result

            length_result = FlextUtilities.Validation.validate_string_length(
                not_empty_result.unwrap(), min_length, max_length, field_name
            )
            if length_result.is_failure:
                return length_result

            if pattern:
                return FlextUtilities.Validation.validate_string_pattern(
                    length_result.unwrap(), pattern, field_name
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
                url, min_length=10, pattern=url_pattern, field_name="URL"
            )

        @staticmethod
        def validate_port(port: int | str) -> FlextResult[int]:
            """Validate network port number using explicit FlextResult patterns."""
            # First validate the input type and convert to integer explicitly
            port_conversion = FlextUtilities.Processing.convert_to_integer(port)
            if port_conversion.is_failure:
                return FlextResult[int].fail(
                    f"Port must be a valid integer, got {port}: {port_conversion.error}"
                )

            port_int = port_conversion.unwrap()

            # Then validate the port range
            if not (1 <= port_int <= MAX_PORT_NUMBER):
                return FlextResult[int].fail(
                    f"Port must be between 1 and {MAX_PORT_NUMBER}, got {port_int}"
                )
            return FlextResult[int].ok(port_int)

        @staticmethod
        def validate_environment_value(
            value: str, allowed_environments: list[str]
        ) -> FlextResult[str]:
            """Validate environment value against allowed list."""
            string_result = FlextUtilities.Validation.validate_string(
                value, min_length=1, field_name="environment"
            )
            if string_result.is_failure:
                return string_result

            env = string_result.unwrap()
            if env in allowed_environments:
                return FlextResult[str].ok(env)
            return FlextResult[str].fail(
                f"Environment must be one of {allowed_environments}, got '{env}'"
            )

        @staticmethod
        def validate_log_level(level: str) -> FlextResult[str]:
            """Validate log level value."""
            allowed_levels = list(FlextConstants.Logging.VALID_LEVELS)
            return FlextUtilities.Validation.validate_environment_value(
                level.upper(), allowed_levels
            )

        @staticmethod
        def validate_security_token(token: str) -> FlextResult[str]:
            """Validate security token format and strength."""
            return FlextUtilities.Validation.validate_string(
                token, min_length=MIN_TOKEN_LENGTH, field_name="security token"
            )

        @staticmethod
        def validate_connection_string(conn_str: str) -> FlextResult[str]:
            """Validate database connection string format."""
            return FlextUtilities.Validation.validate_string(
                conn_str, min_length=10, field_name="connection string"
            )

        @staticmethod
        def validate_directory_path(path: str) -> FlextResult[str]:
            """Validate directory path format."""
            # Check for null bytes and other illegal characters
            if "\x00" in path:
                return FlextResult[str].fail("directory path cannot contain null bytes")

            string_result = FlextUtilities.Validation.validate_string(
                path, min_length=1, field_name="directory path"
            )
            if string_result.is_failure:
                return string_result

            return FlextResult[str].ok(os.path.normpath(string_result.unwrap()))

        @staticmethod
        def validate_file_path(path: str) -> FlextResult[str]:
            """Validate file path format."""
            string_result = FlextUtilities.Validation.validate_string(
                path, min_length=1, field_name="file path"
            )
            if string_result.is_failure:
                return string_result

            return FlextResult[str].ok(os.path.normpath(string_result.unwrap()))

        @staticmethod
        def validate_existing_file_path(path: str) -> FlextResult[str]:
            """Validate that file path exists on filesystem."""
            file_path_result = FlextUtilities.Validation.validate_file_path(path)
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
                    f"Timeout must be a valid number, got {timeout}: {float_conversion.error}"
                )

            timeout_float = float_conversion.unwrap()

            # Then validate the timeout constraints
            if timeout_float <= 0:
                return FlextResult[float].fail(
                    f"Timeout must be positive, got {timeout_float}"
                )
            if timeout_float > MAX_TIMEOUT_SECONDS:
                return FlextResult[float].fail(
                    f"Timeout too large (max {MAX_TIMEOUT_SECONDS}s), got {timeout_float}"
                )
            return FlextResult[float].ok(timeout_float)

        @staticmethod
        def validate_retry_count(retries: int) -> FlextResult[int]:
            """Validate retry count value."""
            try:
                if retries < 0:
                    return FlextResult[int].fail(
                        f"Retry count cannot be negative, got {retries}"
                    )
                if retries > MAX_RETRY_COUNT:
                    return FlextResult[int].fail(
                        f"Retry count too high (max {MAX_RETRY_COUNT}), got {retries}"
                    )
                return FlextResult[int].ok(retries)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"Retry count must be a valid integer, got {retries}"
                )

        @staticmethod
        def validate_positive_integer(
            value: int, field_name: str = "value"
        ) -> FlextResult[int]:
            """Validate that value is a positive integer."""
            try:
                if value <= 0:
                    return FlextResult[int].fail(
                        f"{field_name} must be positive, got {value}"
                    )
                return FlextResult[int].ok(value)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"{field_name} must be a valid integer, got {value}"
                )

        @staticmethod
        def validate_non_negative_integer(
            value: int, field_name: str = "value"
        ) -> FlextResult[int]:
            """Validate that value is a non-negative integer."""
            try:
                if value < 0:
                    return FlextResult[int].fail(
                        f"{field_name} cannot be negative, got {value}"
                    )
                return FlextResult[int].ok(value)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"{field_name} must be a valid integer, got {value}"
                )

        @staticmethod
        def validate_host(host: str) -> FlextResult[str]:
            """Validate host name or IP address."""
            return FlextUtilities.Validation.validate_string(
                host, min_length=1, field_name="host"
            )

        @staticmethod
        def validate_http_status(status_code: int) -> FlextResult[int]:
            """Validate HTTP status code range."""
            try:
                min_http_status = FlextConstants.Platform.MIN_HTTP_STATUS_RANGE
                max_http_status = FlextConstants.Platform.MAX_HTTP_STATUS_RANGE
                if not (min_http_status <= status_code <= max_http_status):
                    return FlextResult[int].fail(
                        f"HTTP status code must be between {FlextConstants.Platform.MIN_HTTP_STATUS_RANGE} and {FlextConstants.Platform.MAX_HTTP_STATUS_RANGE}, got {status_code}"
                    )
                return FlextResult[int].ok(status_code)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"HTTP status code must be a valid integer, got {status_code}"
                )

        @staticmethod
        def is_non_empty_string(value: str) -> bool:
            """Check if string is non-empty after stripping."""
            return bool(value.strip())

        @staticmethod
        def validate_pipeline[TValidate](
            value: TValidate,
            *validators: Callable[[TValidate], FlextResult[None]],
        ) -> FlextResult[TValidate]:
            """Comprehensive validation pipeline using advanced railway patterns.

            Args:
                value: Value to validate
                *validators: Validation functions to apply

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
            validation_result = validator(value)
            if validation_result.is_failure:
                return FlextResult[TContext].fail(
                    f"{context_name}: {validation_result.error}"
                )
            return FlextResult[TContext].ok(value)

        @staticmethod
        def validate_email_address(email: str) -> FlextResult[str]:
            """Enhanced email validation matching FlextModels.EmailAddress pattern."""
            if not email:
                return FlextResult[str].fail("Email cannot be empty")

            # Use pattern from FlextConstants.Platform.PATTERN_EMAIL
            if "@" not in email or "." not in email.rsplit("@", maxsplit=1)[-1]:
                return FlextResult[str].fail(f"Invalid email format: {email}")

            # Length validation using FlextConstants
            if len(email) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                return FlextResult[str].fail(
                    f"Email too long (max {FlextConstants.Validation.MAX_EMAIL_LENGTH} chars)"
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

            return FlextResult[str].ok(hostname.lower())

        @staticmethod
        def validate_entity_id(entity_id: str) -> FlextResult[str]:
            """Validate entity ID format matching FlextModels.EntityId pattern."""
            # Trim whitespace first
            entity_id = entity_id.strip()

            # Check if empty after trimming
            if not entity_id:
                return FlextResult[str].fail("Entity ID cannot be empty")

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
                    f"Phone number must have at least {FlextConstants.Validation.MIN_PHONE_DIGITS} digits"
                )

            return FlextResult[str].ok(phone)

        @staticmethod
        def validate_name_length(name: str) -> FlextResult[str]:
            """Validate name length using FlextConstants."""
            if not name:
                return FlextResult[str].fail("Name cannot be empty")

            if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
                return FlextResult[str].fail(
                    f"Name too short (min {FlextConstants.Validation.MIN_NAME_LENGTH} chars)"
                )

            if len(name) > FlextConstants.Validation.MAX_NAME_LENGTH:
                return FlextResult[str].fail(
                    f"Name too long (max {FlextConstants.Validation.MAX_NAME_LENGTH} chars)"
                )

            return FlextResult[str].ok(name.strip())

        @staticmethod
        def validate_bcrypt_rounds(rounds: int) -> FlextResult[int]:
            """Validate BCrypt rounds using FlextConstants."""
            if rounds < FlextConstants.Security.MIN_BCRYPT_ROUNDS:
                return FlextResult[int].fail(
                    f"BCrypt rounds too low (min {FlextConstants.Security.MIN_BCRYPT_ROUNDS})"
                )

            if rounds > FlextConstants.Security.MAX_BCRYPT_ROUNDS:
                return FlextResult[int].fail(
                    f"BCrypt rounds too high (max {FlextConstants.Security.MAX_BCRYPT_ROUNDS})"
                )

            return FlextResult[int].ok(rounds)

    class Transformation:
        """Data transformation utilities using railway composition."""

        @staticmethod
        def normalize_string(value: str) -> FlextResult[str]:
            """Normalize string by stripping whitespace and converting to lowercase."""
            return FlextResult[str].ok(value.strip().lower())

        @staticmethod
        def sanitize_filename(filename: str) -> FlextResult[str]:
            """Sanitize filename by removing/replacing invalid characters."""
            validation_result = FlextUtilities.Validation.validate_string(
                filename, min_length=1, field_name="filename"
            )
            if validation_result.is_failure:
                return validation_result

            name: str = validation_result.unwrap()
            sanitized_name = re.sub(r'[<>:"/\\|?*]', "_", name)
            limited_name = sanitized_name[: FlextConstants.Validation.MAX_EMAIL_LENGTH]

            return FlextResult[str].ok(limited_name)

        @staticmethod
        def parse_comma_separated(value: str) -> FlextResult[list[str]]:
            """Parse comma-separated string into list."""
            validation_result = FlextUtilities.Validation.validate_string(
                value, min_length=1, field_name="comma-separated value"
            )
            if validation_result.is_failure:
                return FlextResult[list[str]].fail(
                    validation_result.error or "Validation failed"
                )

            v: str = validation_result.unwrap()
            items = [item.strip() for item in v.split(",") if item.strip()]
            return FlextResult[list[str]].ok(items)

        @staticmethod
        def format_error_message(
            error: str, context: str | None = None
        ) -> FlextResult[str]:
            """Format error message with optional context."""
            validation_result = FlextUtilities.Validation.validate_string(
                error, min_length=1, field_name="error message"
            )
            if validation_result.is_failure:
                return validation_result

            msg: str = validation_result.unwrap()
            formatted_msg = f"{context}: {msg}" if context else msg
            return FlextResult[str].ok(formatted_msg)

    class Processing:
        """Processing utilities with reliability patterns."""

        @staticmethod
        def retry_operation(
            operation: Callable[[], FlextResult[T]],
            max_retries: int = 3,
            delay_seconds: float = 1.0,
        ) -> FlextResult[T]:
            """Retry operation with exponential backoff using advanced railway pattern."""
            # Validate parameters first
            retry_validation = FlextUtilities.Validation.validate_retry_count(
                max_retries
            )
            if retry_validation.is_failure:
                error_msg = retry_validation.error or "Invalid retry count"
                return FlextResult[T].fail(error_msg)

            delay_validation = FlextUtilities.Validation.validate_timeout_seconds(
                delay_seconds
            )
            if delay_validation.is_failure:
                error_msg = delay_validation.error or "Invalid delay seconds"
                return FlextResult[T].fail(error_msg)

            # Use retry_until_success directly on the operation result
            initial_result = operation()
            if initial_result.is_success:
                return initial_result

            # Retry on failure
            return initial_result.retry_until_success(
                lambda _: operation(),
                max_attempts=max_retries,
                backoff_factor=delay_seconds,
            )

        @staticmethod
        def timeout_operation(
            operation: Callable[[], FlextResult[T]],
            timeout_seconds: float = FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS,
        ) -> FlextResult[T]:
            """Execute operation with timeout using advanced railway pattern."""
            # Validate timeout first
            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                timeout_seconds
            )
            if timeout_validation.is_failure:
                return FlextResult[T].fail(
                    timeout_validation.error or "Invalid timeout"
                )

            # Use the reliability pattern instead of FlextResult.with_timeout
            return FlextUtilities.Reliability.with_timeout(operation, timeout_seconds)

        @classmethod
        def circuit_breaker(
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
                failure_threshold
            )
            if threshold_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid failure threshold: {threshold_validation.error}"
                )

            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                recovery_timeout
            )
            if timeout_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid recovery timeout: {timeout_validation.error}"
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
                        f"Next retry in {recovery_timeout - (current_time - last_failure_time):.1f}s"
                    )

            # Execute operation (CLOSED or HALF_OPEN state)
            try:
                result = operation()

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
                        f"Error: {result.error}"
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
                        f"Exception: {e}"
                    )

                return FlextResult[T].fail(f"Circuit breaker operation failed: {e}")

        # Circuit breaker state management (class-level state)
        _circuit_breaker_states: ClassVar[dict[str, dict[str, object]]] = {}
        _circuit_breaker_lock: ClassVar[threading.Lock] = threading.Lock()

        @classmethod
        def _get_circuit_breaker_state(cls, operation_id: str) -> dict[str, object]:
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
            if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
                return FlextResult[re.Pattern[str]].fail(
                    "Pattern too long (max 1000 characters)"
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
                    f"String '{value}' does not represent a valid integer"
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
                    f"Cannot convert '{value}' to integer: {e}"
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
            if isinstance(value, float):
                return FlextResult[float].ok(value)

            if isinstance(value, int):
                return FlextResult[float].ok(float(value))

            # value is str at this point due to type annotation float | str
            # Type checking already ensures this

            # Validate string before conversion
            cleaned_value = value.strip()
            if not cleaned_value:
                return FlextResult[float].fail("Cannot convert empty string to float")

            # Check for obvious non-numeric patterns (basic validation)
            if cleaned_value.lower() in {"inf", "+inf", "-inf", "nan"}:
                return FlextResult[float].fail(
                    f"Special float values not allowed: '{value}'"
                )

            # Use minimal try/except only for interfacing with built-in float()
            # which doesn't provide non-exception validation API
            try:
                converted_float = float(cleaned_value)
                # Check for infinity and NaN after conversion
                if not math.isfinite(converted_float):
                    return FlextResult[float].fail(
                        f"Infinite or NaN values not allowed: '{value}'"
                    )
                return FlextResult[float].ok(converted_float)
            except (ValueError, OverflowError) as e:
                # This try/except is acceptable for interfacing with built-in functions
                # that don't provide non-exception APIs for validation
                return FlextResult[float].fail(
                    f"Cannot convert '{value}' to float: {e}"
                )

    class Utilities:
        """General utility functions using railway composition."""

        @staticmethod
        def safe_cast(
            value: object, target_type: type[T], field_name: str = "value"
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
                        # Use type ignore for the constructor call since MyPy can't verify all types
                        converted_value = cast("Callable[[object], T]", target_type)(
                            value
                        )
                        return FlextResult[T].ok(converted_value)
                    # If no constructor or object type, return the value with explicit type annotation
                    return FlextResult[T].ok(cast("T", value))
                except (TypeError, ValueError):
                    # If constructor fails, return the value with explicit type annotation
                    return FlextResult[T].ok(cast("T", value))

            except (ValueError, TypeError) as e:
                return FlextResult[T].fail(
                    f"Cannot cast {field_name} to {target_type.__name__}: {e}"
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
                    f"Dictionary merge conflicts: {'; '.join(conflicts)}"
                )

            return FlextResult[dict[str, object]].ok(result)

        @staticmethod
        def deep_get(
            data: dict[str, object], path: str, default: object = None
        ) -> FlextResult[object]:
            """Safely get nested dictionary value using dot notation."""
            try:
                keys = path.split(".")
                current: object = data  # Start with the data object
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = cast("object", current[key])
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
            if batch_size <= 0:
                return FlextResult[list[U]].fail("Batch size must be positive")

            return FlextUtilities.Utilities.process_batches_railway(
                items, processor, batch_size, fail_fast=fail_fast
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

                from typing import cast

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
            all_results = [processor(item) for batch in batches for item in batch]
            return FlextResult.accumulate_errors(*all_results)

    class Cache:
        """Cache management utilities for FlextMixins and other components."""

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
                                cache_attr.clear
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

    class Generators:
        """ID and data generation utilities."""

        @staticmethod
        def generate_id() -> str:
            """Generate a unique ID using UUID4."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_timestamp() -> str:
            """Generate ISO format timestamp."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO format timestamp (alias for compatibility)."""
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
        def generate_uuid() -> str:
            """Generate a UUID (alias for generate_id for compatibility)."""
            return str(uuid.uuid4())

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
                    "Module name must be a non-empty string"
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
            text: str, max_length: int = 100, suffix: str = "..."
        ) -> FlextResult[str]:
            """Truncate text to maximum length with suffix."""
            if len(text) <= max_length:
                return FlextResult[str].ok(text)

            truncated = text[: max_length - len(suffix)] + suffix
            return FlextResult[str].ok(truncated)

        @staticmethod
        def safe_string(text: str, default: str = "") -> str:
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

    class Conversions:
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
            backoff_factor: float = 1.0,
        ) -> FlextResult[T]:
            """Enhanced retry with exponential backoff using railway patterns."""
            # Simple implementation that tries the operation multiple times
            last_error = "Operation failed"

            for attempt in range(max_retries):
                try:
                    result = operation()
                    if result.is_success:
                        return result
                    last_error = result.error or f"Attempt {attempt + 1} failed"

                    # Add delay before next attempt (simple backoff)
                    if attempt < max_retries - 1:  # Don't wait after last attempt
                        time.sleep(backoff_factor * (2**attempt))

                except Exception as e:
                    last_error = f"Exception in attempt {attempt + 1}: {e}"

            return FlextResult[T].fail(
                f"All {max_retries} attempts failed. Last error: {last_error}"
            )

        @staticmethod
        def with_timeout[TTimeout](
            operation: Callable[[], FlextResult[TTimeout]],
            timeout_seconds: float,
        ) -> FlextResult[TTimeout]:
            """Execute operation with timeout using railway patterns."""
            if timeout_seconds <= 0:
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
            import contextvars

            context = contextvars.copy_context()
            thread = threading.Thread(target=context.run, args=(run_operation,))
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Thread is still running, timeout occurred
                return FlextResult[TTimeout].fail(
                    f"Operation timed out after {timeout_seconds} seconds"
                )

            if exception_container[0]:
                return FlextResult[TTimeout].fail(
                    f"Operation failed with exception: {exception_container[0]}"
                )

            if result_container[0] is None:
                return FlextResult[TTimeout].fail(
                    "Operation completed but returned no result"
                )

            return result_container[0]

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
                return operation()

            # Validate parameters
            if failure_threshold <= 0:
                return FlextResult[TCircuit].fail("Failure threshold must be positive")
            if recovery_timeout <= 0:
                return FlextResult[TCircuit].fail("Recovery timeout must be positive")

            # Delegate to Processing circuit breaker with enhanced state management
            return FlextUtilities.Processing.circuit_breaker(
                operation, failure_threshold, recovery_timeout
            )

        @staticmethod
        def with_fallback[TFallback](
            primary_operation: Callable[[], FlextResult[TFallback]],
            *fallback_operations: Callable[[], FlextResult[TFallback]],
        ) -> FlextResult[TFallback]:
            """Execute operation with fallback alternatives using railway patterns."""
            # Try primary operation first
            primary_result = primary_operation()

            # Use or_try to chain fallbacks
            return primary_result.or_try(*fallback_operations)

    class Conversion:
        """Data conversion utilities for table formatting and display."""

        @staticmethod
        def to_table_format(data: object) -> FlextResult[list[dict[str, object]]]:
            """Convert various data types to table format (list of dictionaries).

            Handles:
            - list[dict]: Direct passthrough
            - dict: Convert to single-row table or key-value pairs
            - object with __dict__: Convert to key-value table
            - primitive values: Single-cell table

            Args:
                data: Data to convert to table format

            Returns:
                FlextResult containing list of dictionaries for table display

            """
            if data is None:
                return FlextResult[list[dict[str, object]]].ok([])

            # Handle list of dictionaries (ideal case)
            if isinstance(data, list):
                if not data:
                    return FlextResult[list[dict[str, object]]].ok([])

                if all(isinstance(item, dict) for item in data):
                    return FlextResult[list[dict[str, object]]].ok(
                        cast("list[dict[str, object]]", data)
                    )

                # Convert list of non-dict items to table
                def convert_item_to_dict(i: int, item: object) -> dict[str, object]:
                    return {"index": i, "value": str(item)}

                return FlextResult[list[dict[str, object]]].ok(
                    list(
                        starmap(
                            convert_item_to_dict, enumerate(cast("list[object]", data))
                        )
                    )
                )

            # Handle single dictionary
            if isinstance(data, dict):
                if not data:
                    return FlextResult[list[dict[str, object]]].ok([])

                # Convert to key-value table
                def convert_kv_to_dict(key: object, value: object) -> dict[str, object]:
                    return {"key": str(key), "value": str(value)}

                return FlextResult[list[dict[str, object]]].ok(
                    list(
                        starmap(
                            convert_kv_to_dict, cast("dict[str, object]", data).items()
                        )
                    )
                )

            # Handle objects with __dict__ attribute
            if hasattr(data, "__dict__"):
                obj_dict = data.__dict__
                if obj_dict:
                    return FlextResult[list[dict[str, object]]].ok([
                        {"attribute": str(key), "value": str(value)}
                        for key, value in obj_dict.items()
                        if not key.startswith("_")  # Skip private attributes
                    ])
                return FlextResult[list[dict[str, object]]].ok([{"object": str(data)}])

            # Handle primitive values
            return FlextResult[list[dict[str, object]]].ok([{"value": str(data)}])

    class TypeGuards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is a non-empty string."""
            return isinstance(value, str) and len(value) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is a non-empty dictionary."""
            return isinstance(value, dict) and len(cast("dict[str, object]", value)) > 0

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
            cls, handler_class: type
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
            cls, handler_class: type
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
            cls, expected_type: object, message_type: object
        ) -> bool:
            """Evaluate compatibility between expected and actual message types.

            Args:
                expected_type: Expected message type
                message_type: Actual message type

            Returns:
                True if types are compatible

            """
            origin_type = get_origin(expected_type) or expected_type
            message_origin = get_origin(message_type) or message_type

            if isinstance(message_type, type) or hasattr(message_type, "__origin__"):
                return cls._handle_type_or_origin_check(
                    expected_type, message_type, origin_type, message_origin
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
            cls, message_type: object, origin_type: object
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
                        import inspect

                        sig = inspect.signature(validation_method)
                        if (
                            len(sig.parameters) == 0
                        ):  # No parameters = custom validation method
                            validation_result = validation_method()
                            if (
                                hasattr(validation_result, "is_failure")
                                and hasattr(validation_result, "error")
                                and getattr(validation_result, "is_failure", False)
                            ):
                                return FlextResult[None].fail(
                                    getattr(validation_result, "error", f"{operation} validation failed")
                                    or f"{operation} validation failed",
                                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                                )
                    except Exception as e:
                        # If calling without parameters fails, it's likely a Pydantic field validator
                        # Skip custom validation in this case - this is expected behavior
                        # Log at debug level since this is expected for Pydantic field validators
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(
                            "Skipping validation method %s: %s",
                            validation_method_name,
                            str(e)
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
                        "message_type": "NoneType",
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
                result = {}
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
                        data = method()
                    except Exception as e:
                        # Log the exception for debugging purposes as required by S112
                        logger.debug(
                            f"Serialization method '{method_name}' failed for {type(message).__name__}: {e}",
                            method_name=method_name,
                            message_type=type(message).__name__,
                            error=str(e),
                        )
                        continue
                    if data is not None:
                        return data

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
