"""Core utility functions and helpers for the FLEXT ecosystem.

This module provides essential utility functions and helper classes used
throughout the FLEXT ecosystem. It includes validation utilities, helper
functions, and common patterns that support the foundation libraries.

All utilities are designed to work with FlextResult for consistent error
handling and composability across ecosystem projects.
"""

# ruff: E402, S404
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
import subprocess  # nosec B404 - Required for shell command execution utilities
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import (
    cast,
    get_origin,
    get_type_hints,
)

import orjson
from pydantic import (
    EmailStr,
    HttpUrl,
    TypeAdapter,
    ValidationError as PydanticValidationError,
)
from pydantic.types import conint, constr

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextUtilities:
    """Comprehensive utility functions for FLEXT ecosystem operations."""

    class Validation:
        """Unified validation patterns using railway composition.

        # REQUIRED ACTION:
        # - Move ALL validation logic to FlextConfig.Validation for configuration validation
        # - Move ALL validation logic to FlextModels.Validation for domain validation
        # - Remove this entire Validation class from utilities.py
        - Keep only transformation, processing, and reliability patterns in utilities

        # SHOULD BE USED INSTEAD:
        # - FlextConfig.Validation for configuration validation
        # - FlextModels.Validation for domain model validation
        # - FlextModels.Field validators for Pydantic model validation
        """

        @staticmethod
        def validate_string_not_none(
            value: str | None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate that string is not None.

            # REQUIRED ACTION: Move to FlextModels.Validation.validate_string_not_none()
            # SHOULD BE USED INSTEAD: FlextModels.Field(validator=validate_not_none)

            Returns:
                FlextResult[str]: Success with validated string or failure with error message

            """
            if value is None:
                return FlextResult[str].fail(f"{field_name} cannot be None")
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string_not_empty(
            value: str,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Validate that string is not empty after stripping.

            # REQUIRED ACTION: Move to FlextModels.Validation.validate_string_not_empty()
            # SHOULD BE USED INSTEAD: FlextModels.Field(validator=validate_not_empty)

            Returns:
                FlextResult[str]: Success with validated string or failure with error message

            """
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
            """Validate string length constraints using Pydantic constr."""
            try:
                # Create constrained string type using Pydantic
                if max_length is not None:
                    constrained_str = constr(
                        min_length=min_length, max_length=max_length
                    )
                else:
                    constrained_str = constr(min_length=min_length)

                # Validate using TypeAdapter
                adapter = TypeAdapter(constrained_str)
                validated = adapter.validate_python(value)
                return FlextResult[str].ok(str(validated))
            except (PydanticValidationError, ValueError) as e:
                # Extract meaningful error message
                length = len(value)
                if length < min_length:
                    return FlextResult[str].fail(
                        f"{field_name} must be at least {min_length} characters, got {length}",
                    )
                if max_length is not None and length > max_length:
                    return FlextResult[str].fail(
                        f"{field_name} must be at most {max_length} characters, got {length}",
                    )
                return FlextResult[str].fail(f"{field_name} validation error: {e}")

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
            pattern_validation = FlextUtilities.Validation.validate_regex_pattern(
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
            """Validate email format using FlextRuntime foundation + Pydantic validation.

            Delegates to FlextRuntime.is_valid_email() for initial validation (Layer 0.5),
            then performs comprehensive Pydantic EmailStr validation (RFC 5322 compliant).

            Validation layers:
            1. FlextRuntime.is_valid_email() - Pattern-based type guard from Layer 0.5
            2. Pydantic EmailStr - RFC 5322 compliance, normalization, length constraints

            Args:
                email: Email address string to validate

            Returns:
                FlextResult[str]: Validated email or error message

            """
            # Layer 0.5: Foundation validation using FlextRuntime type guard
            if not FlextRuntime.is_valid_email(email):
                return FlextResult[str].fail(
                    "Invalid email format (failed foundation validation)"
                )

            # Layer 7: Comprehensive Pydantic validation
            try:
                email_adapter: TypeAdapter[EmailStr] = TypeAdapter(EmailStr)
                validated: EmailStr = email_adapter.validate_python(email)
                return FlextResult[str].ok(str(validated))
            except ImportError:
                # email-validator not installed, fall back to foundation validation
                return FlextResult[str].ok(email)
            except (PydanticValidationError, ValueError) as e:
                return FlextResult[str].fail(f"Invalid email format: {e}")

        @staticmethod
        def validate_url(url: str) -> FlextResult[str]:
            """Validate URL format using FlextRuntime foundation + Pydantic validation.

            Delegates to FlextRuntime.is_valid_url() for initial validation (Layer 0.5),
            then performs comprehensive Pydantic HttpUrl validation (RFC 3986 compliant).

            Validation layers:
            1. FlextRuntime.is_valid_url() - Pattern-based type guard from Layer 0.5
            2. Pydantic HttpUrl - Protocol, host, port, path validation per RFC 3986

            Args:
                url: URL string to validate

            Returns:
                FlextResult[str]: Validated URL or error message

            """
            # Layer 0.5: Foundation validation using FlextRuntime type guard
            if not FlextRuntime.is_valid_url(url):
                return FlextResult[str].fail(
                    "Invalid URL format (failed foundation validation)"
                )

            # Layer 7: Comprehensive Pydantic validation
            try:
                url_adapter = TypeAdapter(HttpUrl)
                validated = url_adapter.validate_python(url)
                return FlextResult[str].ok(str(validated))
            except (PydanticValidationError, ValueError) as e:
                return FlextResult[str].fail(f"Invalid URL format: {e}")

        @staticmethod
        def validate_port(port: int | str) -> FlextResult[int]:
            """Validate network port number using Pydantic conint."""
            try:
                # Define port constraints (1-65535) using Pydantic
                port_number = conint(ge=1, le=FlextConstants.Network.MAX_PORT)

                # Validate using TypeAdapter (handles both int and str input)
                adapter = TypeAdapter(port_number)
                validated = adapter.validate_python(port)
                return FlextResult[int].ok(int(validated))
            except (PydanticValidationError, ValueError) as e:
                # Provide meaningful error messages
                if isinstance(port, str) and not port.isdigit():
                    return FlextResult[int].fail(
                        f"Port must be a valid integer, got {port}"
                    )

                try:
                    port_int = int(port)
                    if port_int < 1 or port_int > FlextConstants.Network.MAX_PORT:
                        return FlextResult[int].fail(
                            f"Port must be between 1 and {FlextConstants.Network.MAX_PORT}, got {port_int}",
                        )
                except (ValueError, TypeError):
                    pass

                return FlextResult[int].fail(f"Invalid port number: {e}")

        @staticmethod
        def validate_environment_value(
            value: str,
            allowed_environments: FlextTypes.StringList,
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
            """Validate directory path using FlextRuntime foundation + normalization.

            Delegates to FlextRuntime.is_valid_path() for pattern validation (Layer 0.5),
            then performs null byte checks and path normalization.

            Validation layers:
            1. FlextRuntime.is_valid_path() - Pattern-based type guard from Layer 0.5
            2. Null byte check - Security validation
            3. os.path.normpath - Path normalization

            Args:
                path: Directory path string to validate

            Returns:
                FlextResult[str]: Normalized path or error message

            """
            # Layer 0.5: Foundation validation using FlextRuntime type guard
            if not FlextRuntime.is_valid_path(path):
                return FlextResult[str].fail(
                    "Invalid directory path format (failed foundation validation)"
                )

            # Layer 7: Additional security checks
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
            """Validate file path using FlextRuntime foundation + normalization.

            Delegates to FlextRuntime.is_valid_path() for pattern validation (Layer 0.5),
            then performs path normalization.

            Validation layers:
            1. FlextRuntime.is_valid_path() - Pattern-based type guard from Layer 0.5
            2. os.path.normpath - Path normalization

            Args:
                path: File path string to validate

            Returns:
                FlextResult[str]: Normalized path or error message

            """
            # Layer 0.5: Foundation validation using FlextRuntime type guard
            if not FlextRuntime.is_valid_path(path):
                return FlextResult[str].fail(
                    "Invalid file path format (failed foundation validation)"
                )

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
            float_conversion = FlextUtilities.Validation.convert_to_float(timeout)
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
                        f"Retry count too high (max {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}), got {retries}"
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
                        f"HTTP status code must be between {min_http_status} and {max_http_status}, got {status_code}"
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
            """Validate phone number using FlextRuntime foundation + digit count check.

            Delegates to FlextRuntime.is_valid_phone() for pattern validation (Layer 0.5),
            then validates minimum digit requirements using FlextConstants.

            Validation layers:
            1. FlextRuntime.is_valid_phone() - Pattern-based type guard from Layer 0.5
            2. MIN_PHONE_DIGITS check - Ensures minimum digit count per FlextConstants

            Args:
                phone: Phone number string to validate

            Returns:
                FlextResult[str]: Validated phone number or error message

            """
            if not phone:
                return FlextResult[str].fail("Phone number cannot be empty")

            # Layer 0.5: Foundation validation using FlextRuntime type guard
            if not FlextRuntime.is_valid_phone(phone):
                return FlextResult[str].fail(
                    "Invalid phone number format (failed foundation validation)"
                )

            # Layer 7: Additional digit count validation using FlextConstants
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
            data: FlextTypes.Dict,
            required_fields: FlextTypes.StringList
            | dict[str, type | tuple[type, ...]]
            | None = None,
        ) -> FlextResult[FlextTypes.Dict]:
            """Validate dictionary data with optional required fields and type checking.

            Args:
                data: Dictionary to validate
                required_fields: Optional list of field names or FlextTypes.Dict mapping fields to types

            Returns:
                FlextResult[FlextTypes.Dict]: Validated data or error

            """
            if required_fields:
                if isinstance(required_fields, list):
                    # Simple presence check
                    missing_fields = [
                        field for field in required_fields if field not in data
                    ]
                    if missing_fields:
                        return FlextResult[FlextTypes.Dict].fail(
                            f"Missing required fields: {', '.join(missing_fields)}"
                        )
                else:
                    # Type validation
                    for field, expected_type in required_fields.items():
                        if field not in data:
                            return FlextResult[FlextTypes.Dict].fail(
                                f"Missing required field: {field}"
                            )
                        if not isinstance(data[field], expected_type):
                            return FlextResult[FlextTypes.Dict].fail(
                                f"Field '{field}' has incorrect type: expected {expected_type}, got {type(data[field])}"
                            )

            return FlextResult[FlextTypes.Dict].ok(data)

        class Providers:
            """Dependency injection providers for validation services.

            Enables validation services to be registered and injected via DI container.
            This follows Phase 2 enhancement pattern for making utilities injectable.
            """

            @staticmethod
            def create_validation_service_provider() -> object:  # providers.Singleton
                """Create singleton provider for validation service.

                Returns:
                    providers.Singleton: Singleton validation service provider

                Example:
                    >>> from flext_core import FlextUtilities
                    >>> provider = FlextUtilities.Validation.Providers.create_validation_service_provider()
                    >>> # Register in container
                    >>> container.register("validation_service", provider)

                """
                providers_module = FlextRuntime.dependency_providers()

                class ValidationService:
                    """Injectable validation service using FlextUtilities patterns."""

                    @staticmethod
                    def validate_string(
                        value: str | None,
                        field_name: str = "string",
                        min_length: int = 1,
                        max_length: int | None = None,
                        pattern: str | None = None,
                    ) -> FlextResult[str]:
                        """Validate string with composable railway pattern.

                        Args:
                            value: String to validate
                            field_name: Field name for error messages
                            min_length: Minimum length (default: 1)
                            max_length: Maximum length (default: None)
                            pattern: Optional regex pattern

                        Returns:
                            FlextResult[str]: Validated string or error

                        """
                        return (
                            FlextUtilities.Validation.validate_string_not_none(
                                value, field_name
                            )
                            .flat_map(
                                lambda v: FlextUtilities.Validation.validate_string_not_empty(
                                    v, field_name
                                )
                            )
                            .flat_map(
                                lambda v: FlextUtilities.Validation.validate_string_length(
                                    v, min_length, max_length, field_name
                                )
                            )
                            .flat_map(
                                lambda v: FlextUtilities.Validation.validate_string_pattern(
                                    v, pattern, field_name
                                )
                                if pattern
                                else FlextResult[str].ok(v)
                            )
                        )

                    @staticmethod
                    def validate_email(
                        value: str | None, field_name: str = "email"
                    ) -> FlextResult[str]:
                        """Validate email address.

                        Args:
                            value: Email to validate
                            field_name: Field name for error messages

                        Returns:
                            FlextResult[str]: Validated email or error

                        """
                        # First validate not none
                        if value is None:
                            return FlextResult[str].fail(f"{field_name} cannot be None")
                        return FlextUtilities.Validation.validate_email(value)

                    @staticmethod
                    def validate_url(
                        value: str | None, field_name: str = "url"
                    ) -> FlextResult[str]:
                        """Validate URL.

                        Args:
                            value: URL to validate
                            field_name: Field name for error messages

                        Returns:
                            FlextResult[str]: Validated URL or error

                        """
                        # First validate not none
                        if value is None:
                            return FlextResult[str].fail(f"{field_name} cannot be None")
                        return FlextUtilities.Validation.validate_url(value)

                return providers_module.Singleton(ValidationService)

            @staticmethod
            def register_in_container(container: FlextContainer) -> FlextResult[None]:
                """Register validation service in DI container.

                Args:
                    container: DI container (FlextContainer or dependency_injector container)

                Returns:
                    FlextResult[None]: Success or failure

                Example:
                    >>> from flext_core import FlextContainer, FlextUtilities
                    >>> container = FlextContainer.get_global()
                    >>> result = (
                    ...     FlextUtilities.Validation.Providers.register_in_container(
                    ...         container
                    ...     )
                    ... )

                """
                try:
                    validation_provider = FlextUtilities.Validation.Providers.create_validation_service_provider()

                    # Register with FlextContainer
                    if hasattr(container, "register"):
                        result = container.register(
                            "validation_service", validation_provider
                        )
                        if result.is_failure:
                            return FlextResult[None].fail(
                                f"Registration failed: {result.error}"
                            )
                        return FlextResult[None].ok(None)

                    return FlextResult[None].fail(
                        "Container does not support service registration"
                    )

                except Exception as e:
                    return FlextResult[None].fail(
                        f"Validation provider registration failed: {e}"
                    )

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
                            # Clear FlextTypes.Dict-like caches
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
            try:
                json_bytes = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
                return json_bytes.decode(FlextConstants.Utilities.DEFAULT_ENCODING)
            except Exception:
                # Fallback to standard library json with sorted keys
                return json.dumps(value, sort_keys=True, default=str)

        @staticmethod
        def normalize_component(value: object) -> object:
            """Normalize arbitrary objects into cache-friendly deterministic structures."""
            if value is None or isinstance(value, (bool, int, float, str)):
                return value

            if isinstance(value, bytes):
                return ("bytes", value.hex())

            if isinstance(value, FlextProtocols.Foundation.HasModelDump):
                try:
                    dumped: FlextTypes.Dict = value.model_dump()
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
                # Return sorted FlextTypes.Dict for cache-friendly deterministic ordering
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
                value_vars_dict: FlextTypes.Dict = cast(
                    "FlextTypes.Dict",
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
                if isinstance(command, FlextProtocols.Foundation.HasModelDump):
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
                        cast("FlextTypes.Dict", command),
                    )
                    return f"{command_type.__name__}_{hash(str(dict_sorted_data))}"  # type: ignore[misc]

                # For other objects, use string representation
                command_str = str(command) if command is not None else "None"
                command_hash = hash(command_str)  # type: ignore[misc]
                return f"{command_type.__name__}_{command_hash}"

            except Exception:
                # Fallback to string representation if anything fails
                command_str_fallback = str(command) if command is not None else "None"
                command_str_fallback = command_str_fallback.encode(
                    "utf-8", errors="ignore"
                ).decode("utf-8", errors="ignore")
                try:
                    command_hash_fallback = hash(command_str_fallback)  # type: ignore[misc]
                    return f"{command_type.__name__}_{command_hash_fallback}"
                except TypeError:
                    # If hash fails, use a deterministic fallback
                    return f"{command_type.__name__}_{abs(hash(command_str_fallback.encode(FlextConstants.Utilities.DEFAULT_ENCODING)))}"  # type: ignore[misc]

        @staticmethod
        def sort_dict_keys(obj: object) -> object:
            """Recursively sort dictionary keys for deterministic ordering.

            Args:
                obj: Object to sort (FlextTypes.Dict, list, or other)

            Returns:
                Object with sorted keys

            """
            if isinstance(obj, dict):
                dict_obj: FlextTypes.Dict = cast("FlextTypes.Dict", obj)
                sorted_items: list[tuple[object, object]] = sorted(
                    dict_obj.items(),
                    key=lambda x: str(x[0]),
                )
                return {
                    str(k): FlextUtilities.Cache.sort_dict_keys(v)
                    for k, v in sorted_items
                }
            if isinstance(obj, list):
                obj_list: FlextTypes.List = cast("FlextTypes.List", obj)
                return [FlextUtilities.Cache.sort_dict_keys(item) for item in obj_list]
            if isinstance(obj, tuple):
                obj_tuple: tuple[object, ...] = cast("tuple[object, ...]", obj)
                return tuple(
                    FlextUtilities.Cache.sort_dict_keys(item) for item in obj_tuple
                )
            return obj

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
            """Generate an entity version number using FlextConstants.Context."""
            return (
                int(
                    datetime.now(UTC).timestamp()
                    * FlextConstants.Context.MILLISECONDS_PER_SECOND
                )
                % 1000000
            )

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
            max_length: int = FlextConstants.Utilities.DEFAULT_BATCH_SIZE,
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

    class TypeConversions:
        """Type conversion utilities using railway composition.

        This class handles type conversions (str->bool, str->int), while "Conversion" handles table formatting.
        """

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

            return result_container[0]

    class TypeGuards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is a non-empty string (excluding whitespace-only strings)."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is a non-empty dictionary."""
            return isinstance(value, dict) and len(cast("FlextTypes.Dict", value)) > 0

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is a non-empty list."""
            return isinstance(value, list) and len(cast("FlextTypes.List", value)) > 0

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
            message_types: FlextTypes.List = []
            message_types.extend(cls._extract_generic_message_types(handler_class))

            if not message_types:
                explicit_type = cls._extract_message_type_from_handle(handler_class)
                if explicit_type is not None:
                    message_types.append(explicit_type)

            return tuple(message_types)

        @classmethod
        def _extract_generic_message_types(cls, handler_class: type) -> FlextTypes.List:
            """Extract message types from generic base annotations.

            Args:
                handler_class: Handler class to analyze

            Returns:
                List of message types from generic annotations

            """
            message_types: FlextTypes.List = []
            for base in getattr(handler_class, "__orig_bases__", ()) or ():
                # Layer 0.5: Use FlextRuntime for type introspection
                origin = get_origin(base)
                # Check by name to avoid circular import
                if origin and origin.__name__ == "FlextHandlers":
                    # Use FlextRuntime.extract_generic_args() from Layer 0.5
                    args = FlextRuntime.extract_generic_args(base)
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

            # object type should be compatible with everything
            if (
                hasattr(expected_type, "__name__")
                and getattr(expected_type, "__name__", "") == "object"
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

    @staticmethod
    def run_external_command(
        cmd: FlextTypes.StringList,
        *,
        capture_output: bool = True,
        check: bool = True,
        env: FlextTypes.StringDict | None = None,
        cwd: str | pathlib.Path | None = None,
        timeout: float | None = None,
        command_input: str | bytes | None = None,
        text: bool | None = None,
    ) -> FlextResult[subprocess.CompletedProcess[str]]:
        """Execute external command with proper error handling using FlextResult pattern.

        Args:
            cmd: Command to execute as list of strings
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit code
            env: Environment variables for the command
            cwd: Working directory for the command
            timeout: Command timeout in seconds
            input: Input to send to the command
            text: Whether to decode stdout/stderr as text (Python 3.7+)

        Returns:
            FlextResult containing CompletedProcess on success or error details on failure

        Example:
            ```python
            result = FlextUtilities.run_external_command(
                ["python", "script.py"], capture_output=True, timeout=60.0
            )
            if result.is_success:
                process = result.value
                print(f"Exit code: {process.returncode}")
                print(f"Output: {process.stdout}")
            ```

        """
        try:
            # Validate command for security - ensure all parts are safe strings
            # This prevents shell injection since we use list form, not shell=True
            if not cmd or not all(part for part in cmd):
                return FlextResult[subprocess.CompletedProcess[str]].fail(
                    "Command must be a non-empty list of strings",
                    error_code="INVALID_COMMAND",
                )

            # Execute subprocess.run with explicit parameters to avoid overload issues
            # S603: Command is validated above to ensure it's a safe list of strings
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=capture_output,
                check=check,
                env=env,
                cwd=cwd,
                timeout=timeout,
                input=command_input,
                text=text if text is not None else True,
            )

            return FlextResult[subprocess.CompletedProcess[str]].ok(result)

        except subprocess.CalledProcessError as e:
            return FlextResult[subprocess.CompletedProcess[str]].fail(
                f"Command failed with exit code {e.returncode}",
                error_code="COMMAND_FAILED",
                error_data={
                    "cmd": cmd,
                    "returncode": e.returncode,
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                },
            )
        except subprocess.TimeoutExpired as e:
            return FlextResult[subprocess.CompletedProcess[str]].fail(
                f"Command timed out after {timeout} seconds",
                error_code="COMMAND_TIMEOUT",
                error_data={
                    "cmd": cmd,
                    "timeout": timeout,
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                },
            )
        except FileNotFoundError:
            return FlextResult[subprocess.CompletedProcess[str]].fail(
                f"Command not found: {cmd[0]}",
                error_code="COMMAND_NOT_FOUND",
                error_data={"cmd": cmd, "executable": cmd[0]},
            )
        except Exception as e:
            return FlextResult[subprocess.CompletedProcess[str]].fail(
                f"Unexpected error running command: {e!s}",
                error_code="COMMAND_ERROR",
                error_data={"cmd": cmd, "error": str(e)},
            )

    generate_id = Generators.generate_id


__all__ = [
    "FlextUtilities",
]
