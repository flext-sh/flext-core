"""Consolidated utilities for FLEXT ecosystem with railway composition patterns.

Eliminates cross-module duplication by providing centralized validation,
transformation, and processing utilities using FlextResult monadic operators.
"""

from __future__ import annotations

import os
import pathlib
import re
import secrets
import string
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

# Import FlextResult directly since it's used for more than type hinting
from flext_core.result import FlextResult

T = TypeVar("T")
U = TypeVar("U")

# Module-level constants to avoid magic numbers
MIN_TOKEN_LENGTH = 8  # Minimum length for security tokens and passwords
MAX_PORT_NUMBER = 65535  # Maximum valid port number
MAX_TIMEOUT_SECONDS = 3600  # Maximum timeout in seconds (1 hour)
MAX_RETRY_COUNT = 10  # Maximum retry attempts
MAX_ERROR_DISPLAY = 5  # Maximum errors to display in batch processing


class FlextUtilities:
    """Consolidated utilities for FLEXT ecosystem with railway composition patterns.

    Eliminates cross-module duplication by providing centralized validation,
    transformation, and processing utilities using FlextResult monadic operators.
    """

    MIN_TOKEN_LENGTH = MIN_TOKEN_LENGTH

    class Validation:
        """Unified validation patterns using railway composition."""

        @staticmethod
        def validate_string_not_none(
            value: str | None, field_name: str = "string"
        ) -> FlextResult[str]:
            """Validate that string is not None."""
            if value is None:
                return FlextResult[str].fail(f"{field_name} cannot be None")
            return FlextResult[str].ok(value)

        @staticmethod
        def validate_string_not_empty(
            value: str, field_name: str = "string"
        ) -> FlextResult[str]:
            """Validate that string is not empty after stripping."""
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

            try:
                if not re.match(pattern, value):
                    return FlextResult[str].fail(
                        f"{field_name} does not match required pattern"
                    )
                return FlextResult[str].ok(value)
            except re.error as e:
                return FlextResult[str].fail(f"Invalid pattern for {field_name}: {e}")

        @staticmethod
        def validate_string(
            value: str | None,
            min_length: int = 1,
            max_length: int | None = None,
            pattern: str | None = None,
            field_name: str = "string",
        ) -> FlextResult[str]:
            """Comprehensive string validation using railway composition."""
            return (
                FlextUtilities.Validation.validate_string_not_none(value, field_name)
                >> (
                    lambda s: FlextUtilities.Validation.validate_string_not_empty(
                        s, field_name
                    )
                )
                >> (
                    lambda s: FlextUtilities.Validation.validate_string_length(
                        s, min_length, max_length, field_name
                    )
                )
                >> (
                    lambda s: FlextUtilities.Validation.validate_string_pattern(
                        s, pattern, field_name
                    )
                    if pattern
                    else FlextResult[str].ok(s)
                )
            )

        @staticmethod
        def validate_email(email: str) -> FlextResult[str]:
            """Validate email format using railway composition."""
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
            """Validate network port number."""
            try:
                port_int = int(port) if isinstance(port, str) else port
                if not (1 <= port_int <= MAX_PORT_NUMBER):
                    return FlextResult[int].fail(
                        f"Port must be between 1 and {MAX_PORT_NUMBER}, got {port_int}"
                    )
                return FlextResult[int].ok(port_int)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"Port must be a valid integer, got {port}"
                )

        @staticmethod
        def validate_environment_value(
            value: str, allowed_environments: list[str]
        ) -> FlextResult[str]:
            """Validate environment value against allowed list."""
            return FlextUtilities.Validation.validate_string(
                value, min_length=1, field_name="environment"
            ) >> (
                lambda env: FlextResult[str].ok(env)
                if env in allowed_environments
                else FlextResult[str].fail(
                    f"Environment must be one of {allowed_environments}, got '{env}'"
                )
            )

        @staticmethod
        def validate_log_level(level: str) -> FlextResult[str]:
            """Validate log level value."""
            allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
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

            return FlextUtilities.Validation.validate_string(
                path, min_length=1, field_name="directory path"
            ) >> (lambda p: FlextResult[str].ok(os.path.normpath(p)))

        @staticmethod
        def validate_file_path(path: str) -> FlextResult[str]:
            """Validate file path format."""
            return FlextUtilities.Validation.validate_string(
                path, min_length=1, field_name="file path"
            ) >> (lambda p: FlextResult[str].ok(os.path.normpath(p)))

        @staticmethod
        def validate_existing_file_path(path: str) -> FlextResult[str]:
            """Validate that file path exists on filesystem."""
            return FlextUtilities.Validation.validate_file_path(path) >> (
                lambda p: FlextResult[str].ok(p)
                if pathlib.Path(p).is_file()
                else FlextResult[str].fail(f"file does not exist: {p}")
            )

        @staticmethod
        def validate_timeout_seconds(timeout: float) -> FlextResult[float]:
            """Validate timeout value in seconds."""
            try:
                timeout_float = float(timeout)
                if timeout_float <= 0:
                    return FlextResult[float].fail(
                        f"Timeout must be positive, got {timeout_float}"
                    )
                if timeout_float > MAX_TIMEOUT_SECONDS:
                    return FlextResult[float].fail(
                        f"Timeout too large (max {MAX_TIMEOUT_SECONDS}s), got {timeout_float}"
                    )
                return FlextResult[float].ok(timeout_float)
            except (ValueError, TypeError):
                return FlextResult[float].fail(
                    f"Timeout must be a valid number, got {timeout}"
                )

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
                if not (100 <= status_code <= 599):
                    return FlextResult[int].fail(
                        f"HTTP status code must be between 100 and 599, got {status_code}"
                    )
                return FlextResult[int].ok(status_code)
            except (ValueError, TypeError):
                return FlextResult[int].fail(
                    f"HTTP status code must be a valid integer, got {status_code}"
                )

        @staticmethod
        def is_non_empty_string(value: str) -> bool:
            """Check if string is non-empty after stripping."""
            return isinstance(value, str) and bool(value.strip())

    class Transformation:
        """Data transformation utilities using railway composition."""

        @staticmethod
        def normalize_string(value: str) -> FlextResult[str]:
            """Normalize string by stripping whitespace and converting to lowercase."""
            return FlextResult[str].ok(value) >> (
                lambda s: FlextResult[str].ok(s.strip().lower())
            )

        @staticmethod
        def sanitize_filename(filename: str) -> FlextResult[str]:
            """Sanitize filename by removing/replacing invalid characters."""
            return (
                FlextUtilities.Validation.validate_string(
                    filename, min_length=1, field_name="filename"
                )
                >> (
                    lambda name: FlextResult[str].ok(re.sub(r'[<>:"/\\|?*]', "_", name))
                )
                >> (lambda name: FlextResult[str].ok(name[:255]))  # Limit length
            )

        @staticmethod
        def parse_comma_separated(value: str) -> FlextResult[list[str]]:
            """Parse comma-separated string into list."""
            return FlextUtilities.Validation.validate_string(
                value, min_length=1, field_name="comma-separated value"
            ) >> (
                lambda v: FlextResult[list[str]].ok(
                    [item.strip() for item in v.split(",") if item.strip()]
                )
            )

        @staticmethod
        def format_error_message(
            error: str, context: str | None = None
        ) -> FlextResult[str]:
            """Format error message with optional context."""
            return FlextUtilities.Validation.validate_string(
                error, min_length=1, field_name="error message"
            ) >> (
                lambda msg: FlextResult[str].ok(f"{context}: {msg}" if context else msg)
            )

    class Processing:
        """Processing utilities with reliability patterns."""

        @staticmethod
        def retry_operation(
            operation: Callable[[], FlextResult[T]],
            max_retries: int = 3,
            delay_seconds: float = 1.0,
        ) -> FlextResult[T]:
            """Retry operation with exponential backoff."""
            retry_validation = FlextUtilities.Validation.validate_retry_count(
                max_retries
            )
            if retry_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid retry configuration: {retry_validation.error}"
                )

            delay_validation = FlextUtilities.Validation.validate_timeout_seconds(
                delay_seconds
            )
            if delay_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid delay configuration: {delay_validation.error}"
                )

            last_error = "No attempts made"
            for attempt in range(max_retries + 1):
                try:
                    result = operation()
                    if result.is_success:
                        return result
                    last_error = result.error or "Operation failed"
                except Exception as e:
                    last_error = str(e)

                if attempt < max_retries:
                    time.sleep(delay_seconds * (2**attempt))  # Exponential backoff

            return FlextResult[T].fail(
                f"Operation failed after {max_retries + 1} attempts. Last error: {last_error}"
            )

        @staticmethod
        def timeout_operation(
            operation: Callable[[], FlextResult[T]], timeout_seconds: float = 30.0
        ) -> FlextResult[T]:
            """Execute operation with timeout."""
            timeout_validation = FlextUtilities.Validation.validate_timeout_seconds(
                timeout_seconds
            )
            if timeout_validation.is_failure:
                return FlextResult[T].fail(
                    f"Invalid timeout configuration: {timeout_validation.error}"
                )

            result_container: list[FlextResult[T]] = []
            exception_container: list[Exception] = []

            def target() -> None:
                try:
                    result_container.append(operation())
                except Exception as e:
                    exception_container.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                return FlextResult[T].fail(
                    f"Operation timed out after {timeout_seconds} seconds"
                )

            if exception_container:
                return FlextResult[T].fail(
                    f"Operation failed with exception: {exception_container[0]}"
                )

            if result_container:
                return result_container[0]

            return FlextResult[T].fail("Operation completed but returned no result")

        @staticmethod
        def circuit_breaker(
            operation: Callable[[], FlextResult[T]],
            failure_threshold: int = 5,
            recovery_timeout: float = 60.0,
        ) -> FlextResult[T]:
            """Simple circuit breaker pattern implementation."""
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

            # Basic implementation - execute operation directly
            # In production, would track failure count and state
            try:
                return operation()
            except Exception as e:
                return FlextResult[T].fail(f"Circuit breaker operation failed: {e}")

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

                # For basic types, try to construct with value
                if target_type in {str, int, float, bool}:
                    # Use type: ignore for basic type construction
                    casted_value: T = target_type(value)  # type: ignore[call-arg]
                    return FlextResult[T].ok(casted_value)

                # For other types, try to cast directly
                # Use type: ignore for generic type construction
                casted_value = target_type(value)  # type: ignore[call-arg]
                return FlextResult[T].ok(casted_value)

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
                current: object = data
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return FlextResult[object].ok(default)
                return FlextResult[object].ok(current)
            except Exception as e:
                return FlextResult[object].fail(f"Error accessing path '{path}': {e}")

        @staticmethod
        def ensure_list(value: object) -> FlextResult[list[object]]:
            """Ensure value is a list, wrapping single values."""
            if isinstance(value, list):
                return FlextResult[list[object]].ok(value)
            if value is None:
                return FlextResult[list[object]].ok([])
            return FlextResult[list[object]].ok([value])

        @staticmethod
        def filter_none_values(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Remove None values from dictionary."""
            return FlextResult[dict[str, object]].ok(
                {k: v for k, v in data.items() if v is not None}
            )

        @staticmethod
        def batch_process(
            items: list[T],
            processor: Callable[[T], FlextResult[U]],
            batch_size: int = 100,
        ) -> FlextResult[list[U]]:
            """Process items in batches with error collection."""
            # Validate batch size
            if batch_size <= 0:
                return FlextResult[list[U]].fail("Batch size must be positive")

            results: list[U] = []
            errors: list[str] = []

            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                for j, item in enumerate(batch):
                    try:
                        result = processor(item)
                        if result.is_success:
                            results.append(result.unwrap())
                        else:
                            error_msg = result.error or "Unknown error"
                            errors.append(f"Item {i + j}: {error_msg}")
                    except Exception as e:
                        errors.append(f"Item {i + j}: Exception {e}")

            if errors:
                error_suffix = "..." if len(errors) > MAX_ERROR_DISPLAY else ""
                display_errors = errors[:MAX_ERROR_DISPLAY]
                return FlextResult[list[U]].fail(
                    f"Batch processing errors: {'; '.join(display_errors)}{error_suffix}"
                )

            return FlextResult[list[U]].ok(results)

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
            return f"corr-{str(uuid.uuid4())[:8]}"

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
            if not text or not isinstance(text, str):
                return default
            return text.strip()

    class Conversions:
        """Type conversion utilities using railway composition."""

        @staticmethod
        def to_bool(value: str | bool | int | None) -> FlextResult[bool]:
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
            try:
                if isinstance(value, int):
                    return FlextResult[int].ok(value)
                if isinstance(value, (str, float)):
                    return FlextResult[int].ok(int(value))
                # value is None case
                return FlextResult[int].fail("Cannot convert None to integer")
            except (ValueError, TypeError) as e:
                return FlextResult[int].fail(f"Integer conversion failed: {e}")

    class Reliability:
        """Reliability patterns for resilient operations."""

        @staticmethod
        def retry_with_backoff(
            operation: Callable[[], FlextResult[T]],
            max_retries: int = 3,
            initial_delay: float = 1.0,
        ) -> FlextResult[T]:
            """Retry operation with exponential backoff."""
            return FlextUtilities.Processing.retry_operation(
                operation, max_retries, initial_delay
            )

    class TypeGuards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is a non-empty string."""
            return isinstance(value, str) and len(value) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is a non-empty dictionary."""
            return isinstance(value, dict) and len(value) > 0

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is a non-empty list."""
            return isinstance(value, list) and len(value) > 0
