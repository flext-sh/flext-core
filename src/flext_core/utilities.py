"""FLEXT Utilities - Comprehensive utility functions with hierarchical organization and type safety.

Comprehensive utility collection providing FlextUtilities class with ID generation, text processing,
validation, performance monitoring, JSON handling, and decorator patterns organized in nested classes
following SOLID principles with FlextResult integration for consistent error handling.

Module Role in Architecture:
    FlextUtilities provides essential utility functions for all FLEXT ecosystem components,
    organized hierarchically from Generators to Decorators, enabling consistent ID generation,
    text processing, validation, and performance monitoring across the ecosystem.

Classes and Methods:
    FlextUtilities:                         # Hierarchical utility collection
        # Generators - ID and Timestamp Generation:
        Generators.generate_uuid() -> str               # Generate RFC4122 UUID
        Generators.generate_entity_id(prefix="entity") -> str # Generate entity identifier
        Generators.generate_correlation_id(prefix="corr") -> str # Generate correlation ID
        Generators.generate_request_id(prefix="req") -> str # Generate request identifier
        Generators.generate_timestamp() -> str          # Generate ISO timestamp
        Generators.generate_hash(data) -> str           # Generate SHA-256 hash

        # TextProcessor - Text Processing and Formatting:
        TextProcessor.truncate(text, max_length=100) -> str # Truncate text safely
        TextProcessor.safe_string(value) -> str         # Convert any value to safe string
        TextProcessor.sanitize_filename(filename) -> str # Sanitize filename for filesystem
        TextProcessor.extract_keywords(text) -> list[str] # Extract keywords from text
        TextProcessor.format_bytes(size_bytes) -> str   # Format byte size human-readable
        TextProcessor.slugify(text) -> str              # Convert text to URL slug
        TextProcessor.generate_camel_case_alias(field_name) -> str # Convert snake_case to camelCase

        # Validators - Data Validation Utilities:
        Validators.validate_email(email) -> FlextResult[str] # Validate email format
        Validators.validate_url(url) -> FlextResult[str] # Validate URL format
        Validators.validate_json(json_str) -> FlextResult[dict] # Validate JSON string
        Validators.validate_uuid(uuid_str) -> FlextResult[str] # Validate UUID format
        Validators.is_valid_identifier(name) -> bool    # Check if valid Python identifier
        Validators.validate_phone(phone) -> FlextResult[str] # Validate phone number

        # Performance - Performance Monitoring and Timing:
        Performance.time_function(func) -> Callable     # Decorator for function timing
        Performance.measure_execution(func, *args) -> tuple[object, float] # Measure function execution
        Performance.benchmark_operations(operations) -> dict # Benchmark multiple operations
        Performance.memory_usage() -> dict              # Get current memory usage
        Performance.system_metrics() -> dict            # Get system performance metrics

        # JSON - JSON Processing and Serialization:
        JSON.safe_loads(json_str) -> FlextResult[dict]  # Safe JSON deserialization
        JSON.safe_dumps(obj) -> FlextResult[str]        # Safe JSON serialization
        JSON.merge_dicts(*dicts) -> dict               # Deep merge multiple dictionaries
        JSON.flatten_dict(nested_dict) -> dict         # Flatten nested dictionary
        JSON.unflatten_dict(flat_dict) -> dict         # Unflatten dictionary

        # Decorators - Utility Decorators and Function Wrappers:
        Decorators.retry(max_attempts=3) -> Callable    # Retry decorator with exponential backoff
        Decorators.timeout(seconds=30) -> Callable      # Timeout decorator
        Decorators.cache_result(ttl=300) -> Callable    # Simple result caching decorator
        Decorators.log_calls(logger=None) -> Callable   # Log function calls decorator
        Decorators.validate_args(*validators) -> Callable # Argument validation decorator

        # Configuration Methods:
        configure_utilities_system(config) -> FlextResult[ConfigDict] # Configure utility system
        get_utilities_system_config() -> FlextResult[ConfigDict] # Get current config
        create_environment_utilities_config(environment) -> FlextResult[ConfigDict] # Environment config
        optimize_utilities_performance(performance_level) -> FlextResult[ConfigDict] # Performance optimization

Usage Examples:
    ID Generation:
        uuid = FlextUtilities.Generators.generate_uuid()
        entity_id = FlextUtilities.Generators.generate_entity_id("user")
        correlation_id = FlextUtilities.Generators.generate_correlation_id()

    Text Processing:
        truncated = FlextUtilities.TextProcessor.truncate("Long text here", 50)
        slug = FlextUtilities.TextProcessor.slugify("My Article Title")
        safe_filename = FlextUtilities.TextProcessor.sanitize_filename("file<name>.txt")

    Validation with FlextResult:
        email_result = FlextUtilities.Validators.validate_email("user@example.com")
        if email_result.success:
            validated_email = email_result.value

    Performance monitoring:
        @FlextUtilities.Performance.time_function
        def my_function():
            # Function implementation
            pass

    Configuration:
        config = {
            "environment": "production",
            "enable_caching": True,
            "default_timeout": 30,
        }
        FlextUtilities.configure_utilities_system(config)

Integration:
    FlextUtilities integrates with FlextResult for error handling, FlextTypes.Config
    for configuration, FlextConstants for limits and defaults, providing comprehensive
    utility functions with consistent patterns across the entire FLEXT ecosystem.
"""

from __future__ import annotations

import functools
import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from pydantic import ConfigDict, ValidationError

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, P, R, T

logger = FlextLogger(__name__)

# Performance metrics storage.
PERFORMANCE_METRICS: dict[str, dict[str, object]] = {}


class FlextUtilities:
    """Comprehensive utility functions organized by functional domain.

    This class serves as the central container for all FLEXT utility functions,
    organized into nested classes by functional area. Provides common operations
    for ID generation, text processing, validation, performance monitoring,
    JSON handling, and more.

    Architecture:
        The utilities are organized into nested classes following single
        responsibility principle:
        - Generators: ID and timestamp generation utilities
        - TextProcessor: Text manipulation and formatting
        - Validators: Data validation with FlextResult patterns
        - Performance: Timing and performance monitoring
        - JSON: JSON processing and serialization
        - Decorators: Function decorators and wrappers

    Examples:
        ID generation utilities::

            uuid = FlextUtilities.Generators.generate_uuid()
            entity_id = FlextUtilities.Generators.generate_entity_id()
            correlation_id = FlextUtilities.Generators.generate_correlation_id()

        Text processing utilities::

            truncated = FlextUtilities.TextProcessor.truncate(text, 100)
            cleaned = FlextUtilities.TextProcessor.safe_string(value)

        Validation utilities::

            email_result = FlextUtilities.Validators.validate_email(email)
            if email_result.success:
                process_valid_email(email)

    Note:
        All utilities follow FLEXT patterns including FlextResult for error
        handling, FlextConstants for configuration, and structured logging.

    """

    # ==========================================================================
    # CLASS CONSTANTS
    # ==========================================================================

    MIN_PORT: int = FlextConstants.Network.MIN_PORT
    MAX_PORT: int = FlextConstants.Network.MAX_PORT

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class Generators:
        """ID and timestamp generation utilities with consistent prefixing.

        Provides various ID generation methods with semantic prefixes for
        different use cases. All IDs use UUID4 for uniqueness with shortened
        hex representations for readability.

        Examples:
            Generate different types of IDs::

                uuid = Generators.generate_uuid()  # Full UUID4
                entity_id = Generators.generate_entity_id()  # entity_xxxx
                correlation_id = Generators.generate_correlation_id()  # corr_xxxx
                session_id = Generators.generate_session_id()  # sess_xxxx

        """

        @staticmethod
        def generate_uuid() -> str:
            """Generate standard UUID4."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_id() -> str:
            """Generate unique ID with flext_ prefix."""
            return f"flext_{uuid.uuid4().hex[:8]}"

        @staticmethod
        def generate_entity_id() -> str:
            """Generate entity ID with entity_ prefix."""
            return f"entity_{uuid.uuid4().hex[:12]}"

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate correlation ID for request tracing."""
            return f"corr_{uuid.uuid4().hex[:16]}"

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO timestamp in UTC."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_session_id() -> str:
            """Generate session ID for web sessions."""
            return f"sess_{uuid.uuid4().hex[:16]}"

        @staticmethod
        def generate_request_id() -> str:
            """Generate request ID for web requests."""
            return f"req_{uuid.uuid4().hex[:12]}"

    class TextProcessor:
        """Text processing, formatting, and safe conversion utilities.

        Provides safe text processing functions with proper error handling
        and consistent formatting. Handles edge cases and provides fallbacks
        for robust text manipulation.

        Examples:
            Text processing operations::

                truncated = TextProcessor.truncate(long_text, 50, "...")
                safe_str = TextProcessor.safe_string(any_object, "default")

        """

        @staticmethod
        def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
            """Truncate text to maximum length with suffix."""
            if len(text) <= max_length:
                return text
            if max_length <= len(suffix):
                return text[:max_length]
            return text[: max_length - len(suffix)] + suffix

        @staticmethod
        def safe_string(value: object, default: str = "") -> str:
            """Convert any value to string safely."""
            try:
                return str(value) if value is not None else default
            except Exception:
                return default

        @staticmethod
        def clean_text(text: str) -> str:
            """Clean text by removing extra whitespace and controlling characters."""
            if not text:
                return ""
            # Remove control characters except tabs and newlines
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
            # Normalize whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned.strip()

        @staticmethod
        def slugify(text: str) -> str:
            """Convert text to URL-safe slug format."""
            if not text:
                return ""
            # Convert to lowercase and remove extra whitespace
            slug = text.lower().strip()
            # Replace non-alphanumeric characters with hyphens
            slug = re.sub(r"[^a-z0-9]+", "-", slug)
            # Remove leading/trailing hyphens
            return slug.strip("-")

        @staticmethod
        def mask_sensitive(
            text: str,
            mask_char: str = "*",
            visible_chars: int = 4,
            show_first: int | None = None,
            show_last: int | None = None,
        ) -> str:
            """Mask sensitive information with flexible visibility options.

            Args:
                text: Text to mask
                mask_char: Character to use for masking
                visible_chars: Number of characters to show at the end (legacy parameter)
                show_first: Number of characters to show at the start
                show_last: Number of characters to show at the end

            """
            if not text:
                return mask_char * 8  # Default masked length

            # Handle new API with show_first and show_last
            if show_first is not None or show_last is not None:
                first_chars = show_first or 0
                last_chars = show_last or 0

                if len(text) <= (first_chars + last_chars):
                    return mask_char * len(text)

                first_part = text[:first_chars] if first_chars > 0 else ""
                last_part = text[-last_chars:] if last_chars > 0 else ""
                middle_length = len(text) - first_chars - last_chars
                masked_part = mask_char * middle_length

                return first_part + masked_part + last_part

            # Legacy API - show only last characters
            if len(text) <= visible_chars:
                return mask_char * len(text)

            visible_part = text[-visible_chars:]
            masked_part = mask_char * (len(text) - visible_chars)
            return masked_part + visible_part

        @staticmethod
        def generate_camel_case_alias(field_name: str) -> str:
            """Generate camelCase alias from snake_case field name.

            Used for Pydantic field aliasing to convert Python snake_case field names
            to camelCase for JSON serialization and API compatibility.

            Args:
                field_name: Snake_case field name to convert

            Returns:
                camelCase version of the field name
            Examples:
                >>> TextProcessor.generate_camel_case_alias("user_name")
                "userName"
                >>> TextProcessor.generate_camel_case_alias("is_active")
                "isActive"
                >>> TextProcessor.generate_camel_case_alias("created_at")
                "createdAt"

            """
            if not field_name:
                return ""
            components = field_name.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

    class TimeUtils:
        """Time and duration utilities with formatting and conversion.

        Provides utilities for time formatting, duration calculations, and
        timestamp operations with proper timezone handling.

        Examples:
            Time operations::

                formatted = TimeUtils.format_duration(123.45)  # "2.1m"
                utc_now = TimeUtils.get_timestamp_utc()

        """

        @staticmethod
        def format_duration(seconds: float) -> str:
            """Format duration in human-readable format."""
            if seconds < 1:
                return f"{seconds * 1000:.1f}ms"
            if seconds < FlextConstants.Utilities.SECONDS_PER_MINUTE:
                return f"{seconds:.1f}s"
            if seconds < FlextConstants.Utilities.SECONDS_PER_HOUR:
                return f"{seconds / FlextConstants.Utilities.SECONDS_PER_MINUTE:.1f}m"
            return f"{seconds / FlextConstants.Utilities.SECONDS_PER_HOUR:.1f}h"

        @staticmethod
        def get_timestamp_utc() -> datetime:
            """Get current UTC timestamp."""
            return datetime.now(UTC)

    class Performance:
        """Performance tracking and monitoring utilities.

        Provides decorators and utilities for tracking function performance,
        recording metrics, and monitoring operation success rates.

        Examples:
            Performance tracking::

                @Performance.track_performance("user_creation")
                def create_user(data):
                    return process_user_data(data)


                metrics = Performance.get_metrics("user_creation")

        """

        @staticmethod
        def track_performance(
            operation_name: str,
        ) -> Callable[[Callable[P, R]], Callable[P, R]]:
            """Decorator for tracking performance metrics."""

            def decorator(func: Callable[P, R]) -> Callable[P, R]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.perf_counter() - start_time
                        FlextUtilities.Performance.record_metric(
                            operation_name, duration, success=True
                        )
                        return result
                    except Exception as e:
                        duration = time.perf_counter() - start_time
                        FlextUtilities.Performance.record_metric(
                            operation_name, duration, success=False, error=str(e)
                        )
                        raise

                return wrapper

            return decorator

        @staticmethod
        def record_metric(
            operation: str,
            duration: float,
            *,
            success: bool = True,
            error: str | None = None,
        ) -> None:
            """Record performance metric."""
            if operation not in PERFORMANCE_METRICS:
                PERFORMANCE_METRICS[operation] = {
                    "total_calls": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                }

            metrics = PERFORMANCE_METRICS[operation]
            total_calls = cast("int", metrics["total_calls"]) + 1
            total_duration = cast("float", metrics["total_duration"]) + duration

            metrics["total_calls"] = total_calls
            metrics["total_duration"] = total_duration
            metrics["avg_duration"] = total_duration / total_calls

            if success:
                metrics["success_count"] = cast("int", metrics["success_count"]) + 1
            else:
                metrics["error_count"] = cast("int", metrics["error_count"]) + 1
                if error:
                    metrics["last_error"] = error

        @staticmethod
        def get_metrics(operation: str | None = None) -> dict[str, object]:
            """Get performance metrics."""
            if operation:
                return PERFORMANCE_METRICS.get(operation, {})
            return dict(PERFORMANCE_METRICS)

    class Conversions:
        """Safe type conversion utilities with fallback handling.

        Provides robust type conversion functions that handle edge cases
        and provide sensible defaults when conversion fails.

        Examples:
            Safe conversions::

                num = Conversions.safe_int("123", 0)  # 123
                flag = Conversions.safe_bool("true")  # True
                val = Conversions.safe_float(None, 0.0)  # 0.0

        """

        @staticmethod
        def safe_int(value: object, default: int = 0) -> int:
            """Convert value to int safely."""
            if value is None:
                return default
            try:
                # Type narrowing: handle string and numeric types
                if isinstance(value, (str, int, float)):
                    return int(value)
                # For other objects, try str conversion first
                return int(str(value))
            except (ValueError, TypeError, OverflowError):
                return default

        @staticmethod
        def safe_float(value: object, default: float = 0.0) -> float:
            """Convert value to float safely."""
            if value is None:
                return default
            try:
                # Type narrowing: handle string and numeric types
                if isinstance(value, (str, int, float)):
                    return float(value)
                # For other objects, try str conversion first
                return float(str(value))
            except (ValueError, TypeError, OverflowError):
                return default

        @staticmethod
        def safe_bool(value: object, *, default: bool = False) -> bool:
            """Convert value to bool safely."""
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "on"}
            try:
                return bool(value)
            except (ValueError, TypeError):
                return default

    class TypeGuards:
        """Type checking and validation guard utilities.

        Provides type guard functions for runtime type checking and
        validation with proper type narrowing support.

        Examples:
            Type validation::

                if TypeGuards.is_string_non_empty(value):
                    process_string(value)  # value is now str

                if TypeGuards.is_dict_non_empty(data):
                    process_dict(data)  # data is now dict

        """

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is non-empty string."""
            return isinstance(value, str) and len(value) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is non-empty dict."""
            if isinstance(value, dict):
                sized_dict = cast("FlextProtocols.Foundation.SizedDict", value)
                return len(sized_dict) > 0
            return False

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is non-empty list."""
            if isinstance(value, list):
                sized_list = cast("FlextProtocols.Foundation.SizedList", value)
                return len(sized_list) > 0
            return False

        @staticmethod
        def has_attribute(obj: object, attr: str) -> bool:
            """Check if object has attribute."""
            return hasattr(obj, attr)

        @staticmethod
        def is_not_none(value: object) -> bool:
            """Check if value is not None."""
            return value is not None

    class Formatters:
        """Data formatting utilities for human-readable output.

        Provides formatting functions for common data types including
        byte sizes, percentages, and other human-readable formats.

        Examples:
            Data formatting::

                size = Formatters.format_bytes(1024)  # "1.0 KB"
                percent = Formatters.format_percentage(0.85)  # "85.0%"

        """

        @staticmethod
        def format_bytes(bytes_count: int) -> str:
            """Format byte count in human-readable format."""
            kb = FlextConstants.Utilities.BYTES_PER_KB
            if bytes_count < kb:
                return f"{bytes_count} B"
            if bytes_count < kb**2:
                return f"{bytes_count / kb:.1f} KB"
            if bytes_count < kb**3:
                return f"{bytes_count / (kb**2):.1f} MB"
            return f"{bytes_count / (kb**3):.1f} GB"

        @staticmethod
        def format_percentage(value: float, precision: int = 1) -> str:
            """Format value as percentage."""
            return f"{value * 100:.{precision}f}%"

    class ProcessingUtils:
        """Data processing utilities for JSON, models, and structured data.

        Provides safe processing functions for JSON parsing, model extraction,
        and data validation with FlextResult error handling.

        Examples:
            Data processing::

                data = ProcessingUtils.safe_json_parse(json_str, {})
                json_str = ProcessingUtils.safe_json_stringify(obj)
                result = ProcessingUtils.parse_json_to_model(json_str, MyModel)

        """

        @staticmethod
        def safe_json_parse(
            json_str: str, default: dict[str, object] | None = None
        ) -> dict[str, object]:
            """Safely parse JSON string."""
            try:
                result: object = json.loads(json_str)
                if isinstance(result, dict):
                    return cast("dict[str, object]", result)
                return default or {}
            except (json.JSONDecodeError, TypeError):
                return default or {}

        @staticmethod
        def safe_json_stringify(obj: object, default: str = "{}") -> str:
            """Safely stringify object to JSON."""
            try:
                return json.dumps(obj, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                return default

        @staticmethod
        def extract_model_data(obj: object) -> dict[str, object]:
            """Extract data from Pydantic model or dict."""
            if hasattr(obj, "model_dump"):
                model_obj = cast("FlextProtocols.Foundation.HasModelDump", obj)
                return model_obj.model_dump()
            if hasattr(obj, "dict"):
                dict_obj = cast("FlextProtocols.Foundation.HasDict", obj)
                return dict_obj.dict()
            if isinstance(obj, dict):
                return cast("dict[str, object]", obj)
            return {}

        @staticmethod
        def parse_json_to_model[TModel](
            json_text: str, model_class: type[TModel]
        ) -> FlextResult[TModel]:
            """Parse JSON and validate using appropriate model instantiation strategy.

            This method provides type-safe JSON parsing with automatic detection of:
            - Pydantic v2 models (using model_validate)
            - Dictionary-constructible classes (using **kwargs)
            - Default constructible classes (using no args)

            Args:
                json_text: JSON string to parse and validate
                model_class: Target model class for instantiation

            Returns:
                FlextResult containing validated model instance or error details

            Note:
                Uses Python 3.13+ generic syntax with strategic casting to handle
                dynamic method invocation while preserving type safety.

            """
            try:
                parsed_data: object = json.loads(json_text)

                # Strategy 1: Pydantic v2 model_validate (preferred for validation)
                if hasattr(model_class, "model_validate") and callable(
                    getattr(model_class, "model_validate", None)
                ):
                    # Dynamic invocation through cast to protocol
                    pydantic_class = cast(
                        "type[FlextProtocols.Foundation.HasModelValidate]",
                        model_class,  # pyright: ignore[reportGeneralTypeIssues]
                    )
                    validated_obj = pydantic_class.model_validate(parsed_data)
                    # Direct cast to target type
                    instance = cast("TModel", validated_obj)
                    return FlextResult[TModel].ok(instance)

                # Strategy 2: Dictionary constructor (for dict-like data)
                if isinstance(parsed_data, dict):
                    dict_data = cast("dict[str, object]", parsed_data)
                    instance = model_class(**dict_data)
                    return FlextResult[TModel].ok(instance)

                # Strategy 3: Default constructor (fallback)
                instance = model_class()
                return FlextResult[TModel].ok(instance)

            except json.JSONDecodeError as json_error:
                error_message = f"Invalid JSON format: {json_error}"
            except ValidationError as pydantic_error:
                # Handle Pydantic ValidationError specifically for better error messages
                error_message = f"Pydantic validation failed: {pydantic_error}"
            except (TypeError, ValueError) as validation_error:
                error_message = f"Model validation failed: {validation_error}"
            except Exception as unexpected_error:
                error_message = (
                    f"Unexpected error during model creation: {unexpected_error}"
                )

            # Single return point for all error cases
            return FlextResult[TModel].fail(
                error_message,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    class ResultUtils:
        """FlextResult processing and composition utilities.

        Provides utilities for working with FlextResult types including
        result chaining, batch processing, and error collection.

        Examples:
            Result operations::

                combined = ResultUtils.chain_results(result1, result2)
                successes, errors = ResultUtils.batch_process(items, processor)

        """

        @staticmethod
        def chain_results(*results: FlextResult[T]) -> FlextResult[list[T]]:
            """Chain multiple FlextResults into a single result."""
            values: list[T] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[T]].fail(
                        result.error or "Chain operation failed"
                    )
                values.append(result.value)
            return FlextResult[list[T]].ok(values)

        @staticmethod
        def batch_process[TInput, TOutput](
            items: list[TInput],
            processor: Callable[[TInput], FlextResult[TOutput]],
        ) -> tuple[list[TOutput], list[str]]:
            """Process list of items, collecting successes and errors."""
            successes: list[TOutput] = []
            errors: list[str] = []

            for item in items:
                result = processor(item)
                if result.is_success:
                    successes.append(result.value)
                else:
                    errors.append(result.error or "Unknown error")

            return successes, errors

    class Configuration:
        """Configuration utilities with FlextTypes.Config and StrEnum integration.

        Provides comprehensive configuration management utilities including
        environment detection, configuration validation, and system configuration
        generation using FlextTypes.Config hierarchical structure.

        Examples:
            Configuration utilities::

                config = Configuration.create_default_config("production")
                result = Configuration.validate_config(config_dict)
                env_config = Configuration.get_environment_configuration("staging")

        """

        @staticmethod
        def create_default_config(
            environment: FlextTypes.Config.Environment = "development",
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Create default configuration for specified environment using FlextTypes.Config.

            Args:
                environment: Target environment using FlextTypes.Config.Environment.

            Returns:
                FlextResult containing default configuration dictionary.

            """
            try:
                # Validate environment is a valid StrEnum value
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if environment not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment: {environment}. Must be one of: {valid_environments}"
                    )

                # Create environment-specific configuration
                config: FlextTypes.Config.ConfigDict = {
                    "environment": environment,
                    "log_level": (
                        FlextConstants.Config.LogLevel.ERROR.value
                        if environment
                        == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                        else FlextConstants.Config.LogLevel.DEBUG.value
                    ),
                    "validation_level": (
                        FlextConstants.Config.ValidationLevel.STRICT.value
                        if environment
                        == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                        else FlextConstants.Config.ValidationLevel.NORMAL.value
                    ),
                    "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
                    "debug": environment
                    != FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                    "performance_monitoring": True,
                    "request_timeout": (
                        60000
                        if environment
                        == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                        else 30000
                    ),
                    "max_retries": 3,
                    "enable_caching": True,
                }

                return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Default config creation failed: {e}"
                )

        @staticmethod
        def validate_configuration_with_types(  # noqa: PLR0911, PLR0912
            config: FlextTypes.Config.ConfigDict,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Validate configuration using FlextTypes.Config with comprehensive StrEnum validation.

            Args:
                config: Configuration dictionary to validate.

            Returns:
                FlextResult containing validated configuration or validation errors.

            """
            # Configuration validation constants
            min_timeout_ms = 100
            max_timeout_ms = 300000
            max_retries = 10

            try:
                validated: FlextTypes.Config.ConfigDict = {}

                # Environment validation
                if "environment" not in config:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        "Required field 'environment' missing"
                    )

                env_value = config["environment"]
                valid_environments = {
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                }
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {sorted(valid_environments)}"
                    )
                validated["environment"] = env_value

                # Log level validation
                log_level = config.get(
                    "log_level", FlextConstants.Config.LogLevel.INFO.value
                )
                valid_log_levels = {
                    level.value for level in FlextConstants.Config.LogLevel
                }
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {sorted(valid_log_levels)}"
                    )
                validated["log_level"] = log_level

                # Validation level validation
                validation_level = config.get(
                    "validation_level",
                    FlextConstants.Config.ValidationLevel.NORMAL.value,
                )
                valid_validation_levels = {
                    v.value for v in FlextConstants.Config.ValidationLevel
                }
                if validation_level not in valid_validation_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{validation_level}'. Valid options: {sorted(valid_validation_levels)}"
                    )
                validated["validation_level"] = validation_level

                # Config source validation
                config_source = config.get(
                    "config_source",
                    FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
                )
                valid_config_sources = {
                    s.value for s in FlextConstants.Config.ConfigSource
                }
                if config_source not in valid_config_sources:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid config_source '{config_source}'. Valid options: {sorted(valid_config_sources)}"
                    )
                validated["config_source"] = config_source

                # Boolean validations
                for bool_field in ["debug", "performance_monitoring", "enable_caching"]:
                    if bool_field in config:
                        value = config[bool_field]
                        if not isinstance(value, bool):
                            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                                f"Field '{bool_field}' must be a boolean"
                            )
                        validated[bool_field] = value

                # Numeric validations
                if "request_timeout" in config:
                    timeout = config["request_timeout"]
                    if (
                        not isinstance(timeout, (int, float))
                        or timeout < min_timeout_ms
                        or timeout > max_timeout_ms
                    ):
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            "request_timeout must be a number between 100 and 300000 milliseconds"
                        )
                    validated["request_timeout"] = timeout

                if "max_retries" in config:
                    retries = config["max_retries"]
                    if (
                        not isinstance(retries, int)
                        or retries < 0
                        or retries > max_retries
                    ):
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            "max_retries must be an integer between 0 and 10"
                        )
                    validated["max_retries"] = retries

                return FlextResult[FlextTypes.Config.ConfigDict].ok(validated)

            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Configuration validation failed: {e}"
                )

        @staticmethod
        def get_environment_configuration(
            environment: FlextTypes.Config.Environment,
        ) -> FlextResult[dict[str, object]]:
            """Get comprehensive environment-specific configuration using FlextTypes.Config.

            Args:
                environment: Target environment for configuration.

            Returns:
                FlextResult containing environment-specific configuration details.

            """
            try:
                # Create base configuration
                config_result = FlextUtilities.Configuration.create_default_config(
                    environment
                )
                if config_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        config_result.error or "Failed to create base configuration"
                    )

                base_config: ConfigDict = cast("ConfigDict", config_result.value)

                # Create comprehensive environment configuration
                env_config: dict[str, object] = {
                    "base_configuration": base_config,
                    "environment_metadata": {
                        "name": environment,
                        "is_production": environment
                        == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                        "is_development": environment
                        == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                        "is_testing": environment
                        == FlextConstants.Config.ConfigEnvironment.TEST.value,
                    },
                    "available_environments": [
                        e.value for e in FlextConstants.Config.ConfigEnvironment
                    ],
                    "available_log_levels": [
                        level.value for level in FlextConstants.Config.LogLevel
                    ],
                    "available_validation_levels": [
                        v.value for v in FlextConstants.Config.ValidationLevel
                    ],
                    "available_config_sources": [
                        s.value for s in FlextConstants.Config.ConfigSource
                    ],
                    "performance_settings": {
                        "request_timeout_ms": base_config.get("request_timeout", 30000),
                        "max_retries": base_config.get("max_retries", 3),
                        "caching_enabled": base_config.get("enable_caching", True),
                        "monitoring_enabled": base_config.get(
                            "performance_monitoring", True
                        ),
                    },
                    "security_settings": {
                        "debug_mode": base_config.get("debug", False),
                        "strict_validation": base_config.get("validation_level")
                        == "strict",
                        "log_level": base_config.get("log_level", "INFO"),
                    },
                }

                return FlextResult[dict[str, object]].ok(env_config)

            except Exception as e:
                return FlextResult[dict[str, object]].fail(
                    f"Environment configuration generation failed: {e}"
                )

    # ==========================================================================
    # MAIN CLASS METHODS - Delegate to nested classes
    # ==========================================================================

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID (delegates to Generators)."""
        return cls.Generators.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate ID (delegates to Generators)."""
        return cls.Generators.generate_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID (delegates to Generators)."""
        return cls.Generators.generate_entity_id()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID (delegates to Generators)."""
        return cls.Generators.generate_correlation_id()

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text (delegates to TextProcessor)."""
        return cls.TextProcessor.truncate(text, max_length, suffix)

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration (delegates to TimeUtils)."""
        return cls.TimeUtils.format_duration(seconds)

    @classmethod
    def track_performance(
        cls, operation_name: str
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track performance (delegates to Performance)."""
        return cls.Performance.track_performance(operation_name)

    @classmethod
    def safe_json_parse(
        cls, json_str: str, default: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Parse JSON safely (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.safe_json_parse(json_str, default)

    @classmethod
    def safe_json_stringify(cls, obj: object, default: str = "{}") -> str:
        """Stringify object to JSON safely (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.safe_json_stringify(obj, default)

    @classmethod
    def parse_json_to_model(
        cls, json_text: str, model_class: type[T]
    ) -> FlextResult[T]:
        """Parse JSON to model (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.parse_json_to_model(json_text, model_class)

    @classmethod
    def safe_int(cls, value: object, default: int = 0) -> int:
        """Convert to int safely (delegates to Conversions)."""
        return cls.Conversions.safe_int(value, default)

    @classmethod
    def batch_process[TInput, TOutput](
        cls,
        items: list[TInput],
        processor: Callable[[TInput], FlextResult[TOutput]],
    ) -> tuple[list[TOutput], list[str]]:
        """Process batch (delegates to ResultUtils)."""
        return cls.ResultUtils.batch_process(items, processor)

    # Additional methods needed by legacy compatibility layer
    @classmethod
    def safe_int_conversion(
        cls, value: object, default: int | None = None
    ) -> int | None:
        """Convert value to int safely with optional default."""
        if value is None:
            return default
        try:
            # Type narrowing: handle string and numeric types
            if isinstance(value, (str, int, float)):
                return int(value)
            # For other objects, try str conversion first
            return int(str(value))
        except (ValueError, TypeError, OverflowError):
            return default

    @classmethod
    def safe_int_conversion_with_default(cls, value: object, default: int) -> int:
        """Convert value to int safely with guaranteed default."""
        return cls.safe_int_conversion(value, default) or default

    @classmethod
    def safe_bool_conversion(cls, value: object, *, default: bool = False) -> bool:
        """Convert value to bool safely (delegates to Conversions)."""
        return cls.Conversions.safe_bool(value, default=default)

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO timestamp (delegates to Generators)."""
        return cls.Generators.generate_iso_timestamp()

    @classmethod
    def get_performance_metrics(cls) -> dict[str, object]:
        """Get all performance metrics (delegates to Performance)."""
        return cls.Performance.get_metrics()

    @classmethod
    def record_performance(
        cls,
        operation: str,
        duration: float,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record performance metric (delegates to Performance)."""
        return cls.Performance.record_metric(
            operation, duration, success=success, error=error
        )

    # Additional delegator methods needed by flext-cli
    @classmethod
    def is_non_empty_string(cls, value: object) -> bool:
        """Check if value is non-empty string (delegates to TypeGuards)."""
        return cls.TypeGuards.is_string_non_empty(value)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean text (delegates to TextProcessor)."""
        return cls.TextProcessor.clean_text(text)

    @classmethod
    def generate_timestamp(cls) -> str:
        """Generate timestamp (delegates to Generators)."""
        return cls.Generators.generate_iso_timestamp()

    @classmethod
    def parse_iso_timestamp(cls, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string to datetime."""
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # Return current UTC time as fallback
            return datetime.now(UTC)

    @classmethod
    def get_elapsed_time(cls, start_time: datetime) -> float:
        """Get elapsed time from start_time to now in seconds."""
        current_time = datetime.now(UTC)
        # Ensure both timestamps are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)

        delta = current_time - start_time
        return delta.total_seconds()


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextUtilities",  # ONLY main class exported
]
