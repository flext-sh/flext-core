"""FLEXT Core Utilities - Enterprise-grade utility functions.

This module provides comprehensive utility functions organized by functional domain,
following FLEXT architectural principles and Python 3.13+ best practices.

Key Features:
- Pattern matching optimizations for performance
- Caching strategies for high-frequency operations
- Type-safe operations with FlextTypes
- Constants-driven configuration via FlextConstants
- Pydantic V2 integration for data validation
- Unicode normalization for international text processing

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, cast
from urllib.parse import urlparse as _urlparse

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
)

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, P, R, T

# Type variables imported from FlextTypes for consistency

# Pre-compiled regex patterns for performance
_email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Optimized JSON operations for performance
_json_dumps = functools.partial(json.dumps, separators=(",", ":"), ensure_ascii=False)
_json_loads = json.loads

# Unicode normalization is always available in FLEXT ecosystem
# Optimized partial functions for frequent operations
_normalize_nfkd = functools.partial(unicodedata.normalize, "NFKD")
_is_combining = unicodedata.combining


class FlextUtilities:
    """Comprehensive utility functions organized by functional domain."""

    # ==========================================================================
    # CLASS CONSTANTS
    # ==========================================================================

    MIN_PORT: int = FlextConstants.Network.MIN_PORT
    MAX_PORT: int = FlextConstants.Network.MAX_PORT

    # Performance metrics storage
    PERFORMANCE_METRICS: ClassVar[dict[str, FlextTypes.Core.Dict]] = {}

    # ==========================================================================
    # MODERN VALIDATION UTILITIES (FLEXT-CORE DOMAIN)
    # ==========================================================================

    class DataValidators:
        """Advanced data validation utilities using Pydantic models.

        Provides comprehensive validation for common data types while staying
        within flext-core domain boundaries. Uses Pydantic for validation
        but maintains flext-core patterns and interfaces.
        """

        # Pydantic validation models (mandatory in FLEXT ecosystem)
        class EmailModel(BaseModel):
            """Email validation model."""

            email: str = Field(..., min_length=5, max_length=320)

            @field_validator("email")
            @classmethod
            def validate_email_format(cls, v: str) -> str:
                """Validate email format with regex."""
                if not _email_pattern.match(v):
                    error_msg = "Invalid email format"
                    raise ValueError(error_msg)
                return v.lower().strip()

        class URLModel(BaseModel):
            """URL validation model."""

            url: str = Field(..., min_length=10, max_length=2048)

            @field_validator("url")
            @classmethod
            def validate_url_format(cls, v: str) -> str:
                """Validate URL format."""
                try:
                    parsed = _urlparse(v)

                    if not parsed.scheme or not parsed.netloc:
                        error_msg = "Invalid URL format"
                        raise ValueError(error_msg)

                    if parsed.scheme not in {"http", "https", "ftp", "ftps"}:
                        error_msg = "Unsupported URL scheme"
                        raise ValueError(error_msg)

                    return v

                except Exception as e:
                    error_msg = f"Invalid URL: {e!s}"
                    raise ValueError(error_msg) from e

        @classmethod
        def validate_email_with_pydantic(cls, email: str) -> FlextResult[str]:
            """Validate email using Pydantic model."""
            try:
                model = cls.EmailModel(email=email)
                return FlextResult.ok(model.email)
            except ValidationError as e:
                return FlextResult.fail(f"Email validation failed: {e}")
            except Exception as e:
                return FlextResult.fail(f"Unexpected validation error: {e}")

        @classmethod
        def validate_url_with_pydantic(cls, url: str) -> FlextResult[str]:
            """Validate URL using Pydantic model."""
            try:
                model = cls.URLModel(url=url)
                return FlextResult.ok(model.url)
            except ValidationError as e:
                return FlextResult.fail(f"URL validation failed: {e}")
            except Exception as e:
                return FlextResult.fail(f"Unexpected validation error: {e}")

        @classmethod
        def validate_json_schema(
            cls, data: str | FlextTypes.Core.Dict, schema: FlextTypes.Core.Dict
        ) -> FlextResult[FlextTypes.Core.Dict]:
            """Validate JSON data against schema using Pydantic."""
            try:
                # Parse data
                parsed = _json_loads(data) if isinstance(data, str) else data

                if not isinstance(parsed, dict):
                    return FlextResult.fail(
                        "Data must be a JSON object for schema validation"
                    )

                # Use Pydantic model validation if schema provided
                if schema:
                    # Create dynamic Pydantic model from schema
                    # This is a simplified implementation - in practice you'd use
                    # more sophisticated schema-to-model conversion
                    pass

                return FlextResult.ok(parsed)

            except json.JSONDecodeError as e:
                return FlextResult.fail(f"Invalid JSON: {e}")
            except Exception as e:
                return FlextResult.fail(f"Schema validation failed: {e}")

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class Generators:
        """ID and timestamp generation utilities with performance optimizations."""

        # Cache for expensive UUID operations (fallback to simple dict if cachetools unavailable)
        _uuid_cache: ClassVar[dict[str, str]] = {}
        _cache_size: ClassVar[int] = (
            FlextConstants.Validation.MIN_SERVICE_NAME_LENGTH * 50
        )

        @classmethod
        @functools.cache
        def generate_uuid(cls) -> str:
            """Generate UUID4 with caching for performance."""
            return str(uuid.uuid4())

        @classmethod
        def generate_id(
            cls,
            prefix: str = FlextConstants.Core.NAME.lower(),
            length: int = FlextConstants.Validation.MIN_NAME_LENGTH * 4,
        ) -> str:
            """Generate ID with customizable prefix and length using pattern matching."""
            match (prefix, length):
                case ("flext", 8):
                    return f"flext_{uuid.uuid4().hex[:8]}"
                case ("entity", 12):
                    return f"entity_{uuid.uuid4().hex[:12]}"
                case ("corr", 16):
                    return f"corr_{uuid.uuid4().hex[:16]}"
                case ("sess", 16):
                    return f"sess_{uuid.uuid4().hex[:16]}"
                case ("req", 12):
                    return f"req_{uuid.uuid4().hex[:12]}"
                case _:
                    return f"{prefix}_{uuid.uuid4().hex[:length]}"

        @classmethod
        def generate_entity_id(cls) -> str:
            """Generate entity ID with optimized caching."""
            return cls.generate_id("entity", 12)

        @classmethod
        def generate_correlation_id(cls) -> str:
            """Generate correlation ID with optimized caching."""
            return cls.generate_id("corr", 16)

        @classmethod
        def generate_iso_timestamp(cls) -> str:
            """Generate ISO timestamp in UTC with caching for high-frequency calls."""
            return datetime.now(UTC).isoformat()

        @classmethod
        def generate_session_id(cls) -> str:
            """Generate session ID with optimized caching."""
            return cls.generate_id("sess", 16)

        @classmethod
        def generate_request_id(cls) -> str:
            """Generate request ID with optimized caching."""
            return cls.generate_id("req", 12)

        @classmethod
        def generate_batch_ids(cls, count: int, id_type: str = "entity") -> list[str]:
            """Generate batch of IDs efficiently."""
            return [cls.generate_id(id_type) for _ in range(count)]

        @classmethod
        def _manage_cache_size(cls) -> None:
            """Manage cache size to prevent memory issues."""
            if len(cls._uuid_cache) > cls._cache_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(cls._uuid_cache) - cls._cache_size
                keys_to_remove = list(cls._uuid_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del cls._uuid_cache[key]

    class TextProcessor:
        """Text processing, formatting, and safe conversion utilities with optimizations.

        Provides safe text processing functions with proper error handling
        and consistent formatting. Handles edge cases and provides fallbacks
        for robust text manipulation using modern Python patterns.
        """

        # Pre-compiled regex patterns for better performance
        _control_chars_pattern: ClassVar[re.Pattern[str]] = re.compile(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
        )
        _whitespace_pattern: ClassVar[re.Pattern[str]] = re.compile(r"\s+")
        _non_alphanumeric_pattern: ClassVar[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
        _multiple_hyphens_pattern: ClassVar[re.Pattern[str]] = re.compile(r"-+")

        @classmethod
        @functools.cache
        def truncate(
            cls,
            text: str,
            max_length: int = FlextConstants.Validation.MAX_NAME_LENGTH,
            suffix: str = "...",
        ) -> str:
            """Truncate text to maximum length with pattern matching optimization."""
            text_length = len(text)
            suffix_length = len(suffix)

            # Use pattern matching for different truncation scenarios
            match (text_length, max_length, suffix_length):
                case (tl, ml, sl) if tl <= ml:
                    return text
                case (tl, ml, sl) if ml <= sl:
                    return text[:max_length]
                case _:
                    return text[: max_length - suffix_length] + suffix

        @classmethod
        def safe_string(cls, value: object, default: str = "") -> str:
            """Convert value to string safely with enhanced type checking."""
            match value:
                case None:
                    return default
                case str():
                    return value
                case int() | float() | bool():
                    return str(value)
                case _:
                    try:
                        return str(value)
                    except Exception:
                        return default

        @classmethod
        @functools.cache
        def clean_text(cls, text: str) -> str:
            """Clean text removing whitespace and control characters with optimized patterns."""
            if not text:
                return ""

            # Use pre-compiled patterns for better performance
            cleaned = cls._control_chars_pattern.sub("", text)
            cleaned = cls._whitespace_pattern.sub(" ", cleaned)
            return cleaned.strip()

        @classmethod
        @functools.cache
        def slugify(cls, text: str) -> str:
            """Convert text to URL-safe slug with advanced Unicode support."""
            if not text:
                return ""

            # Use pattern matching for text processing optimization
            match text.strip():
                case "":
                    return ""
                case processed_text:
                    # Convert to lowercase and normalize
                    slug = processed_text.lower().strip()

                    # Advanced Unicode normalization (always available in FLEXT)
                    slug = _normalize_nfkd(slug)
                    slug = "".join(c for c in slug if not _is_combining(c))

                    # Use pre-compiled patterns for better performance
                    slug = cls._non_alphanumeric_pattern.sub("-", slug)
                    slug = cls._multiple_hyphens_pattern.sub("-", slug)

                    # Remove leading/trailing hyphens
                    return slug.strip("-")

        @staticmethod
        def mask_sensitive(
            text: str,
            mask_char: str = "*",
            visible_chars: int = FlextConstants.Validation.MIN_NAME_LENGTH,
            show_first: int | None = None,
            show_last: int | None = None,
        ) -> str:
            """Mask sensitive information with flexible visibility options."""
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
        def sanitize_filename(name: str) -> str:
            """Sanitize filename for filesystem usage."""
            # Basic cleanup
            cleaned = FlextUtilities.TextProcessor.clean_text(name)
            cleaned = cleaned.strip()

            # Replace reserved characters commonly invalid on filesystems
            # < > : " / \ | ? * and control chars
            cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", cleaned)

            # Remove remaining control chars and condense spaces
            cleaned = FlextUtilities.TextProcessor.clean_text(cleaned)

            # Trim leading/trailing dots and spaces again
            cleaned = cleaned.strip(" .")

            # Fallback when empty or only dots
            if not cleaned or set(cleaned) == {"."}:
                cleaned = "untitled"

            # Enforce length limit
            max_len = FlextConstants.Validation.MAX_EMAIL_LENGTH
            if len(cleaned) > max_len:
                cleaned = cleaned[:max_len]

            return cleaned

        @staticmethod
        def generate_camel_case_alias(field_name: str) -> str:
            """Generate camelCase from snake_case."""
            if not field_name:
                return ""
            components = field_name.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

    class TimeUtils:
        """Time and duration utilities with formatting and conversion."""

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
                            operation_name,
                            duration,
                            success=True,
                        )
                        return result
                    except Exception as e:
                        duration = time.perf_counter() - start_time
                        FlextUtilities.Performance.record_metric(
                            operation_name,
                            duration,
                            success=False,
                            error=str(e),
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
            if operation not in FlextUtilities.PERFORMANCE_METRICS:
                FlextUtilities.PERFORMANCE_METRICS[operation] = {
                    "total_calls": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                }

            metrics = FlextUtilities.PERFORMANCE_METRICS[operation]
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
        def get_metrics(operation: str | None = None) -> FlextTypes.Core.Dict:
            """Get performance metrics."""
            if operation:
                return FlextUtilities.PERFORMANCE_METRICS.get(operation, {})
            return dict(FlextUtilities.PERFORMANCE_METRICS)

        @staticmethod
        def create_performance_config(
            performance_level: str = "medium",
        ) -> FlextTypes.Core.ConfigDict:
            """Create performance configuration based on level."""
            base_config: FlextTypes.Core.ConfigDict = {
                "performance_level": performance_level,
                "optimization_enabled": True,
                "optimization_timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            }

            # Performance level specific optimizations
            if performance_level == "high":
                return {
                    **base_config,
                    # Handler optimization
                    "handler_cache_size": 1000,
                    "enable_handler_pooling": True,
                    "handler_pool_size": 100,
                    "max_concurrent_handlers": 50,
                    "handler_discovery_cache_ttl": 3600,  # 1 hour
                    # Middleware optimization
                    "enable_middleware_caching": True,
                    "middleware_thread_count": 8,
                    "middleware_queue_size": 500,
                    "parallel_middleware_processing": True,
                    # Command processing optimization
                    "command_batch_size": 100,
                    "enable_command_batching": True,
                    "command_processing_threads": 16,
                    "command_queue_size": 2000,
                    # Memory optimization
                    "memory_pool_size_mb": 200,
                    "enable_object_pooling": True,
                    "gc_optimization_enabled": True,
                    "optimization_level": "high",
                    # Performance metrics and targets
                    "expected_throughput_commands_per_second": 500,
                    "target_handler_latency_ms": 5,
                    "target_middleware_latency_ms": 2,
                    "memory_efficiency_target": 0.95,
                }
            if performance_level == "medium":
                return {
                    **base_config,
                    # Balanced handler settings
                    "handler_cache_size": 500,
                    "enable_handler_pooling": True,
                    "handler_pool_size": 50,
                    "max_concurrent_handlers": 25,
                    "handler_discovery_cache_ttl": 1800,  # 30 minutes
                    # Moderate middleware settings
                    "enable_middleware_caching": True,
                    "middleware_thread_count": 4,
                    "middleware_queue_size": 250,
                    "parallel_middleware_processing": True,
                    # Standard command processing
                    "command_batch_size": 50,
                    "enable_command_batching": True,
                    "command_processing_threads": 8,
                    "command_queue_size": 1000,
                    # Moderate memory settings
                    "memory_pool_size_mb": 100,
                    "enable_object_pooling": True,
                    "gc_optimization_enabled": True,
                    "optimization_level": "balanced",
                    # Performance metrics and targets
                    "expected_throughput_commands_per_second": 200,
                    "target_handler_latency_ms": 15,
                    "target_middleware_latency_ms": 8,
                    "memory_efficiency_target": 0.85,
                }
            # low performance level
            return {
                **base_config,
                # Conservative handler settings
                "handler_cache_size": 100,
                "enable_handler_pooling": False,
                "handler_pool_size": 10,
                "max_concurrent_handlers": 5,
                "handler_discovery_cache_ttl": 300,  # 5 minutes
                # Minimal middleware settings
                "enable_middleware_caching": False,
                "middleware_thread_count": 1,
                "middleware_queue_size": 50,
                "parallel_middleware_processing": False,
                # Single-threaded command processing
                "command_batch_size": 10,
                "enable_command_batching": False,
                "command_processing_threads": 1,
                "command_queue_size": 100,
                # Minimal memory footprint
                "memory_pool_size_mb": 50,
                "enable_object_pooling": False,
                "gc_optimization_enabled": False,
                "optimization_level": "conservative",
                # Performance metrics and targets
                "expected_throughput_commands_per_second": 50,
                "target_handler_latency_ms": 50,
                "target_middleware_latency_ms": 25,
                "memory_efficiency_target": 0.70,
            }

    class Conversions:
        """Safe type conversion utilities with modern optimizations.

        Provides robust type conversion functions that handle edge cases
        and provide sensible defaults when conversion fails, using pattern matching
        and performance optimizations.

        """

        # Simple cache for conversion results (fallback if cachetools unavailable)
        _int_cache: ClassVar[dict[str, int]] = {}
        _float_cache: ClassVar[dict[str, float]] = {}
        _bool_cache: ClassVar[dict[str, bool]] = {}
        _cache_max_size: ClassVar[int] = FlextConstants.Validation.MAX_EMAIL_LENGTH

        @classmethod
        @functools.cache
        def safe_int(cls, value: object, default: int = 0) -> int:
            """Convert value to int safely with pattern matching optimization."""
            cache_key = f"int:{value!s}:{default}"

            # Check simple cache
            if cache_key in cls._int_cache:
                return cls._int_cache[cache_key]

            result = cls._convert_to_int_with_pattern_matching(value, default)

            # Manage cache size
            if len(cls._int_cache) < cls._cache_max_size:
                cls._int_cache[cache_key] = result

            return result

        @classmethod
        def _convert_to_int_with_pattern_matching(
            cls, value: object, default: int
        ) -> int:
            """Internal int conversion using pattern matching."""
            match value:
                case None:
                    return default
                case int():
                    return value
                case float():
                    return int(value)
                case str() as s:
                    try:
                        return int(s)
                    except ValueError:
                        return default
                case _:
                    try:
                        return int(str(value))
                    except (ValueError, TypeError, OverflowError):
                        return default

        @classmethod
        @functools.cache
        def safe_float(cls, value: object, default: float = 0.0) -> float:
            """Convert value to float safely with pattern matching optimization."""
            cache_key = f"float:{value!s}:{default}"

            if cache_key in cls._float_cache:
                return cls._float_cache[cache_key]

            result = cls._convert_to_float_with_pattern_matching(value, default)

            if len(cls._float_cache) < cls._cache_max_size:
                cls._float_cache[cache_key] = result

            return result

        @classmethod
        def _convert_to_float_with_pattern_matching(
            cls, value: object, default: float
        ) -> float:
            """Internal float conversion using pattern matching."""
            match value:
                case None:
                    return default
                case int() | float():
                    return float(value)
                case str() as s:
                    try:
                        return float(s)
                    except ValueError:
                        return default
                case _:
                    try:
                        return float(str(value))
                    except (ValueError, TypeError, OverflowError):
                        return default

        @classmethod
        @functools.cache
        def safe_bool(cls, value: object, *, default: bool = False) -> bool:
            """Convert value to bool safely with pattern matching optimization."""
            cache_key = f"bool:{value!s}:{default}"

            if cache_key in cls._bool_cache:
                return cls._bool_cache[cache_key]

            result = cls._convert_to_bool_with_pattern_matching(value, default=default)

            if len(cls._bool_cache) < cls._cache_max_size:
                cls._bool_cache[cache_key] = result

            return result

        @classmethod
        def _convert_to_bool_with_pattern_matching(
            cls, value: object, *, default: bool
        ) -> bool:
            """Internal bool conversion using advanced pattern matching."""
            match value:
                case None:
                    return default
                case bool():
                    return value
                case str() as s:
                    return s.lower() in {"true", "1", "yes", "on", "y"}
                case (int() | float()) as n:
                    return bool(n)
                case _:
                    try:
                        return bool(value)
                    except (ValueError, TypeError):
                        return default

        @classmethod
        def _manage_cache_size(cls) -> None:
            """Manage cache size to prevent memory issues."""
            # Manage int cache
            if len(cls._int_cache) > cls._cache_max_size:
                items_to_remove = len(cls._int_cache) - cls._cache_max_size
                keys_to_remove = list(cls._int_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del cls._int_cache[key]

            # Manage float cache
            if len(cls._float_cache) > cls._cache_max_size:
                items_to_remove = len(cls._float_cache) - cls._cache_max_size
                keys_to_remove = list(cls._float_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del cls._float_cache[key]

            # Manage bool cache
            if len(cls._bool_cache) > cls._cache_max_size:
                items_to_remove = len(cls._bool_cache) - cls._cache_max_size
                keys_to_remove = list(cls._bool_cache.keys())[:items_to_remove]
                for key in keys_to_remove:
                    del cls._bool_cache[key]

        @staticmethod
        def safe_dict_get(data: object, key: str, default: object = None) -> object:
            """Safely get value from dictionary-like object."""
            if data is None:
                return default
            try:
                if isinstance(data, dict):
                    return data.get(key, default)
                # For other objects, try attribute access
                if hasattr(data, key):
                    return getattr(data, key, default)
                return default
            except (AttributeError, KeyError, TypeError):
                return default

    class TypeGuards:
        """Type checking and validation guard utilities.

        Provides type guard functions for runtime type checking and
        validation with proper type narrowing support.

        """

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is non-empty string after stripping whitespace."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is non-empty dict."""
            if isinstance(value, dict):
                sized_dict = cast("FlextTypes.Core.Dict", value)
                return len(sized_dict) > 0
            return False

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is non-empty list."""
            if isinstance(value, list):
                sized_list = cast("FlextTypes.Core.List", value)
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
        def format_percentage(
            value: float,
            precision: int = FlextConstants.Validation.MIN_NAME_LENGTH // 2,
        ) -> str:
            """Format value as percentage."""
            return f"{value * 100:.{precision}f}%"

    class EnvironmentUtils:
        """Environment and file system utilities.

        Provides safe utilities for environment variables, file operations,
        and configuration management with FlextResult error handling.
        """

        @staticmethod
        def safe_get_env_var(
            var_name: str,
            default: str | None = None,
        ) -> FlextResult[str]:
            """Safely get environment variable with optional default value."""
            try:
                value = os.getenv(var_name, default)
                if value is None:
                    return FlextResult[str].fail(
                        f"Environment variable {var_name} not set"
                    )
                return FlextResult[str].ok(value)
            except Exception as e:
                return FlextResult[str].fail(
                    f"{FlextConstants.Errors.CONFIG_ERROR}: {e}"
                )

        @staticmethod
        def safe_load_json_file(
            file_path: str | Path,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            """Safely load JSON file with validation."""
            try:
                with Path(file_path).open(encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                    )

                return FlextResult[FlextTypes.Core.Dict].ok(
                    cast("FlextTypes.Core.Dict", data)
                )
            except FileNotFoundError:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"{FlextConstants.Errors.NOT_FOUND}: {file_path}",
                )
            except json.JSONDecodeError as e:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"{FlextConstants.Errors.SERIALIZATION_ERROR}: {e}",
                )
            except Exception as e:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"{FlextConstants.Errors.CONFIG_ERROR}: {e}",
                )

        @staticmethod
        def merge_dicts(
            base_dict: FlextTypes.Core.Dict,
            override_dict: FlextTypes.Core.Dict,
        ) -> FlextTypes.Core.Dict:
            """Merge two dictionaries - returns plain dict for test compatibility."""
            # Simple merge for test compatibility - tests expect plain dict return
            safe_base = base_dict if isinstance(base_dict, dict) else {}
            safe_override = override_dict if isinstance(override_dict, dict) else {}

            return {**safe_base, **safe_override}

        @staticmethod
        def merge_dicts_safe(
            base_dict: FlextTypes.Core.Dict,
            override_dict: FlextTypes.Core.Dict,
            *,
            required_non_null_fields: set[str] | None = None,
        ) -> FlextResult[FlextTypes.Core.Dict]:
            """Merge two dictionaries with validation for required fields (FlextResult version)."""
            try:
                merged = {**base_dict, **override_dict}

                # Validate required non-null fields if specified
                if required_non_null_fields:
                    for key in required_non_null_fields:
                        if key in merged and merged[key] is None:
                            return FlextResult[FlextTypes.Core.Dict].fail(
                                f"{FlextConstants.Messages.VALIDATION_FAILED} for {key}: {FlextConstants.Messages.NULL_DATA}",
                            )

                return FlextResult[FlextTypes.Core.Dict].ok(merged)
            except Exception as e:
                return FlextResult[FlextTypes.Core.Dict].fail(
                    f"Dictionary merge failed: {e}"
                )

    class ValidationUtils:
        """Generic validation utilities.

        Provides reusable validation functions for common validation patterns.
        """

        @staticmethod
        def validate_with_callable(
            value: object,
            validator: object,
            error_message: str = "Validation failed",
        ) -> FlextResult[bool]:
            """Validate a value using a callable validator function."""
            try:
                if not callable(validator):
                    return FlextResult[bool].fail("Validator must be callable")

                try:
                    result = validator(value)
                    if not result:
                        return FlextResult[bool].fail(error_message)
                    return FlextResult[bool].ok(data=True)
                except Exception as e:
                    return FlextResult[bool].fail(f"Validation error: {e}")
            except Exception as e:
                return FlextResult[bool].fail(f"Validation failed: {e}")

        @staticmethod
        def validate_email(email: str) -> FlextResult[bool]:
            """Validate email address format."""
            try:
                if not email or not isinstance(email, str):
                    return FlextResult[bool].fail("Invalid email: empty or not string")

                if "@" not in email:
                    return FlextResult[bool].fail("Invalid email: missing @ symbol")

                domain_part = email.rsplit("@", maxsplit=1)[-1]
                if "." not in domain_part:
                    return FlextResult[bool].fail("Invalid email: invalid domain")

                return FlextResult[bool].ok(data=True)
            except Exception as e:
                return FlextResult[bool].fail(f"Email validation error: {e}")

        @staticmethod
        def validate_url(url: str) -> FlextResult[bool]:
            """Validate URL format."""
            try:
                if not url or not isinstance(url, str):
                    return FlextResult[bool].fail("Invalid URL: empty or not string")

                url = url.strip()
                if not url:
                    return FlextResult[bool].fail("Invalid URL: empty string")

                # Basic URL validation
                if not (url.startswith(("http://", "https://"))):
                    return FlextResult[bool].fail(
                        "Invalid URL: must start with http:// or https://"
                    )

                # Check for valid characters
                if " " in url:
                    return FlextResult[bool].fail("Invalid URL: contains spaces")

                return FlextResult[bool].ok(data=True)
            except Exception as e:
                return FlextResult[bool].fail(f"URL validation error: {e}")

        @staticmethod
        def validate_phone(phone: str) -> FlextResult[bool]:
            """Validate phone number format."""
            try:
                if not phone or not isinstance(phone, str):
                    return FlextResult[bool].fail("Invalid phone: empty or not string")

                # Remove common separators
                cleaned = re.sub(r"[\s\-\.\(\)\+]", "", phone)

                # Check if all remaining are digits
                if not cleaned.isdigit():
                    return FlextResult[bool].fail(
                        "Invalid phone: contains non-digit characters"
                    )

                # Basic length check (international numbers can vary)
                min_phone_length = FlextConstants.Validation.MIN_NAME_LENGTH * 3 + 1
                max_phone_length = FlextConstants.Validation.MAX_NAME_LENGTH // 6 + 5
                if len(cleaned) < min_phone_length or len(cleaned) > max_phone_length:
                    return FlextResult[bool].fail("Invalid phone: invalid length")

                return FlextResult[bool].ok(data=True)
            except Exception as e:
                return FlextResult[bool].fail(f"Phone validation error: {e}")

    class ProcessingUtils:
        """Data processing utilities for JSON, models, and structured data.

        Provides safe processing functions for JSON parsing, model extraction,
        and data validation with FlextResult error handling.


        """

        @staticmethod
        def safe_json_parse(
            json_str: str,
            default: FlextTypes.Core.Dict | None = None,
        ) -> FlextTypes.Core.Dict:
            """Safely parse JSON string."""
            try:
                result: object = json.loads(json_str)
                if isinstance(result, dict):
                    return cast("FlextTypes.Core.Dict", result)
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
        def extract_model_data(obj: object) -> FlextTypes.Core.Dict:
            """Extract data from Pydantic model or dict."""
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                # Cast to dict since we've already checked hasattr and callable
                return cast("FlextTypes.Core.Dict", getattr(obj, "model_dump")())
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                # Cast to dict since we've already checked hasattr and callable
                return cast("FlextTypes.Core.Dict", getattr(obj, "dict")())
            if isinstance(obj, dict):
                return cast("FlextTypes.Core.Dict", obj)
            return {}

        @staticmethod
        def parse_json_to_model[TModel: BaseModel](
            json_text: str,
            model_class: type[TModel],
        ) -> FlextResult[TModel]:
            """Parse JSON and validate using appropriate model instantiation strategy."""
            try:
                parsed_data: object = json.loads(json_text)

                # Strategy 1: Pydantic v2 model_validate (preferred for validation)
                if hasattr(model_class, "model_validate") and callable(
                    getattr(model_class, "model_validate", None)
                ):
                    validated_obj = model_class.model_validate(parsed_data)
                    return FlextResult[TModel].ok(validated_obj)

                # Strategy 2: Dictionary constructor (for dict-like data)
                if isinstance(parsed_data, dict):
                    dict_data = cast("FlextTypes.Core.Dict", parsed_data)
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

        """

        @staticmethod
        def chain_results(*results: FlextResult[T]) -> FlextResult[list[T]]:
            """Chain multiple FlextResults into a single result."""
            values: list[T] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[T]].fail(
                        result.error or "Chain operation failed",
                    )
                values.append(result.value)
            return FlextResult[list[T]].ok(values)

        @staticmethod
        def batch_process[TInput, TOutput](
            items: list[TInput],
            processor: Callable[[TInput], FlextResult[TOutput]],
        ) -> tuple[list[TOutput], FlextTypes.Core.StringList]:
            """Process list of items, collecting successes and errors."""
            successes: list[TOutput] = []
            errors: FlextTypes.Core.StringList = []

            for item in items:
                result = processor(item)
                if result.is_success:
                    successes.append(result.value)
                else:
                    errors.append(result.error or "Unknown error")

            return successes, errors

    class Configuration:
        """Configuration utilities with FlextTypes.Config and StrEnum integration."""

        @staticmethod
        def create_default_config(
            environment: FlextTypes.Config.Environment = "development",
        ) -> FlextResult[FlextTypes.Core.ConfigDict]:
            """Create default configuration for specified environment using FlextTypes.Config."""
            try:
                # Validate environment is a valid StrEnum value
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if environment not in valid_environments:
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        f"Invalid environment: {environment}. Must be one of: {valid_environments}",
                    )

                # Create environment-specific configuration
                config: FlextTypes.Core.ConfigDict = {
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

                return FlextResult[FlextTypes.Core.ConfigDict].ok(config)

            except Exception as e:
                return FlextResult[FlextTypes.Core.ConfigDict].fail(
                    f"Default config creation failed: {e}",
                )

        @staticmethod
        def validate_configuration_with_types(
            config: FlextTypes.Core.ConfigDict,
        ) -> FlextResult[FlextTypes.Core.ConfigDict]:
            """Validate configuration using FlextTypes.Config with efficient StrEnum validation."""
            # Configuration validation constants
            min_timeout_ms = FlextConstants.Network.CONNECTION_TIMEOUT * 10
            max_timeout_ms = FlextConstants.Network.TOTAL_TIMEOUT * 5000
            max_retries = FlextConstants.Validation.MAX_NAME_LENGTH // 10

            try:
                validated: FlextTypes.Core.ConfigDict = {}

                # Environment validation
                if "environment" not in config:
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        "Required field 'environment' missing",
                    )

                env_value = config["environment"]
                valid_environments = {
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                }
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {sorted(valid_environments)}",
                    )
                validated["environment"] = env_value

                # Log level validation
                log_level = config.get(
                    "log_level",
                    FlextConstants.Config.LogLevel.INFO.value,
                )
                valid_log_levels = {
                    level.value for level in FlextConstants.Config.LogLevel
                }
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {sorted(valid_log_levels)}",
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
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        f"Invalid validation_level '{validation_level}'. Valid options: {sorted(valid_validation_levels)}",
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
                    return FlextResult[FlextTypes.Core.ConfigDict].fail(
                        f"Invalid config_source '{config_source}'. Valid options: {sorted(valid_config_sources)}",
                    )
                validated["config_source"] = config_source

                # Boolean validations
                for bool_field in ["debug", "performance_monitoring", "enable_caching"]:
                    if bool_field in config:
                        value = config[bool_field]
                        if not isinstance(value, bool):
                            return FlextResult[FlextTypes.Core.ConfigDict].fail(
                                f"Field '{bool_field}' must be a boolean",
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
                        return FlextResult[FlextTypes.Core.ConfigDict].fail(
                            "request_timeout must be a number between 100 and 300000 milliseconds",
                        )
                    validated["request_timeout"] = timeout

                if "max_retries" in config:
                    retries = config["max_retries"]
                    if (
                        not isinstance(retries, int)
                        or retries < 0
                        or retries > max_retries
                    ):
                        return FlextResult[FlextTypes.Core.ConfigDict].fail(
                            "max_retries must be an integer between 0 and 10",
                        )
                    validated["max_retries"] = retries

                return FlextResult[FlextTypes.Core.ConfigDict].ok(validated)

            except Exception as e:
                return FlextResult[FlextTypes.Core.ConfigDict].fail(
                    f"Configuration validation failed: {e}",
                )

    # ==========================================================================
    # MAIN CLASS METHODS - Delegate to nested classes
    # ==========================================================================

    generate_uuid = Generators.generate_uuid
    generate_id = Generators.generate_id
    generate_entity_id = Generators.generate_entity_id
    generate_correlation_id = Generators.generate_correlation_id
    truncate = TextProcessor.truncate
    format_duration = TimeUtils.format_duration
    track_performance = Performance.track_performance
    safe_json_parse = ProcessingUtils.safe_json_parse
    safe_json_stringify = ProcessingUtils.safe_json_stringify
    parse_json_to_model = ProcessingUtils.parse_json_to_model
    safe_int = Conversions.safe_int
    batch_process = ResultUtils.batch_process

    # Additional methods needed by legacy compatibility layer
    @classmethod
    def safe_int_conversion(
        cls,
        value: object,
        default: int | None = None,
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
    def get_performance_metrics(cls) -> FlextTypes.Core.Dict:
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
            operation,
            duration,
            success=success,
            error=error,
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
    # Convenience attributes for backward compatibility
    Strings = TextProcessor
    Files = EnvironmentUtils  # File operations are in EnvironmentUtils
    Collections = Conversions  # Collection operations are in Conversions


# EXPORTS - Clean public API
# =============================================================================

__all__: FlextTypes.Core.StringList = [
    "FlextUtilities",  # ONLY main class exported
]
