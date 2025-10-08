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
import hashlib
import inspect
import json
import logging
import operator
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

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextUtilities:
    """Comprehensive utility functions for FLEXT ecosystem operations."""

    class Cache:
        """Cache utility functions for data normalization and sorting."""

        @staticmethod
        def normalize_component(
            component: object,
        ) -> object:
            """Normalize a component for consistent representation."""
            if isinstance(component, dict):
                component_dict = cast("FlextTypes.Dict", component)
                return {
                    str(k): FlextUtilities.Cache.normalize_component(v)
                    for k, v in component_dict.items()
                }
            if isinstance(component, (list, tuple)):
                sequence = cast("Sequence[object]", component)
                return [
                    FlextUtilities.Cache.normalize_component(item) for item in sequence
                ]
            if isinstance(component, set):
                set_component = cast("set[object]", component)
                return {
                    FlextUtilities.Cache.normalize_component(item)
                    for item in set_component
                }
            return component

        @staticmethod
        def sort_key(key: object) -> tuple[int, str]:
            """Generate a sort key for consistent ordering."""
            if isinstance(key, str):
                return (0, key.lower())
            if isinstance(key, (int, float)):
                return (1, str(key))
            return (2, str(key))

        @staticmethod
        def sort_dict_keys(data: object) -> object:
            """Sort dictionary keys for consistent representation."""
            if isinstance(data, dict):
                data_dict = cast("FlextTypes.Dict", data)
                return cast(
                    "object",
                    {
                        k: FlextUtilities.Cache.sort_dict_keys(data_dict[k])
                        for k in sorted(
                            data_dict.keys(), key=FlextUtilities.Cache.sort_key
                        )
                    },
                )
            return data

        @staticmethod
        def clear_object_cache(obj: object) -> FlextResult[None]:
            """Clear any caches on an object."""
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
                return FlextResult[None].fail(f"Failed to clear caches: {e}")

        @staticmethod
        def has_cache_attributes(obj: object) -> bool:
            """Check if object has any cache-related attributes."""
            cache_attributes = [
                "_cache",
                "__cache__",
                "cache",
                "_cached_data",
                "_memoized",
            ]
            return any(hasattr(obj, attr) for attr in cache_attributes)

        @staticmethod
        def generate_cache_key(*args: object, **kwargs: object) -> str:
            """Generate a cache key from arguments."""
            key_data = str(args) + str(sorted(kwargs.items()))
            return hashlib.sha256(key_data.encode()).hexdigest()

    @staticmethod
    def generate_id() -> str:
        """Generate a unique identifier."""
        return str(uuid.uuid4())

    class Validation:
        """Unified validation patterns using railway composition."""

        @staticmethod
        def validate_string_not_none(value: object) -> FlextResult[None]:
            """Validate that a string value is not None."""
            if value is None:
                return FlextResult[None].fail("Value cannot be None")
            if not isinstance(value, str):
                return FlextResult[None].fail("Value must be a string")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_string_not_empty(value: str) -> FlextResult[None]:
            """Validate that a string value is not empty."""
            if not value:
                return FlextResult[None].fail("String cannot be empty")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_string_length(
            value: str, min_length: int = 0, max_length: int | None = None
        ) -> FlextResult[None]:
            """Validate string length constraints."""
            if len(value) < min_length:
                return FlextResult[None].fail(f"String too short (min: {min_length})")
            if max_length is not None and len(value) > max_length:
                return FlextResult[None].fail(f"String too long (max: {max_length})")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_string_pattern(
            value: str, pattern: str | None
        ) -> FlextResult[None]:
            """Validate string against a regex pattern."""
            if pattern is None:
                return FlextResult[None].ok(None)

            try:
                if not re.match(pattern, value):
                    return FlextResult[None].fail(
                        f"String does not match pattern: {pattern}"
                    )
                return FlextResult[None].ok(None)
            except re.error as e:
                return FlextResult[None].fail(f"Invalid regex pattern: {e}")

        @staticmethod
        def validate_string(value: str) -> FlextResult[None]:
            """Validate that value is a non-empty string."""
            if not value.strip():
                return FlextResult[None].fail(
                    "String cannot be empty or whitespace-only"
                )
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_email(value: str) -> FlextResult[None]:
            """Validate email format."""
            if "@" not in value or "." not in value:
                return FlextResult[None].fail("Invalid email format")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_positive_integer(value: object) -> FlextResult[None]:
            """Validate that value is a positive integer."""
            if not isinstance(value, int) or value <= 0:
                return FlextResult[None].fail("Value must be a positive integer")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_non_negative_integer(value: object) -> FlextResult[None]:
            """Validate that value is a non-negative integer."""
            if not isinstance(value, int) or value < 0:
                return FlextResult[None].fail("Value must be a non-negative integer")
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_directory_path(value: str) -> FlextResult[None]:
            """Validate directory path exists."""
            if not pathlib.Path(value).exists():
                return FlextResult[None].fail(f"Directory does not exist: {value}")
            if not pathlib.Path(value).is_dir():
                return FlextResult[None].fail(f"Path is not a directory: {value}")
            return FlextResult[None].ok(None)

        @staticmethod
        def is_non_empty_string(value: object) -> bool:
            """Check if value is a non-empty string."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def clear_all_caches(obj: object) -> FlextResult[None]:
            """Clear all caches on an object to prevent memory leaks.

            Args:
                obj: Object to clear caches on

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
                return FlextResult[None].fail(f"Failed to clear caches: {e}")

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
        def sort_key(value: FlextTypes.SerializableType) -> str:
            """Return a deterministic string for ordering normalized cache components."""
            try:
                json_bytes = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
                return json_bytes.decode(FlextConstants.Utilities.DEFAULT_ENCODING)
            except Exception as e:
                # Use proper logger instead of root logger
                logger = logging.getLogger(__name__)
                logger.debug("orjson dumps failed: %s", e)
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

                # Sort by string representation for deterministic ordering
                set_items.sort(key=str)

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
                    return f"{command_type.__name__}_{hash(str(dict_sorted_data))}"

                # For other objects, use string representation
                command_str = str(command) if command is not None else "None"
                command_hash = hash(command_str)
                return f"{command_type.__name__}_{command_hash}"

            except Exception:
                # Fallback to string representation if anything fails
                command_str_fallback = str(command) if command is not None else "None"  # type: ignore[arg-type]
                command_str_fallback = command_str_fallback.encode(
                    "utf-8", errors="ignore"
                ).decode("utf-8", errors="ignore")
                try:
                    command_hash_fallback = hash(command_str_fallback)
                    return f"{command_type.__name__}_{command_hash_fallback}"
                except TypeError:
                    # If hash fails, use a deterministic fallback
                    return f"{command_type.__name__}_{abs(hash(command_str_fallback.encode(FlextConstants.Utilities.DEFAULT_ENCODING)))}"

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
                env: Environment variables dictionary for the command
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


__all__ = [
    "FlextUtilities",
]
