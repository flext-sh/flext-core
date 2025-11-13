"""Utilities module - FlextUtilitiesGenerators.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import secrets
import string
import time
import uuid
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
_logger = logging.getLogger(__name__)


class FlextUtilitiesGenerators:
    """ID and data generation utilities."""

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: object,
        length: int = FlextConstants.Utilities.SHORT_UUID_LENGTH,
    ) -> str:
        """Factory method for generating prefixed IDs with UUID.

        **INTERNAL METHOD**: This is a private implementation detail used
        by public ID generation methods. Do not call directly - use the
        specific public methods instead (generate_correlation_id,
        generate_batch_id, etc.).

        This method consolidates 12+ similar ID generation methods following
        DRY principle (Don't Repeat Yourself).

        Args:
            prefix: ID prefix (e.g., 'corr', 'batch', 'txn')
            *parts: Optional middle parts (e.g., batch_size, context)
            length: UUID truncation length (SHORT or LONG)

        Returns:
            Formatted ID string: {prefix}_{parts}_{uuid[:length]}

        Example:
            >>> _generate_prefixed_id("corr")
            'corr_a1b2c3d4'
            >>> _generate_prefixed_id("batch", 100)
            'batch_100_a1b2c3d4'

        """
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

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
        """Generate ISO format timestamp without microseconds.

        .. deprecated:: 0.9.9
            Use :func:`generate_iso_timestamp` instead. This method is identical
            and will be removed in version 2.0.0.

        """
        warnings.warn(
            "generate_timestamp() is deprecated, use generate_iso_timestamp() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlextUtilitiesGenerators.generate_iso_timestamp()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp without microseconds."""
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a correlation ID for tracking."""
        return FlextUtilitiesGenerators._generate_prefixed_id("corr")

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
    def generate_correlation_id_with_context(context: str) -> str:
        """Generate a correlation ID with context prefix."""
        return FlextUtilitiesGenerators._generate_prefixed_id(context)

    @staticmethod
    def generate_batch_id(batch_size: int) -> str:
        """Generate a batch ID with size information."""
        return FlextUtilitiesGenerators._generate_prefixed_id("batch", batch_size)

    @staticmethod
    def generate_transaction_id() -> str:
        """Generate a transaction ID for distributed transactions."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "txn", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_saga_id() -> str:
        """Generate a saga ID for distributed transaction patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "saga", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_event_id() -> str:
        """Generate an event ID for domain events."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "evt", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_command_id() -> str:
        """Generate a command ID for CQRS patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "cmd", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_query_id() -> str:
        """Generate a query ID for CQRS patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "qry", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_aggregate_id(aggregate_type: str) -> str:
        """Generate an aggregate ID with type prefix."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            aggregate_type, length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_entity_version() -> int:
        """Generate an entity version number using FlextConstants.Context."""
        return (
            int(
                datetime.now(UTC).timestamp()
                * FlextConstants.Context.MILLISECONDS_PER_SECOND
            )
            % FlextConstants.Utilities.VERSION_MODULO
        ) + 1

    @staticmethod
    def ensure_id(obj: FlextTypes.CachedObjectType) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                new_id = FlextUtilitiesGenerators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    @staticmethod
    def ensure_trace_context(
        context: dict[str, object] | object,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> dict[str, object]:
        """Ensure context dict has distributed tracing fields (trace_id, span_id, etc).

        This generic helper consolidates duplicate context enrichment logic
        across multiple Pydantic models (service.py, config.py).

        If context is not dict-like, creates new empty dict. Generates UUIDs
        for missing trace_id and span_id fields. Optionally adds correlation_id
        and ISO timestamp.

        Args:
            context: Context dictionary or object to enrich (can be any type)
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists (ISO 8601)

        Returns:
            dict[str, object]: Enriched context with requested fields

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> # Basic: trace_id + span_id
            >>> ctx = FlextUtilities.Generators.ensure_trace_context({})
            >>> "trace_id" in ctx and "span_id" in ctx
            True
            >>> # With correlation_id
            >>> ctx = FlextUtilities.Generators.ensure_trace_context(
            ...     {}, include_correlation_id=True
            ... )
            >>> "correlation_id" in ctx
            True
            >>> # With timestamp
            >>> ctx = FlextUtilities.Generators.ensure_trace_context(
            ...     {}, include_timestamp=True
            ... )
            >>> "timestamp" in ctx
            True
            >>> # Existing values preserved
            >>> ctx = FlextUtilities.Generators.ensure_trace_context({
            ...     "trace_id": "abc"
            ... })
            >>> ctx["trace_id"]
            'abc'

        """
        # Convert to dict if not dict-like
        if not isinstance(context, dict):
            # Try converting dict-like objects (has .items())
            if hasattr(context, "items"):
                # Cast is safe here as we checked for dict-like protocol
                try:
                    # Use getattr to safely access items() method
                    items_method = getattr(context, "items", None)
                    if callable(items_method):
                        # Cast to ensure type checker understands this returns dict items
                        items_result = cast(
                            "Iterable[tuple[str, object]]", items_method()
                        )
                        context_dict = dict(items_result)
                    else:
                        context_dict = {}
                except (AttributeError, TypeError):
                    context_dict = {}
            else:
                context_dict = {}
        else:
            context_dict = context

        # Ensure all keys are strings (already validated above)

        # Ensure trace_id exists (always included)
        if "trace_id" not in context_dict:
            context_dict["trace_id"] = FlextUtilitiesGenerators.generate_uuid()

        # Ensure span_id exists (always included)
        if "span_id" not in context_dict:
            context_dict["span_id"] = FlextUtilitiesGenerators.generate_uuid()

        # Optionally ensure correlation_id
        if include_correlation_id and "correlation_id" not in context_dict:
            context_dict["correlation_id"] = FlextUtilitiesGenerators.generate_uuid()

        # Optionally ensure timestamp (ISO 8601 format)
        if include_timestamp and "timestamp" not in context_dict:
            context_dict["timestamp"] = (
                FlextUtilitiesGenerators.generate_iso_timestamp()
            )

        return context_dict

    @staticmethod
    def ensure_dict(
        value: object, *, allow_none_as_empty: bool = True
    ) -> dict[str, object]:
        """Ensure value is a dict, converting or defaulting to empty dict.

        This generic helper consolidates duplicate dict normalization logic
        across multiple Pydantic validators (context.py) and dispatcher metadata
        handling. Supports Pydantic models, dict-like objects, and None values.

        Conversion Strategy:
            1. Already dict → return as-is
            2. Pydantic BaseModel → call model_dump()
            3. Dict-like (has .items()) → convert via dict()
            4. None → empty dict (if allow_none_as_empty=True)
            5. Fallback → cast to dict[str, object]

        Args:
            value: Value to normalize (dict, BaseModel, dict-like, None, or other)
            allow_none_as_empty: If True, None becomes {} (default: True)

        Returns:
            dict[str, object]: Normalized dict

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> FlextUtilities.Generators.ensure_dict({"a": 1})
            {'a': 1}
            >>> FlextUtilities.Generators.ensure_dict(None)
            {}
            >>> # Pydantic model
            >>> from pydantic import BaseModel
            >>> class MyModel(BaseModel):
            ...     field: str = "value"
            >>> model = MyModel()
            >>> FlextUtilities.Generators.ensure_dict(model)
            {'field': 'value'}

        """
        # Strategy 1: Already a dict - return as-is
        if isinstance(value, dict):
            return value

        # Strategy 2: Pydantic BaseModel - use model_dump()
        if isinstance(value, BaseModel):
            try:
                return value.model_dump()
            except (AttributeError, TypeError):
                # Fallback for older Pydantic versions or custom implementations
                pass

        # Strategy 3: Dict-like object (has .items()) - convert
        if hasattr(value, "items"):
            try:
                # Use getattr to safely access items() method
                items_method = getattr(value, "items", None)
                if callable(items_method):
                    # Cast to ensure type checker understands this returns dict items
                    items_result = cast("Iterable[tuple[str, object]]", items_method())
                    return dict(items_result)
            except (AttributeError, TypeError):
                pass

        # Strategy 4: None - return empty dict if allowed
        if value is None and allow_none_as_empty:
            return {}

        # Strategy 5: Fallback - cast to dict (may fail at runtime)
        return cast("dict[str, object]", value) if value is not None else {}

    @staticmethod
    def generate_operation_id(message_type: str, message: object) -> str:
        """Generate unique operation ID for dispatch operations.

        Args:
            message_type: Type of message being dispatched
            message: Message object

        Returns:
            str: Unique operation ID

        """
        timestamp = int(time.time() * 1000000)
        message_id = id(message)
        return f"{message_type}_{message_id}_{timestamp}"


def create_dynamic_type_subclass(
    name: str,
    base_class: object,  # Can be a class or Self
    attributes: dict[str, object],
) -> type:
    """Create a dynamic subclass using type() for metaprogramming.

    This helper function encapsulates the creation of dynamic classes
    to isolate type checker issues with metaprogramming.

    Args:
        name: Name of the subclass
        base_class: Base class to inherit from
        attributes: Dictionary of attributes to add to the subclass

    Returns:
        The dynamically created subclass

    """
    # pyrefly doesn't understand type() for dynamic class creation
    # This is valid Python metaprogramming
    return type(name, (cast("type", base_class),), attributes)


__all__ = ["FlextUtilitiesGenerators"]
