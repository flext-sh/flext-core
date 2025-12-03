"""ID and data generation helpers shared across dispatcher flows.

These primitives centralize correlation, batch, and timestamp generation so
dispatcher handlers and services produce consistent identifiers and audit
metadata without duplicating randomness or formatting concerns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import secrets
import string
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextGenerators:
    """Generate deterministic IDs and timestamps for CQRS workflows.

    Nested classes organize related generators:
    - Random: Random/short ID generation
    - Type: Dynamic type generation
    """

    class Random:
        """Random ID generation helpers."""

        @staticmethod
        def generate_short_id(length: int = 8) -> str:
            """Generate a short random ID."""
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(length))

    # NOTE: create_dynamic_type_subclass is available as static method - no nested Type class needed

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: t.GeneralValueType,
        length: int = FlextConstants.Utilities.SHORT_UUID_LENGTH,
    ) -> str:
        """Factory method for generating prefixed IDs with UUID.

        This private helper keeps the public generators consistent while
        limiting duplication across the correlation, batch, and transaction
        ID factories.

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
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp without microseconds.

        Note: For precise duration calculations, use generate_datetime_utc() instead
        as this method removes microseconds which affects timing precision.
        """
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def generate_datetime_utc() -> datetime:
        """Generate current UTC datetime with full precision (preserves microseconds).

        Use this method when you need precise datetime objects for duration calculations.
        Unlike generate_iso_timestamp(), this preserves microseconds for accurate timing.

        Returns:
            datetime: Current UTC datetime with full microsecond precision

        Example:
            >>> start = uGenerators.generate_datetime_utc()
            >>> # ... operation ...
            >>> end = uGenerators.generate_datetime_utc()
            >>> duration = (end - start).total_seconds()  # Precise duration

        """
        return datetime.now(UTC)

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a correlation ID for tracking."""
        return FlextGenerators._generate_prefixed_id("corr")

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate a short random ID (delegates to Random.generate_short_id)."""
        return uGenerators.Random.generate_short_id(length)

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
        return FlextGenerators._generate_prefixed_id(context)

    @staticmethod
    def generate_batch_id(batch_size: int) -> str:
        """Generate a batch ID with size information."""
        return FlextGenerators._generate_prefixed_id("batch", batch_size)

    @staticmethod
    def generate_transaction_id() -> str:
        """Generate a transaction ID for distributed transactions."""
        return FlextGenerators._generate_prefixed_id(
            "txn",
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_saga_id() -> str:
        """Generate a saga ID for distributed transaction patterns."""
        return FlextGenerators._generate_prefixed_id(
            "saga",
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_event_id() -> str:
        """Generate an event ID for domain events."""
        return FlextGenerators._generate_prefixed_id(
            "evt",
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_command_id() -> str:
        """Generate a command ID for CQRS patterns."""
        return FlextGenerators._generate_prefixed_id(
            "cmd",
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_query_id() -> str:
        """Generate a query ID for CQRS patterns."""
        return FlextGenerators._generate_prefixed_id(
            "qry",
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_aggregate_id(aggregate_type: str) -> str:
        """Generate an aggregate ID with type prefix."""
        return FlextGenerators._generate_prefixed_id(
            aggregate_type,
            length=FlextConstants.Utilities.LONG_UUID_LENGTH,
        )

    @staticmethod
    def generate_entity_version() -> int:
        """Generate an entity version number using FlextConstants.Context."""
        return (
            int(
                uGenerators.generate_datetime_utc().timestamp()
                * FlextConstants.Context.MILLISECONDS_PER_SECOND,
            )
            % FlextConstants.Utilities.VERSION_MODULO
        ) + 1

    @staticmethod
    def ensure_id(obj: p.HasModelDump) -> None:
        """Ensure object has an ID using u and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                new_id = uGenerators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    @staticmethod
    def _normalize_context_to_dict(
        context: dict[str, t.GeneralValueType] | object,
    ) -> dict[str, t.GeneralValueType]:
        """Normalize context to dict - fast fail validation.

        Args:
            context: Context to normalize

        Returns:
            dict[str, t.GeneralValueType]: Normalized context dict

        Raises:
            TypeError: If context cannot be normalized

        """
        if isinstance(context, dict):
            return context
        if isinstance(context, Mapping):
            try:
                return dict(context.items())
            except (AttributeError, TypeError) as e:
                msg = (
                    f"Failed to convert Mapping {type(context).__name__} to dict: "
                    f"{type(e).__name__}: {e}"
                )
                raise TypeError(msg) from e
        if isinstance(context, BaseModel):
            try:
                return context.model_dump()
            except (AttributeError, TypeError) as e:
                msg = (
                    f"Failed to dump BaseModel {type(context).__name__}: "
                    f"{type(e).__name__}: {e}"
                )
                raise TypeError(msg) from e
        if context is None:
            msg = (
                "Context cannot be None. Use explicit empty dict {} "
                "or handle None in calling code."
            )
            raise TypeError(msg)
        msg = (
            f"Context must be dict, Mapping, or BaseModel, got {type(context).__name__}"
        )
        raise TypeError(msg)

    @staticmethod
    def _enrich_context_fields(
        context_dict: dict[str, str],
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> None:
        """Enrich context dict with tracing fields.

        Args:
            context_dict: Context dict to enrich (modified in place, all values must be strings)
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists

        """
        if "trace_id" not in context_dict:
            context_dict["trace_id"] = uGenerators.generate_id()
        if "span_id" not in context_dict:
            context_dict["span_id"] = uGenerators.generate_id()

        # Optionally ensure correlation_id
        if include_correlation_id and "correlation_id" not in context_dict:
            context_dict["correlation_id"] = uGenerators.generate_id()

        # Optionally ensure timestamp (ISO 8601 format)
        if include_timestamp and "timestamp" not in context_dict:
            context_dict["timestamp"] = uGenerators.generate_iso_timestamp()

    @staticmethod
    def ensure_trace_context(
        context: dict[str, str] | object,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> dict[str, str]:
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
            dict[str, str]: Enriched context with requested fields (all string values)

        Example:
            >>> from flext_core.utilities import u
            >>> # Basic: trace_id + span_id
            >>> ctx = u.Generators.ensure_trace_context({})
            >>> "trace_id" in ctx and "span_id" in ctx
            True
            >>> # With correlation_id
            >>> ctx = u.Generators.ensure_trace_context({}, include_correlation_id=True)
            >>> "correlation_id" in ctx
            True
            >>> # With timestamp
            >>> ctx = u.Generators.ensure_trace_context({}, include_timestamp=True)
            >>> "timestamp" in ctx
            True
            >>> # Existing values preserved
            >>> ctx = u.Generators.ensure_trace_context({"trace_id": "abc"})
            >>> ctx["trace_id"]
            'abc'

        """
        normalized_dict = FlextGenerators._normalize_context_to_dict(context)
        # Convert all values to strings for trace context (trace_id, span_id, etc. are strings)
        context_dict: dict[str, str] = {
            k: str(v) if not isinstance(v, str) else v
            for k, v in normalized_dict.items()
        }
        FlextGenerators._enrich_context_fields(
            context_dict,
            include_correlation_id=include_correlation_id,
            include_timestamp=include_timestamp,
        )
        return context_dict

    @staticmethod
    def ensure_dict(
        value: t.GeneralValueType,
        default: dict[str, t.GeneralValueType] | None = None,
    ) -> dict[str, t.GeneralValueType]:
        """Ensure value is a dict, converting from Pydantic models or dict-like.

        This generic helper consolidates duplicate dict normalization logic
        across multiple Pydantic validators (context.py) and dispatcher metadata
        handling. Supports Pydantic models and dict-like objects.

        Conversion Strategy:
            1. None + default provided → return default
            2. Already dict → return as-is
            3. Pydantic BaseModel → call model_dump()
            4. Dict-like (has .items()) → convert via dict()
            5. None (no default) → fast fail (raises TypeError)
            6. Other → fast fail (raises TypeError)

        Args:
            value: Value to normalize (dict, BaseModel, or dict-like)
            default: Default value to return if value is None (optional)

        Returns:
            dict[str, t.GeneralValueType]: Normalized dict or default

        Raises:
            TypeError: If value is None (and no default) or cannot be converted

        Example:
            >>> from flext_core.utilities import u
            >>> u.Generators.ensure_dict({"a": 1})
            {'a': 1}
            >>> u.Generators.ensure_dict(None, default={})
            {}
            >>> # Pydantic model
            >>> from pydantic import BaseModel
            >>> class MyModel(BaseModel):
            ...     field: str = "value"
            >>> model = MyModel()
            >>> u.Generators.ensure_dict(model)
            {'field': 'value'}

        """
        # Strategy 1: Already a dict - return as-is
        if isinstance(value, dict):
            return value

        # Strategy 2: Pydantic BaseModel - use model_dump()
        if isinstance(value, BaseModel):
            try:
                result = value.model_dump()
                if isinstance(result, dict):
                    normalized = FlextRuntime.normalize_to_general_value(result)
                    if isinstance(normalized, dict):
                        return normalized
                return {}
            except (AttributeError, TypeError):
                pass

        # Strategy 3: Mapping (dict-like) - convert via dict() (fast fail)
        if isinstance(value, Mapping):
            # Fast fail: Mapping.items() must succeed
            try:
                return dict(value.items())
            except (AttributeError, TypeError) as e:
                msg = (
                    f"Failed to convert Mapping {type(value).__name__} to dict: "
                    f"{type(e).__name__}: {e}"
                )
                raise TypeError(msg) from e

        # Strategy 4: None - use default or fast fail
        if value is None:
            if default is not None:
                return default
            msg = (
                "Value cannot be None. Use explicit empty dict {} "
                "or pass default={} parameter."
            )
            raise TypeError(msg)

        # Strategy 5: Fast fail - unsupported type
        msg = (
            f"Cannot convert {type(value).__name__} to dict. "
            "Supported types: dict, BaseModel, Mapping."
        )
        raise TypeError(msg)

    @staticmethod
    def generate_operation_id(message_type: str, message: t.GeneralValueType) -> str:
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

    @staticmethod
    def create_dynamic_type_subclass(
        name: str,
        base_class: type,  # Base class for dynamic subclass
        attributes: Mapping[str, t.GeneralValueType] | dict[str, t.GeneralValueType],
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
        # Validate base_class is a type before using
        if not isinstance(base_class, type):
            msg = f"base_class must be a type, got {type(base_class).__name__}"
            raise TypeError(msg)
        # Convert Mapping to dict if needed - accept both Mapping and dict per FLEXT standards
        if isinstance(attributes, Mapping) and not isinstance(attributes, dict):
            attributes = dict(attributes.items())
        base_type: type = base_class
        return type(name, (base_type,), attributes)


uGenerators = FlextGenerators  # noqa: N816

__all__ = [
    "FlextGenerators",
    "uGenerators",
]
