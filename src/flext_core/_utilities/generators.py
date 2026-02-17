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

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesGenerators:
    """Generate deterministic IDs and timestamps for CQRS workflows.

    Nested classes organize related generators:
    - Random: Random/short ID generation
    - Type: Dynamic type generation
    """

    class Random:
        """Random ID generation helpers."""

        @staticmethod
        def generate_short_id(length: int = c.Utilities.SHORT_UUID_LENGTH) -> str:
            """Generate a short random ID (public API for backward compatibility)."""
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(length))

    # NOTE: create_dynamic_type_subclass is available as static method - no nested Type class needed

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: t.GeneralValueType,
        length: int = c.Utilities.SHORT_UUID_LENGTH,
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
            >>> generate_prefixed_id("corr")
            'corr_a1b2c3d4'
            >>> generate_prefixed_id("batch", 100)
            'batch_100_a1b2c3d4'

        """
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID using UUID4 (private helper)."""
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
            >>> start = FlextUtilitiesGenerators.generate_datetime_utc()
            >>> # ... operation ...
            >>> end = FlextUtilitiesGenerators.generate_datetime_utc()
            >>> duration = (end - start).total_seconds()  # Precise duration

        """
        return datetime.now(UTC)

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
            # Type narrowing: context is dict, isinstance provides type narrowing to ConfigurationDict
            context_dict_result: dict[str, t.GeneralValueType] = context
            return context_dict_result
        if isinstance(context, Mapping):
            try:
                # Type narrowing: context is Mapping, convert to dict
                context_dict_mapping: dict[str, t.GeneralValueType] = dict(
                    context.items(),
                )
                return context_dict_mapping
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
            context_dict["trace_id"] = FlextUtilitiesGenerators._generate_id()
        if "span_id" not in context_dict:
            context_dict["span_id"] = FlextUtilitiesGenerators._generate_id()

        # Optionally ensure correlation_id
        if include_correlation_id and "correlation_id" not in context_dict:
            context_dict["correlation_id"] = FlextUtilitiesGenerators._generate_id()

        # Optionally ensure timestamp (ISO 8601 format)
        if include_timestamp and "timestamp" not in context_dict:
            context_dict["timestamp"] = (
                FlextUtilitiesGenerators.generate_iso_timestamp()
            )

    @staticmethod
    def ensure_trace_context(
        context: Mapping[str, str] | object,
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
        normalized_dict = FlextUtilitiesGenerators._normalize_context_to_dict(context)
        # Convert all values to strings for trace context (trace_id, span_id, etc. are strings)
        context_dict: dict[str, str] = {
            k: v if isinstance(v, str) else str(v) for k, v in normalized_dict.items()
        }
        FlextUtilitiesGenerators._enrich_context_fields(
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
            # BaseModel.model_dump() returns dict-like object, check with is_type guard
            result = value.model_dump()
            if FlextUtilitiesGuards.is_type(result, dict):
                # normalize_to_general_value preserves dict structure
                # so normalized will be dict[str, t.GeneralValueType]
                normalized = FlextRuntime.normalize_to_general_value(result)
                # Type narrowing: isinstance narrows normalized to dict
                if isinstance(normalized, dict):
                    # Type narrowed: normalized is dict[str, t.GeneralValueType]
                    return normalized
                # Fallback: if normalization changed type, return empty dict
                return {}

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
    def _determine_prefix(kind: str | None, prefix: str | None) -> str | None:
        """Determine actual prefix from kind or custom prefix.

        Args:
            kind: ID kind string.
            prefix: Custom prefix (overrides kind).

        Returns:
            Actual prefix string or None.

        """
        if prefix is not None:
            return prefix

        if kind is None:
            return None

        kind_prefix_map: dict[str, str] = {
            "correlation": "corr",
            "entity": "ent",
            "batch": c.Cqrs.ProcessingMode.BATCH,
            "transaction": "txn",
            "saga": "saga",
            "event": "evt",
            "command": "cmd",
            "query": "qry",
            # "aggregate": None,  # Requires prefix parameter - removed to avoid dict type error
        }
        return kind_prefix_map.get(kind)

    @staticmethod
    def _should_generate_uuid(kind: str | None, actual_prefix: str | None) -> bool:
        """Check if UUID generation should be used.

        Args:
            kind: ID kind string.
            actual_prefix: Determined prefix or None.

        Returns:
            True if UUID should be generated.

        """
        return kind == "uuid" or (kind is None and actual_prefix is None)

    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        prefix: str | None = None,
        parts: tuple[t.GeneralValueType, ...] | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
    ) -> str:
        """Generate ID by kind or custom prefix.

        This is the ONLY public method for ID generation. All other generate* methods
        are private helpers used internally.

        Args:
            kind: ID kind ("uuid", "correlation", "entity", "batch", "transaction",
                "saga", "event", "command", "query", "aggregate", "ulid", "id").
                If None, generates UUID.
            prefix: Custom prefix (overrides kind prefix if provided).
            parts: Additional parts to include in ID (e.g., batch_size, aggregate_type).
            length: Custom length for generated ID (only for ulid/short IDs or LONG UUID).
            include_timestamp: Include timestamp in ID.
            separator: Separator between prefix, parts, and ID (default: "_").

        Returns:
            Generated ID string.

        """
        actual_prefix = FlextUtilitiesGenerators._determine_prefix(kind, prefix)

        # Handle UUID/ULID/id special cases
        if FlextUtilitiesGenerators._should_generate_uuid(kind, actual_prefix):
            return FlextUtilitiesGenerators._generate_id()

        if kind == "ulid":
            ulid_length = (
                length if length is not None else c.Utilities.SHORT_UUID_LENGTH
            )
            return FlextUtilitiesGenerators.Random.generate_short_id(ulid_length)

        if kind == "id" and actual_prefix is None:
            return FlextUtilitiesGenerators._generate_id()

        # Generate prefixed ID
        if actual_prefix is not None:
            # Build parts list
            all_parts: list[t.GeneralValueType] = []
            if include_timestamp:
                timestamp = int(datetime.now(UTC).timestamp())
                all_parts.append(timestamp)
            if parts:
                all_parts.extend(parts)

            id_length = length if length is not None else c.Utilities.SHORT_UUID_LENGTH

            # Custom separator requires manual construction
            if separator != "_" or include_timestamp:
                uuid_part = str(uuid.uuid4())[:id_length]
                if all_parts:
                    middle = str(separator).join(str(p) for p in all_parts)
                    return f"{actual_prefix}{separator}{middle}{separator}{uuid_part}"
                return f"{actual_prefix}{separator}{uuid_part}"

            # Standard prefixed ID generation (separator is "_")
            if all_parts:
                return FlextUtilitiesGenerators._generate_prefixed_id(
                    actual_prefix,
                    *all_parts,
                    length=id_length,
                )
            return FlextUtilitiesGenerators._generate_prefixed_id(
                actual_prefix,
                length=id_length,
            )

        # Fallback: UUID
        return FlextUtilitiesGenerators._generate_id()

    @staticmethod
    def generate_operation_id(message_type: str, message: t.GeneralValueType) -> str:
        """Generate unique operation ID for dispatch operations.

        Args:
            message_type: Type of message being dispatched
            message: Message object

        Returns:
            str: Unique operation ID

        """
        timestamp = int(time.time() * c.MICROSECONDS_MULTIPLIER)
        message_id = id(message)
        return f"{message_type}_{message_id}_{timestamp}"

    @staticmethod
    def create_dynamic_type_subclass(
        name: str,
        base_class: type,  # Base class for dynamic subclass
        attributes: t.ConfigurationMapping | dict[str, t.GeneralValueType],
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
        # Runtime validation: base_class parameter is typed as type
        # Type system ensures base_class is a type, so no runtime check needed
        # ConfigurationMapping and ConfigurationDict are both Mapping, so isinstance is redundant
        # Convert to dict for type() call
        attributes_dict: dict[str, t.GeneralValueType] = dict(attributes)
        base_type: type = base_class
        return type(name, (base_type,), attributes_dict)


__all__ = [
    "FlextUtilitiesGenerators",
]
