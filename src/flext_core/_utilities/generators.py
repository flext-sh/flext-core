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
from typing import TypeIs

from pydantic import BaseModel

from flext_core import FlextRuntime, c, r, t


class FlextUtilitiesGenerators:
    """Generate deterministic IDs and timestamps for CQRS workflows.

    Nested classes organize related generators:
    - Random: Random/short ID generation
    - Type: Dynamic type generation
    """

    class Random:
        """Random ID generation helpers."""

        @staticmethod
        def generate_short_id(length: int = c.SHORT_UUID_LENGTH) -> str:
            """Generate a short random ID (public API for backward compatibility)."""
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def _determine_prefix(kind: str | None, prefix: str | None) -> r[str]:
        """Determine actual prefix from kind or custom prefix.

        Args:
            kind: ID kind string.
            prefix: Custom prefix (overrides kind).

        Returns:
            Actual prefix string wrapped in r.

        """
        if prefix is not None:
            return r[str].ok(prefix)
        if kind is None:
            return r[str].fail("No kind provided for prefix resolution")
        kind_prefix_map: Mapping[str, str] = {
            "correlation": "corr",
            "entity": "ent",
            "batch": c.ProcessingMode.BATCH,
            "transaction": "txn",
            "saga": "saga",
            c.HandlerType.EVENT: "evt",
            c.HandlerType.COMMAND: "cmd",
            c.HandlerType.QUERY: "qry",
        }
        resolved_prefix = kind_prefix_map.get(kind)
        if resolved_prefix is None:
            return r[str].fail(f"Unsupported generator kind: {kind}")
        return r[str].ok(resolved_prefix)

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
        if include_correlation_id and c.KEY_CORRELATION_ID not in context_dict:
            context_dict[c.KEY_CORRELATION_ID] = FlextUtilitiesGenerators._generate_id()
        if include_timestamp and "timestamp" not in context_dict:
            context_dict["timestamp"] = (
                FlextUtilitiesGenerators.generate_iso_timestamp()
            )

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID using UUID4 (private helper)."""
        return str(uuid.uuid4())

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: t.NormalizedValue,
        length: int = c.SHORT_UUID_LENGTH,
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
    def _is_config_mapping(
        value: t.NormalizedValue,
    ) -> TypeIs[Mapping[str, t.NormalizedValue]]:
        return isinstance(value, Mapping)

    @staticmethod
    def _normalize_context_to_dict(
        context: Mapping[str, t.NormalizedValue] | BaseModel | None,
    ) -> Mapping[str, t.NormalizedValue]:
        """Normalize context to dict - fast fail validation.

        Args:
            context: Context to normalize

        Returns:
            Mapping[str, t.NormalizedValue]: Normalized context dict

        Raises:
            TypeError: If context cannot be normalized

        """
        if context is None:
            msg = "Context cannot be None. Use explicit empty dict {} or handle None in calling code."
            raise TypeError(msg)
        if isinstance(context, Mapping):
            try:
                return dict(context.items())
            except (TypeError, ValueError, AttributeError) as e:
                msg = f"Failed to convert Mapping {context.__class__.__name__}: {e}"
                raise TypeError(msg) from e
        try:
            model_data = context.model_dump()
            model_data_typed: dict[str, t.NormalizedValue] = model_data
            return model_data_typed
        except (AttributeError, TypeError, ValueError) as e:
            msg = f"Failed to dump BaseModel {type(context).__name__}: {type(e).__name__}: {e}"
            raise TypeError(msg) from e

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
    def create_dynamic_type_subclass(
        name: str,
        base_class: type,
        attributes: t.ConfigMap,
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
        attributes_dict = dict(attributes)
        base_type: type = base_class
        return type(name, (base_type,), attributes_dict)

    @staticmethod
    def ensure_dict(
        value: t.ValueOrModel | Mapping[str, t.NormalizedValue] | None,
        default: Mapping[str, t.NormalizedValue] | None = None,
    ) -> Mapping[str, t.NormalizedValue]:
        """Ensure value is a dict, converting from Pydantic models or dict-like.

        This generic helper consolidates duplicate dict normalization logic
        across multiple Pydantic validators (context.py) and dispatcher metadata
        handling. Supports Pydantic models and dict-like objects.

        Conversion Strategy:
            1. None + default provided → return default
            2. None (no default) → return {}
            3. Pydantic model-like value → model_dump() then validate as ConfigMap
            4. Mapping-like value → validate as ConfigMap
            5. Scalar/other value → wrap as {"value": value}

        Args:
            value: Value to normalize (dict, BaseModel, or dict-like)
            default: Default value to return if value is None (optional)

        Returns:
            Mapping[str, t.NormalizedValue]: Normalized dict or default

        Example:
            >>> from flext_core import FlextUtilitiesGuards
            >>> u.ensure_dict({"a": 1})
            {'a': 1}
            >>> u.ensure_dict(None, default={})
            {}
            >>> # Pydantic model
            >>> from pydantic import BaseModel
            >>> class MyModel(BaseModel):
            ...     field: str = "value"
            >>> model = MyModel()
            >>> u.ensure_dict(model)
            {'field': 'value'}

        """
        if value is None:
            if default is not None:
                return default
            msg = "Value cannot be None"
            raise TypeError(msg)
        if isinstance(value, dict):
            return value
        if isinstance(value, Mapping):
            try:
                return {str(key): item_value for key, item_value in value.items()}
            except (TypeError, ValueError, AttributeError) as e:
                msg = f"Failed to convert Mapping {type(value).__name__}: {e}"
                raise TypeError(msg) from e
        if isinstance(value, BaseModel):
            try:
                dumped = value.model_dump()
                dumped_typed: dict[str, t.NormalizedValue] = dumped
                return dumped_typed
            except (AttributeError, TypeError, ValueError) as e:
                msg = f"Failed to convert BaseModel {type(value).__name__} to dict: {e}"
                raise TypeError(msg) from e
        msg = f"Cannot convert {value.__class__.__name__} to dict"
        raise TypeError(msg)

    @staticmethod
    def ensure_trace_context_dict(
        context: Mapping[str, t.NormalizedValue] | BaseModel | None,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> Mapping[str, str]:
        """Ensure context dict has distributed tracing fields (trace_id, span_id, etc).

        This generic helper consolidates duplicate context enrichment logic
        across multiple Pydantic models (service.py, config.py).

        If context is not dict-like, creates new empty dict. Generates UUIDs
        for missing trace_id and span_id fields. Optionally adds correlation_id
        and ISO timestamp.

        Args:
            context: Context dictionary or t.NormalizedValue to enrich (can be any type)
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists (ISO 8601)

        Returns:
            Mapping[str, str]: Enriched context with requested fields (all string values)

        Example:
            >>> from flext_core import FlextUtilitiesGuards
            >>> # Basic: trace_id + span_id
            >>> ctx = u.ensure_trace_context({})
            >>> "trace_id" in ctx and "span_id" in ctx
            True
            >>> # With correlation_id
            >>> ctx = u.ensure_trace_context({}, include_correlation_id=True)
            >>> "correlation_id" in ctx
            True
            >>> # With timestamp
            >>> ctx = u.ensure_trace_context({}, include_timestamp=True)
            >>> "timestamp" in ctx
            True
            >>> # Existing values preserved
            >>> ctx = u.ensure_trace_context({"trace_id": "abc"})
            >>> ctx["trace_id"]
            'abc'

        """
        normalized_dict = FlextUtilitiesGenerators._normalize_context_to_dict(context)
        context_dict: dict[str, str] = {k: str(v) for k, v in normalized_dict.items()}
        FlextUtilitiesGenerators._enrich_context_fields(
            context_dict,
            include_correlation_id=include_correlation_id,
            include_timestamp=include_timestamp,
        )
        return context_dict

    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        prefix: str | None = None,
        parts: tuple[t.NormalizedValue, ...] | None = None,
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
        actual_prefix_result = FlextUtilitiesGenerators._determine_prefix(kind, prefix)
        actual_prefix = (
            actual_prefix_result.value if actual_prefix_result.is_success else None
        )
        if FlextUtilitiesGenerators._should_generate_uuid(kind, actual_prefix):
            return FlextUtilitiesGenerators._generate_id()
        if kind == "ulid":
            ulid_length = length if length is not None else c.SHORT_UUID_LENGTH
            return FlextUtilitiesGenerators.Random.generate_short_id(ulid_length)
        if kind == "id" and actual_prefix is None:
            return FlextUtilitiesGenerators._generate_id()
        if actual_prefix is not None:
            all_parts: list[t.NormalizedValue] = []
            if include_timestamp:
                timestamp = int(datetime.now(UTC).timestamp())
                all_parts.append(timestamp)
            if parts:
                all_parts.extend(parts)
            id_length = length if length is not None else c.SHORT_UUID_LENGTH
            if separator != "_" or include_timestamp:
                uuid_part = str(uuid.uuid4())[:id_length]
                if all_parts:
                    middle = str(separator).join(str(p) for p in all_parts)
                    return f"{actual_prefix}{separator}{middle}{separator}{uuid_part}"
                return f"{actual_prefix}{separator}{uuid_part}"
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
        return FlextUtilitiesGenerators._generate_id()

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
        return FlextRuntime.generate_datetime_utc()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp without microseconds.

        Note: For precise duration calculations, use generate_datetime_utc() instead
        as this method removes microseconds which affects timing precision.
        """
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def generate_operation_id(message_type: str, message: t.ValueOrModel) -> str:
        """Generate unique operation ID for dispatch operations.

        Args:
            message_type: Type of message being dispatched
            message: Message t.NormalizedValue

        Returns:
            str: Unique operation ID

        """
        timestamp = int(time.time() * c.MICROSECONDS_MULTIPLIER)
        message_id = id(message)
        return f"{message_type}_{message_id}_{timestamp}"


__all__ = ["FlextUtilitiesGenerators"]
