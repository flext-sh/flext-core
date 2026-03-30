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
import uuid
from datetime import UTC, datetime

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
            Actual prefix string wrapped in p.Result.

        """
        if prefix is not None:
            return r[str].ok(prefix)
        if kind is None:
            return r[str].fail("No kind provided for prefix resolution")
        kind_prefix_map: t.StrMapping = {
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

        """
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

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
            all_parts: t.MutableContainerList = []
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
            datetime: Current UTC datetime with full microsecond precisionon

        """
        return FlextRuntime.generate_datetime_utc()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp without microseconds.

        Note: For precise duration calculations, use generate_datetime_utc() instead
        as this method removes microseconds which affects timing precision.

        """
        return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = ["FlextUtilitiesGenerators"]
