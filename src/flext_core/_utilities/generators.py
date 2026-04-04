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

from flext_core import c, r, t


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
        """Resolve ID prefix from kind or custom override."""
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
        *parts: t.RecursiveContainer,
        length: int = c.SHORT_UUID_LENGTH,
    ) -> str:
        """Generate {prefix}_{parts}_{uuid[:length]} formatted ID."""
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

    @staticmethod
    def _should_generate_uuid(kind: str | None, actual_prefix: str | None) -> bool:
        """Check if UUID generation should be used."""
        return kind == "uuid" or (kind is None and actual_prefix is None)

    @staticmethod
    def _build_parts_list(
        parts: tuple[t.RecursiveContainer, ...] | None,
        *,
        include_timestamp: bool,
    ) -> t.MutableContainerList:
        """Collect ID parts including optional timestamp prefix."""
        all_parts: t.MutableContainerList = []
        if include_timestamp:
            all_parts.append(int(datetime.now(UTC).timestamp()))
        if parts:
            all_parts.extend(parts)
        return all_parts

    @staticmethod
    def _generate_custom_separator_id(
        actual_prefix: str,
        all_parts: t.MutableContainerList,
        separator: str,
        id_length: int,
    ) -> str:
        """Generate ID with custom separator."""
        uuid_part = str(uuid.uuid4())[:id_length]
        if all_parts:
            middle = str(separator).join(str(p) for p in all_parts)
            return f"{actual_prefix}{separator}{middle}{separator}{uuid_part}"
        return f"{actual_prefix}{separator}{uuid_part}"

    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        prefix: str | None = None,
        parts: tuple[t.RecursiveContainer, ...] | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
    ) -> str:
        """Generate ID by kind or custom prefix (the ONLY public ID generation method)."""
        actual_prefix_result = FlextUtilitiesGenerators._determine_prefix(kind, prefix)
        actual_prefix = (
            actual_prefix_result.value if actual_prefix_result.is_success else None
        )
        match (kind, actual_prefix):
            case (("uuid" | None), None):
                return FlextUtilitiesGenerators._generate_id()
            case ("ulid", _):
                return FlextUtilitiesGenerators.Random.generate_short_id(
                    length if length is not None else c.SHORT_UUID_LENGTH,
                )
            case ("id", None):
                return FlextUtilitiesGenerators._generate_id()
            case (_, str() as pfx):
                all_parts = FlextUtilitiesGenerators._build_parts_list(
                    parts, include_timestamp=include_timestamp
                )
                id_length = length if length is not None else c.SHORT_UUID_LENGTH
                if separator != "_" or include_timestamp:
                    return FlextUtilitiesGenerators._generate_custom_separator_id(
                        pfx,
                        all_parts,
                        separator,
                        id_length,
                    )
                return FlextUtilitiesGenerators._generate_prefixed_id(
                    pfx,
                    *all_parts,
                    length=id_length,
                )
            case _:
                return FlextUtilitiesGenerators._generate_id()

    @staticmethod
    def generate_datetime_utc() -> datetime:
        """Generate current UTC datetime with full microsecond precision."""
        return datetime.now(UTC)

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID using UUID4."""
        return FlextUtilitiesGenerators._generate_id()

    @staticmethod
    def generate_prefixed_id(prefix: str, length: int | None = None) -> str:
        """Generate prefixed ID using UUID4 with optional truncation."""
        base_id = str(uuid.uuid4()).replace("-", "")
        if length is not None:
            base_id = base_id[:length]
        return f"{prefix}_{base_id}" if prefix else base_id

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO timestamp without microseconds (use generate_datetime_utc for precision)."""
        return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = ["FlextUtilitiesGenerators"]
