"""ID and data generation helpers shared across dispatcher flows.

These primitives centralize correlation, batch, and timestamp generation so
dispatcher handlers and services produce consistent identifiers and audit
metadata without duplicating randomness or formatting concerns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextConstants as c, FlextTypes as t


class FlextUtilitiesGenerators:
    """Generate deterministic IDs and timestamps for CQRS workflows.

    Centralizes random, prefixed, and timestamp-based identifier generation.
    """

    class GenerateOptions(BaseModel):
        """Typed options envelope for public ID generation."""

        model_config = ConfigDict(extra="forbid")

        prefix: str | None = Field(default=None, description="Custom ID prefix")
        parts: t.VariadicTuple[t.JsonValue] | None = Field(
            default=None,
            description="Optional parts inserted between prefix and random suffix",
        )
        length: int | None = Field(
            default=None,
            description="Optional random suffix length override",
        )
        include_timestamp: bool = Field(
            default=False,
            description="Whether to prepend a UTC timestamp to parts",
        )
        separator: str = Field(
            default="_",
            description="Separator used for custom formatted IDs",
        )

    @staticmethod
    def _determine_prefix(
        kind: str | None,
        prefix: str | None,
    ) -> t.Pair[bool, str | None]:
        """Resolve ID prefix from kind or custom override."""
        if prefix is not None:
            return (True, prefix)
        if kind is None:
            return (False, None)
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
            return (False, None)
        return (True, resolved_prefix)

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID using UUID4 (private helper)."""
        return str(uuid.uuid4())

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: t.JsonValue,
        length: int = c.SHORT_UUID_LENGTH,
    ) -> str:
        """Generate {prefix}_{parts}_{uuid[:length]} formatted ID."""
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

    @staticmethod
    def _build_parts_list(
        parts: t.VariadicTuple[t.JsonValue] | None,
        *,
        include_timestamp: bool,
    ) -> t.SequenceOf[t.JsonValue]:
        """Collect ID parts including optional timestamp prefix."""
        all_parts: t.JsonValueList = []
        if include_timestamp:
            all_parts.append(int(datetime.now(UTC).timestamp()))
        if parts:
            all_parts.extend(parts)
        return all_parts

    @staticmethod
    def _generate_custom_separator_id(
        actual_prefix: str,
        all_parts: t.SequenceOf[t.JsonValue],
        separator: str,
        id_length: int,
    ) -> str:
        """Generate ID with custom separator."""
        uuid_part = str(uuid.uuid4())[:id_length]
        if all_parts:
            middle = separator.join(str(p) for p in all_parts)
            return f"{actual_prefix}{separator}{middle}{separator}{uuid_part}"
        return f"{actual_prefix}{separator}{uuid_part}"


__all__: list[str] = ["FlextUtilitiesGenerators"]
