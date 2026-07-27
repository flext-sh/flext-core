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
from datetime import UTC, datetime, tzinfo
from importlib import import_module
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextConstants as c, FlextTypes as t


# NOTE (multi-agent): mro-i6nq.12 — consolidated _generators_parts/part_01..02 (one
# FlextUtilitiesGenerators class split across a numbered MRO chain) into this single
# facade module.
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
            default=None, description="Optional random suffix length override"
        )
        include_timestamp: bool = Field(
            default=False, description="Whether to prepend a UTC timestamp to parts"
        )
        separator: str = Field(
            default="_", description="Separator used for custom formatted IDs"
        )

    @staticmethod
    def _determine_prefix(
        kind: str | None, prefix: str | None
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
        prefix: str, *parts: t.JsonValue, length: int = c.SHORT_UUID_LENGTH
    ) -> str:
        """Generate {prefix}_{parts}_{uuid[:length]} formatted ID."""
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

    @staticmethod
    def _build_parts_list(
        parts: t.VariadicTuple[t.JsonValue] | None, *, include_timestamp: bool
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

    @staticmethod
    def generate(
        kind: str | None = None, *, options: GenerateOptions | None = None
    ) -> str:
        """Generate ID by kind or custom prefix (the ONLY public ID generation method)."""
        resolved_options = options or FlextUtilitiesGenerators.GenerateOptions()
        _prefix_resolved, actual_prefix = FlextUtilitiesGenerators._determine_prefix(
            kind, resolved_options.prefix
        )
        generated_id: str
        match (kind, actual_prefix):
            case (("uuid" | None), None):
                generated_id = FlextUtilitiesGenerators._generate_id()
            case ("ulid", _):
                alphabet = string.ascii_letters + string.digits
                short_length = (
                    resolved_options.length
                    if resolved_options.length is not None
                    else c.SHORT_UUID_LENGTH
                )
                generated_id = "".join(
                    secrets.choice(alphabet) for _ in range(short_length)
                )
            case ("id", None):
                generated_id = FlextUtilitiesGenerators._generate_id()
            case (_, str() as pfx):
                all_parts = FlextUtilitiesGenerators._build_parts_list(
                    resolved_options.parts,
                    include_timestamp=resolved_options.include_timestamp,
                )
                id_length = (
                    resolved_options.length
                    if resolved_options.length is not None
                    else c.SHORT_UUID_LENGTH
                )
                if (
                    resolved_options.separator != "_"
                    or resolved_options.include_timestamp
                ):
                    generated_id = (
                        FlextUtilitiesGenerators._generate_custom_separator_id(
                            pfx, all_parts, resolved_options.separator, id_length
                        )
                    )
                else:
                    generated_id = FlextUtilitiesGenerators._generate_prefixed_id(
                        pfx, *all_parts, length=id_length
                    )
            case _:
                generated_id = FlextUtilitiesGenerators._generate_id()
        return generated_id

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

    @staticmethod
    def resolve_timezone(name: str) -> tzinfo:
        """Resolve an IANA timezone name to a tzinfo (``UTC`` maps to ``datetime.UTC``)."""
        return UTC if name.upper() == "UTC" else ZoneInfo(name)

    @staticmethod
    def configured_timezone() -> tzinfo:
        """Resolve the configured timezone from ``FlextSettings.timezone``."""
        settings_module = import_module("flext_core._settings")
        settings_cls = settings_module.FlextSettings
        return FlextUtilitiesGenerators.resolve_timezone(
            settings_cls.fetch_global().timezone
        )

    @staticmethod
    def now() -> datetime:
        """Return the current timezone-aware datetime in the configured timezone (FlextSettings.timezone)."""
        return datetime.now(FlextUtilitiesGenerators.configured_timezone())

    @staticmethod
    def now_iso() -> str:
        """Return the current ISO timestamp (no microseconds) in the configured timezone."""
        return (
            datetime
            .now(FlextUtilitiesGenerators.configured_timezone())
            .replace(microsecond=0)
            .isoformat()
        )

    @staticmethod
    def from_timestamp(timestamp: float) -> datetime:
        """Convert a POSIX timestamp to an aware datetime in the configured timezone."""
        return datetime.fromtimestamp(
            timestamp, FlextUtilitiesGenerators.configured_timezone()
        )

    @staticmethod
    def from_iso(value: str) -> datetime:
        """Parse an ISO-8601 string, assuming the configured timezone when naive."""
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=FlextUtilitiesGenerators.configured_timezone())
        return parsed


__all__: list[str] = ["FlextUtilitiesGenerators"]
