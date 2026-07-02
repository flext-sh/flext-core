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

from flext_core import FlextConstants as c

from .generators_part_01 import (
    FlextUtilitiesGenerators as FlextUtilitiesGeneratorsPart01,
)


class FlextUtilitiesGenerators(FlextUtilitiesGeneratorsPart01):
    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        options: FlextUtilitiesGeneratorsPart01.GenerateOptions | None = None,
    ) -> str:
        """Generate ID by kind or custom prefix (the ONLY public ID generation method)."""
        resolved_options = options or FlextUtilitiesGenerators.GenerateOptions()
        _prefix_resolved, actual_prefix = FlextUtilitiesGenerators._determine_prefix(
            kind,
            resolved_options.prefix,
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
                            pfx,
                            all_parts,
                            resolved_options.separator,
                            id_length,
                        )
                    )
                else:
                    generated_id = FlextUtilitiesGenerators._generate_prefixed_id(
                        pfx,
                        *all_parts,
                        length=id_length,
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
        settings_module = import_module("flext_core.settings")
        settings_cls = settings_module.FlextSettings
        return FlextUtilitiesGenerators.resolve_timezone(
            settings_cls.fetch_global().timezone,
        )

    @staticmethod
    def now() -> datetime:
        """Current timezone-aware datetime in the configured timezone (FlextSettings.timezone)."""
        return datetime.now(FlextUtilitiesGenerators.configured_timezone())

    @staticmethod
    def now_iso() -> str:
        """Current ISO timestamp (no microseconds) in the configured timezone."""
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
            timestamp, FlextUtilitiesGenerators.configured_timezone(),
        )

    @staticmethod
    def from_iso(value: str) -> datetime:
        """Parse an ISO-8601 string, assuming the configured timezone when naive."""
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=FlextUtilitiesGenerators.configured_timezone())
        return parsed


__all__: list[str] = ["FlextUtilitiesGenerators"]
