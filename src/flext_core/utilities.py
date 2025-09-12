"""FLEXT Core Utilities - Refactored based on REAL usage analysis.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from datetime import UTC, datetime


class FlextUtilities:
    """Essential utilities based on REAL usage analysis across FLEXT ecosystem."""

    class Validation:
        """Centralized validation utilities to eliminate code duplication."""

        @staticmethod
        def is_non_empty_string(value: str | None) -> bool:
            """Check if string is non-empty after stripping whitespace."""
            return bool(value and value.strip())

    class Generators:
        """ID and timestamp generation utilities - ACTUALLY USED."""

        @staticmethod
        def generate_id() -> str:
            """Generate short unique ID."""
            return str(uuid.uuid4())[:8]

        @staticmethod
        def generate_uuid() -> str:
            """Generate full UUID."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO timestamp."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate correlation ID."""
            return f"corr_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_request_id() -> str:
            """Generate request ID."""
            return f"req_{str(uuid.uuid4())[:12]}"

        @staticmethod
        def generate_entity_id() -> str:
            """Generate entity ID."""
            return f"ent_{str(uuid.uuid4())[:12]}"

    # ==========================================================================
    # ACTUALLY USED CLASS: TEXT PROCESSOR (96 usages across ecosystem)
    # ==========================================================================

    class TextProcessor:
        """Text processing utilities - ACTUALLY USED."""

        @staticmethod
        def safe_string(value: object) -> str:
            """Convert any value to safe string."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            return str(value)

        @staticmethod
        def clean_text(text: str) -> str:
            """Clean and normalize text."""
            if not text:
                return ""
            # Basic cleaning - remove extra whitespace, normalize unicode
            text = re.sub(r"\s+", " ", text.strip())
            return unicodedata.normalize("NFKD", text)

        @staticmethod
        def is_non_empty_string(value: object) -> bool:
            """Check if value is a non-empty string."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def slugify(text: str) -> str:
            """Convert text to URL-friendly slug."""
            if not text:
                return ""
            # Basic slugification - remove special chars, lowercase, replace spaces with hyphens
            text = re.sub(r"[^\w\s-]", "", text.lower())
            text = re.sub(r"[-\s]+", "-", text)
            return text.strip("-")

    class Conversions:
        """Type conversion utilities."""

        @staticmethod
        def safe_int(value: str | float, *, default: int = 0) -> int:
            """Convert value to int safely."""
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        @staticmethod
        def safe_bool(value: object, *, default: bool = False) -> bool:
            """Convert value to bool safely."""
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower_val = value.lower()
                if lower_val in {"true", "1", "yes", "on"}:
                    return True
                if lower_val in {"false", "0", "no", "off"}:
                    return False
                return default
            try:
                return bool(value) if value else default
            except (ValueError, TypeError):
                return default

        @staticmethod
        def safe_float(value: str | float, *, default: float = 0.0) -> float:
            """Convert value to float safely."""
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

    class TypeGuards:
        """Type checking utilities."""

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is non-empty dict."""
            return isinstance(value, dict) and len(value) > 0

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is non-empty string."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is non-empty list."""
            return isinstance(value, list) and len(value) > 0

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO timestamp - direct usage compatibility."""
        return FlextUtilities.Generators.generate_iso_timestamp()

    @staticmethod
    def safe_json_stringify(data: object, *, default: str = "{}") -> str:
        """Convert data to JSON string safely."""
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return default


__all__: list[str] = [
    "FlextUtilities",
]
