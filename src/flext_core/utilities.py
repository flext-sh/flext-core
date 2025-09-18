"""Utility helpers referenced by the FLEXT 1.0.0 modernization plan.

These helpers back the validation, ID generation, and text-processing stories
described in ``README.md`` and ``docs/architecture.md``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
import uuid
from collections.abc import Buffer, Callable
from datetime import UTC, datetime
from typing import SupportsFloat, SupportsIndex, cast

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import P, R


class FlextUtilities:
    """Utility namespace reused across the FLEXT ecosystem.

    Convenience functions live here so packages share the same guardrails and
    generators while we stabilise the 1.0.0 surface area.
    """

    class Validation:
        """Validation helpers shared across modules."""

        @staticmethod
        def is_non_empty_string(value: str | None) -> bool:
            """Check if string is non-empty after stripping whitespace."""
            return bool(value and value.strip())

    class Generators:
        """ID and timestamp generators for traceability."""

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
        """Text processing helpers for normalization and sanitisation."""

        @staticmethod
        def safe_string(value: object) -> str:
            """Convert any value to safe string."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            return str(value)

        @staticmethod
        def clean_text(text: str | None) -> str:
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
        def slugify(text: str | None) -> str:
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
        def safe_int(value: object, *, default: int = 0) -> int:
            """Convert value to int safely."""
            try:
                if isinstance(value, (str, int, float)):
                    return int(value)
                return default
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
        def safe_float(value: object, *, default: float = 0.0) -> float:
            """Convert value to float safely."""
            try:
                # Try conversion for any input; rely on exceptions for invalid types

                return float(
                    cast("str | Buffer | SupportsFloat | SupportsIndex", value)
                )
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

    class Reliability:
        """Reliability-focused helpers."""

        @staticmethod
        def safe_result(func: Callable[P, R]) -> Callable[P, FlextResult[R]]:
            """Wrap callable execution in FlextResult for safe error handling."""

            @functools.wraps(func)
            def _wrapped(*args: P.args, **kwargs: P.kwargs) -> FlextResult[R]:
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, FlextResult):
                        return result
                    return FlextResult[R].ok(result)
                except Exception as exc:
                    return FlextResult[R].fail(
                        str(exc),
                        error_code=FlextConstants.Errors.UNKNOWN_ERROR,
                        error_data={"exception": type(exc).__name__},
                    )

            return _wrapped

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
