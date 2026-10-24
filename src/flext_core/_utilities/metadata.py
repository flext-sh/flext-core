"""Shared public-metadata redaction at the JSON boundary.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import ValidationError

from flext_core import t
from flext_core._utilities import FlextUtilitiesTypingAdapterJson


class FlextUtilitiesMetadata:
    """Sanitize structured diagnostics without creating another logging engine."""

    @staticmethod
    def redact_metadata(
        value: t.JsonMapping,
        *,
        keys: t.StrSequence,
        values: t.StrSequence = (),
        replacement: str | None = None,
    ) -> t.JsonDict:
        """Remove or replace sensitive keys before any diagnostic serialization.

        Valid JSON objects embedded in text retain their representation as text.
        Ordinary text is not required to be JSON.

        Returns:
            Sanitized JSON data, without changing the supplied value.

        """
        return FlextUtilitiesMetadata._redact_mapping(
            value,
            protected=frozenset(map(str.casefold, keys)),
            secrets=tuple(filter(None, values)),
            replacement=replacement,
        )

    @staticmethod
    def _redact_mapping(
        item: t.JsonMapping,
        *,
        protected: frozenset[str],
        secrets: t.StrSequence,
        replacement: str | None,
    ) -> t.JsonDict:
        """Drop or replace protected keys and redact every retained value.

        Returns:
            A new mapping without the protected keys' original values.

        """
        return {
            key: replacement
            if key.casefold() in protected
            else FlextUtilitiesMetadata._redact_value(
                nested, protected=protected, secrets=secrets, replacement=replacement
            )
            for key, nested in item.items()
            if replacement is not None or key.casefold() not in protected
        }

    @staticmethod
    def _redact_value(
        item: t.JsonValue,
        *,
        protected: frozenset[str],
        secrets: t.StrSequence,
        replacement: str | None,
    ) -> t.JsonValue:
        """Redact one JSON value by its shape: object, array, text or scalar.

        Returns:
            The redacted value; scalars other than text are returned unchanged.

        """
        if isinstance(item, dict):
            return FlextUtilitiesMetadata._redact_mapping(
                item, protected=protected, secrets=secrets, replacement=replacement
            )
        if isinstance(item, list):
            return [
                FlextUtilitiesMetadata._redact_value(
                    nested,
                    protected=protected,
                    secrets=secrets,
                    replacement=replacement,
                )
                for nested in item
            ]
        if isinstance(item, str):
            return FlextUtilitiesMetadata._redact_text(
                item, protected=protected, secrets=secrets, replacement=replacement
            )
        return item

    @staticmethod
    def _redact_text(
        text: str,
        *,
        protected: frozenset[str],
        secrets: t.StrSequence,
        replacement: str | None,
    ) -> str:
        """Redact embedded JSON text in place of its literal, then mask secrets.

        Returns:
            The text with embedded protected keys and secret values redacted.

        """
        adapter = FlextUtilitiesTypingAdapterJson.json_value_adapter()
        try:
            decoded = adapter.validate_json(text)
        except ValidationError:
            decoded = text
        if isinstance(decoded, (dict, list)):
            cleaned = FlextUtilitiesMetadata._redact_value(
                decoded, protected=protected, secrets=secrets, replacement=replacement
            )
            if cleaned != decoded:
                text = adapter.dump_json(cleaned).decode()
        for secret in secrets:
            text = text.replace(secret, replacement or "")
        return text


__all__: list[str] = ["FlextUtilitiesMetadata"]
