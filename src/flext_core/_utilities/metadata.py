"""Shared public-metadata redaction at the JSON boundary."""

from __future__ import annotations

import json

from pydantic import ValidationError

from flext_core import t
from flext_core._utilities import (
    FlextUtilitiesPydantic,
    FlextUtilitiesTypingAdapterJson,
)


class FlextUtilitiesMetadata:
    """Sanitize structured diagnostics without creating another logging engine."""

    @staticmethod
    def normalize_to_metadata[T](val: T) -> t.JsonValue:
        """Serialize metadata values without inventing text for unsupported inputs."""
        return FlextUtilitiesPydantic.normalize_to_json_value(val)

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
        protected = frozenset(key.casefold() for key in keys)
        adapter = FlextUtilitiesTypingAdapterJson.json_value_adapter()

        def text_payload(text: str) -> str:
            try:
                decoded = adapter.validate_json(text)
            except ValidationError:
                decoded = text
            if isinstance(decoded, dict | list):
                cleaned = redact(decoded)
                if cleaned != decoded:
                    text = json.dumps(cleaned, ensure_ascii=False)
            for secret in values:
                if secret:
                    text = text.replace(secret, replacement or "")
            return text

        def redact(item: t.JsonValue) -> t.JsonValue:
            if isinstance(item, dict):
                return mapping(item)
            if isinstance(item, list):
                return [redact(nested) for nested in item]
            if isinstance(item, str):
                return text_payload(item)
            return item

        def mapping(item: t.JsonMapping) -> t.JsonDict:
            return {
                key: replacement if key.casefold() in protected else redact(nested)
                for key, nested in item.items()
                if replacement is not None or key.casefold() not in protected
            }

        return mapping(value)


__all__: list[str] = ["FlextUtilitiesMetadata"]
