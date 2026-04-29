"""Exception internal helpers — safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import no_type_check

from pydantic import ValidationError as PydanticValidationError

from flext_core import (
    FlextConstants as c,
    FlextModelsBase as m,
    FlextProtocols as p,
    FlextRuntime,
    FlextTypes as t,
)


@no_type_check
class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.JsonPayload | None],
    ) -> tuple[tuple[str, t.JsonValue], ...]:
        """Collect normalized metadata entries from context and kwargs once."""
        entries: list[tuple[str, t.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            try:
                source_mapping = FlextRuntime.normalize_metadata_input_mapping(
                    source_value,
                )
            except (PydanticValidationError, TypeError, ValueError):
                continue
            if not source_mapping:
                continue
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((key, FlextRuntime.normalize_to_metadata(value)))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: p.HasModelDump | Mapping[str, t.JsonPayload | None] | t.JsonValue | None,
    ) -> m.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        metadata: m.Metadata | None = None
        if value is not None:
            try:
                metadata = m.Metadata.model_validate(value, from_attributes=True)
            except (PydanticValidationError, TypeError):
                if isinstance(value, (Mapping, p.HasModelDump)):
                    try:
                        attrs_map = FlextRuntime.normalize_metadata_input_mapping(value)
                    except (PydanticValidationError, TypeError, ValueError):
                        attrs_map = None
                    if attrs_map is not None:
                        attrs = {
                            key: item
                            for key, item in attrs_map.items()
                            if item is not None
                        }
                        metadata = m.Metadata.model_validate({
                            c.FIELD_ATTRIBUTES: attrs
                        })
        return metadata

    @staticmethod
    def safe_optional_str(value: t.JsonPayload | type | None) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def build_context_map(
        context: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.JsonPayload | None],
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> dict[str, t.JsonValue]:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context,
                extra_kwargs,
            )
            if key not in excluded
        }

    @staticmethod
    def build_param_map(
        context: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.JsonPayload | None],
        keys: set[str] | frozenset[str],
    ) -> dict[str, t.JsonValue]:
        """Build parameter map restricted to declared param keys."""
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context,
                extra_kwargs,
            )
            if key in keys
        }


__all__: list[str] = ["FlextExceptionsHelpers"]
