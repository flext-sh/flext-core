"""Exception internal helpers - safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextRuntime, c, m, p

if TYPE_CHECKING:
    from flext_core import t


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: p.MetadataInput,
        extra_kwargs: t.MetadataMapping,
    ) -> tuple[tuple[str, t.JsonValue], ...]:
        """Collect normalized metadata entries from context and kwargs once.

        Returns:
            The resulting ``tuple[tuple[str, tb.JsonValue], ...]``.

        """
        entries: list[tuple[str, t.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            source_mapping = (
                source_value.model_dump(mode="json")
                if isinstance(source_value, p.HasModelDump)
                else source_value
            )
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((key, FlextRuntime.normalize_to_json_value(value)))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: p.HasModelDump
        | t.MappingKV[str, t.JsonPayload | None]
        | t.JsonValue
        | None,
    ) -> m.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model.

        Returns:
            The resulting ``m.Metadata | None``.

        """
        if value is None or isinstance(value, m.Metadata):
            return value
        return m.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: (
                value.model_dump(mode="json")
                if isinstance(value, p.HasModelDump)
                else value
            )
        })

    @staticmethod
    def safe_optional_str(
        value: t.JsonPayload | type | None,
    ) -> str | None:
        """Extract optional strict string from dynamic values.

        Returns:
            The resulting ``str | None``.

        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def build_context_map(
        context: p.MetadataInput,
        extra_kwargs: t.MetadataMapping,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> t.JsonDict:
        """Build normalized context map from context and kwargs.

        Returns:
            The resulting ``tb.JsonDict``.

        """
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
        context: p.MetadataInput,
        extra_kwargs: t.MetadataMapping,
        keys: set[str] | frozenset[str],
    ) -> t.JsonDict:
        """Build parameter map restricted to declared param keys.

        Returns:
            The resulting ``tb.JsonDict``.

        """
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context,
                extra_kwargs,
            )
            if key in keys
        }


__all__: list[str] = ["FlextExceptionsHelpers"]
