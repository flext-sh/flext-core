"""Exception internal helpers - safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import flext_core
from flext_core._utilities import FlextUtilitiesPydantic

if TYPE_CHECKING:
    pass


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: flext_core.p.MetadataInput,
        extra_kwargs: flext_core.t.MetadataMapping,
    ) -> tuple[tuple[str, flext_core.t.JsonValue], ...]:
        """Collect normalized metadata entries from context and kwargs once.

        Returns:
            The resulting ``tuple[tuple[str, tb.JsonValue], ...]``.

        """
        entries: list[tuple[str, flext_core.t.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            source_mapping = (
                source_value.model_dump(mode="json")
                if isinstance(source_value, flext_core.p.HasModelDump)
                else source_value
            )
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((
                        key,
                        FlextUtilitiesPydantic.normalize_to_json_value(value),
                    ))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: flext_core.p.HasModelDump
        | Mapping[str, flext_core.t.JsonPayload | None]
        | flext_core.t.JsonValue
        | None,
    ) -> flext_core.p.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model.

        Returns:
            The resulting ``m.Metadata | None``.

        """
        if value is None or isinstance(value, flext_core.m.Metadata):
            return value
        return flext_core.m.Metadata.model_validate({
            flext_core.c.FIELD_ATTRIBUTES: (
                value.model_dump(mode="json")
                if isinstance(value, flext_core.p.HasModelDump)
                else value
            )
        })

    @staticmethod
    def safe_optional_str(
        value: flext_core.t.JsonPayload | type | None,
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
        context: flext_core.p.MetadataInput,
        extra_kwargs: flext_core.t.MetadataMapping,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> flext_core.t.JsonDict:
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
        context: flext_core.p.MetadataInput,
        extra_kwargs: flext_core.t.MetadataMapping,
        keys: set[str] | frozenset[str],
    ) -> flext_core.t.JsonDict:
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
