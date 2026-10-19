"""Exception internal helpers - safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c
from flext_core._models import FlextModelsBase
from flext_core._protocols import FlextProtocolsResult

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesServices, FlextTypingBase


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None]
        | FlextProtocolsResult.HasModelDump
        | None,
        extra_kwargs: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ],
    ) -> tuple[tuple[str, FlextTypingBase.JsonValue], ...]:
        """Collect normalized metadata entries from context and kwargs once.

        Returns:
            The resulting ``tuple[tuple[str, tb.JsonValue], ...]``.

        """
        from flext_core import m

        entries: list[tuple[str, FlextTypingBase.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            source_mapping = m.ConfigMap.model_validate(
                source_value.model_dump(mode="json")
                if isinstance(source_value, FlextProtocolsResult.HasModelDump)
                else source_value
            ).model_dump(mode="json")
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((key, value))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: FlextProtocolsResult.HasModelDump
        | FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None]
        | FlextTypingBase.JsonValue
        | None,
    ) -> FlextModelsBase.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model.

        Returns:
            The resulting ``FlextModelsBase.Metadata | None``.

        """
        if value is None or isinstance(value, FlextModelsBase.Metadata):
            return value
        return FlextModelsBase.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: (
                value.model_dump(mode="json")
                if isinstance(value, FlextProtocolsResult.HasModelDump)
                else value
            )
        })

    @staticmethod
    def safe_optional_str(
        value: FlextTypesServices.JsonPayload | type | None,
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
        context: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None]
        | FlextProtocolsResult.HasModelDump
        | None,
        extra_kwargs: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ],
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> FlextTypingBase.JsonDict:
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
        context: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None]
        | FlextProtocolsResult.HasModelDump
        | None,
        extra_kwargs: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ],
        keys: set[str] | frozenset[str],
    ) -> FlextTypingBase.JsonDict:
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
