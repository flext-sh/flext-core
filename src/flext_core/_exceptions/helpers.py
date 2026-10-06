"""Exception internal helpers - safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from flext_core._constants._errors_parts.flextconstantserrors_part_03 import (
    FlextConstantsErrorsValidationExceptions,
)
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._models.base import FlextModelsBase
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._runtime._metadata_validation import FlextRuntimeMetadataValidation

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.services import FlextTypesServices


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
        entries: list[tuple[str, FlextTypingBase.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            try:
                source_mapping = (
                    FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(
                        source_value,
                    )
                )
            except FlextConstantsErrorsValidationExceptions.EXC_PYDANTIC_TYPE_VALUE:
                continue
            if not source_mapping:
                continue
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((
                        key,
                        FlextRuntimeMetadataValidation.normalize_to_metadata(value),
                    ))
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
        metadata: FlextModelsBase.Metadata | None = None
        if value is not None:
            try:
                metadata = FlextModelsBase.Metadata.model_validate(
                    value,
                    from_attributes=True,
                )
            except (PydanticValidationError, TypeError):
                if isinstance(value, (Mapping, FlextProtocolsResult.HasModelDump)):
                    try:
                        attrs_map = FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(
                            value,
                        )
                    except (
                        FlextConstantsErrorsValidationExceptions.EXC_PYDANTIC_TYPE_VALUE
                    ):
                        attrs_map = None
                    if attrs_map is not None:
                        attrs = {
                            key: item
                            for key, item in attrs_map.items()
                            if item is not None
                        }
                        metadata = FlextModelsBase.Metadata.model_validate({
                            FlextConstantsMixins.FIELD_ATTRIBUTES: attrs,
                        })
        return metadata

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
