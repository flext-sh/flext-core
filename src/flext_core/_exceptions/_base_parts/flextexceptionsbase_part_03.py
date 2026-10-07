"""Exception base facade implementation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core import c
from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
    FlextBaseErrorStateMixin,
)
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core._runtime._metadata_validation import FlextRuntimeMetadataValidation
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_core import m
    from flext_core._protocols import FlextProtocolsResult
    from flext_core._typings.services import FlextTypesServices


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    class BaseError(FlextBaseErrorStateMixin, Exception):
        """Base exception with correlation metadata and error codes."""

        params_cls: ClassVar[FlextTypesServices.ModelClass[m.BaseModel] | None] = None
        excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
        _default_error_code: ClassVar[str] = c.ErrorCode.UNKNOWN_ERROR

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.ErrorCode.UNKNOWN_ERROR,
            context: FlextTypingBase.MappingKV[
                str,
                FlextTypesServices.JsonPayload | None,
            ]
            | FlextProtocolsResult.HasModelDump
            | None = None,
            metadata: FlextProtocolsResult.HasModelDump
            | FlextTypingBase.JsonValue
            | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: FlextTypingBase.MappingKV[
                str,
                FlextTypesServices.JsonPayload | None,
            ]
            | FlextProtocolsResult.HasModelDump
            | None = None,
            params: m.BaseModel | None = None,
            **extra_kwargs: FlextTypingBase.JsonValue,
        ) -> None:
            """Initialize base error with message and optional metadata."""
            declaredparams_cls = self.__class__.params_cls
            if declaredparams_cls is None:
                self._init_error_identity(message, error_code)
                self._init_error_correlation(
                    correlation_id,
                    auto_correlation=auto_correlation,
                )
                self._init_error_channels(
                    self._merge_error_channel_sources(
                        context,
                        merged_kwargs,
                        extra_kwargs,
                    ),
                    metadata,
                    auto_log=auto_log,
                )
                return
            resolved_error_code = self._resolve_declared_error_code(error_code)
            combined_extra = self._build_declared_extra(merged_kwargs, extra_kwargs)
            declared_param_keys = frozenset(declaredparams_cls.model_fields)
            remaining_extra, preserved_metadata, declared_correlation_id = (
                self._extract_declared_remaining(combined_extra)
            )
            resolved, ctx = self._resolve_declared_params(
                declaredparams_cls,
                context,
                remaining_extra,
                params,
            )
            self._init_error_identity(message, resolved_error_code)
            self._init_error_correlation(
                correlation_id
                if correlation_id is not None
                else declared_correlation_id,
                auto_correlation=auto_correlation,
            )
            self._init_error_channels(
                self._merge_error_channel_sources(ctx or None, None, {}),
                metadata if metadata is not None else preserved_metadata,
                auto_log=auto_log,
            )
            for key in declared_param_keys:
                setattr(self, key, getattr(resolved, key))

        def _resolve_declared_error_code(self, error_code: str) -> str:
            """Resolve the effective error code for a declared-params error.

            Returns:
                The resulting ``str``.

            """
            return (
                self._default_error_code
                if error_code == c.ErrorCode.UNKNOWN_ERROR
                else error_code
            )

        @staticmethod
        def _build_declared_extra(
            merged_kwargs: FlextTypingBase.MappingKV[
                str,
                FlextTypesServices.JsonPayload | None,
            ]
            | FlextProtocolsResult.HasModelDump
            | None,
            extra_kwargs: FlextTypingBase.JsonDict,
        ) -> MutableMapping[str, FlextTypesServices.JsonPayload | None]:
            """Normalize merged and extra kwargs into the declared-extra mapping.

            Returns:
                The resulting ``MutableMapping``.

            """
            combined_extra: MutableMapping[
                str,
                FlextTypesServices.JsonPayload | None,
            ] = {}
            try:
                merged_kwargs_map = (
                    FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(
                        merged_kwargs,
                    )
                )
            except c.EXC_PYDANTIC_TYPE_VALUE:
                merged_kwargs_map = None
            if merged_kwargs_map:
                combined_extra.update({
                    key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                    for key, value in merged_kwargs_map.items()
                    if value is not None
                })
            combined_extra.update({
                key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                for key, value in extra_kwargs.items()
            })
            return combined_extra

        @staticmethod
        def _extract_declared_remaining(
            combined_extra: MutableMapping[str, FlextTypesServices.JsonPayload | None],
        ) -> tuple[
            FlextTypingBase.MutableJsonMapping,
            FlextTypingBase.JsonValue | None,
            str | None,
        ]:
            """Split declared params from remaining extras in the combined mapping.

            Returns:
                The resulting ``tuple``.

            """
            remaining_extra: FlextTypingBase.MutableJsonMapping = {}
            if combined_extra:
                remaining_extra.update({
                    key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                    for key, value in combined_extra.items()
                    if value is not None
                })
            preserved_metadata_raw = remaining_extra.pop(c.FIELD_METADATA, None)
            preserved_metadata = (
                FlextRuntimeMetadataValidation.normalize_to_metadata(
                    preserved_metadata_raw,
                )
                if preserved_metadata_raw is not None
                else None
            )
            correlation_id_raw = remaining_extra.pop(c.ContextKey.CORRELATION_ID, None)
            correlation_id_str = FlextExceptionsHelpers.safe_optional_str(
                correlation_id_raw,
            )
            return remaining_extra, preserved_metadata, correlation_id_str

        def _resolve_declared_params(
            self,
            declaredparams_cls: FlextTypesServices.ModelClass[m.BaseModel],
            context: FlextTypingBase.MappingKV[
                str,
                FlextTypesServices.JsonPayload | None,
            ]
            | FlextProtocolsResult.HasModelDump
            | None,
            remaining_extra: FlextTypingBase.MutableJsonMapping,
            params: m.BaseModel | None,
        ) -> tuple[m.BaseModel, FlextTypingBase.MutableJsonMapping]:
            """Validate declared params and build the structured context mapping.

            Returns:
                The resulting ``tuple``.

            """
            declared_param_keys = frozenset(declaredparams_cls.model_fields)
            param_values = FlextExceptionsHelpers.build_param_map(
                context,
                remaining_extra,
                keys=declared_param_keys,
            )
            for key, value in self._extract_declared_named(
                declared_param_keys,
                remaining_extra,
            ).items():
                if value is None:
                    continue
                normalized_value = FlextRuntimeMetadataValidation.normalize_to_metadata(
                    value,
                )
                param_values[key] = (
                    normalized_value
                    if isinstance(normalized_value, c.SCALAR_TYPES)
                    else str(normalized_value)
                )
            resolved = (
                params
                if params is not None
                else declaredparams_cls.model_validate(param_values)
            )
            ctx = FlextExceptionsHelpers.build_context_map(
                context,
                remaining_extra,
                excluded_keys=type(self).excluded_context_keys,
            )
            resolved_fields = declaredparams_cls.__pydantic_fields__
            for key in declared_param_keys:
                attr_val = getattr(resolved, key, None)
                if attr_val is not None:
                    ctx[key] = FlextRuntimeMetadataValidation.normalize_to_metadata(
                        attr_val,
                    )
                field_info = resolved_fields.get(key)
                if field_info is None:
                    continue
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    ctx[f"{key}_description"] = field_help
            return resolved, ctx

        @staticmethod
        def _extract_declared_named(
            declared_param_keys: frozenset[str],
            remaining_extra: FlextTypingBase.MutableJsonMapping,
        ) -> MutableMapping[str, FlextTypesServices.JsonPayload | None]:
            """Pop declared param defaults from the remaining extras.

            Returns:
                The resulting ``MutableMapping``.

            """
            resolved_named: MutableMapping[
                str,
                FlextTypesServices.JsonPayload | None,
            ] = {}
            for key in declared_param_keys:
                resolved_named.setdefault(key, remaining_extra.pop(key, None))
            return resolved_named


__all__: list[str] = ["FlextExceptionsBase"]
