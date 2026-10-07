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
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as FlextRuntime,
)
from flext_core._typings.base import FlextTypingBase as tb

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_core import m
    from flext_core._protocols.result import FlextProtocolsResult as pr
    from flext_core._typings.services import FlextTypesServices as ts


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    class BaseError(FlextBaseErrorStateMixin, Exception):
        """Base exception with correlation metadata and error codes."""

        params_cls: ClassVar[ts.ModelClass[m.BaseModel] | None] = None
        excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
        _default_error_code: ClassVar[str] = c.ErrorCode.UNKNOWN_ERROR

        def __init__(  # ruff: ignore[too-many-arguments] -- the keyword contract mirrors the public BaseError constructor; every argument is a distinct documented exception field.
            self,
            message: str,
            *,
            error_code: str = c.ErrorCode.UNKNOWN_ERROR,
            context: tb.MappingKV[str, ts.JsonPayload | None]
            | pr.HasModelDump
            | None = None,
            metadata: pr.HasModelDump | tb.JsonValue | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: tb.MappingKV[str, ts.JsonPayload | None]
            | pr.HasModelDump
            | None = None,
            params: m.BaseModel | None = None,
            **extra_kwargs: tb.JsonValue,
        ) -> None:
            """Initialize base error with message and optional metadata."""
            declaredparams_cls = self.__class__.params_cls
            if declaredparams_cls is None:
                self._initialize_base_state(
                    message,
                    error_code=error_code,
                    context=context,
                    metadata=metadata,
                    correlation_id=correlation_id,
                    auto_correlation=auto_correlation,
                    auto_log=auto_log,
                    merged_kwargs=merged_kwargs,
                    extra_kwargs=extra_kwargs,
                )
                return
            resolved_error_code = (
                self._default_error_code
                if error_code == c.ErrorCode.UNKNOWN_ERROR
                else error_code
            )
            combined_extra = self._combined_extra_map(merged_kwargs, extra_kwargs)
            resolved, remaining_extra, correlation_id_str, preserved_metadata = (
                self._resolve_declared_params(
                    declaredparams_cls,
                    combined_extra,
                    context,
                    params,
                )
            )
            ctx = self._context_from_resolved(
                declaredparams_cls,
                resolved,
                remaining_extra,
                context,
            )
            self._initialize_base_state(
                message,
                error_code=resolved_error_code,
                context=ctx or None,
                metadata=metadata if metadata is not None else preserved_metadata,
                correlation_id=(
                    correlation_id if correlation_id is not None else correlation_id_str
                ),
                auto_correlation=auto_correlation,
                auto_log=auto_log,
                merged_kwargs=None,
                extra_kwargs={},
            )
            for key in frozenset(declaredparams_cls.model_fields):
                setattr(self, key, getattr(resolved, key))

        @staticmethod
        def _combined_extra_map(
            merged_kwargs: tb.MappingKV[str, ts.JsonPayload | None]
            | pr.HasModelDump
            | None,
            extra_kwargs: tb.MappingKV[str, ts.JsonPayload | None],
        ) -> MutableMapping[str, ts.JsonPayload | None]:
            """Merge the merged-kwargs and extra-kwargs maps into one mapping.

            Returns:
                The resulting ``MutableMapping[str, ts.JsonPayload | None]``.

            """
            combined_extra: MutableMapping[str, ts.JsonPayload | None] = {}
            try:
                merged_kwargs_map = FlextRuntime.normalize_metadata_input_mapping(
                    merged_kwargs,
                )
            except c.EXC_PYDANTIC_TYPE_VALUE:
                merged_kwargs_map = None
            if merged_kwargs_map:
                combined_extra.update({
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in merged_kwargs_map.items()
                    if value is not None
                })
            combined_extra.update({
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in extra_kwargs.items()
            })
            return combined_extra

        @staticmethod
        def _resolve_declared_params(
            declared_cls: ts.ModelClass[m.BaseModel],
            combined_extra: MutableMapping[str, ts.JsonPayload | None],
            context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
            params: m.BaseModel | None,
        ) -> tuple[m.BaseModel, tb.MutableJsonMapping, str | None, tb.JsonValue | None]:
            """Resolve the declared params model plus the remaining extras.

            Returns:
                The resulting ``tuple[m.BaseModel, tb.MutableJsonMapping, str | None,
                tb.JsonValue | None]``.

            """
            declared_param_keys = frozenset(declared_cls.model_fields)
            remaining_extra: tb.MutableJsonMapping = {}
            if combined_extra:
                remaining_extra.update({
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in combined_extra.items()
                    if value is not None
                })
            resolved_named: MutableMapping[str, ts.JsonPayload | None] = {}
            for key in declared_param_keys:
                resolved_named.setdefault(key, remaining_extra.pop(key, None))
            preserved_metadata_raw = remaining_extra.pop(c.FIELD_METADATA, None)
            preserved_metadata = (
                FlextRuntime.normalize_to_metadata(preserved_metadata_raw)
                if preserved_metadata_raw is not None
                else None
            )
            correlation_id_raw = remaining_extra.pop(
                c.ContextKey.CORRELATION_ID,
                None,
            )
            correlation_id_str = FlextExceptionsHelpers.safe_optional_str(
                correlation_id_raw,
            )
            param_values = FlextExceptionsHelpers.build_param_map(
                context,
                remaining_extra,
                keys=declared_param_keys,
            )
            for key, value in resolved_named.items():
                if value is None:
                    continue
                normalized_value = FlextRuntime.normalize_to_metadata(value)
                param_values[key] = (
                    normalized_value
                    if isinstance(normalized_value, c.SCALAR_TYPES)
                    else str(normalized_value)
                )
            resolved = (
                params
                if params is not None
                else declared_cls.model_validate(param_values)
            )
            return resolved, remaining_extra, correlation_id_str, preserved_metadata

        @classmethod
        def _context_from_resolved(
            cls,
            declared_cls: ts.ModelClass[m.BaseModel],
            resolved: m.BaseModel,
            remaining_extra: tb.MutableJsonMapping,
            context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
        ) -> tb.JsonDict:
            """Build the context map enriched from the resolved params fields.

            Returns:
                The resulting ``tb.JsonDict``.

            """
            ctx = FlextExceptionsHelpers.build_context_map(
                context,
                remaining_extra,
                excluded_keys=cls.excluded_context_keys,
            )
            resolved_fields = declared_cls.__pydantic_fields__
            for key in frozenset(declared_cls.model_fields):
                attr_val = getattr(resolved, key, None)
                if attr_val is not None:
                    ctx[key] = FlextRuntime.normalize_to_metadata(attr_val)
                field_info = resolved_fields.get(key)
                if field_info is None:
                    continue
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    ctx[f"{key}_description"] = field_help
            return ctx


__all__: list[str] = ["FlextExceptionsBase"]
