"""Exception base facade implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.infrastructure import FlextConstantsInfrastructure as ci
from flext_core._constants.mixins import FlextConstantsMixins as cm
from flext_core._constants.validation import FlextConstantsValidation as cv
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as FlextRuntime,
)
from flext_core._typings.base import FlextTypingBase as tb

from .flextexceptionsbase_part_02 import FlextBaseErrorStateMixin

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_core._protocols.base import FlextProtocolsBase as p
    from flext_core._protocols.result import FlextProtocolsResult as pr
    from flext_core._typings.services import FlextTypesServices as ts


class FlextBaseError(FlextBaseErrorStateMixin, Exception):
    """Base exception with correlation metadata and error codes."""

    _params_cls: ClassVar[type[p.BaseModel] | None] = None
    _excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None

    def __init__(
        self,
        message: str,
        *,
        error_code: str = cv.ErrorCode.UNKNOWN_ERROR,
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
        params: p.BaseModel | None = None,
        **extra_kwargs: tb.JsonValue,
    ) -> None:
        """Initialize base error with message and optional metadata."""
        declared_params_cls = self.__class__._params_cls
        if declared_params_cls is not None:
            resolved_error_code = (
                str(getattr(type(self), "_default_error_code", error_code))
                if error_code == cv.ErrorCode.UNKNOWN_ERROR
                else error_code
            )
            combined_extra: MutableMapping[str, ts.JsonPayload | None] = {}
            try:
                merged_kwargs_map = FlextRuntime.normalize_metadata_input_mapping(
                    merged_kwargs
                )
            except ce.EXC_PYDANTIC_TYPE_VALUE:
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
            declared_param_keys = frozenset(declared_params_cls.__pydantic_fields__)
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
            preserved_metadata_raw = remaining_extra.pop(cm.FIELD_METADATA, None)
            preserved_metadata = (
                FlextRuntime.normalize_to_metadata(preserved_metadata_raw)
                if preserved_metadata_raw is not None
                else None
            )
            correlation_id_raw = remaining_extra.pop(ci.ContextKey.CORRELATION_ID, None)
            correlation_id_str = FlextExceptionsHelpers.safe_optional_str(
                correlation_id_raw
            )
            param_values = FlextExceptionsHelpers.build_param_map(
                context, remaining_extra, keys=declared_param_keys
            )
            for key, value in resolved_named.items():
                if value is None:
                    continue
                normalized_value = FlextRuntime.normalize_to_metadata(value)
                param_values[key] = (
                    normalized_value
                    if isinstance(normalized_value, tb.SCALAR_TYPES)
                    else str(normalized_value)
                )
            resolved = (
                params
                if params is not None
                else declared_params_cls.model_validate(param_values)
            )
            ctx = FlextExceptionsHelpers.build_context_map(
                context,
                remaining_extra,
                excluded_keys=type(self)._excluded_context_keys,
            )
            resolved_fields = declared_params_cls.__pydantic_fields__
            for key in declared_param_keys:
                attr_val = getattr(resolved, key, None)
                if attr_val is not None:
                    ctx[key] = FlextRuntime.normalize_to_metadata(attr_val)
                field_info = resolved_fields.get(key)
                if field_info is None:
                    continue
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    ctx[f"{key}_description"] = field_help
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
            for key in declared_param_keys:
                setattr(self, key, getattr(resolved, key))
            return
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


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    BaseError = FlextBaseError


__all__: list[str] = ["FlextExceptionsBase"]
