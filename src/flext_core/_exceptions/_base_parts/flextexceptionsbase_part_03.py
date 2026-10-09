"""Exception base facade implementation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core import c, m
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

    from flext_core._protocols.result import FlextProtocolsResult as pr
    from flext_core._typings.services import FlextTypesServices as ts


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    class BaseError(FlextBaseErrorStateMixin, Exception):
        """Base exception with correlation metadata and error codes."""

        params_cls: ClassVar[ts.ModelClass[m.BaseModel] | None] = None
        excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
        _default_error_code: ClassVar[str] = c.ErrorCode.UNKNOWN_ERROR

        def __init__(
            self,
            message: str,
            *format_args: ts.JsonPayload,
            options: m.ExceptionInitOptions | None = None,
            params: m.BaseModel | None = None,
            metadata: pr.HasModelDump | tb.JsonValue | None = None,
            **extra_kwargs: tb.JsonValue,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Positional arguments after ``message`` are %-format values applied
            to the message template (the standard logging convention).
            ``metadata`` is the typed channel for structured error metadata:
            it accepts a dumpable model (e.g. a domain ``ErrorMetadata``) or a
            JSON payload, mirroring ``ExceptionInitOptions.metadata``.
            """
            resolved_message = message % format_args if format_args else message
            opts = options
            if opts is None:
                # Keyword-form construction: the tested public contract accepts
                # the ExceptionInitOptions fields as direct keywords (e.g.
                # ``BaseError("m", error_code="E_BASE", auto_log=False)``).
                # Extract exactly the remaining option keys from the kwargs so
                # they feed the typed options instead of falling into the
                # generic extra bucket; ``metadata`` is a named parameter.
                option_kwargs: dict[str, pr.HasModelDump | tb.JsonValue | None] = {
                    key: extra_kwargs.pop(key)
                    for key in (
                        "error_code",
                        "context",
                        "correlation_id",
                        "auto_correlation",
                        "auto_log",
                        "merged_kwargs",
                    )
                    if key in extra_kwargs
                }
                if metadata is not None:
                    option_kwargs["metadata"] = metadata
                opts = m.ExceptionInitOptions.model_validate(option_kwargs)
            elif metadata is not None:
                opts = opts.model_copy(update={"metadata": metadata})
            declaredparams_cls = self.__class__.params_cls
            if declaredparams_cls is None:
                self._initialize_base_state(
                    resolved_message,
                    error_code=opts.error_code,
                    options=opts,
                    extra_kwargs=extra_kwargs,
                )
                return
            resolved_error_code = self._resolve_declared_error_code(opts.error_code)
            combined_extra = self._build_declared_extra(
                opts.merged_kwargs,
                extra_kwargs,
            )
            remaining_extra, preserved_metadata, declared_correlation_id = (
                self._extract_declared_remaining(combined_extra)
            )
            resolved, ctx = self._resolve_declared_params(
                declaredparams_cls,
                opts.context,
                remaining_extra,
                params,
            )
            resolved_options = self._resolve_declared_options(
                opts,
                ctx,
                preserved_metadata,
                declared_correlation_id,
            )
            self._initialize_base_state(
                resolved_message,
                error_code=resolved_error_code,
                options=resolved_options,
                extra_kwargs={},
            )
            for key in frozenset(declaredparams_cls.model_fields):
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
            merged_kwargs: tb.MappingKV[str, ts.JsonPayload | None]
            | pr.HasModelDump
            | None,
            extra_kwargs: tb.MappingKV[str, tb.JsonValue],
        ) -> MutableMapping[str, ts.JsonPayload | None]:
            """Normalize merged and extra kwargs into the declared-extra mapping.

            Returns:
                The resulting ``MutableMapping``.

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
        def _extract_declared_remaining(
            combined_extra: MutableMapping[str, ts.JsonPayload | None],
        ) -> tuple[tb.MutableJsonMapping, tb.JsonValue | None, str | None]:
            """Split declared params from remaining extras in the combined mapping.

            Returns:
                The resulting ``tuple``.

            """
            remaining_extra: tb.MutableJsonMapping = {}
            if combined_extra:
                remaining_extra.update({
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in combined_extra.items()
                    if value is not None
                })
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
            return remaining_extra, preserved_metadata, correlation_id_str

        def _resolve_declared_params(
            self,
            declaredparams_cls: ts.ModelClass[m.BaseModel],
            context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
            remaining_extra: tb.MutableJsonMapping,
            params: m.BaseModel | None,
        ) -> tuple[m.BaseModel, tb.MutableJsonMapping]:
            """Validate declared params and build the structured context mapping.

            Returns:
                The resulting ``tuple``.

            """
            declared_param_keys = frozenset(declaredparams_cls.model_fields)
            resolved_named: MutableMapping[str, ts.JsonPayload | None] = {}
            for key in declared_param_keys:
                resolved_named.setdefault(key, remaining_extra.pop(key, None))
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
                    ctx[key] = FlextRuntime.normalize_to_metadata(attr_val)
                field_info = resolved_fields.get(key)
                if field_info is None:
                    continue
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    ctx[f"{key}_description"] = field_help
            return resolved, ctx

        @staticmethod
        def _resolve_declared_options(
            options: m.ExceptionInitOptions,
            ctx: tb.MutableJsonMapping,
            preserved_metadata: tb.JsonValue | None,
            declared_correlation_id: str | None,
        ) -> m.ExceptionInitOptions:
            """Fold declared-params resolutions back into the init options.

            Returns:
                The resulting ``m.ExceptionInitOptions``.

            """
            return options.model_copy(
                update={
                    "context": ctx or None,
                    "metadata": (
                        options.metadata
                        if options.metadata is not None
                        else preserved_metadata
                    ),
                    "correlation_id": (
                        options.correlation_id
                        if options.correlation_id is not None
                        else declared_correlation_id
                    ),
                    "merged_kwargs": None,
                },
            )


__all__: list[str] = ["FlextExceptionsBase"]
