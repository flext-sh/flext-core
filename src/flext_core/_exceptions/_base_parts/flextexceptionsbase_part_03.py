"""Exception base facade implementation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core import c, m
from flext_core._exceptions import FlextExceptionsHelpers
from flext_core._exceptions._base_parts import FlextBaseErrorStateMixin
from flext_core._typings import FlextTypingBase

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_core._protocols import FlextProtocolsResult
    from flext_core._typings import FlextTypesServices


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
            *format_args: FlextTypesServices.JsonPayload,
            options: m.ExceptionInitOptions | None = None,
            params: m.BaseModel | None = None,
            metadata: FlextProtocolsResult.HasModelDump
            | FlextTypingBase.JsonValue
            | None = None,
            **extra_kwargs: FlextTypingBase.JsonValue,
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
                option_kwargs: FlextTypingBase.MutableMappingKV[
                    str,
                    FlextProtocolsResult.HasModelDump
                    | FlextTypingBase.JsonValue
                    | None,
                ] = {
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
            merged_kwargs: FlextTypingBase.MappingKV[
                str, FlextTypesServices.JsonPayload | None
            ]
            | FlextProtocolsResult.HasModelDump
            | None,
            extra_kwargs: FlextTypingBase.MappingKV[str, FlextTypingBase.JsonValue],
        ) -> MutableMapping[str, FlextTypesServices.JsonPayload | None]:
            """Normalize merged and extra kwargs into the declared-extra mapping.

            Returns:
                The resulting ``MutableMapping``.

            """
            return dict(
                FlextExceptionsHelpers.build_context_map(merged_kwargs, extra_kwargs)
            )

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
            remaining_extra = FlextExceptionsHelpers.build_context_map(
                combined_extra, {}
            )
            preserved_metadata = remaining_extra.pop(c.FIELD_METADATA, None)
            correlation_id_raw = remaining_extra.pop(
                c.ContextKey.CORRELATION_ID,
                None,
            )
            correlation_id_str = FlextExceptionsHelpers.safe_optional_str(
                correlation_id_raw,
            )
            return remaining_extra, preserved_metadata, correlation_id_str

        @staticmethod
        def _resolve_named_param_values(
            context: FlextTypingBase.MappingKV[
                str, FlextTypesServices.JsonPayload | None
            ]
            | FlextProtocolsResult.HasModelDump
            | None,
            remaining_extra: FlextTypingBase.MutableJsonMapping,
            declared_param_keys: frozenset[str],
            resolved_named: MutableMapping[str, FlextTypesServices.JsonPayload | None],
        ) -> FlextTypingBase.JsonDict:
            """Fold resolved named params into the declared param-value map.

            Returns:
                The resulting ``FlextTypingBase.JsonDict``.

            """
            param_values = FlextExceptionsHelpers.build_param_map(
                context,
                remaining_extra,
                keys=declared_param_keys,
            )
            param_values.update(
                FlextExceptionsHelpers.build_param_map(
                    None, resolved_named, keys=declared_param_keys
                )
            )
            return param_values

        @staticmethod
        def _attach_declared_field_context(
            ctx: FlextTypingBase.MutableJsonMapping,
            declaredparams_cls: FlextTypesServices.ModelClass[m.BaseModel],
            resolved: m.BaseModel,
            declared_param_keys: frozenset[str],
        ) -> None:
            """Attach declared field values and help text to the context map."""
            resolved_fields = declaredparams_cls.__pydantic_fields__
            ctx.update(
                FlextExceptionsHelpers.build_param_map(
                    resolved, {}, keys=declared_param_keys
                )
            )
            for key in declared_param_keys:
                field_info = resolved_fields.get(key)
                if field_info is None:
                    continue
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    ctx[f"{key}_description"] = field_help

        def _resolve_declared_params(
            self,
            declaredparams_cls: FlextTypesServices.ModelClass[m.BaseModel],
            context: FlextTypingBase.MappingKV[
                str, FlextTypesServices.JsonPayload | None
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
            resolved_named: MutableMapping[
                str, FlextTypesServices.JsonPayload | None
            ] = {}
            for key in declared_param_keys:
                resolved_named.setdefault(key, remaining_extra.pop(key, None))
            param_values = self._resolve_named_param_values(
                context,
                remaining_extra,
                declared_param_keys,
                resolved_named,
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
            self._attach_declared_field_context(
                ctx,
                declaredparams_cls,
                resolved,
                declared_param_keys,
            )
            return resolved, ctx

        @staticmethod
        def _resolve_declared_options(
            options: m.ExceptionInitOptions,
            ctx: FlextTypingBase.MutableJsonMapping,
            preserved_metadata: FlextTypingBase.JsonValue | None,
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
