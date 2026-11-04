"""Exception base facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, MutableMapping
from typing import ClassVar, cast, override

from flext_core import c, m, p, t
from flext_core._exceptions import FlextExceptionsHelpers


class FlextExceptionsBase:
    """Public exception base namespace."""

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes."""

        @staticmethod
        def normalize_metadata(
            metadata: p.HasModelDump | t.JsonValue | None,
            merged_kwargs: Mapping[str, t.JsonPayload],
        ) -> p.Metadata:
            """Normalize metadata from various input types to m.Metadata model.

            Returns:
                The resulting ``m.Metadata``.

            """
            metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
            source = (
                metadata_model.model_dump(mode="json")
                if metadata_model is not None
                else {}
            )
            merged_attrs: MutableMapping[str, t.JsonPayload | None] = (
                dict(metadata_model.attributes) if metadata_model is not None else {}
            )
            merged_attrs.update(merged_kwargs)
            return m.Metadata.model_validate({
                **source,
                c.FIELD_ATTRIBUTES: merged_attrs,
            })

        message: str
        error_code: str
        correlation_id: str | None
        metadata: p.Metadata
        timestamp: float
        auto_log: bool
        args: tuple[str, ...]

        _error_domains: ClassVar[Mapping[str, c.ErrorDomain]] = {
            c.ErrorCode.VALIDATION_ERROR: c.ErrorDomain.VALIDATION,
            c.ErrorCode.TYPE_ERROR: c.ErrorDomain.VALIDATION,
            c.ErrorCode.ALREADY_EXISTS: c.ErrorDomain.VALIDATION,
            c.ErrorCode.CONFIG_ERROR: c.ErrorDomain.INTERNAL,
            c.ErrorCode.CONFIGURATION_ERROR: c.ErrorDomain.INTERNAL,
            c.ErrorCode.ATTRIBUTE_ERROR: c.ErrorDomain.INTERNAL,
            c.ErrorCode.OPERATION_ERROR: c.ErrorDomain.INTERNAL,
            c.ErrorCode.AUTHENTICATION_ERROR: c.ErrorDomain.AUTH,
            c.ErrorCode.AUTHORIZATION_ERROR: c.ErrorDomain.AUTH,
            c.ErrorCode.PERMISSION_ERROR: c.ErrorDomain.AUTH,
            c.ErrorCode.CONNECTION_ERROR: c.ErrorDomain.NETWORK,
            c.ErrorCode.EXTERNAL_SERVICE_ERROR: c.ErrorDomain.NETWORK,
            c.ErrorCode.TIMEOUT_ERROR: c.ErrorDomain.TIMEOUT,
            c.ErrorCode.NOT_FOUND_ERROR: c.ErrorDomain.NOT_FOUND,
            c.ErrorCode.NOT_FOUND: c.ErrorDomain.NOT_FOUND,
            c.ErrorCode.RESOURCE_NOT_FOUND: c.ErrorDomain.NOT_FOUND,
            c.ErrorCode.UNKNOWN_ERROR: c.ErrorDomain.UNKNOWN,
        }

        @property
        def error_domain(self) -> str | None:
            """Canonical routing domain derived from the structured error code."""
            if not self.error_code:
                return None
            domain = self._error_domains.get(self.error_code, c.ErrorDomain.UNKNOWN)
            return domain.value

        @property
        def error_message(self) -> str | None:
            """Human-readable message used by structured error consumers."""
            return self.message

        def matches_error_domain(self, domain: str) -> bool:
            """Whether this error belongs to the provided routing domain.

            Returns:
                The resulting ``bool``.

            """
            return self.error_domain == domain

        def _initialize_base_state(
            self,
            message: str,
            *,
            error_code: str,
            options: p.ExceptionInitOptions,
            extra_kwargs: Mapping[str, t.JsonPayload | None],
        ) -> None:
            """Initialize the shared base error state without metaprogramming."""
            self.args = (message,)
            self.message = message
            self.error_code = error_code
            final_kwargs_dict: t.JsonDict = {}
            for source_value in (options.merged_kwargs, options.context, extra_kwargs):
                if source_value is None:
                    continue
                final_kwargs_dict.update(
                    FlextExceptionsHelpers.build_context_map(source_value, {})
                )
            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if options.auto_correlation and not options.correlation_id
                else options.correlation_id
            )
            self.metadata = type(self).normalize_metadata(
                options.metadata,
                final_kwargs_dict,
            )
            self.timestamp = time.time()
            self.auto_log = options.auto_log

        @override
        def __str__(self) -> str:
            """Return string representation with error code if present.

            Returns:
                String representation with error code if present.

            """
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        params_cls: ClassVar[t.ModelClass[t.BaseModelType] | None] = None
        excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
        _default_error_code: ClassVar[str] = c.ErrorCode.UNKNOWN_ERROR

        def __init__(
            self,
            message: str,
            *format_args: t.JsonPayload,
            options: p.ExceptionInitOptions | None = None,
            params: p.BaseModel | None = None,
            metadata: p.HasModelDump | t.JsonValue | None = None,
            **extra_kwargs: t.JsonValue,
        ) -> None:
            """Initialize base error with message and optional metadata.

            Positional arguments after ``message`` are %-format values applied
            to the message template (the standard logging convention).
            ``metadata`` is the typed channel for structured error metadata:
            it accepts a dumpable model (e.g. a domain ``ErrorMetadata``) or a
            JSON payload, mirroring ``ExceptionInitOptions.metadata``.
            """
            resolved_message = message % format_args if format_args else message
            opts = (
                cast("m.ExceptionInitOptions", options) if options is not None else None
            )
            if opts is None:
                # Keyword-form construction: the tested public contract accepts
                # the ExceptionInitOptions fields as direct keywords (e.g.
                # ``BaseError("m", error_code="E_BASE", auto_log=False)``).
                # Extract exactly the remaining option keys from the kwargs so
                # they feed the typed options instead of falling into the
                # generic extra bucket; ``metadata`` is a named parameter.
                option_kwargs: MutableMapping[
                    str,
                    p.HasModelDump | t.JsonValue | None,
                ] = {
                    key: extra_kwargs.pop(key)
                    for key in (
                        "error_code",
                        "context",
                        c.ContextKey.CORRELATION_ID,
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
            merged_kwargs: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
            extra_kwargs: Mapping[str, t.JsonValue],
        ) -> MutableMapping[str, t.JsonPayload | None]:
            """Normalize merged and extra kwargs into the declared-extra mapping.

            Returns:
                The resulting ``MutableMapping``.

            """
            return dict(
                FlextExceptionsHelpers.build_context_map(merged_kwargs, extra_kwargs)
            )

        @staticmethod
        def _extract_declared_remaining(
            combined_extra: MutableMapping[str, t.JsonPayload | None],
        ) -> tuple[
            t.MutableJsonMapping,
            t.JsonValue | None,
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
            context: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
            remaining_extra: t.MutableJsonMapping,
            declared_param_keys: frozenset[str],
            resolved_named: MutableMapping[str, t.JsonPayload | None],
        ) -> t.JsonDict:
            """Fold resolved named params into the declared param-value map.

            Returns:
                The resulting ``t.JsonDict``.

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
            ctx: t.MutableJsonMapping,
            declaredparams_cls: t.ModelClass[t.BaseModelType],
            resolved: p.BaseModel,
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
            declaredparams_cls: t.ModelClass[t.BaseModelType],
            context: Mapping[str, t.JsonPayload | None] | p.HasModelDump | None,
            remaining_extra: t.MutableJsonMapping,
            params: p.BaseModel | None,
        ) -> tuple[p.BaseModel, t.MutableJsonMapping]:
            """Validate declared params and build the structured context mapping.

            Returns:
                The resulting ``tuple``.

            """
            declared_param_keys = frozenset(declaredparams_cls.model_fields)
            resolved_named: MutableMapping[str, t.JsonPayload | None] = {}
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
            options: p.ExceptionInitOptions,
            ctx: t.MutableJsonMapping,
            preserved_metadata: t.JsonValue | None,
            declared_correlation_id: str | None,
        ) -> p.ExceptionInitOptions:
            """Fold declared-params resolutions back into the init options.

            Returns:
                The resulting ``m.ExceptionInitOptions``.

            """
            return cast("m.ExceptionInitOptions", options).model_copy(
                update={
                    "context": ctx or None,
                    "metadata": (
                        options.metadata
                        if options.metadata is not None
                        else preserved_metadata
                    ),
                    c.ContextKey.CORRELATION_ID: (
                        options.correlation_id
                        if options.correlation_id is not None
                        else declared_correlation_id
                    ),
                    "merged_kwargs": None,
                },
            )


__all__: list[str] = ["FlextExceptionsBase"]
