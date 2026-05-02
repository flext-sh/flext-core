"""Exception base class — BaseError with full correlation metadata.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import (
    Mapping,
    MutableMapping,
)
from typing import ClassVar, override

from flext_core import (
    FlextConstants as c,
    FlextExceptionsHelpers,
    FlextModelsBase as m,
    FlextModelsContainers as mc,
    FlextModelsErrors,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
    FlextRuntime,
    FlextTypes as t,
)


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        _params_cls: ClassVar[t.ModelClass[mp.BaseModel] | None] = None
        _excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
        message: str
        error_code: str
        correlation_id: str | None
        metadata: m.Metadata
        timestamp: float
        auto_log: bool
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
            domain = self._error_domains.get(
                self.error_code,
                c.ErrorDomain.UNKNOWN,
            )
            return domain.value

        @property
        def error_message(self) -> str | None:
            """Human-readable message used by structured error consumers."""
            return self.message

        def matches_error_domain(self, domain: str) -> bool:
            """Whether this error belongs to the provided routing domain."""
            return self.error_domain == domain

        def __init__(
            self,
            message: str,
            *,
            error_code: str = c.ErrorCode.UNKNOWN_ERROR,
            context: t.MappingKV[str, t.JsonPayload | None]
            | p.HasModelDump
            | None = None,
            metadata: p.HasModelDump | t.JsonValue | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: t.MappingKV[str, t.JsonPayload | None]
            | p.HasModelDump
            | None = None,
            params: mp.BaseModel | None = None,
            **extra_kwargs: t.JsonValue,
        ) -> None:
            """Initialize base error with message and optional metadata."""
            declared_params_cls = self.__class__._params_cls
            if declared_params_cls is not None:
                resolved_error_code = (
                    str(getattr(type(self), "_default_error_code", error_code))
                    if error_code == c.ErrorCode.UNKNOWN_ERROR
                    else error_code
                )
                combined_extra: MutableMapping[str, t.JsonPayload | None] = {}
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
                declared_param_keys = frozenset(declared_params_cls.model_fields)
                remaining_extra: MutableMapping[str, t.JsonValue] = {}
                if combined_extra:
                    remaining_extra.update({
                        key: FlextRuntime.normalize_to_metadata(value)
                        for key, value in combined_extra.items()
                        if value is not None
                    })
                resolved_named: MutableMapping[str, t.JsonPayload | None] = {}
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
                        if isinstance(normalized_value, t.SCALAR_TYPES)
                        else str(normalized_value)
                    )
                resolved = (
                    params
                    if params is not None
                    else declared_params_cls.model_validate(dict(param_values))
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
                        correlation_id
                        if correlation_id is not None
                        else correlation_id_str
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

        def _initialize_base_state(
            self,
            message: str,
            *,
            error_code: str,
            context: t.MappingKV[str, t.JsonPayload | None] | p.HasModelDump | None,
            metadata: p.HasModelDump | t.JsonValue | None,
            correlation_id: str | None,
            auto_correlation: bool,
            auto_log: bool,
            merged_kwargs: t.MappingKV[str, t.JsonPayload | None]
            | p.HasModelDump
            | None,
            extra_kwargs: t.MappingKV[str, t.JsonPayload | None],
        ) -> None:
            """Initialize the shared base error state without subclass metaprogramming."""
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            final_kwargs_dict: dict[str, t.JsonValue] = {}
            for source_value in (merged_kwargs, context, extra_kwargs):
                if source_value is None:
                    continue
                try:
                    source_dict = FlextRuntime.normalize_metadata_input_mapping(
                        source_value,
                    )
                except c.EXC_PYDANTIC_TYPE_VALUE:
                    continue
                if not source_dict:
                    continue
                for key, value in source_dict.items():
                    if value is not None:
                        final_kwargs_dict[key] = FlextRuntime.normalize_to_metadata(
                            value,
                        )
            final_kwargs = mc.ConfigMap.model_validate(final_kwargs_dict)
            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and (not correlation_id)
                else correlation_id
            )
            self.metadata = FlextExceptionsBase.BaseError._normalize_metadata(
                metadata,
                final_kwargs.root,
            )
            self.timestamp = time.time()
            self.auto_log = auto_log

        @override
        def __str__(self) -> str:
            """Return string representation with error code if present."""
            if self.error_code:
                return f"[{self.error_code}] {self.message}"
            return self.message

        @staticmethod
        def _normalize_metadata(
            metadata: p.HasModelDump | t.JsonValue | None,
            merged_kwargs: t.MappingKV[str, t.JsonPayload],
        ) -> m.Metadata:
            """Normalize metadata from various input types to m.Metadata model."""
            if metadata is None:
                normalized_attrs = {
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in merged_kwargs.items()
                }
                resolved_metadata = m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: normalized_attrs,
                })
            else:
                metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
                if metadata_model is not None:
                    merged_attrs = {
                        key: FlextRuntime.normalize_to_metadata(value)
                        for key, value in metadata_model.attributes.items()
                        if value is not None
                    }
                    for key, value in merged_kwargs.items():
                        if value is None:
                            continue
                        merged_attrs[key] = FlextRuntime.normalize_to_metadata(value)
                    resolved_metadata = m.Metadata.model_validate({
                        c.FIELD_ATTRIBUTES: merged_attrs,
                    })
                else:
                    metadata_dict: t.MappingKV[str, t.JsonPayload | None] | None = None
                    if isinstance(metadata, (Mapping, p.HasModelDump)):
                        try:
                            metadata_dict = (
                                FlextRuntime.normalize_metadata_input_mapping(
                                    metadata,
                                )
                            )
                        except c.EXC_PYDANTIC_TYPE_VALUE:
                            metadata_dict = None
                    resolved_metadata = (
                        FlextExceptionsBase.BaseError._normalize_metadata_from_dict(
                            metadata_dict,
                            merged_kwargs,
                        )
                        if metadata_dict is not None
                        else m.Metadata.model_validate({
                            c.FIELD_ATTRIBUTES: {"value": str(metadata)},
                        })
                    )
            return resolved_metadata

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: t.MappingKV[str, t.JsonPayload | None],
            merged_kwargs: t.MappingKV[str, t.JsonPayload],
        ) -> m.Metadata:
            """Normalize metadata from dict-like recursive containers."""
            merged_attrs: MutableMapping[str, t.JsonValue | None] = {}
            for k, v in metadata_dict.items():
                if v is None:
                    continue
                merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    if v is None:
                        continue
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in merged_attrs.items()
                    if v is not None
                },
            })

        def to_dict(self) -> t.MappingKV[str, t.JsonPayload | None]:
            """Convert exception to dictionary representation."""
            result: MutableMapping[str, t.JsonPayload | None] = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "error_domain": self.error_domain,
                c.ContextKey.CORRELATION_ID: self.correlation_id,
                "timestamp": self.timestamp,
            }
            if self.metadata and self.metadata.attributes:
                filtered_attrs: MutableMapping[str, t.JsonPayload | None] = {
                    k: v for k, v in self.metadata.attributes.items() if k not in result
                }
            else:
                filtered_attrs = {}
            snapshot: FlextModelsErrors.StructuredErrorSnapshot = (
                FlextModelsErrors.StructuredErrorSnapshot.model_validate({
                    "error_type": type(self).__name__,
                    "message": self.message,
                    "error_code": self.error_code,
                    "error_domain": self.error_domain,
                    "correlation_id": self.correlation_id,
                    "timestamp": self.timestamp,
                    "attributes": mc.ConfigMap.model_validate(filtered_attrs).root,
                })
            )
            payload: dict[str, t.JsonPayload | None] = {
                "error_type": snapshot.error_type,
                "message": snapshot.message,
                "error_code": snapshot.error_code,
                "error_domain": snapshot.error_domain,
                "correlation_id": snapshot.correlation_id,
                "timestamp": snapshot.timestamp,
                "error_message": snapshot.error_message,
            }
            for key, value in snapshot.attributes.items():
                if key not in payload:
                    payload[key] = value
            return payload


__all__: list[str] = ["FlextExceptionsBase"]
