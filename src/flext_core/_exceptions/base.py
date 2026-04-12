"""Exception base class — BaseError with full correlation metadata.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, MutableMapping
from typing import ClassVar, override

from flext_core import c, m, t
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core.runtime import FlextRuntime


class FlextExceptionsBase:
    """BaseError and all typed exception subclasses."""

    class BaseError(Exception):
        """Base exception with correlation metadata and error codes.

        All FLEXT exceptions inherit from this to ensure consistent error
        handling, logging, and correlation tracking across the ecosystem.
        """

        _params_cls: ClassVar[t.ModelClass[t.ModelCarrier] | None] = None
        _param_keys: ClassVar[frozenset[str]] = frozenset()
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
            return self._error_domains.get(
                self.error_code,
                c.ErrorDomain.UNKNOWN,
            ).value

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
            context: Mapping[str, t.MetadataValue] | t.ConfigMap | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            correlation_id: str | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap | None = None,
            params: t.ModelCarrier | None = None,
            **extra_kwargs: t.Container,
        ) -> None:
            """Initialize base error with message and optional metadata."""
            declared_params_cls = type(self)._params_cls
            if (
                type(self) is not FlextExceptionsBase.BaseError
                and declared_params_cls is not None
            ):
                resolved_error_code = (
                    str(getattr(type(self), "_default_error_code", error_code))
                    if error_code == c.ErrorCode.UNKNOWN_ERROR
                    else error_code
                )
                combined_extra: MutableMapping[str, t.MetadataOrValue | None] = {}
                if merged_kwargs:
                    combined_extra.update({
                        key: FlextRuntime.normalize_to_metadata(value)
                        for key, value in merged_kwargs.items()
                    })
                combined_extra.update({
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in extra_kwargs.items()
                })
                self._init_declared_error(
                    message,
                    error_code=resolved_error_code,
                    context=context,
                    params=params,
                    correlation_id=correlation_id,
                    metadata=metadata,
                    auto_correlation=auto_correlation,
                    auto_log=auto_log,
                    extra_kwargs=combined_extra,
                )
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
            context: Mapping[str, t.MetadataValue] | t.ConfigMap | None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None,
            correlation_id: str | None,
            auto_correlation: bool,
            auto_log: bool,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap | None,
            extra_kwargs: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
        ) -> None:
            """Initialize the shared base error state without subclass metaprogramming."""
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            final_kwargs: t.ConfigMap = t.ConfigMap(root={})
            if merged_kwargs:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in merged_kwargs.items()
                })
            if context:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in context.items()
                })
            if extra_kwargs:
                final_kwargs.update({
                    k: FlextRuntime.normalize_to_container(
                        FlextRuntime.normalize_to_metadata(v),
                    )
                    for k, v in extra_kwargs.items()
                })
            self.correlation_id = (
                f"exc_{uuid.uuid4().hex[:8]}"
                if auto_correlation and (not correlation_id)
                else correlation_id
            )
            self.metadata = FlextExceptionsBase.BaseError._normalize_metadata(
                metadata,
                final_kwargs,
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
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        ) -> m.Metadata:
            """Normalize metadata from various input types to m.Metadata model."""
            if metadata is None:
                if merged_kwargs:
                    normalized_attrs = {
                        k: FlextRuntime.normalize_to_metadata(v)
                        for k, v in merged_kwargs.items()
                    }
                    return m.Metadata.model_validate({
                        c.FIELD_ATTRIBUTES: normalized_attrs,
                    })
                return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: {}})
            metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
            if metadata_model is not None:
                if not merged_kwargs:
                    return metadata_model
                merged_attrs = {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in metadata_model.attributes.items()
                }
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
                return m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: merged_attrs,
                })
            metadata_dict = FlextExceptionsHelpers.safe_settings_map(metadata)
            if metadata_dict is not None:
                return FlextExceptionsBase.BaseError._normalize_metadata_from_dict(
                    metadata_dict,
                    merged_kwargs,
                )
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {"value": str(metadata)},
            })

        @staticmethod
        def _normalize_metadata_from_dict(
            metadata_dict: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
            merged_kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
        ) -> m.Metadata:
            """Normalize metadata from dict-like recursive containers."""
            merged_attrs: MutableMapping[str, t.MetadataValue | None] = {}
            for k, v in metadata_dict.items():
                merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            if merged_kwargs:
                for k, v in merged_kwargs.items():
                    merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in merged_attrs.items()
                },
            })

        def to_dict(self) -> Mapping[str, t.ValueOrModel | None]:
            """Convert exception to dictionary representation."""
            result: MutableMapping[str, t.MetadataValue | None] = {
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "error_domain": self.error_domain,
                c.ContextKey.CORRELATION_ID: self.correlation_id,
                "timestamp": self.timestamp,
            }
            if self.metadata and self.metadata.attributes:
                filtered_attrs: MutableMapping[str, t.MetadataValue | None] = {
                    k: v for k, v in self.metadata.attributes.items() if k not in result
                }
            else:
                filtered_attrs = {}
            snapshot = m.StructuredErrorSnapshot.model_validate({
                "error_type": type(self).__name__,
                "message": self.message,
                "error_code": self.error_code,
                "error_domain": self.error_domain,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
                "attributes": t.ConfigMap.model_validate(filtered_attrs),
            })
            return snapshot.to_payload().root

        def _init_declared_error(
            self,
            message: str,
            *,
            error_code: str,
            context: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap | None,
            params: t.ModelCarrier | None,
            named_params: Mapping[str, t.RuntimeData | None] | None = None,
            extra_kwargs: Mapping[str, t.MetadataOrValue | None]
            | t.ConfigMap
            | None = None,
            param_keys: frozenset[str] | None = None,
            correlation_id: str | None = None,
            metadata: m.Metadata | t.ConfigMap | t.MetadataValue | None = None,
            auto_correlation: bool = False,
            auto_log: bool = True,
        ) -> None:
            """Initialize a typed error: resolve params, call BaseError.__init__, assign attrs."""
            declared_params_cls = type(self)._params_cls
            if declared_params_cls is None:
                raise ValueError(
                    c.ERR_EXCEPTIONS_PARAMS_CLS_MISSING.format(
                        class_name=type(self).__qualname__,
                    ),
                )
            declared_param_keys = (
                param_keys if param_keys is not None else type(self)._param_keys
            )
            remaining_extra: MutableMapping[str, t.MetadataValue] = {}
            if extra_kwargs:
                remaining_extra.update({
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in extra_kwargs.items()
                })
            resolved_named: MutableMapping[str, t.RuntimeData | None] = dict(
                named_params or {},
            )
            for key in declared_param_keys:
                resolved_named.setdefault(key, remaining_extra.pop(key, None))
            resolved, ctx, meta, corr = FlextExceptionsHelpers.init_error_params(
                context,
                remaining_extra,
                resolved_named,
                declared_params_cls,
                params,
                declared_param_keys,
                excluded_context_keys=type(self)._excluded_context_keys,
            )
            self._initialize_base_state(
                message,
                error_code=error_code,
                context=ctx,
                metadata=metadata if metadata is not None else meta,
                correlation_id=correlation_id if correlation_id is not None else corr,
                auto_correlation=auto_correlation,
                auto_log=auto_log,
                merged_kwargs=None,
                extra_kwargs=t.ConfigMap(root={}),
            )
            for key in declared_param_keys:
                setattr(self, key, getattr(resolved, key))


__all__: list[str] = ["FlextExceptionsBase"]
