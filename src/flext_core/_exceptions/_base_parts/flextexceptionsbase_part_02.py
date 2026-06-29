"""Exception base state behavior."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import ClassVar, override

from flext_core import (
    FlextConstants as c,
    FlextModelsBase as m,
    FlextModelsContainers as mc,
    FlextProtocols as p,
    FlextRuntime,
    FlextTypes as t,
)
from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
    FlextBaseErrorMetadataMixin,
)


class FlextBaseErrorStateMixin(FlextBaseErrorMetadataMixin):
    message: str
    error_code: str
    correlation_id: str | None
    metadata: m.Metadata
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
        merged_kwargs: t.MappingKV[str, t.JsonPayload | None] | p.HasModelDump | None,
        extra_kwargs: t.MappingKV[str, t.JsonPayload | None],
    ) -> None:
        """Initialize the shared base error state without subclass metaprogramming."""
        self.args = (message,)
        self.message = message
        self.error_code = error_code
        final_kwargs_dict: t.JsonDict = {}
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
        self.metadata = type(self)._normalize_metadata(
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


__all__: list[str] = ["FlextBaseErrorStateMixin"]
