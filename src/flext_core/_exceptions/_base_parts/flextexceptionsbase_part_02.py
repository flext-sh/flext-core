"""Exception base state behavior."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, ClassVar, override

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.validation import FlextConstantsValidation as cv
from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as FlextRuntime,
)

from .flextexceptionsbase_part_01 import FlextBaseErrorMetadataMixin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core._models.base import FlextModelsBase as m
    from flext_core._protocols.result import FlextProtocolsResult as pr
    from flext_core._typings.base import FlextTypingBase as tb
    from flext_core._typings.services import FlextTypesServices as ts


class FlextBaseErrorStateMixin(FlextBaseErrorMetadataMixin):
    message: str
    error_code: str
    correlation_id: str | None
    metadata: m.Metadata
    timestamp: float
    auto_log: bool
    args: tuple[str, ...]

    _error_domains: ClassVar[Mapping[str, ce.ErrorDomain]] = {
        cv.ErrorCode.VALIDATION_ERROR: ce.ErrorDomain.VALIDATION,
        cv.ErrorCode.TYPE_ERROR: ce.ErrorDomain.VALIDATION,
        cv.ErrorCode.ALREADY_EXISTS: ce.ErrorDomain.VALIDATION,
        cv.ErrorCode.CONFIG_ERROR: ce.ErrorDomain.INTERNAL,
        cv.ErrorCode.CONFIGURATION_ERROR: ce.ErrorDomain.INTERNAL,
        cv.ErrorCode.ATTRIBUTE_ERROR: ce.ErrorDomain.INTERNAL,
        cv.ErrorCode.OPERATION_ERROR: ce.ErrorDomain.INTERNAL,
        cv.ErrorCode.AUTHENTICATION_ERROR: ce.ErrorDomain.AUTH,
        cv.ErrorCode.AUTHORIZATION_ERROR: ce.ErrorDomain.AUTH,
        cv.ErrorCode.PERMISSION_ERROR: ce.ErrorDomain.AUTH,
        cv.ErrorCode.CONNECTION_ERROR: ce.ErrorDomain.NETWORK,
        cv.ErrorCode.EXTERNAL_SERVICE_ERROR: ce.ErrorDomain.NETWORK,
        cv.ErrorCode.TIMEOUT_ERROR: ce.ErrorDomain.TIMEOUT,
        cv.ErrorCode.NOT_FOUND_ERROR: ce.ErrorDomain.NOT_FOUND,
        cv.ErrorCode.NOT_FOUND: ce.ErrorDomain.NOT_FOUND,
        cv.ErrorCode.RESOURCE_NOT_FOUND: ce.ErrorDomain.NOT_FOUND,
        cv.ErrorCode.UNKNOWN_ERROR: ce.ErrorDomain.UNKNOWN,
    }

    @property
    def error_domain(self) -> str | None:
        """Canonical routing domain derived from the structured error code."""
        if not self.error_code:
            return None
        domain = self._error_domains.get(self.error_code, ce.ErrorDomain.UNKNOWN)
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
        context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
        metadata: pr.HasModelDump | tb.JsonValue | None,
        correlation_id: str | None,
        auto_correlation: bool,
        auto_log: bool,
        merged_kwargs: tb.MappingKV[str, ts.JsonPayload | None]
        | pr.HasModelDump
        | None,
        extra_kwargs: tb.MappingKV[str, ts.JsonPayload | None],
    ) -> None:
        """Initialize the shared base error state without subclass metaprogramming."""
        self.args = (message,)
        self.message = message
        self.error_code = error_code
        final_kwargs_dict: tb.JsonDict = {}
        for source_value in (merged_kwargs, context, extra_kwargs):
            if source_value is None:
                continue
            try:
                source_dict = FlextRuntime.normalize_metadata_input_mapping(
                    source_value
                )
            except ce.EXC_PYDANTIC_TYPE_VALUE:
                continue
            if not source_dict:
                continue
            for key, value in source_dict.items():
                if value is not None:
                    final_kwargs_dict[key] = FlextRuntime.normalize_to_metadata(value)
        final_kwargs = mc.ConfigMap.model_validate(final_kwargs_dict)
        self.correlation_id = (
            f"exc_{uuid.uuid4().hex[:8]}"
            if auto_correlation and (not correlation_id)
            else correlation_id
        )
        self.metadata = type(self)._normalize_metadata(metadata, final_kwargs.root)
        self.timestamp = time.time()
        self.auto_log = auto_log

    @override
    def __str__(self) -> str:
        """Return string representation with error code if present."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


__all__: list[str] = ["FlextBaseErrorStateMixin"]
