"""Exception base state behavior.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import c, m
from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
    FlextBaseErrorMetadataMixin,
)
from flext_core._runtime._metadata_validation import FlextRuntimeMetadataValidation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core._protocols import FlextProtocolsResult
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.services import FlextTypesServices


class FlextBaseErrorStateMixin(FlextBaseErrorMetadataMixin):
    message: str
    error_code: str
    correlation_id: str | None
    metadata: m.Metadata
    timestamp: float
    auto_log: bool
    args: FlextTypingBase.VariadicTuple[str]

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

    def _init_error_identity(self, message: str, error_code: str) -> None:
        """Initialize the identity fields of the shared base error state."""
        self.args = (message,)
        self.message = message
        self.error_code = error_code

    def _init_error_correlation(
        self,
        correlation_id: str | None,
        *,
        auto_correlation: bool,
    ) -> None:
        """Resolve and assign the correlation id of the shared base error state."""
        self.correlation_id = (
            f"exc_{uuid.uuid4().hex[:8]}"
            if auto_correlation and (not correlation_id)
            else correlation_id
        )

    @staticmethod
    def _merge_error_channel_sources(
        context: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None]
        | FlextProtocolsResult.HasModelDump
        | None,
        merged_kwargs: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ]
        | FlextProtocolsResult.HasModelDump
        | None,
        extra_kwargs: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ],
    ) -> m.ConfigMap:
        """Merge context, merged kwargs, and extra kwargs into one mapping.

        Returns:
            The resulting ``m.ConfigMap``.

        """
        final_kwargs_dict: FlextTypingBase.JsonDict = {}
        for source_value in (merged_kwargs, context, extra_kwargs):
            if source_value is None:
                continue
            try:
                source_dict = (
                    FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(
                        source_value,
                    )
                )
            except c.EXC_PYDANTIC_TYPE_VALUE:
                continue
            if not source_dict:
                continue
            for key, value in source_dict.items():
                if value is not None:
                    final_kwargs_dict[key] = (
                        FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                    )
        return m.ConfigMap.model_validate(final_kwargs_dict)

    def _init_error_channels(
        self,
        final_kwargs: m.ConfigMap,
        metadata: FlextProtocolsResult.HasModelDump | FlextTypingBase.JsonValue | None,
        *,
        auto_log: bool,
    ) -> None:
        """Assign metadata, timestamp, and logging behavior of the error state."""
        self.metadata = type(self).normalize_metadata(metadata, final_kwargs.root)
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


__all__: list[str] = ["FlextBaseErrorStateMixin"]
