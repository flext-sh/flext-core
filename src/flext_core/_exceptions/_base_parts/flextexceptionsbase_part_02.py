"""Exception base state behavior.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import c, m
from flext_core._exceptions import FlextExceptionsHelpers
from flext_core._exceptions._base_parts import FlextBaseErrorMetadataMixin

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core._typings import FlextTypesServices, FlextTypingBase


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

    def _initialize_base_state(
        self,
        message: str,
        *,
        error_code: str,
        options: m.ExceptionInitOptions,
        extra_kwargs: FlextTypingBase.MappingKV[
            str, FlextTypesServices.JsonPayload | None
        ],
    ) -> None:
        """Initialize the shared base error state without subclass metaprogramming."""
        self.args = (message,)
        self.message = message
        self.error_code = error_code
        final_kwargs_dict: FlextTypingBase.JsonDict = {}
        for source_value in (options.merged_kwargs, options.context, extra_kwargs):
            if source_value is None:
                continue
            final_kwargs_dict.update(
                FlextExceptionsHelpers.build_context_map(source_value, {})
            )
        final_kwargs = m.ConfigMap.model_validate(final_kwargs_dict)
        self.correlation_id = (
            f"exc_{uuid.uuid4().hex[:8]}"
            if options.auto_correlation and not options.correlation_id
            else options.correlation_id
        )
        self.metadata = type(self).normalize_metadata(
            options.metadata,
            final_kwargs.root,  # type: ignore
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


__all__: list[str] = ["FlextBaseErrorStateMixin"]
