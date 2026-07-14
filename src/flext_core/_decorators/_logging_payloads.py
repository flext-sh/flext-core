"""Structured logging payload builders for decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core._constants.base import FlextConstantsBase as cb
from flext_core._constants.infrastructure import FlextConstantsInfrastructure as ci
from flext_core._decorators._base import FlextDecoratorsBase

from flext_core._typings.base import FlextTypingBase as tb


class FlextDecoratorsLoggingPayloads(FlextDecoratorsBase):
    """Build structured log payloads for operation decorators."""

    @staticmethod
    def _start_log_payload(
        *, func_name: str, func_module: str, correlation_id: str | None
    ) -> tb.MutableJsonMapping:
        """Build structured operation-start log payload."""
        payload: tb.MutableJsonMapping = {
            "function": func_name,
            "func_module": func_module,
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id
        return payload

    @staticmethod
    def _success_log_payload(
        *,
        func_name: str,
        correlation_id: str | None,
        track_perf: bool,
        start_time: float,
    ) -> tb.MutableJsonMapping:
        """Build structured operation-success log payload."""
        payload: tb.MutableJsonMapping = {"function": func_name, "success": True}
        if correlation_id is not None:
            payload[ci.ContextKey.CORRELATION_ID] = correlation_id
        if track_perf:
            duration = time.perf_counter() - start_time
            payload["duration_ms"] = duration * cb.DEFAULT_SIZE
            payload[ci.MetadataKey.DURATION_SECONDS] = duration
        return payload


__all__: list[str] = ["FlextDecoratorsLoggingPayloads"]
