"""Structured logging payload builders for decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._decorators import FlextDecoratorsBase


class FlextDecoratorsLoggingPayloads(FlextDecoratorsBase):
    """Build structured log payloads for operation decorators."""

    @staticmethod
    def _start_log_payload(
        *,
        func_name: str,
        func_module: str,
        correlation_id: str | None,
    ) -> flext_core.t.MutableJsonMapping:
        """Build structured operation-start log payload.

        Returns:
            The resulting ``tb.MutableJsonMapping``.

        """
        payload: flext_core.t.MutableJsonMapping = {
            "function": func_name,
            "func_module": func_module,
        }
        if correlation_id:
            payload[flext_core.c.ContextKey.CORRELATION_ID] = correlation_id
        return payload

    @staticmethod
    def _success_log_payload(
        *,
        func_name: str,
        correlation_id: str | None,
        track_perf: bool,
        start_time: float,
    ) -> flext_core.t.MutableJsonMapping:
        """Build structured operation-success log payload.

        Returns:
            The resulting ``tb.MutableJsonMapping``.

        """
        payload: flext_core.t.MutableJsonMapping = {
            "function": func_name,
            "success": True,
        }
        if correlation_id is not None:
            payload[flext_core.c.ContextKey.CORRELATION_ID] = correlation_id
        if track_perf:
            duration = time.perf_counter() - start_time
            payload["duration_ms"] = duration * flext_core.c.MS_PER_SECOND
            payload[flext_core.c.MetadataKey.DURATION_SECONDS] = duration
        return payload


__all__: list[str] = ["FlextDecoratorsLoggingPayloads"]
