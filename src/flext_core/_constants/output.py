"""FlextConstantsOutput - output/log message authorities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsOutput:
    """Output and logging message constants for core runtime flows."""

    _TEMPLATE_REGISTERED: Final[str] = "Registered {subject}"
    _TEMPLATE_FAILED_TO: Final[str] = "Failed to {operation}"

    LOG_REGISTERED_AUTO_DISCOVERY_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="auto-discovery handler"
    )
    LOG_REGISTERED_EVENT_SUBSCRIBER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="event subscriber"
    )
    LOG_REGISTERED_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(subject="handler")
    LOG_HANDLER_EXECUTION_FAILED: Final[str] = "Handler execution failed"
    LOG_HANDLER_PIPELINE_FAILURE: Final[str] = "Critical handler pipeline failure"
    LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION: Final[str] = (
        "Tracked operation raised expected exception"
    )
    LOG_SERVICE_INITIALIZED: Final[str] = "Service initialized"
    LOG_RUNTIME_BOOTSTRAP_OPTIONS_LOAD_FAILED: Final[str] = _TEMPLATE_FAILED_TO.format(
        operation="load runtime bootstrap options"
    )
    LOG_SERVICE_REGISTRATION_FAILED: Final[str] = "Service registration failed"


__all__ = ["FlextConstantsOutput"]
