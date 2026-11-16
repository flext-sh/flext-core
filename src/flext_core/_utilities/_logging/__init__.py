# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging.async_writer import (
        FlextUtilitiesLoggingAsyncWriter,
    )
    from flext_core._utilities._logging.bound_logger import (
        FlextUtilitiesLoggingBoundLogger,
    )
    from flext_core._utilities._logging.composition import (
        FlextUtilitiesLoggingComposition,
    )
    from flext_core._utilities._logging.context import (
        FlextUtilitiesLoggingContextBinding,
    )
    from flext_core._utilities._logging.exception_info import (
        FlextUtilitiesLoggingExceptionInfo,
    )
    from flext_core._utilities._logging.handlers import FlextUtilitiesLoggingHandlers
    from flext_core._utilities._logging.levels import FlextUtilitiesLoggingLevels
    from flext_core._utilities._logging.processors import (
        FlextUtilitiesLoggingProcessors,
    )
    from flext_core._utilities._logging.state import FlextUtilitiesLoggingState
    from flext_core._utilities._logging.stream import FlextUtilitiesLoggingStream


__all__: tuple[str, ...] = (
    "FlextUtilitiesLoggingAsyncWriter",
    "FlextUtilitiesLoggingBoundLogger",
    "FlextUtilitiesLoggingComposition",
    "FlextUtilitiesLoggingContextBinding",
    "FlextUtilitiesLoggingExceptionInfo",
    "FlextUtilitiesLoggingHandlers",
    "FlextUtilitiesLoggingLevels",
    "FlextUtilitiesLoggingProcessors",
    "FlextUtilitiesLoggingState",
    "FlextUtilitiesLoggingStream",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextUtilitiesLoggingAsyncWriter": ".async_writer",
        "FlextUtilitiesLoggingBoundLogger": ".bound_logger",
        "FlextUtilitiesLoggingComposition": ".composition",
        "FlextUtilitiesLoggingContextBinding": ".context",
        "FlextUtilitiesLoggingExceptionInfo": ".exception_info",
        "FlextUtilitiesLoggingHandlers": ".handlers",
        "FlextUtilitiesLoggingLevels": ".levels",
        "FlextUtilitiesLoggingProcessors": ".processors",
        "FlextUtilitiesLoggingState": ".state",
        "FlextUtilitiesLoggingStream": ".stream",
    },
    public_exports=__all__,
    deferred_members={
        "FlextUtilitiesLoggingComposition": (
            "structlog",
            "_resolve_logger_factory",
            "compose_native_logger",
            "configure_structlog",
            "ensure_structlog_configured",
            "apply_log_level",
        ),
        "FlextUtilitiesLoggingContextBinding": (
            "_scope_keys",
            "logging_scope",
            "bind_context",
            "bind_global_context",
            "clear_global_context",
            "clear_scope",
            "unbind_global_context",
            "to_container_value",
            "to_container_context",
        ),
        "FlextUtilitiesLoggingExceptionInfo": (
            "resolve_logging_exception_info",
            "_exception_metadata",
            "_format_exception",
        ),
        "FlextUtilitiesLoggingHandlers": (
            "_handler_factory",
            "create_logging_handler",
        ),
        "FlextUtilitiesLoggingLevels": (
            "log_threshold",
            "level_number",
            "resolve_effective_log_level",
            "drop_below_threshold",
            "resolve_logging_settings",
        ),
        "FlextUtilitiesLoggingProcessors": (
            "level_based_context_filter",
            "_console_event",
            "_observe_event",
            "_build_structlog_processors",
        ),
        "FlextUtilitiesLoggingState": (
            "Snapshot",
            "_current",
            "current",
            "configure_logging_state",
            "logging_settings",
            "logging_options",
        ),
    },
)

defer_lazy_bases(
    __name__,
    {
        "FlextUtilitiesLoggingComposition": ((".", "FlextUtilitiesLoggingHandlers"),),
        "FlextUtilitiesLoggingHandlers": ((".", "FlextUtilitiesLoggingProcessors"),),
        "FlextUtilitiesLoggingLevels": ((".", "FlextUtilitiesLoggingState"),),
        "FlextUtilitiesLoggingProcessors": (
            (".", "FlextUtilitiesLoggingLevels"),
            (".", "FlextUtilitiesLoggingExceptionInfo"),
        ),
    },
)
