"""FlextSettingsCore — base fields (app metadata, logging flags, caching).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, Field, computed_field

from flext_core import FlextConstants as c, FlextTypes as t, FlextUtilities as u


class FlextSettingsCore:
    """Core settings fields shared by every FLEXT application."""

    app_name: Annotated[str, Field(description="Application name")] = c.DEFAULT_APP_NAME
    version: Annotated[str, Field(description="Application version")] = ""
    debug: Annotated[bool, Field(description="Enable debug mode")] = False
    trace: Annotated[bool, Field(description="Enable trace mode")] = False
    log_level: Annotated[
        c.LogLevel,
        BeforeValidator(lambda v: c.LogLevel(v.upper()) if isinstance(v, str) else v),
        Field(description="Log level"),
    ] = c.LogLevel.INFO
    async_logging: Annotated[
        bool,
        Field(
            description="Enable asynchronous buffered logging for performance",
        ),
    ] = True
    enable_caching: Annotated[bool, Field(description="Enable caching")] = (
        c.ASYNC_ENABLED
    )
    cache_ttl: Annotated[t.PositiveInt, Field(description="Cache TTL")] = c.CACHE_TTL

    @computed_field
    @property
    def effective_log_level(self) -> c.LogLevel:
        """Get effective log level based on debug/trace flags."""
        return u.resolve_effective_log_level(
            trace=self.trace,
            debug=self.debug,
            log_level=self.log_level,
        )


__all__: list[str] = ["FlextSettingsCore"]
