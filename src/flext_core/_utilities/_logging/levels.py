"""Logging threshold admission, level math and elected-level resolution.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging

import structlog

from flext_core import c, p, t
from flext_core._utilities._logging.state import FlextUtilitiesLoggingState


class FlextUtilitiesLoggingLevels(FlextUtilitiesLoggingState):
    """Own log severity validation, threshold admission and level resolution."""

    @classmethod
    def log_threshold(cls) -> int:
        """Return the elected minimum severity applied before enrichment.

        Raises:
            AttributeError: If no threshold has been elected yet.

        """
        threshold = cls.current().threshold
        if threshold is None:
            msg = "logging threshold is not configured"
            raise AttributeError(msg)
        return threshold

    @staticmethod
    def level_number(level: int | str) -> int:
        """Validate a standard severity and return its native number.

        Returns:
            The native numeric severity.

        Raises:
            ValueError: If the severity is not a supported standard level.

        """
        value = (
            level
            if isinstance(level, int)
            else logging.getLevelNamesMapping().get(level.strip().upper())
        )
        if value is None or logging.getLevelName(value) not in c.LogLevel:
            msg = f"Invalid log level: {level}"
            raise ValueError(msg)
        return value

    @staticmethod
    def resolve_effective_log_level(
        *,
        debug: bool,
        log_level: t.LogLevel,
        log_level_explicit: bool = True,
    ) -> t.LogLevel:
        """Keep trace detail independent of the configured logging threshold.

        Returns:
            The effective severity without changing trace detail.

        """
        return c.LogLevel.DEBUG if debug and not log_level_explicit else log_level

    @staticmethod
    def drop_below_threshold(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: p.LoggingEvent,
        *,
        threshold: int | None = None,
    ) -> p.LoggingEvent:
        """Apply the elected threshold before native enrichment and hooks.

        Returns:
            The admitted event.

        Raises:
            DropEvent: If the event falls below the elected threshold.

        """
        _ = logger, method_name
        level = logging.getLevelNamesMapping()[str(event_dict["level"]).upper()]
        admitted_level = (
            FlextUtilitiesLoggingLevels.log_threshold()
            if threshold is None
            else threshold
        )
        if level < admitted_level:
            raise structlog.DropEvent
        return event_dict

    @classmethod
    def resolve_logging_settings(
        cls,
        *,
        runtime_settings: p.LoggingSettings | None = None,
        log_level: str | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
        logging_profile: str | None = None,
    ) -> p.LoggingSettings:
        """Validate optional overrides without I/O or process-wide mutation.

        Returns:
            The validated logging settings snapshot.

        """
        current = (
            runtime_settings if runtime_settings is not None else cls.logging_settings()
        )
        overrides: dict[str, p.SettingsOverride | None] = {
            key: value
            for key, value in (
                ("log_level", log_level),
                ("debug", debug),
                ("trace", trace),
                ("logging_profile", logging_profile),
            )
            if value is not None
        }
        selected = current.clone(**overrides)
        level = cls.resolve_effective_log_level(
            log_level=c.LogLevel(
                logging.getLevelName(cls.level_number(selected.log_level))
            ),
            debug=selected.debug,
            log_level_explicit=(
                log_level is not None
                or (debug is None and "log_level" in current.model_fields_set)
            ),
        )
        return selected.clone(log_level=level.value)


__all__: list[str] = ["FlextUtilitiesLoggingLevels"]
