"""Single typed owner of the elected process-wide logging state.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core import FlextSettings, m, p

if TYPE_CHECKING:
    from flext_core._utilities._logging.async_writer import (
        FlextUtilitiesLoggingAsyncWriter,
    )


class FlextUtilitiesLoggingState:
    """Hold elected settings, options, threshold and the shared async writer."""

    class Snapshot:
        """Mutable elected state snapshot outside the facade MRO."""

        def __init__(self) -> None:
            """Initialize an unelected state snapshot."""
            self.runtime_settings: p.LoggingSettings | None = None
            self.options: p.StructlogOptions | None = None
            self.threshold: int | None = None
            self.async_writer: FlextUtilitiesLoggingAsyncWriter | None = None

    _current: ClassVar[Snapshot | None] = None

    @staticmethod
    def current() -> FlextUtilitiesLoggingState.Snapshot:
        """Return the one elected state owned by this part.

        Returns:
            The process-wide logging state.

        """
        if FlextUtilitiesLoggingState._current is None:
            FlextUtilitiesLoggingState._current = FlextUtilitiesLoggingState.Snapshot()
        return FlextUtilitiesLoggingState._current

    @classmethod
    def selected_logging_settings(cls) -> p.LoggingSettings | None:
        """Return the explicitly elected logging settings."""
        _ = cls
        return FlextUtilitiesLoggingState.current().runtime_settings

    @classmethod
    def selected_logging_options(cls) -> p.StructlogOptions | None:
        """Return the elected composition, if configured."""
        _ = cls
        return FlextUtilitiesLoggingState.current().options

    @classmethod
    def configure_logging_state(
        cls,
        settings: p.LoggingSettings,
        options: p.StructlogOptions,
        threshold: int,
    ) -> None:
        """Publish the state owned by the shared logging configuration."""
        _ = cls
        state = FlextUtilitiesLoggingState.current()
        state.runtime_settings = settings
        state.options = options
        state.threshold = threshold

    @classmethod
    def logging_settings(cls) -> p.LoggingSettings:
        """Return the elected application provider, or the actual Core settings."""
        selected = cls.selected_logging_settings()
        return selected if selected is not None else FlextSettings.fetch_global()

    @classmethod
    def logging_options(cls) -> p.StructlogOptions:
        """Return the active composition for scoped lifecycle restoration."""
        selected = cls.selected_logging_options()
        return selected if selected is not None else m.StructlogOptions()


__all__: list[str] = ["FlextUtilitiesLoggingState"]
