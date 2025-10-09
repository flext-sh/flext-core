"""FlextCore - Unified facade for the complete flext-core ecosystem.

This thin facade exposes all flext-core components through a single import
while inheriting the foundational behaviour from :class:`flext_core.base.FlextBase`.
`FlextCore` keeps the long-standing public API intact and is the recommended
entry point for examples, scripts, and dependent projects that expect the
classic facade semantics (e.g. ``FlextCore.Result[T]``).

The implementation now delegates namespace wiring to :class:`FlextBase`, which
provides ready-to-extend nested classes for constants, models, protocols,
handlers, utilities, and more. Subprojects can subclass :class:`FlextBase`
directly when they need domain-specific extensions, while ``FlextCore``
remains the curated façade for general consumption. ``FlextBase`` is therefore
an extension point—not a replacement—for domain libraries that want to expand
patterns without modifying the facade itself.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any, cast

from flext_core.__version__ import __version__, __version_info__
from flext_core.base import FlextBase
from flext_core.container import FlextContainer


class FlextCore(FlextBase):
    """Unified facade for complete flext-core ecosystem integration.

    This facade provides:
    - All FlextBase functionality (railway pattern, DI, logging, context)
    - Version information for package introspection
    - Additional convenience factory methods for common patterns
    - Streamlined API for reducing boilerplate in examples and services
    """

    # =================================================================
    # VERSION INFORMATION (v0.9.9+ Enhancement)
    # =================================================================
    # Direct access to version information through FlextCore facade

    version: str = __version__
    version_info: tuple[int | str, ...] = __version_info__

    def __init__(self) -> None:
        """Initialise the unified core facade with base helpers ready."""
        super().__init__()

    # =================================================================
    # CONVENIENCE FACTORY METHODS (v0.9.9+ Enhancement)
    # =================================================================
    # Additional shortcuts for common patterns to reduce boilerplate

    @classmethod
    def create_logger(cls, name: str | None = None) -> FlextBase.Logger:
        """Create a configured logger instance.

        Args:
            name: Logger name (defaults to "flext-core")

        Returns:
            FlextLogger: Configured logger instance

        Example:
            >>> logger = FlextCore.create_logger("my-service")
            >>> logger.info("Service started")

        """
        return cls.Logger(name or "flext-core")

    @classmethod
    def create_config(cls, **overrides: object) -> FlextBase.Config:
        """Create configuration with optional overrides.

        Args:
            **overrides: Configuration field overrides

        Returns:
            FlextConfig: Configuration instance

        Example:
            >>> config = FlextCore.create_config(debug=True, log_level="DEBUG")

        """
        return cls.Config(**cast("Any", overrides))

    @classmethod
    def get_container(cls) -> FlextContainer:
        """Get the global dependency injection container.

        Returns:
            FlextContainer: Global container singleton

        Example:
            >>> container = FlextCore.get_container()
            >>> container.register("logger", logger)

        """
        return cls.Container.get_global()


__all__ = [
    "FlextCore",
]
