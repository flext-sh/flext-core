"""Unified facade for the flext-core ecosystem.

This module provides FlextCore, a unified facade that exposes all flext-core
components through a single import while inheriting foundational behavior
from FlextBase.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.__version__ import __version__, __version_info__
from flext_core.base import FlextBase
from flext_core.loggings import FlextLogger


class FlextCore(FlextBase):
    """Unified facade for the flext-core ecosystem.

    Provides unified access to all flext-core components while inheriting
    foundational behavior from FlextBase.

    Features:
    - Unified access to all flext-core components
    - Version information and package metadata
    - Convenience factory methods for common patterns
    - Backward-compatible API for existing code
    - Integration with global dependency container

    Usage:
        >>> from flext_core import FlextCore
        >>>
        >>> core = FlextCore()
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
    # FACTORY METHODS (v0.9.9+ Enhancement)
    # =================================================================

    @classmethod
    def create_logger(cls, name: str) -> FlextLogger:
        """Create a logger instance for the given name.

        Args:
            name: Logger name (typically module __name__)

        Returns:
            FlextLogger: Configured logger instance

        """
        return FlextBase.Logger.create_module_logger(name)

    @classmethod
    def create_config(cls, **kwargs: object) -> FlextBase.Config:
        """Create a configuration instance with optional overrides.

        Args:
            **kwargs: Configuration overrides

        Returns:
            FlextConfig: Configured instance

        """
        return FlextBase.Config(**kwargs)

    @classmethod
    def get_container(cls) -> FlextBase.Container:
        """Get the global dependency injection container.

        Returns:
            FlextContainer: Global container instance

        """
        return FlextBase.Container.get_global()


__all__ = [
    "FlextCore",
]
