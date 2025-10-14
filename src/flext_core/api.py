"""Unified facade for the flext-core ecosystem.

This module provides FlextCore, a unified facade that exposes all flext-core
components through a single import while inheriting foundational behavior
from FlextBase.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import NotRequired, TypedDict, Unpack

from pydantic import SecretStr

from flext_core.__version__ import __version__, __version_info__
from flext_core.base import FlextBase
from flext_core.container import FlextContainer
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

    class FlextConfigKwargs(TypedDict, total=False):
        """Type definition for FlextConfig initialization parameters.

        All fields are optional since FlextConfig provides defaults.
        """

        app_name: NotRequired[str]
        version: NotRequired[str]
        debug: NotRequired[bool]
        trace: NotRequired[bool]
        log_level: NotRequired[str]
        json_output: NotRequired[bool]
        include_source: NotRequired[bool]
        structured_output: NotRequired[bool]
        log_verbosity: NotRequired[str]
        include_context: NotRequired[bool]
        include_correlation_id: NotRequired[bool]
        log_file: NotRequired[str | None]
        log_file_max_size: NotRequired[int]
        log_file_backup_count: NotRequired[int]
        console_enabled: NotRequired[bool]
        console_color_enabled: NotRequired[bool]
        mask_sensitive_data: NotRequired[bool]
        database_url: NotRequired[str | None]
        database_pool_size: NotRequired[int]
        cache_ttl: NotRequired[int]
        cache_max_size: NotRequired[int]
        secret_key: NotRequired[SecretStr | None]
        api_key: NotRequired[SecretStr | None]
        max_retry_attempts: NotRequired[int]
        timeout_seconds: NotRequired[int]
        dispatcher_auto_context: NotRequired[bool]
        dispatcher_timeout_seconds: NotRequired[int]
        dispatcher_enable_metrics: NotRequired[bool]
        dispatcher_enable_logging: NotRequired[bool]
        circuit_breaker_threshold: NotRequired[int]
        rate_limit_max_requests: NotRequired[int]
        rate_limit_window_seconds: NotRequired[float]
        enable_timeout_executor: NotRequired[bool]
        executor_workers: NotRequired[int]
        retry_delay: NotRequired[float]
        enable_caching: NotRequired[bool]
        enable_metrics: NotRequired[bool]
        enable_tracing: NotRequired[bool]
        max_workers: NotRequired[int]
        max_batch_size: NotRequired[int]
        max_name_length: NotRequired[int]
        min_phone_digits: NotRequired[int]
        validation_timeout_ms: NotRequired[int]
        validation_strict_mode: NotRequired[bool]

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
    def create_config(
        cls, **kwargs: Unpack[FlextCore.FlextConfigKwargs]
    ) -> FlextBase.Config:
        """Create a configuration instance with optional overrides.

        Args:
            **kwargs: Configuration overrides (accepts any FlextConfig field)

        Returns:
            FlextConfig: Configured instance

        """
        return FlextBase.Config(**kwargs)

    @classmethod
    def get_container(cls) -> FlextContainer:
        """Get the global dependency injection container.

        Returns:
            FlextContainer: Global container instance

        """
        return FlextBase.Container.get_global()


__all__ = [
    "FlextCore",
]
