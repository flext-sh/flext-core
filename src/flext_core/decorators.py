"""Decorator utilities for FLEXT ecosystem.

SIMPLIFIED: Simple stub decorators for examples and development.
PATTERN: Minimal implementations that pass MyPy strict mode.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core.typings import T


class FlextDecorators:
    """Decorator utilities for FLEXT ecosystem."""

    @staticmethod
    def retry(max_attempts: int = 3) -> Callable[[T], T]:
        """Retry decorator stub."""

        def decorator(func: T) -> T:
            _ = max_attempts  # Use parameter to avoid unused argument warning
            return func

        return decorator

    @staticmethod
    def timeout(seconds: int) -> Callable[[T], T]:
        """Timeout decorator stub."""

        def decorator(func: T) -> T:
            _ = seconds  # Use parameter to avoid unused argument warning
            return func

        return decorator

    @staticmethod
    def cache(ttl: int = 300) -> Callable[[T], T]:
        """Cache decorator stub."""

        def decorator(func: T) -> T:
            _ = ttl  # Use parameter to avoid unused argument warning
            return func

        return decorator

    class Reliability:
        """Reliability decorators for examples."""

        @staticmethod
        def safe_result(func: T) -> T:
            """Safe result decorator stub."""
            return func

        @staticmethod
        def circuit_breaker(_failure_threshold: int = 5) -> Callable[[T], T]:
            """Circuit breaker decorator stub."""

            def decorator(func: T) -> T:
                return func

            return decorator

        @staticmethod
        def bulkhead(_max_concurrent: int = 10) -> Callable[[T], T]:
            """Bulkhead decorator stub."""

            def decorator(func: T) -> T:
                return func

            return decorator

    class Observability:
        """Observability decorators for examples."""

        @staticmethod
        def trace(func: T) -> T:
            """Trace decorator stub."""
            return func

        @staticmethod
        def metrics(_name: str) -> Callable[[T], T]:
            """Metrics decorator stub."""

            def decorator(func: T) -> T:
                return func

            return decorator

        @staticmethod
        def log_execution(func: T) -> T:
            """Log execution decorator stub."""
            return func

    class Performance:
        """Performance decorators for examples."""

        @staticmethod
        def cached(_ttl: int = 300) -> Callable[[T], T]:
            """Cached decorator stub."""

            def decorator(func: T) -> T:
                return func

            return decorator

        @staticmethod
        def monitored(func: T) -> T:
            """Monitored decorator stub."""
            return func


__all__ = ["FlextDecorators"]
