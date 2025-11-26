"""Timeout enforcement for FlextDispatcher.

Contains TimeoutEnforcer class extracted from FlextDispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures


class TimeoutEnforcer:
    """Manages timeout enforcement and thread pool execution."""

    def __init__(
        self,
        *,
        use_timeout_executor: bool,
        executor_workers: int,
    ) -> None:
        """Initialize timeout enforcer.

        Args:
            use_timeout_executor: Whether to use timeout executor
            executor_workers: Number of executor worker threads

        """
        self._use_timeout_executor = use_timeout_executor
        self._executor_workers = max(executor_workers, 1)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def should_use_executor(self) -> bool:
        """Check if timeout executor should be used."""
        return self._use_timeout_executor

    def reset_executor(self) -> None:
        """Reset executor (after shutdown)."""
        self._executor = None

    def resolve_workers(self) -> int:
        """Get the configured worker count."""
        return self._executor_workers

    def ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create the shared executor on demand (lazy initialization)."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._executor_workers,
                thread_name_prefix="flext-dispatcher",
            )
        return self._executor

    def get_executor_status(self) -> dict[str, object]:
        """Get executor status information."""
        return {
            "executor_active": self._executor is not None,
            "executor_workers": self._executor_workers if self._executor else 0,
        }

    def cleanup(self) -> None:
        """Cleanup executor resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


__all__ = ["TimeoutEnforcer"]
