"""Timeout helpers used by ``FlextDispatcher``.

Expose ``TimeoutEnforcer`` to provide deterministic timeout enforcement for
dispatcher-managed handlers. The helper keeps executor configuration isolated
from orchestration code while retaining the same behavior as the consolidated
dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures

from flext_core.constants import c
from flext_core.typings import t


class TimeoutEnforcer:
    """Manage timeout enforcement and dispatcher thread-pool execution."""

    def __init__(
        self,
        *,
        use_timeout_executor: bool,
        executor_workers: int,
    ) -> None:
        """Initialize the timeout coordinator.

        Args:
            use_timeout_executor: Whether to route handler execution through a
                dedicated timeout executor
            executor_workers: Number of worker threads to provision when the
                executor is enabled

        """
        super().__init__()
        self._use_timeout_executor = use_timeout_executor
        self._executor_workers = max(executor_workers, c.Reliability.RETRY_COUNT_MIN)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def should_use_executor(self) -> bool:
        """Return ``True`` when a dedicated timeout executor is enabled.

        Returns:
            bool: True if timeout executor is enabled, False otherwise.

        """
        return self._use_timeout_executor

    def reset_executor(self) -> None:
        """Reset executor after shutdown to allow lazy re-creation."""
        self._executor = None

    def resolve_workers(self) -> int:
        """Return the configured worker count for the dispatcher executor.

        Returns:
            int: Number of worker threads configured for the executor.

        """
        return self._executor_workers

    def ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create the shared executor on demand with lazy initialization.

        Returns:
            ThreadPoolExecutor: The shared thread pool executor instance.

        """
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._executor_workers,
                thread_name_prefix=c.Dispatcher.THREAD_NAME_PREFIX,
            )
        return self._executor

    def get_executor_status(self) -> t.Types.ConfigurationDict:
        """Return executor status metadata for diagnostics and metrics.

        Returns:
            ConfigurationDict: Dictionary with executor status information.

        """
        return {
            "executor_active": self._executor is not None,
            "executor_workers": self._executor_workers if self._executor else 0,
        }

    def cleanup(self) -> None:
        """Release executor resources used by dispatcher timeout handling."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


__all__ = ["TimeoutEnforcer"]
