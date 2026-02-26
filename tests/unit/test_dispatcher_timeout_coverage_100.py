"""Real tests to achieve 100% timeout dispatcher coverage - no mocks.

Module: flext_core._dispatcher.timeout
Scope: TimeoutEnforcer - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _dispatcher/timeout.py.

Uses Python 3.13 patterns, advanced pytest techniques, and aggressive
parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

import pytest
from flext_core._dispatcher.timeout import TimeoutEnforcer


@dataclass(frozen=True, slots=True)
class TimeoutEnforcerScenario:
    """TimeoutEnforcer test scenario."""

    name: str
    use_timeout_executor: bool
    executor_workers: int
    expected_workers: int
    should_use_executor: bool


class TimeoutEnforcerScenarios:
    """Centralized timeout enforcer test scenarios."""

    INIT_SCENARIOS: ClassVar[list[TimeoutEnforcerScenario]] = [
        TimeoutEnforcerScenario(
            name="executor_enabled_multiple_workers",
            use_timeout_executor=True,
            executor_workers=5,
            expected_workers=5,
            should_use_executor=True,
        ),
        TimeoutEnforcerScenario(
            name="executor_enabled_single_worker",
            use_timeout_executor=True,
            executor_workers=1,
            expected_workers=1,
            should_use_executor=True,
        ),
        TimeoutEnforcerScenario(
            name="executor_disabled",
            use_timeout_executor=False,
            executor_workers=3,
            expected_workers=3,
            should_use_executor=False,
        ),
        TimeoutEnforcerScenario(
            name="executor_enabled_zero_workers_clamped",
            use_timeout_executor=True,
            executor_workers=0,
            expected_workers=1,  # Clamped to minimum 1
            should_use_executor=True,
        ),
        TimeoutEnforcerScenario(
            name="executor_enabled_negative_workers_clamped",
            use_timeout_executor=True,
            executor_workers=-5,
            expected_workers=1,  # Clamped to minimum 1
            should_use_executor=True,
        ),
        TimeoutEnforcerScenario(
            name="executor_enabled_large_worker_count",
            use_timeout_executor=True,
            executor_workers=100,
            expected_workers=100,
            should_use_executor=True,
        ),
    ]


class TestTimeoutEnforcerInitialization:
    """Test TimeoutEnforcer initialization."""

    @pytest.mark.parametrize(
        "scenario",
        TimeoutEnforcerScenarios.INIT_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_initialization(
        self,
        scenario: TimeoutEnforcerScenario,
    ) -> None:
        """Test TimeoutEnforcer initialization with various scenarios."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=scenario.use_timeout_executor,
            executor_workers=scenario.executor_workers,
        )

        assert enforcer.should_use_executor() == scenario.should_use_executor
        assert enforcer.resolve_workers() == scenario.expected_workers
        status = enforcer.get_executor_status()
        # executor_workers is 0 when executor is None, otherwise it's the configured value
        assert status["executor_active"] is False  # Not created yet
        assert status["executor_workers"] == 0  # Returns 0 when executor is None


class TestTimeoutEnforcerExecutorManagement:
    """Test TimeoutEnforcer executor management."""

    def test_ensure_executor_creates_on_demand(self) -> None:
        """Test ensure_executor creates executor on demand."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=3,
        )

        # Initially no executor
        assert enforcer.get_executor_status()["executor_active"] is False

        # Create executor
        executor = enforcer.ensure_executor()
        assert executor is not None
        assert enforcer.get_executor_status()["executor_active"] is True

        # Second call returns same executor
        executor2 = enforcer.ensure_executor()
        assert executor2 is executor

    def test_ensure_executor_with_executor_disabled(self) -> None:
        """Test ensure_executor can be called even when executor is disabled."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=False,
            executor_workers=3,
        )

        # Can still create executor (method doesn't check flag)
        executor = enforcer.ensure_executor()
        assert executor is not None
        assert enforcer.get_executor_status()["executor_active"] is True

    def test_reset_executor_clears_executor(self) -> None:
        """Test reset_executor clears the executor."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=2,
        )

        # Create executor
        enforcer.ensure_executor()
        assert enforcer.get_executor_status()["executor_active"] is True

        # Reset executor
        enforcer.reset_executor()
        assert enforcer.get_executor_status()["executor_active"] is False

        # Can recreate after reset
        executor = enforcer.ensure_executor()
        assert executor is not None
        assert enforcer.get_executor_status()["executor_active"] is True

    def test_get_executor_status_before_creation(self) -> None:
        """Test get_executor_status before executor creation."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=5,
        )

        status = enforcer.get_executor_status()
        assert status["executor_active"] is False
        # executor_workers returns 0 when executor is None (not yet created)
        assert status["executor_workers"] == 0

    def test_get_executor_status_after_creation(self) -> None:
        """Test get_executor_status after executor creation."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=7,
        )

        enforcer.ensure_executor()
        status = enforcer.get_executor_status()
        assert status["executor_active"] is True
        assert status["executor_workers"] == 7

    def test_get_executor_status_after_reset(self) -> None:
        """Test get_executor_status after reset."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=4,
        )

        enforcer.ensure_executor()
        enforcer.reset_executor()
        status = enforcer.get_executor_status()
        assert status["executor_active"] is False
        assert status["executor_workers"] == 0


class TestTimeoutEnforcerCleanup:
    """Test TimeoutEnforcer cleanup."""

    def test_cleanup_with_executor(self) -> None:
        """Test cleanup with active executor."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=2,
        )

        enforcer.ensure_executor()
        assert enforcer.get_executor_status()["executor_active"] is True

        # Cleanup should shutdown executor
        enforcer.cleanup()
        assert enforcer.get_executor_status()["executor_active"] is False

        # Can recreate after cleanup
        executor = enforcer.ensure_executor()
        assert executor is not None

    def test_cleanup_without_executor(self) -> None:
        """Test cleanup when no executor exists."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=3,
        )

        # Cleanup should not raise error when no executor
        enforcer.cleanup()
        assert enforcer.get_executor_status()["executor_active"] is False

    def test_cleanup_multiple_times(self) -> None:
        """Test cleanup can be called multiple times safely."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=2,
        )

        enforcer.ensure_executor()
        enforcer.cleanup()
        enforcer.cleanup()  # Second cleanup should not raise
        assert enforcer.get_executor_status()["executor_active"] is False

    def test_cleanup_after_reset(self) -> None:
        """Test cleanup after reset."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=3,
        )

        enforcer.ensure_executor()
        enforcer.reset_executor()
        enforcer.cleanup()  # Should not raise even though executor is None
        assert enforcer.get_executor_status()["executor_active"] is False


class TestTimeoutEnforcerEdgeCases:
    """Test TimeoutEnforcer edge cases."""

    def test_executor_thread_name_prefix(self) -> None:
        """Test executor uses correct thread name prefix."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=1,
        )

        executor = enforcer.ensure_executor()
        # Verify executor was created (thread name prefix is set internally)
        assert executor is not None
        # The thread_name_prefix is set internally, we can't directly verify it
        # but we can verify the executor works
        future = executor.submit(lambda: 42)
        assert future.result() == 42

    def test_executor_submit_task(self) -> None:
        """Test executor can submit and execute tasks."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=2,
        )

        executor = enforcer.ensure_executor()

        # Submit multiple tasks
        futures = [executor.submit(lambda x: x * 2, i) for i in range(5)]
        results = [f.result() for f in futures]

        assert results == [0, 2, 4, 6, 8]

    def test_executor_concurrent_execution(self) -> None:
        """Test executor handles concurrent execution."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=True,
            executor_workers=3,
        )

        executor = enforcer.ensure_executor()

        def slow_task(delay: float) -> float:
            time.sleep(delay)
            return delay

        # Submit tasks that should execute concurrently
        start = time.time()
        futures = [executor.submit(slow_task, 0.1) for _ in range(3)]
        results = [f.result() for f in futures]
        elapsed = time.time() - start

        assert all(r == pytest.approx(0.1) for r in results)
        # Should complete faster than sequential (3 * 0.1 = 0.3s)
        # but account for overhead
        assert elapsed < 0.5


__all__ = [
    "TestTimeoutEnforcerCleanup",
    "TestTimeoutEnforcerEdgeCases",
    "TestTimeoutEnforcerExecutorManagement",
    "TestTimeoutEnforcerInitialization",
    "TimeoutEnforcerScenarios",
]
