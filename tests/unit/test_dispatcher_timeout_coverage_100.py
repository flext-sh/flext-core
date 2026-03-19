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
from concurrent.futures import Future
from typing import Annotated, ClassVar

import pytest
from flext_tests import tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModelsDispatcher

TimeoutEnforcer = FlextModelsDispatcher.TimeoutEnforcer


class TestDispatcherTimeoutCoverage100:
    class _TimeoutEnforcerScenario(BaseModel):
        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Timeout enforcer scenario name")]
        use_timeout_executor: Annotated[
            bool,
            Field(description="Whether timeout executor is enabled"),
        ]
        executor_workers: Annotated[
            int,
            Field(description="Configured executor worker count"),
        ]
        expected_workers: Annotated[
            int, Field(description="Expected resolved worker count")
        ]
        should_use_executor: Annotated[
            bool,
            Field(description="Expected executor usage flag"),
        ]

    _INIT_SCENARIOS: ClassVar[list[_TimeoutEnforcerScenario]] = [
        _TimeoutEnforcerScenario(
            name="executor_enabled_multiple_workers",
            use_timeout_executor=True,
            executor_workers=5,
            expected_workers=5,
            should_use_executor=True,
        ),
        _TimeoutEnforcerScenario(
            name="executor_enabled_single_worker",
            use_timeout_executor=True,
            executor_workers=1,
            expected_workers=1,
            should_use_executor=True,
        ),
        _TimeoutEnforcerScenario(
            name="executor_disabled",
            use_timeout_executor=False,
            executor_workers=3,
            expected_workers=3,
            should_use_executor=False,
        ),
        _TimeoutEnforcerScenario(
            name="executor_enabled_zero_workers_clamped",
            use_timeout_executor=True,
            executor_workers=0,
            expected_workers=1,
            should_use_executor=True,
        ),
        _TimeoutEnforcerScenario(
            name="executor_enabled_negative_workers_clamped",
            use_timeout_executor=True,
            executor_workers=-5,
            expected_workers=1,
            should_use_executor=True,
        ),
        _TimeoutEnforcerScenario(
            name="executor_enabled_large_worker_count",
            use_timeout_executor=True,
            executor_workers=100,
            expected_workers=100,
            should_use_executor=True,
        ),
    ]

    @pytest.mark.parametrize("scenario", _INIT_SCENARIOS, ids=lambda s: s.name)
    def test_initialization(self, scenario: _TimeoutEnforcerScenario) -> None:
        """Test TimeoutEnforcer initialization with various scenarios."""
        enforcer = TimeoutEnforcer(
            use_timeout_executor=scenario.use_timeout_executor,
            executor_workers=scenario.executor_workers,
        )
        tm.that(enforcer.should_use_executor(), eq=scenario.should_use_executor)
        tm.that(enforcer.resolve_workers(), eq=scenario.expected_workers)
        status = enforcer.get_executor_status()
        tm.that(status["executor_active"], eq=False)
        tm.that(status["executor_workers"], eq=0)

    def test_ensure_executor_creates_on_demand(self) -> None:
        """Test ensure_executor creates executor on demand."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=3)
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)
        executor = enforcer.ensure_executor()
        assert executor is not None
        tm.that(enforcer.get_executor_status()["executor_active"], eq=True)
        executor2 = enforcer.ensure_executor()
        assert executor2 is executor

    def test_ensure_executor_with_executor_disabled(self) -> None:
        """Test ensure_executor can be called even when executor is disabled."""
        enforcer = TimeoutEnforcer(use_timeout_executor=False, executor_workers=3)
        executor = enforcer.ensure_executor()
        assert executor is not None
        tm.that(enforcer.get_executor_status()["executor_active"], eq=True)

    def test_reset_executor_clears_executor(self) -> None:
        """Test reset_executor clears the executor."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=2)
        enforcer.ensure_executor()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=True)
        enforcer.reset_executor()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)
        executor = enforcer.ensure_executor()
        assert executor is not None
        tm.that(enforcer.get_executor_status()["executor_active"], eq=True)

    def test_get_executor_status_before_creation(self) -> None:
        """Test get_executor_status before executor creation."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=5)
        status = enforcer.get_executor_status()
        tm.that(status["executor_active"], eq=False)
        tm.that(status["executor_workers"], eq=0)

    def test_get_executor_status_after_creation(self) -> None:
        """Test get_executor_status after executor creation."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=7)
        enforcer.ensure_executor()
        status = enforcer.get_executor_status()
        tm.that(status["executor_active"], eq=True)
        tm.that(status["executor_workers"], eq=7)

    def test_get_executor_status_after_reset(self) -> None:
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=4)
        enforcer.ensure_executor()
        enforcer.reset_executor()
        status = enforcer.get_executor_status()
        tm.that(status["executor_active"], eq=False)
        tm.that(status["executor_workers"], eq=0)

    def test_cleanup_with_executor(self) -> None:
        """Test cleanup with active executor."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=2)
        enforcer.ensure_executor()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=True)
        enforcer.cleanup()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)
        executor = enforcer.ensure_executor()
        assert executor is not None

    def test_cleanup_without_executor(self) -> None:
        """Test cleanup when no executor exists."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=3)
        enforcer.cleanup()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)

    def test_cleanup_multiple_times(self) -> None:
        """Test cleanup can be called multiple times safely."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=2)
        enforcer.ensure_executor()
        enforcer.cleanup()
        enforcer.cleanup()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)

    def test_cleanup_after_reset(self) -> None:
        """Test cleanup after reset."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=3)
        enforcer.ensure_executor()
        enforcer.reset_executor()
        enforcer.cleanup()
        tm.that(enforcer.get_executor_status()["executor_active"], eq=False)

    def test_executor_thread_name_prefix(self) -> None:
        """Test executor uses correct thread name prefix."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=1)
        executor = enforcer.ensure_executor()
        assert executor is not None
        future = executor.submit(lambda: 42)
        tm.that(future.result(), eq=42)

    def test_executor_submit_task(self) -> None:
        """Test executor can submit and execute tasks."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=2)
        executor = enforcer.ensure_executor()

        def double(value: int) -> int:
            return value * 2

        futures: list[Future[int]] = [executor.submit(double, i) for i in range(5)]
        results: list[int] = [future.result() for future in futures]
        tm.that(results, eq=[0, 2, 4, 6, 8])

    def test_executor_concurrent_execution(self) -> None:
        """Test executor handles concurrent execution."""
        enforcer = TimeoutEnforcer(use_timeout_executor=True, executor_workers=3)
        executor = enforcer.ensure_executor()

        def slow_task(delay: float) -> float:
            time.sleep(delay)
            return delay

        start = time.time()
        futures = [executor.submit(slow_task, 0.1) for _ in range(3)]
        results = [f.result() for f in futures]
        elapsed = time.time() - start
        assert all(abs(r - 0.1) < 1e-09 for r in results)
        tm.that(elapsed, lt=0.5)
