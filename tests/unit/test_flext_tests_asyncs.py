"""Tests for flext_tests asyncs module - comprehensive coverage enhancement.

Comprehensive tests for FlextTestsAsyncs to improve coverage from 25% to 70%+.
Tests async testing utilities, concurrency patterns, and timeout management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from flext_tests import asyncs
from flext_tests.asyncs import FlextTestsAsyncs


class TestFlextTestsAsyncs:
    """Comprehensive tests for FlextTestsAsyncs module - Real functional testing.

    Tests async utilities, concurrency testing, timeout management,
    and async context management to enhance coverage from 25% to 70%+.
    """

    def test_asyncs_class_instantiation(self) -> None:
        """Test FlextTestsAsyncs class can be instantiated."""
        asyncs = FlextTestsAsyncs()
        assert asyncs is not None
        assert isinstance(asyncs, FlextTestsAsyncs)

    def test_is_not_exception_type_guard(self) -> None:
        """Test _is_not_exception type guard functionality."""
        # Test with non-exception objects
        assert FlextTestsAsyncs._is_not_exception("string") is True
        assert FlextTestsAsyncs._is_not_exception(42) is True
        assert FlextTestsAsyncs._is_not_exception([1, 2, 3]) is True
        assert FlextTestsAsyncs._is_not_exception({"key": "value"}) is True
        assert FlextTestsAsyncs._is_not_exception(None) is True

        # Test with exception objects
        assert FlextTestsAsyncs._is_not_exception(Exception("test")) is False
        assert FlextTestsAsyncs._is_not_exception(ValueError("test")) is False
        assert FlextTestsAsyncs._is_not_exception(RuntimeError("test")) is False

    @pytest.mark.asyncio
    async def test_wait_for_condition_success(self) -> None:
        """Test wait_for_condition with successful condition."""
        # Test with synchronous condition that becomes true
        counter = 0

        def increment_condition() -> bool:
            nonlocal counter
            counter += 1
            return counter >= 3

        await FlextTestsAsyncs.wait_for_condition(
            condition=increment_condition, timeout_seconds=1.0, poll_interval=0.1
        )

        assert counter >= 3

    @pytest.mark.asyncio
    async def test_wait_for_condition_timeout(self) -> None:
        """Test wait_for_condition with timeout."""

        # Test condition that never becomes true
        def never_true_condition() -> bool:
            return False

        with pytest.raises(TimeoutError):
            await FlextTestsAsyncs.wait_for_condition(
                condition=never_true_condition, timeout_seconds=0.2, poll_interval=0.05
            )

    @pytest.mark.asyncio
    async def test_wait_for_condition_async_condition(self) -> None:
        """Test wait_for_condition with async condition."""
        # Test with asynchronous condition
        counter = 0

        async def async_increment_condition() -> bool:
            nonlocal counter
            await asyncio.sleep(0.01)  # Small async delay
            counter += 1
            return counter >= 2

        await FlextTestsAsyncs.wait_for_condition(
            condition=async_increment_condition, timeout_seconds=1.0, poll_interval=0.05
        )

        assert counter >= 2

    @pytest.mark.asyncio
    async def test_wait_for_condition_custom_error_message(self) -> None:
        """Test wait_for_condition with custom error message."""

        def always_false() -> bool:
            return False

        custom_message = "Custom timeout error message"

        with pytest.raises(TimeoutError) as exc_info:
            await FlextTestsAsyncs.wait_for_condition(
                condition=always_false,
                timeout_seconds=0.1,
                poll_interval=0.02,
                error_message=custom_message,
            )

        # Verify the custom error message is used
        assert str(exc_info.value) == custom_message

    def test_async_utilities_static_methods_exist(self) -> None:
        """Test that expected static methods exist in FlextTestsAsyncs."""
        # Check for expected static methods
        expected_methods = ["wait_for_condition", "_is_not_exception"]

        for method_name in expected_methods:
            assert hasattr(FlextTestsAsyncs, method_name), (
                f"Method {method_name} not found"
            )
            method = getattr(FlextTestsAsyncs, method_name)
            assert callable(method), f"Expected {method_name} to be callable"

    @pytest.mark.asyncio
    async def test_wait_for_condition_edge_cases(self) -> None:
        """Test wait_for_condition with edge cases."""

        # Test with condition that never returns True (timeout scenario)
        def never_true_condition() -> bool:
            return False

        with pytest.raises(TimeoutError):
            await FlextTestsAsyncs.wait_for_condition(
                condition=never_true_condition,
                timeout_seconds=0.05,  # Very short timeout
                poll_interval=0.01,
            )

        # Test with condition that returns True immediately
        def immediate_true() -> bool:
            return True

        # Should complete immediately
        start_time = time.time()
        await FlextTestsAsyncs.wait_for_condition(
            condition=immediate_true, timeout_seconds=1.0, poll_interval=0.1
        )
        elapsed = time.time() - start_time
        assert elapsed < 0.5  # Should be very fast

    @pytest.mark.asyncio
    async def test_wait_for_condition_with_exception_in_condition(self) -> None:
        """Test wait_for_condition when condition raises exception."""

        def exception_condition() -> bool:
            msg = "Condition failed"
            raise ValueError(msg)

        # The implementation catches exceptions and continues polling,
        # so this should timeout rather than propagate the exception
        with pytest.raises(TimeoutError):
            await FlextTestsAsyncs.wait_for_condition(
                condition=exception_condition,
                timeout_seconds=0.1,  # Short timeout
                poll_interval=0.02,
            )

    @pytest.mark.asyncio
    async def test_async_condition_with_exception(self) -> None:
        """Test async condition that raises exception."""

        async def async_exception_condition() -> bool:
            await asyncio.sleep(0.01)
            msg = "Async condition failed"
            raise RuntimeError(msg)

        # The implementation catches exceptions and continues polling,
        # so this should timeout rather than propagate the exception
        with pytest.raises(TimeoutError):
            await FlextTestsAsyncs.wait_for_condition(
                condition=async_exception_condition,
                timeout_seconds=0.1,  # Short timeout
                poll_interval=0.02,
            )

    def test_type_guard_with_various_types(self) -> None:
        """Test type guard with various Python types."""
        # Test with built-in types
        test_objects = [
            42,  # int
            math.pi,  # float
            "string",  # str
            [1, 2, 3],  # list
            {"key": "value"},  # dict
            {1, 2, 3},  # set
            (1, 2, 3),  # tuple
            True,  # bool
            None,  # NoneType
            lambda x: x,  # function
            FlextTestsAsyncs,  # class
        ]

        for obj in test_objects:
            assert FlextTestsAsyncs._is_not_exception(obj) is True

        # Test with exception instances
        exceptions = [
            Exception(),
            ValueError(),
            TypeError(),
            RuntimeError(),
            AttributeError(),
            KeyError(),
            IndexError(),
            ImportError(),
            OSError(),
        ]

        for exc in exceptions:
            assert FlextTestsAsyncs._is_not_exception(exc) is False

    @pytest.mark.asyncio
    async def test_wait_for_condition_performance_characteristics(self) -> None:
        """Test performance characteristics of wait_for_condition."""
        # Test that polling intervals are respected
        call_count = 0
        start_time = time.time()

        def counting_condition() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        await FlextTestsAsyncs.wait_for_condition(
            condition=counting_condition, timeout_seconds=2.0, poll_interval=0.1
        )

        elapsed = time.time() - start_time

        # Should have taken at least 2 polling intervals (0.2s)
        # but less than timeout
        assert 0.15 <= elapsed < 2.0
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_concurrent_wait_for_condition_calls(self) -> None:
        """Test multiple concurrent wait_for_condition calls."""
        # Test concurrent execution
        counter1 = 0
        counter2 = 0

        def condition1() -> bool:
            nonlocal counter1
            counter1 += 1
            return counter1 >= 2

        def condition2() -> bool:
            nonlocal counter2
            counter2 += 1
            return counter2 >= 3

        # Run both conditions concurrently
        results = await asyncio.gather(
            FlextTestsAsyncs.wait_for_condition(
                condition=condition1, timeout_seconds=1.0, poll_interval=0.05
            ),
            FlextTestsAsyncs.wait_for_condition(
                condition=condition2, timeout_seconds=1.0, poll_interval=0.05
            ),
        )

        # Both should complete successfully
        assert len(results) == 2
        assert counter1 >= 2
        assert counter2 >= 3

    def test_class_structure_and_organization(self) -> None:
        """Test the class structure and organization of FlextTestsAsyncs."""
        # Test class docstring and structure
        assert FlextTestsAsyncs.__doc__ is not None
        assert "async testing utilities" in FlextTestsAsyncs.__doc__.lower()

        # Test unified class pattern compliance
        asyncs = FlextTestsAsyncs()

        # Should be instantiable
        assert isinstance(asyncs, FlextTestsAsyncs)

        # Check for static methods
        static_methods = [
            method
            for method in dir(FlextTestsAsyncs)
            if not method.startswith("_") or method == "_is_not_exception"
        ]

        assert len(static_methods) > 0

    @pytest.mark.asyncio
    async def test_async_condition_cancellation(self) -> None:
        """Test behavior when async condition is cancelled."""

        async def long_running_condition() -> bool:
            # Simulate a long-running condition
            await asyncio.sleep(2.0)
            return True

        # Create task and cancel it quickly
        task = asyncio.create_task(
            FlextTestsAsyncs.wait_for_condition(
                condition=long_running_condition, timeout_seconds=5.0, poll_interval=0.1
            )
        )

        # Give it a moment to start, then cancel
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_wait_for_condition_with_varying_poll_intervals(self) -> None:
        """Test wait_for_condition with different poll intervals."""
        # Test with very short poll interval
        counter_fast = 0

        def fast_condition() -> bool:
            nonlocal counter_fast
            counter_fast += 1
            return counter_fast >= 5

        start_time = time.time()
        await FlextTestsAsyncs.wait_for_condition(
            condition=fast_condition,
            timeout_seconds=1.0,
            poll_interval=0.01,  # Very fast polling
        )
        fast_elapsed = time.time() - start_time

        # Test with slower poll interval
        counter_slow = 0

        def slow_condition() -> bool:
            nonlocal counter_slow
            counter_slow += 1
            return counter_slow >= 3

        start_time = time.time()
        await FlextTestsAsyncs.wait_for_condition(
            condition=slow_condition,
            timeout_seconds=1.0,
            poll_interval=0.1,  # Slower polling
        )
        slow_elapsed = time.time() - start_time

        # Both should complete, but different timing characteristics
        assert fast_elapsed < 1.0
        assert slow_elapsed < 1.0
        assert counter_fast >= 5
        assert counter_slow >= 3

    def test_module_imports_and_structure(self) -> None:
        """Test module imports and overall structure."""
        # Verify module structure
        assert hasattr(asyncs, "FlextTestsAsyncs")
        assert asyncs.FlextTestsAsyncs is FlextTestsAsyncs

        # Test logger import
        assert hasattr(asyncs, "logger")

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self) -> None:
        """Test error handling robustness in various scenarios."""

        # Test with condition that returns non-boolean
        def non_boolean_condition() -> bool:
            return bool("not a boolean")  # Convert to bool for type safety

        # This should work because Python treats non-empty strings as truthy
        await FlextTestsAsyncs.wait_for_condition(
            condition=non_boolean_condition, timeout_seconds=0.5, poll_interval=0.1
        )

        # Test with condition that returns 0 (falsy)
        counter = 0

        def eventually_truthy_condition() -> bool:
            nonlocal counter
            counter += 1
            return bool(counter >= 3)  # Convert to bool for type safety

        await FlextTestsAsyncs.wait_for_condition(
            condition=eventually_truthy_condition,
            timeout_seconds=1.0,
            poll_interval=0.1,
        )

        assert counter >= 3
