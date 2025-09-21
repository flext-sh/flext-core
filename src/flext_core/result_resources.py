"""Resource management utilities for FlextResult.

This module provides resource management utilities for FlextResult,
extracted from the main FlextResult class to reduce complexity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import signal
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from flext_core.typings import T_co, TResource, TTimeout, UResource

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextResultResources:
    """Resource management utilities for FlextResult.

    Provides resource management, timeout handling, and cleanup operations
    extracted from the main FlextResult class.
    """

    # =========================================================================
    # RESOURCE MANAGEMENT - ZERO USAGE (unused pattern)
    # =========================================================================

    @staticmethod
    def with_resource[T_co, TResource, UResource](
        result: FlextResult[T_co],
        resource_factory: Callable[[], TResource],
        operation: Callable[[T_co, TResource], FlextResult[UResource]],
        cleanup: Callable[[TResource], None] | None = None,
    ) -> FlextResult[UResource]:
        """Execute operation with automatic resource management.

        USAGE: 0+ occurrences - UNUSED resource management pattern
        OPTIMIZATION: Streamlined resource management with proper cleanup

        Arguments:
            result: The result to apply the resource management to.
            resource_factory: The factory function to create the resource.
            operation: The operation to execute with the resource.
            cleanup: The cleanup function to call when the resource is no longer needed.

        Returns:
            A FlextResult containing the resource management result.

        """
        from flext_core.result import FlextResult

        if result.is_failure:
            return FlextResult[UResource].fail(
                result.error or "Cannot use resource with failed result",
                error_code=result.error_code,
                error_data=result.error_data,
            )

        try:
            resource = resource_factory()
            try:
                return operation(result.unwrap(), resource)
            finally:
                if cleanup:
                    cleanup(resource)
        except Exception as e:
            return FlextResult[UResource].fail(f"Resource operation failed: {e}")

    @staticmethod
    def with_timeout[T_co, TTimeout](
        result: FlextResult[T_co],
        timeout_seconds: float,
        operation: Callable[[T_co], FlextResult[TTimeout]],
    ) -> FlextResult[TTimeout]:
        """Execute operation with timeout.

        USAGE: 0+ occurrences - UNUSED timeout pattern
        OPTIMIZATION: Streamlined timeout handling with signal management

        Arguments:
            result: The result to apply the timeout to.
            timeout_seconds: The timeout in seconds.
            operation: The operation to execute.

        Returns:
            A FlextResult containing the timeout result.

        """
        from flext_core.result import FlextResult

        if result.is_failure:
            return FlextResult[TTimeout].fail(
                result.error or "Cannot apply timeout to failed result",
                error_code=result.error_code,
                error_data=result.error_data,
            )

        def timeout_handler(_signum: int, _frame: object) -> None:
            """Timeout handler for signal.

            Arguments:
                _signum: The signal number (unused).
                _frame: The frame object (unused).

            Raises:
                TimeoutError: If the operation timed out.

            """
            signal.alarm(0)  # Clear timeout
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg)

        try:
            # Set up timeout signal
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            start_time = time.time()
            operation_result = operation(result.unwrap())
            elapsed = time.time() - start_time

            # Clear timeout
            signal.alarm(0)

            # Add timing metadata
            if operation_result.is_success:
                success_metadata = (
                    dict(operation_result.error_data)
                    if operation_result.error_data
                    else {}
                )
                success_metadata["execution_time"] = elapsed
                return FlextResult[TTimeout].ok(operation_result.unwrap())
            failure_metadata = (
                dict(operation_result.error_data) if operation_result.error_data else {}
            )
            failure_metadata["execution_time"] = elapsed
            return FlextResult[TTimeout].fail(
                operation_result.error or "Timed operation failed",
                error_code=operation_result.error_code,
                error_data=failure_metadata,
            )
        except TimeoutError as e:
            signal.alarm(0)  # Clear timeout

            # Add timing metadata for timeout error
            timeout_metadata: dict[str, object] = {
                "timeout_seconds": timeout_seconds,
                "execution_time": time.time() - start_time,
            }
            return FlextResult[TTimeout].fail(
                str(e),
                error_code="TIMEOUT_ERROR",
                error_data=timeout_metadata,
            )
        except Exception as e:
            signal.alarm(0)  # Clear timeout
            return FlextResult[TTimeout].fail(f"Timeout operation failed: {e}")

    # =========================================================================
    # CONTEXT AND LOGGING OPERATIONS - LOW USAGE (1+ occurrences each)
    # =========================================================================

    @staticmethod
    def with_context[T_co](
        result: FlextResult[T_co],
        context_func: Callable[[str], str],
    ) -> FlextResult[T_co]:
        """Add contextual information to error messages.

        USAGE: 1+ occurrences - Context enhancement pattern
        OPTIMIZATION: Streamlined error enhancement

        Arguments:
            result: The result to apply the context to.
            context_func: The function to add contextual information to the error message.

        Returns:
            A FlextResult containing the contextual result.

        """
        from flext_core.result import FlextResult

        if result.is_success:
            return result

        if result.error:
            enhanced_error = context_func(result.error)
            return FlextResult[T_co].fail(
                enhanced_error,
                error_code=result.error_code,
                error_data=result.error_data,
            )
        return result

    @staticmethod
    def rescue_with_logging[T_co](
        result: FlextResult[T_co],
        logger_func: Callable[[str], None],
    ) -> FlextResult[T_co]:
        """Log error and continue with failure state.

        USAGE: 1+ occurrences - Logging rescue pattern
        OPTIMIZATION: Suppressed exceptions for railway pattern compliance

        Arguments:
            result: The result to apply the logging to.
            logger_func: The function to log the error.

        Returns:
            A FlextResult containing the logging result.

        """
        if result.is_failure and result.error:
            # Railway pattern: Logging failure should not break the chain
            # Using contextlib.suppress as recommended by ruff
            with contextlib.suppress(Exception):
                logger_func(result.error)
        return result
