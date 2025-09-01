"""FLEXT Timing - Performance tracking and timing functionality.

Provides comprehensive performance timing capabilities through hierarchical organization
of timing utilities and mixin classes. Built for operation timing, performance
measurement, and execution statistics with enterprise-grade patterns.

Module Role in Architecture:
    FlextTiming serves as the performance measurement foundation providing timing
    patterns for object-oriented applications. Integrates with all FLEXT ecosystem
    components requiring performance tracking and optimization analysis.
"""

from __future__ import annotations

import time

from flext_core.protocols import FlextProtocols

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT
# =============================================================================


class FlextTiming:
    """Unified performance timing system implementing single class pattern.

    This class serves as the single main export consolidating ALL timing
    functionality with enterprise-grade patterns. Provides comprehensive
    performance measurement capabilities while maintaining clean API.

    Tier 1 Module Pattern: timing.py -> FlextTiming
    All timing functionality is accessible through this single interface.
    """

    # =============================================================================
    # CORE TIMING OPERATIONS
    # =============================================================================

    @staticmethod
    def start_timing(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Start timing operation and return start time.

        Args:
            obj: Object to start timing on

        Returns:
            Start time in seconds since epoch

        """
        start_time = time.time()
        obj._start_time = start_time
        return start_time

    @staticmethod
    def stop_timing(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Stop timing and return elapsed seconds.

        Args:
            obj: Object to stop timing on

        Returns:
            Elapsed time in seconds

        """
        start_time = getattr(obj, "_start_time", None)
        if start_time is None:
            return 0.0

        elapsed: float = time.time() - start_time

        # Store in timing history
        if not hasattr(obj, "_elapsed_times"):
            obj._elapsed_times = []

        elapsed_times = getattr(obj, "_elapsed_times", [])
        elapsed_times.append(elapsed)
        obj._elapsed_times = elapsed_times
        obj._start_time = None

        return elapsed

    @staticmethod
    def get_last_elapsed_time(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get last elapsed time.

        Args:
            obj: Object to get elapsed time from

        Returns:
            Last elapsed time in seconds

        """
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return elapsed_times[-1] if elapsed_times else 0.0

    @staticmethod
    def get_average_elapsed_time(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get average elapsed time.

        Args:
            obj: Object to get average time from

        Returns:
            Average elapsed time in seconds

        """
        elapsed_times = getattr(obj, "_elapsed_times", [])
        return sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    @staticmethod
    def clear_timing_history(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear timing history.

        Args:
            obj: Object to clear timing history from

        """
        obj._elapsed_times = []

    # =============================================================================
    # MIXIN CLASS
    # =============================================================================

    class Timeable:
        """Mixin class providing performance timing functionality.

        This mixin adds performance timing capabilities to any class,
        including operation timing and average calculation.
        """

        def start_timing(self) -> float:
            """Start timing operation and return start time."""
            return FlextTiming.start_timing(self)

        def stop_timing(self) -> float:
            """Stop timing and return elapsed seconds."""
            return FlextTiming.stop_timing(self)

        def get_last_elapsed_time(self) -> float:
            """Get last elapsed time."""
            return FlextTiming.get_last_elapsed_time(self)

        def get_average_elapsed_time(self) -> float:
            """Get average elapsed time."""
            return FlextTiming.get_average_elapsed_time(self)

        def clear_timing_history(self) -> None:
            """Clear timing history."""
            FlextTiming.clear_timing_history(self)


__all__ = [
    "FlextTiming",
]
