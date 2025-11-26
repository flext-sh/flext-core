"""Dispatcher submodule with reliability patterns extracted from FlextDispatcher.

This module contains helper classes that were previously nested inside FlextDispatcher.
Extracting them reduces dispatcher.py from 3400+ lines to ~2400 lines while
maintaining the same functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from flext_core._dispatcher.config import DispatcherConfig
from flext_core._dispatcher.reliability import (
    CircuitBreakerManager,
    RateLimiterManager,
    RetryPolicy,
)
from flext_core._dispatcher.timeout import TimeoutEnforcer

__all__ = [
    "CircuitBreakerManager",
    "DispatcherConfig",
    "RateLimiterManager",
    "RetryPolicy",
    "TimeoutEnforcer",
]
