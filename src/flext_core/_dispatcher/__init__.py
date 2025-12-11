"""Expose dispatcher reliability and timeout helpers.

This package houses helper classes previously nested within
``FlextDispatcher`` so orchestration code remains concise while retaining
identical behavior and typed exports for downstream consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from flext_core._dispatcher.reliability import (
    CircuitBreakerManager,
    RateLimiterManager,
    RetryPolicy,
)
from flext_core._dispatcher.timeout import TimeoutEnforcer
from flext_core.typings import FlextTypes

# Import DispatcherConfig from the correct location in typings
DispatcherConfig = FlextTypes.Dispatcher.DispatcherConfig

__all__ = [
    "CircuitBreakerManager",
    "DispatcherConfig",
    "RateLimiterManager",
    "RetryPolicy",
    "TimeoutEnforcer",
]
