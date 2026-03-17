"""Reliability helpers for ``FlextDispatcher``.

Provide reusable circuit breaking, rate limiting, and retry primitives used by
the dispatcher pipeline to protect CQRS handlers. Splitting these helpers into
their own module keeps orchestration readable while preserving the same
runtime behavior and typed surface exposed by the dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.dispatcher import FlextModelsDispatcher

CircuitBreakerManager = FlextModelsDispatcher.CircuitBreakerManager
RateLimiterManager = FlextModelsDispatcher.RateLimiterManager
RetryPolicy = FlextModelsDispatcher.RetryPolicy


__all__ = ["CircuitBreakerManager", "RateLimiterManager", "RetryPolicy"]
