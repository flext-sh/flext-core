"""Dispatcher state-model namespace.

The concrete ``FlextDispatcher`` service does not currently consume any
Pydantic state model from this module.  Legacy reliability / timeout /
rate-limiter models (``TimeoutEnforcer``, ``CircuitBreakerManager``,
``RateLimiterManager``) were only referenced by their own test suites and
have been removed.  Future reliability support MUST come back through
``_utilities/`` helpers plus state-only models added here.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextModelsDispatcher:
    """Empty namespace kept for import compatibility."""


__all__: list[str] = ["FlextModelsDispatcher"]
