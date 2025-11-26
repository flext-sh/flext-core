"""Dispatcher configuration TypedDict.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypedDict


class DispatcherConfig(TypedDict):
    """Typed dictionary for dispatcher configuration values."""

    dispatcher_timeout_seconds: float
    executor_workers: int
    circuit_breaker_threshold: int
    rate_limit_max_requests: int
    rate_limit_window_seconds: float
    max_retry_attempts: int
    retry_delay: float
    enable_timeout_executor: bool
    dispatcher_enable_logging: bool
    dispatcher_auto_context: bool
    dispatcher_enable_metrics: bool


__all__ = ["DispatcherConfig"]
