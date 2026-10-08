"""Shared exception parameter model lists.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import m


class TestsFlextModelsExceptionParamsSupport:
    """Canonical namespace owner."""

    _ALL_PARAMS_MODELS = [
        m.ValidationErrorParams,
        m.ConfigurationErrorParams,
        m.ConnectionErrorParams,
        m.TimeoutErrorParams,
        m.AuthenticationErrorParams,
        m.AuthorizationErrorParams,
        m.NotFoundErrorParams,
        m.ConflictErrorParams,
        m.RateLimitErrorParams,
        m.CircuitBreakerErrorParams,
        m.TypeErrorParams,
        m.OperationErrorParams,
        m.AttributeAccessErrorParams,
    ]

    _ALL_PARAMS_IDS = [
        "validation",
        "configuration",
        "connection",
        "timeout",
        "authentication",
        "authorization",
        "not-found",
        "conflict",
        "rate-limit",
        "circuit-breaker",
        "type-error",
        "operation",
        "attribute-access",
    ]
