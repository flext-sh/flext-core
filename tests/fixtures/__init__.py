"""Test fixtures package for flext-core.

This package provides centralized test data fixtures for consistent testing
across all flext-core test modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import (
    FlextBus,
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextDispatcher,
    FlextExceptions,
    FlextHandlers,
    FlextLogger,
    FlextMixins,
    FlextModels,
    FlextProcessors,
    FlextProtocols,
    FlextRegistry,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextTypes,
    FlextUtilities,
)

from .error_scenarios import (
    get_test_error_scenarios,
)
from .performance_data import (
    get_benchmark_data,
    get_performance_threshold,
)
from .sample_data import (
    get_error_context,
    get_sample_data,
    get_test_user_data,
)
from .test_constants import get_test_constants
from .test_contexts import get_test_contexts
from .test_payloads import get_test_payloads

__all__ = [
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessors",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "get_benchmark_data",
    "get_error_context",
    "get_performance_threshold",
    "get_sample_data",
    "get_test_constants",
    "get_test_contexts",
    "get_test_error_scenarios",
    "get_test_payloads",
    "get_test_user_data",
]
