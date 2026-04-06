# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Benchmark package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import tests.benchmark.test_container_memory as _tests_benchmark_test_container_memory

    test_container_memory = _tests_benchmark_test_container_memory
    import tests.benchmark.test_container_performance as _tests_benchmark_test_container_performance
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )

    test_container_performance = _tests_benchmark_test_container_performance
    import tests.benchmark.test_refactor_nesting_performance as _tests_benchmark_test_refactor_nesting_performance
    from tests.benchmark.test_container_performance import TestContainerPerformance

    test_refactor_nesting_performance = (
        _tests_benchmark_test_refactor_nesting_performance
    )
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )
_LAZY_IMPORTS = {
    "TestContainerMemory": (
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ),
    "TestContainerPerformance": (
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ),
    "TestPerformanceBenchmarks": (
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ),
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "get_memory_usage": ("tests.benchmark.test_container_memory", "get_memory_usage"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "test_container_memory": "tests.benchmark.test_container_memory",
    "test_container_performance": "tests.benchmark.test_container_performance",
    "test_refactor_nesting_performance": "tests.benchmark.test_refactor_nesting_performance",
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestPerformanceBenchmarks",
    "c",
    "d",
    "e",
    "get_memory_usage",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "test_container_memory",
    "test_container_performance",
    "test_refactor_nesting_performance",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
