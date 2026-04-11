"""Performance benchmarks for lazy export installation and symbol resolution."""

from __future__ import annotations

import importlib
import time
from types import ModuleType

import pytest

import flext_core
from flext_core import install_lazy_exports
from flext_core.lazy import lazy

type LazyImportEntry = str | tuple[str, str]
type LazyImportMap = dict[str, LazyImportEntry]

REAL_SYMBOLS: tuple[str, ...] = (
    "FlextConstants",
    "FlextDispatcher",
    "FlextModels",
    "FlextProtocols",
    "FlextRuntime",
    "FlextService",
    "FlextUtilities",
    "c",
    "m",
    "p",
    "r",
    "t",
    "u",
)

EXTRA_INSTALL_MAPS: tuple[LazyImportMap, ...] = (
    {
        "_types": ".typings:FlextTypes",
        "_models": ".models:FlextModels",
        "_utils": ".utilities:FlextUtilities",
    },
    {
        "_svc": ".service:FlextService",
        "_dispatch": ".dispatcher:FlextDispatcher",
        "_runtime": ".runtime:FlextRuntime",
    },
    {
        "_const": ".constants:FlextConstants",
        "_proto": ".protocols:FlextProtocols",
        "_result": ".result:r",
    },
)


class TestLazyPerformance:
    """Benchmarks for lazy installation and lazy symbol resolution."""

    class LazyBenchmark:
        """Helpers for benchmark execution."""

        @staticmethod
        def _new_virtual_module(module_name: str) -> ModuleType:
            module = ModuleType(module_name)
            module.__dict__.update(
                {
                    "__name__": module_name,
                    "__package__": "flext_core",
                    "__all__": [],
                },
            )
            return module

        @staticmethod
        def _exercise_lazy_path(reset_between_iterations: bool) -> None:
            if reset_between_iterations:
                lazy.reset()

            reloaded_module = importlib.reload(flext_core)

            for index, lazy_map in enumerate(EXTRA_INSTALL_MAPS):
                module_name = f"flext_core._bench_virtual_{index}"
                virtual_module = TestLazyPerformance.LazyBenchmark._new_virtual_module(
                    module_name,
                )
                install_lazy_exports(
                    module_name,
                    virtual_module.__dict__,
                    lazy_map,
                    publish_all=index % 2 == 0,
                )

            for symbol_name in REAL_SYMBOLS:
                getattr(reloaded_module, symbol_name)

        @staticmethod
        def run(reset_between_iterations: bool, iterations: int) -> float:
            start = time.perf_counter()
            for _ in range(iterations):
                TestLazyPerformance.LazyBenchmark._exercise_lazy_path(
                    reset_between_iterations=reset_between_iterations,
                )
            return time.perf_counter() - start

    @pytest.mark.benchmark
    def test_lazy_install_and_resolution_cold_path(self) -> None:
        """Benchmark cold path with cache reset before each iteration."""
        lazy.reset()
        elapsed = self.LazyBenchmark.run(
            reset_between_iterations=True,
            iterations=120,
        )
        assert elapsed > 0.0

    @pytest.mark.benchmark
    def test_lazy_install_and_resolution_warm_path(self) -> None:
        """Benchmark warm path with cache reuse across iterations."""
        lazy.reset()
        _ = self.LazyBenchmark.run(reset_between_iterations=False, iterations=10)
        elapsed = self.LazyBenchmark.run(
            reset_between_iterations=False,
            iterations=120,
        )

        assert elapsed > 0.0
        assert lazy.cache_stats["install_cache"] >= 1
