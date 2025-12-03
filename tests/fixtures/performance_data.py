"""Performance testing fixtures using advanced Python 3.13 patterns.

Provides comprehensive performance threshold and benchmark data factories
for validating enterprise-grade performance standards in flext-core.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flext_core import u  # Use alias for concise code


@dataclass(frozen=True, slots=True)
class PerformanceThreshold:
    """Factory for performance threshold configurations."""

    result_creation: float = 0.001  # 1ms for FlextResult creation
    container_registration: float = 0.005  # 5ms for service registration
    container_retrieval: float = 0.001  # 1ms for service retrieval
    validation: float = 0.01  # 10ms for validation operations
    serialization: float = 0.05  # 50ms for serialization
    database_query: float = 0.1  # 100ms for database operations
    api_call: float = 0.5  # 500ms for external API calls

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format for compatibility."""
        return {
            "result_creation": self.result_creation,
            "container_registration": self.container_registration,
            "container_retrieval": self.container_retrieval,
            "validation": self.validation,
            "serialization": self.serialization,
            "database_query": self.database_query,
            "api_call": self.api_call,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """Factory for benchmark dataset configurations."""

    name: str
    data: object
    size_category: str = "medium"
    description: str = field(default="", compare=False)

    @staticmethod
    def create_small_dataset() -> BenchmarkDataset:
        """Create small benchmark dataset."""
        return BenchmarkDataset(
            name="small_dataset",
            data=list(range(100)),
            size_category="small",
            description="Small dataset with 100 items",
        )

    @staticmethod
    def create_medium_dataset() -> BenchmarkDataset:
        """Create medium benchmark dataset."""
        return BenchmarkDataset(
            name="medium_dataset",
            data=list(range(1000)),
            size_category="medium",
            description="Medium dataset with 1000 items",
        )

    @staticmethod
    def create_large_dataset() -> BenchmarkDataset:
        """Create large benchmark dataset."""
        return BenchmarkDataset(
            name="large_dataset",
            data=list(range(10000)),
            size_category="large",
            description="Large dataset with 10000 items",
        )

    @staticmethod
    def create_complex_dict() -> BenchmarkDataset:
        """Create complex dictionary dataset."""
        return BenchmarkDataset(
            name="complex_dict",
            data={f"key_{i}": f"value_{i}" for i in range(1000)},
            size_category="medium",
            description="Complex dictionary with 1000 key-value pairs",
        )

    @staticmethod
    def create_nested_structure() -> BenchmarkDataset:
        """Create nested structure dataset."""
        return BenchmarkDataset(
            name="nested_structure",
            data={
                "level1": {"level2": {"level3": {"data": list(range(100))}}},
            },
            size_category="small",
            description="Nested dictionary structure",
        )


class PerformanceFactories:
    """Centralized factories for performance testing."""

    @staticmethod
    def get_strict_thresholds() -> PerformanceThreshold:
        """Get strict performance thresholds for critical operations."""
        return PerformanceThreshold(
            result_creation=0.0005,  # 0.5ms
            container_registration=0.002,  # 2ms
            container_retrieval=0.0005,  # 0.5ms
            validation=0.005,  # 5ms
            serialization=0.02,  # 20ms
        )

    @staticmethod
    def get_standard_thresholds() -> PerformanceThreshold:
        """Get standard performance thresholds."""
        return PerformanceThreshold()

    @staticmethod
    def get_relaxed_thresholds() -> PerformanceThreshold:
        """Get relaxed thresholds for integration testing."""
        return PerformanceThreshold(
            result_creation=0.01,  # 10ms
            container_registration=0.05,  # 50ms
            container_retrieval=0.01,  # 10ms
            validation=0.1,  # 100ms
            serialization=0.5,  # 500ms
            database_query=1.0,  # 1s
            api_call=5.0,  # 5s
        )

    @staticmethod
    def get_all_datasets() -> dict[str, BenchmarkDataset]:
        """Get all predefined benchmark datasets."""
        return {
            "small": BenchmarkDataset.create_small_dataset(),
            "medium": BenchmarkDataset.create_medium_dataset(),
            "large": BenchmarkDataset.create_large_dataset(),
            "complex_dict": BenchmarkDataset.create_complex_dict(),
            "nested": BenchmarkDataset.create_nested_structure(),
        }

    @staticmethod
    def get_datasets_by_size(size: str) -> list[BenchmarkDataset]:
        """Get datasets filtered by size category."""
        all_datasets = PerformanceFactories.get_all_datasets()
        return list(
            u.filter(list(all_datasets.values()), lambda ds: ds.size_category == size)
        )


# Backward compatibility functions
def get_performance_threshold() -> dict[str, float]:
    """Provide performance thresholds for testing (backward compatibility).

    Returns:
        Dict containing performance thresholds in legacy format

    """
    return PerformanceFactories.get_standard_thresholds().to_dict()


def get_benchmark_data() -> dict[str, object]:
    """Provide standardized data for performance testing (backward compatibility).

    Returns:
        Dict containing benchmark data sets in legacy format

    """
    datasets = PerformanceFactories.get_all_datasets()
    return {name: ds.data for name, ds in datasets.items()}
