"""Real tests to achieve 100% collections coverage - no mocks.

Module: flext_core._models.collections
Scope: FlextModels.Categories, Statistics, Config, Results, Options

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _models/collections.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from pydantic import Field

from flext_core import FlextModels
from flext_core._models.collections import FlextModelsCollections


@dataclass(frozen=True, slots=True)
class CategoryOperationScenario:
    """Category operation test scenario."""

    name: str
    category: str
    entries: list[str]
    operation: str
    expected_result: object


class CollectionsScenarios:
    """Centralized collections test scenarios using FlextConstants."""

    CATEGORY_OPERATIONS: ClassVar[list[CategoryOperationScenario]] = [
        CategoryOperationScenario("add_new", "users", ["user1", "user2"], "add", True),
        CategoryOperationScenario("add_existing", "users", ["user3"], "add", True),
        CategoryOperationScenario("set_replace", "users", ["user4"], "set", True),
        CategoryOperationScenario("remove", "users", [], "remove", True),
    ]


class TestFlextModelsCollectionsCategories:
    """Real tests for FlextModels.Categories using FlextTestsUtilities."""

    def test_categories_initialization(self) -> None:
        """Test Categories initialization."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        assert categories.categories == {}
        assert len(categories) == 0

    def test_categories_get_entries_empty(self) -> None:
        """Test get_entries with empty category."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        assert categories.get_entries("nonexistent") == []

    @pytest.mark.parametrize(
        "scenario",
        CollectionsScenarios.CATEGORY_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_categories_operations(self, scenario: CategoryOperationScenario) -> None:
        """Test category operations with various scenarios."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        if scenario.operation == "add":
            categories.add_entries(scenario.category, scenario.entries)
            assert categories.get_entries(scenario.category) == scenario.entries
        elif scenario.operation == "set":
            categories.add_entries(scenario.category, ["existing"])
            categories.set_entries(scenario.category, scenario.entries)
            assert categories.get_entries(scenario.category) == scenario.entries
        elif scenario.operation == "remove":
            categories.add_entries(scenario.category, ["temp"])
            categories.remove_category(scenario.category)
            assert not categories.has_category(scenario.category)

    def test_categories_add_entries_existing_category(self) -> None:
        """Test add_entries with existing category."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1"])
        categories.add_entries("users", ["user2", "user3"])
        assert categories.get_entries("users") == ["user1", "user2", "user3"]

    def test_categories_has_category(self) -> None:
        """Test has_category."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        assert not categories.has_category("users")
        categories.add_entries("users", ["user1"])
        assert categories.has_category("users")

    def test_categories_remove_category_nonexistent(self) -> None:
        """Test remove_category with nonexistent category."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.remove_category("nonexistent")

    def test_categories_category_names(self) -> None:
        """Test category_names method."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1"])
        categories.add_entries("groups", ["group1"])
        names = categories.category_names
        assert all(name in names for name in ["users", "groups"])
        assert len(names) == 2

    def test_categories_total_entries(self) -> None:
        """Test total_entries computed field."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        assert categories.total_entries == 3

    def test_categories_summary(self) -> None:
        """Test summary computed field."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        summary = categories.summary
        assert summary["users"] == 2
        assert summary["groups"] == 1

    def test_categories_dict_like_operations(self) -> None:
        """Test dict-like operations."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1"])
        assert ("users", ["user1"]) in categories.items()
        assert "users" in categories
        assert ["user1"] in categories.values()
        assert categories["users"] == ["user1"]
        categories["groups"] = ["group1"]
        assert categories.get_entries("groups") == ["group1"]
        assert "users" in categories
        assert "nonexistent" not in categories
        assert len(categories) == 2

    def test_categories_get_with_default(self) -> None:
        """Test get method with default."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        assert categories.get("nonexistent", ["default"]) == ["default"]
        assert categories.get("nonexistent") == []

    def test_categories_from_dict(self) -> None:
        """Test from_dict class method."""
        data = {"users": ["user1"], "groups": ["group1"]}
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str].from_dict(data)
        )
        assert categories.get_entries("users") == ["user1"]
        assert categories.get_entries("groups") == ["group1"]

    def test_categories_to_dict(self) -> None:
        """Test to_dict method."""
        categories: FlextModelsCollections.Categories[str] = (
            FlextModelsCollections.Categories[str]()
        )
        categories.add_entries("users", ["user1"])
        assert categories.to_dict() == {"users": ["user1"]}


class TestFlextModelsCollectionsStatistics:
    """Real tests for FlextModels.Collections.Statistics using FlextTestsUtilities."""

    def test_statistics_aggregate_empty(self) -> None:
        """Test aggregate with empty list."""

        class TestStats(FlextModels.Statistics):
            count: int = 0

        assert TestStats.aggregate([]) == {}

    def test_statistics_aggregate_numbers(self) -> None:
        """Test aggregate with numeric values."""

        class TestStats(FlextModels.Statistics):
            count: int = 0

        stats1 = TestStats(count=10)
        stats2 = TestStats(count=20)
        result = TestStats.aggregate([stats1, stats2])
        assert result["count"] == 30

    def test_statistics_aggregate_lists(self) -> None:
        """Test aggregate with list values."""

        class TestStats(FlextModels.Statistics):
            items: list[str] = Field(default_factory=list)

        stats1 = TestStats(items=["a", "b"])
        stats2 = TestStats(items=["c"])
        result = TestStats.aggregate([stats1, stats2])
        assert result["items"] == ["a", "b", "c"]

    def test_statistics_aggregate_mixed(self) -> None:
        """Test aggregate with mixed types."""

        class TestStats(FlextModels.Statistics):
            count: int = 0
            items: list[str] = Field(default_factory=list)
            name: str = ""

        stats1 = TestStats(count=10, items=["a"], name="first")
        stats2 = TestStats(count=20, items=["b"], name="second")
        result = TestStats.aggregate([stats1, stats2])
        assert result["count"] == 30
        assert result["items"] == ["a", "b"]
        assert result["name"] == "second"

    def test_statistics_aggregate_none_values(self) -> None:
        """Test aggregate with None values."""

        class TestStats(FlextModels.Statistics):
            count: int | None = None
            name: str | None = None

        stats1 = TestStats(count=10, name="first")
        stats2 = TestStats(count=None, name=None)
        result = TestStats.aggregate([stats1, stats2])
        assert result["count"] == 10
        assert result["name"] == "first"


class TestFlextModelsCollectionsConfig:
    """Real tests for FlextModels.Collections.Config using FlextTestsUtilities."""

    def test_config_merge(self) -> None:
        """Test merge method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30
            retries: int = 3

        config1 = TestConfig(timeout=30, retries=3)
        config2 = TestConfig(timeout=60)
        merged: TestConfig = config1.merge(config2)
        assert merged.timeout == 60
        assert merged.retries == 3

    def test_config_from_dict(self) -> None:
        """Test from_dict class method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30

        data = {"timeout": 60}
        config: TestConfig = TestConfig.from_dict(cast("dict[str, object]", data))
        assert config.timeout == 60

    def test_config_to_dict(self) -> None:
        """Test to_dict method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30

        config = TestConfig(timeout=60)
        assert config.to_dict()["timeout"] == 60

    def test_config_with_updates(self) -> None:
        """Test with_updates method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30
            retries: int = 3

        config = TestConfig(timeout=30, retries=3)
        updated: TestConfig = config.with_updates(timeout=60)
        assert updated.timeout == 60
        assert updated.retries == 3
        assert config.timeout == 30

    def test_config_diff(self) -> None:
        """Test diff method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30
            retries: int = 3

        config1 = TestConfig(timeout=30, retries=3)
        config2 = TestConfig(timeout=60, retries=3)
        diff = config1.diff(config2)
        assert "timeout" in diff
        assert diff["timeout"] == (30, 60)
        assert "retries" not in diff

    def test_config_diff_all_different(self) -> None:
        """Test diff with all fields different."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30
            retries: int = 3

        config1 = TestConfig(timeout=30, retries=3)
        config2 = TestConfig(timeout=60, retries=5)
        diff = config1.diff(config2)
        assert len(diff) == 2
        assert diff["timeout"] == (30, 60)
        assert diff["retries"] == (3, 5)

    def test_config_eq(self) -> None:
        """Test __eq__ method."""

        class TestConfig(FlextModels.Config):
            timeout: int = 30

        config1 = TestConfig(timeout=30)
        config2 = TestConfig(timeout=30)
        config3 = TestConfig(timeout=60)
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"


class TestFlextModelsCollectionsResults:
    """Real tests for FlextModels.Collections.Results using FlextTestsUtilities."""

    def test_results_aggregate_empty(self) -> None:
        """Test aggregate with empty list."""

        class TestResult(FlextModels.Results):
            processed: int = 0

        assert TestResult.aggregate([]) == {}

    def test_results_aggregate_numbers(self) -> None:
        """Test aggregate with numeric values."""

        class TestResult(FlextModels.Results):
            processed: int = 0

        result1 = TestResult(processed=10)
        result2 = TestResult(processed=20)
        aggregated = TestResult.aggregate([result1, result2])
        assert aggregated["processed"] == 30

    def test_results_aggregate_lists(self) -> None:
        """Test aggregate with list values."""

        class TestResult(FlextModels.Results):
            errors: list[str] = Field(default_factory=list)

        result1 = TestResult(errors=["error1"])
        result2 = TestResult(errors=["error2"])
        aggregated = TestResult.aggregate([result1, result2])
        assert aggregated["errors"] == ["error1", "error2"]

    def test_results_aggregate_dicts(self) -> None:
        """Test aggregate with dict values."""

        class TestResult(FlextModels.Results):
            metadata: dict[str, str] = Field(default_factory=dict)

        result1 = TestResult(metadata={"key1": "value1"})
        result2 = TestResult(metadata={"key2": "value2"})
        aggregated = TestResult.aggregate([result1, result2])
        assert aggregated["metadata"] == {"key1": "value1", "key2": "value2"}

    def test_results_aggregate_mixed(self) -> None:
        """Test aggregate with mixed types."""

        class TestResult(FlextModels.Results):
            processed: int = 0
            errors: list[str] = Field(default_factory=list)
            status: str = ""

        result1 = TestResult(processed=10, errors=["a"], status="ok")
        result2 = TestResult(processed=20, errors=["b"], status="done")
        aggregated = TestResult.aggregate([result1, result2])
        assert aggregated["processed"] == 30
        assert aggregated["errors"] == ["a", "b"]
        assert aggregated["status"] == "done"

    def test_results_aggregate_none_values(self) -> None:
        """Test aggregate with None values."""

        class TestResult(FlextModels.Results):
            processed: int | None = None
            status: str | None = None

        result1 = TestResult(processed=10, status="ok")
        result2 = TestResult(processed=None, status=None)
        aggregated = TestResult.aggregate([result1, result2])
        assert aggregated["processed"] == 10
        assert aggregated["status"] == "ok"


class TestFlextModelsCollectionsOptions:
    """Real tests for FlextModels.Collections.Options using FlextTestsUtilities."""

    def test_options_merge(self) -> None:
        """Test merge method."""

        class TestOptions(FlextModels.Options):
            verbose: bool = False
            color: bool = True

        options1 = TestOptions(verbose=False, color=True)
        options2 = TestOptions(verbose=True)
        merged = options1.merge(options2)
        assert merged.verbose is True
        assert merged.color is True

    def test_options_merge_all_fields(self) -> None:
        """Test merge with all fields."""

        class TestOptions(FlextModels.Options):
            verbose: bool = False
            color: bool = True

        options1 = TestOptions(verbose=False, color=True)
        options2 = TestOptions(verbose=True, color=False)
        merged = options1.merge(options2)
        assert merged.verbose is True
        assert merged.color is False


__all__ = [
    "TestFlextModelsCollectionsCategories",
    "TestFlextModelsCollectionsConfig",
    "TestFlextModelsCollectionsOptions",
    "TestFlextModelsCollectionsResults",
    "TestFlextModelsCollectionsStatistics",
]
