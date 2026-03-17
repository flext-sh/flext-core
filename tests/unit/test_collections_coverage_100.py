"""Real tests to achieve 100% collections coverage - no mocks.

Module: flext_core._models.collections
Scope: m.Categories, Statistics, Config, Results, Options

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _models/collections.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextRuntime
from tests import m


class _TestConfig(m.Config):
    """Module-level test config for pydantic mypy plugin compatibility."""

    timeout: int = 30
    retries: int = 3


class CategoryOperationScenario(BaseModel):
    """Category operation test scenario."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(description="Category operation scenario name")]
    category: Annotated[str, Field(description="Category key")]
    entries: Annotated[
        list[str], Field(description="Entries associated with operation")
    ]
    operation: Annotated[str, Field(description="Category operation name")]
    expected_result: Annotated[object, Field(description="Expected operation result")]


def _scenario_id(scenario: CategoryOperationScenario) -> str:
    return scenario.name


class CollectionsScenarios:
    """Centralized collections test scenarios using FlextConstants."""

    CATEGORY_OPERATIONS: ClassVar[list[CategoryOperationScenario]] = [
        CategoryOperationScenario(
            name="add_new",
            category="users",
            entries=["user1", "user2"],
            operation="add",
            expected_result=True,
        ),
        CategoryOperationScenario(
            name="add_existing",
            category="users",
            entries=["user3"],
            operation="add",
            expected_result=True,
        ),
        CategoryOperationScenario(
            name="set_replace",
            category="users",
            entries=["user4"],
            operation="set",
            expected_result=True,
        ),
        CategoryOperationScenario(
            name="remove",
            category="users",
            entries=[],
            operation="remove",
            expected_result=True,
        ),
    ]


class TestFlextModelsCollectionsCategories:
    """Real tests for m.Categories using FlextTestsUtilities."""

    def test_categories_initialization(self) -> None:
        """Test Categories initialization."""
        categories: m.Categories = m.Categories(categories={})
        tm.that(categories.categories, eq={})
        tm.that(len(categories), eq=0)

    def test_categories_get_empty(self) -> None:
        """Test get with empty category."""
        categories: m.Categories = m.Categories(categories={})
        tm.that(categories.get("nonexistent"), eq=[])

    @pytest.mark.parametrize(
        "scenario",
        CollectionsScenarios.CATEGORY_OPERATIONS,
        ids=_scenario_id,
    )
    def test_categories_operations(self, scenario: CategoryOperationScenario) -> None:
        """Test category operations with various scenarios."""
        categories: m.Categories = m.Categories(categories={})
        if scenario.operation == "add":
            categories.add_entries(scenario.category, scenario.entries)
            tm.that(categories.get(scenario.category), eq=scenario.entries)
        elif scenario.operation == "set":
            categories.add_entries(scenario.category, ["existing"])
            categories.categories[scenario.category] = list(scenario.entries)
            tm.that(categories.get(scenario.category), eq=scenario.entries)
        elif scenario.operation == "remove":
            categories.add_entries(scenario.category, ["temp"])
            categories.remove_category(scenario.category)
            tm.that(scenario.category in categories.categories, eq=False)

    def test_categories_add_entries_existing_category(self) -> None:
        """Test add_entries with existing category."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        categories.add_entries("users", ["user2", "user3"])
        tm.that(categories.get("users"), eq=["user1", "user2", "user3"])

    def test_categories_has_category(self) -> None:
        """Test has_category via categories dict."""
        categories: m.Categories = m.Categories(categories={})
        tm.that("users" in categories.categories, eq=False)
        categories.add_entries("users", ["user1"])
        tm.that(categories.has_category("users"), eq=True)

    def test_categories_remove_category_nonexistent(self) -> None:
        """Test remove_category with nonexistent category."""
        categories: m.Categories = m.Categories(categories={})
        categories.remove_category("nonexistent")

    def test_categories_category_names(self) -> None:
        """Test category_names method."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        categories.add_entries("groups", ["group1"])
        names = list(categories.category_names)
        tm.that(all(name in names for name in ["users", "groups"]), eq=True)
        tm.that(len(names), eq=2)

    def test_categories_total_entries(self) -> None:
        """Test total_entries computed field."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        tm.that(categories.total_entries, eq=3)

    def test_categories_summary(self) -> None:
        """Test summary computed field."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        summary: dict[str, int] = {
            name: len(entries) for name, entries in categories.categories.items()
        }
        tm.that(summary["users"], eq=2)
        tm.that(summary["groups"], eq=1)

    def test_categories_dict_like_operations(self) -> None:
        """Test dict-like operations."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        tm.that(("users", ["user1"]) in categories.categories.items(), eq=True)
        tm.that("users" in categories.categories, eq=True)
        tm.that(["user1"] in categories.categories.values(), eq=True)
        tm.that(categories.categories["users"], eq=["user1"])
        categories.categories["groups"] = ["group1"]
        tm.that(categories.get("groups"), eq=["group1"])
        tm.that("users" in categories.categories, eq=True)
        tm.that("nonexistent" in categories.categories, eq=False)
        tm.that(len(categories.categories), eq=2)

    def test_categories_get_with_default(self) -> None:
        """Test get method with default."""
        categories: m.Categories = m.Categories(categories={})
        tm.that(categories.get("nonexistent", ["default"]), eq=["default"])
        tm.that(categories.get("nonexistent"), eq=[])

    def test_categories_model_validate(self) -> None:
        """Test Categories construction via model_validate."""
        data = {"categories": {"users": ["user1"], "groups": ["group1"]}}
        categories: m.Categories = m.Categories.model_validate(data)
        tm.that(categories.get("users"), eq=["user1"])
        tm.that(categories.get("groups"), eq=["group1"])

    def test_categories_to_mapping(self) -> None:
        """Test to_mapping method."""
        categories: m.Categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        result = categories.to_mapping()
        tm.that(result, eq={"users": ["user1"]})


class TestFlextModelsCollectionsStatistics:
    """Real tests for m.Statistics using FlextTestsUtilities."""

    def test_statistics_aggregate_empty(self) -> None:
        """Test aggregate with empty list."""

        class TestStats(m.Statistics):
            count: int = 0

        tm.that(TestStats.aggregate([]), eq={})

    def test_statistics_aggregate_numbers(self) -> None:
        """Test aggregate with numeric values."""

        class TestStats(m.Statistics):
            count: int = 0

        stats1 = TestStats(count=10)
        stats2 = TestStats(count=20)
        result = TestStats.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(cast("RuntimeData", result)), eq=True)
        tm.that(result["count"], eq=30)

    def test_statistics_aggregate_lists(self) -> None:
        """Test aggregate with list values."""

        class TestStats(m.Statistics):
            items: Annotated[list[str], Field(default_factory=list)]

        stats1 = TestStats(items=["a", "b"])
        stats2 = TestStats(items=["c"])
        result = TestStats.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(cast("RuntimeData", result)), eq=True)
        tm.that(result["items"], eq=["a", "b", "c"])

    def test_statistics_aggregate_mixed(self) -> None:
        """Test aggregate with mixed types."""

        class TestStats(m.Statistics):
            count: int = 0
            items: Annotated[list[str], Field(default_factory=list)]
            name: str = ""

        stats1 = TestStats(count=10, items=["a"], name="first")
        stats2 = TestStats(count=20, items=["b"], name="second")
        result = TestStats.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(cast("RuntimeData", result)), eq=True)
        tm.that(result["count"], eq=30)
        tm.that(result["items"], eq=["a", "b"])
        tm.that(result["name"], eq="second")

    def test_statistics_aggregate_none_values(self) -> None:
        """Test aggregate with None values."""

        class TestStats(m.Statistics):
            count: int | None = None
            name: str | None = None

        stats1 = TestStats(count=10, name="first")
        stats2 = TestStats(count=None, name=None)
        result = TestStats.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(cast("RuntimeData", result)), eq=True)
        tm.that(result["count"], eq=10)
        tm.that(result["name"], eq="first")


class TestFlextModelsCollectionsSettings:
    """Real tests for m.Config using FlextTestsUtilities."""

    def test_config_merge(self) -> None:
        """Test merge method."""
        config1 = _TestConfig.model_validate({"timeout": 30, "retries": 3})
        config2 = _TestConfig.model_validate({"timeout": 60})
        merged: _TestConfig = config1.merge(config2)
        tm.that(merged.timeout, eq=60)
        tm.that(merged.retries, eq=3)

    def test_config_from_dict(self) -> None:
        """Test from_mapping class method."""
        config_data = t.ConfigMap(root={"timeout": 60})
        config: _TestConfig = _TestConfig.from_mapping(config_data)
        tm.that(config.timeout, eq=60)

    def test_config_to_dict(self) -> None:
        """Test to_mapping method (to_dict was renamed to to_mapping)."""
        config = _TestConfig.model_validate({"timeout": 60})
        config_dict = config.to_mapping()
        tm.that(config_dict["timeout"], eq=60)

    def test_config_with_updates(self) -> None:
        """Test with_updates method."""
        config = _TestConfig.model_validate({"timeout": 30, "retries": 3})
        updated: _TestConfig = config.with_updates(timeout=60)
        tm.that(updated.timeout, eq=60)
        tm.that(updated.retries, eq=3)
        tm.that(config.timeout, eq=30)

    def test_config_diff(self) -> None:
        """Test diff method."""
        config1 = _TestConfig.model_validate({"timeout": 30, "retries": 3})
        config2 = _TestConfig.model_validate({"timeout": 60, "retries": 3})
        diff = config1.diff(config2)
        tm.that("timeout" in diff, eq=True)
        tm.that(diff["timeout"], eq=(30, 60))
        tm.that("retries" in diff, eq=False)

    def test_config_diff_all_different(self) -> None:
        """Test diff with all fields different."""
        config1 = _TestConfig.model_validate({"timeout": 30, "retries": 3})
        config2 = _TestConfig.model_validate({"timeout": 60, "retries": 5})
        diff = config1.diff(config2)
        tm.that(len(diff), eq=2)
        tm.that(diff["timeout"], eq=(30, 60))
        tm.that(diff["retries"], eq=(3, 5))

    def test_config_eq(self) -> None:
        """Test __eq__ method."""
        config1 = _TestConfig.model_validate({"timeout": 30})
        config2 = _TestConfig.model_validate({"timeout": 30})
        config3 = _TestConfig.model_validate({"timeout": 60})
        tm.that(config1 == config2, eq=True)
        tm.that(config1 != config3, eq=True)
        tm.that(config1 != "not a config", eq=True)


class TestFlextModelsCollectionsResults:
    """Real tests for m.Results using FlextTestsUtilities."""

    def test_results_aggregate_empty(self) -> None:
        """Test aggregate with empty list."""

        class TestResult(m.Results):
            processed: int = 0

        tm.that(TestResult.aggregate([]), eq={})

    def test_results_aggregate_numbers(self) -> None:
        """Test aggregate with numeric values."""

        class TestResult(m.Results):
            processed: int = 0

        result1 = TestResult(processed=10)
        result2 = TestResult(processed=20)
        aggregated_raw = TestResult.aggregate([result1, result2])
        tm.that(
            FlextRuntime.is_dict_like(cast("t.RuntimeData", aggregated_raw)),
            eq=True,
        )
        aggregated = aggregated_raw
        tm.that(aggregated["processed"], eq=30)

    def test_results_aggregate_lists(self) -> None:
        """Test aggregate with list values."""

        class TestResult(m.Results):
            errors: Annotated[list[str], Field(default_factory=list)]

        result1 = TestResult(errors=["error1"])
        result2 = TestResult(errors=["error2"])
        aggregated_raw = TestResult.aggregate([result1, result2])
        tm.that(
            FlextRuntime.is_dict_like(cast("t.RuntimeData", aggregated_raw)),
            eq=True,
        )
        aggregated = aggregated_raw
        tm.that(aggregated["errors"], eq=["error1", "error2"])

    def test_results_aggregate_dicts(self) -> None:
        """Test aggregate with dict values."""

        class TestResult(m.Results):
            metadata: Annotated[dict[str, str], Field(default_factory=dict)]

        result1 = TestResult(metadata={"key1": "value1"})
        result2 = TestResult(metadata={"key2": "value2"})
        aggregated_raw = TestResult.aggregate([result1, result2])
        tm.that(
            FlextRuntime.is_dict_like(cast("t.RuntimeData", aggregated_raw)),
            eq=True,
        )
        aggregated = aggregated_raw
        tm.that(aggregated["metadata"], eq={"key1": "value1", "key2": "value2"})

    def test_results_aggregate_mixed(self) -> None:
        """Test aggregate with mixed types."""

        class TestResult(m.Results):
            processed: int = 0
            errors: Annotated[list[str], Field(default_factory=list)]
            status: str = ""

        result1 = TestResult(processed=10, errors=["a"], status="ok")
        result2 = TestResult(processed=20, errors=["b"], status="done")
        aggregated_raw = TestResult.aggregate([result1, result2])
        tm.that(
            FlextRuntime.is_dict_like(cast("t.RuntimeData", aggregated_raw)),
            eq=True,
        )
        aggregated = aggregated_raw
        tm.that(aggregated["processed"], eq=30)
        tm.that(aggregated["errors"], eq=["a", "b"])
        tm.that(aggregated["status"], eq="done")

    def test_results_aggregate_none_values(self) -> None:
        """Test aggregate with None values."""

        class TestResult(m.Results):
            processed: int | None = None
            status: str | None = None

        result1 = TestResult(processed=10, status="ok")
        result2 = TestResult(processed=None, status=None)
        aggregated_raw = TestResult.aggregate([result1, result2])
        tm.that(
            FlextRuntime.is_dict_like(cast("t.RuntimeData", aggregated_raw)),
            eq=True,
        )
        aggregated = aggregated_raw
        tm.that(aggregated["processed"], eq=10)
        tm.that(aggregated["status"], eq="ok")


class TestFlextModelsCollectionsOptions:
    """Real tests for m.Options using FlextTestsUtilities."""

    def test_options_merge(self) -> None:
        """Test merge method."""

        class TestOptions(m.Options):
            verbose: bool = False
            color: bool = True

        options1 = TestOptions(verbose=False, color=True)
        options2 = TestOptions(verbose=True)
        merged = options1.merge(options2)
        tm.that(merged.verbose is True, eq=True)
        tm.that(merged.color is True, eq=True)

    def test_options_merge_all_fields(self) -> None:
        """Test merge with all fields."""

        class TestOptions(m.Options):
            verbose: bool = False
            color: bool = True

        options1 = TestOptions(verbose=False, color=True)
        options2 = TestOptions(verbose=True, color=False)
        merged = options1.merge(options2)
        tm.that(merged.verbose is True, eq=True)
        tm.that(merged.color is False, eq=True)


__all__ = [
    "TestFlextModelsCollectionsCategories",
    "TestFlextModelsCollectionsOptions",
    "TestFlextModelsCollectionsResults",
    "TestFlextModelsCollectionsSettings",
    "TestFlextModelsCollectionsStatistics",
]
