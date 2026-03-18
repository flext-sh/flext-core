from __future__ import annotations

from typing import Annotated, ClassVar

import pytest
from flext_tests import tm
from pydantic import ConfigDict, Field

from flext_core import FlextRuntime
from tests import m, t


class TestFlextModelsCollectionsCoverage100:
    class ConfigFixture(m.Config):
        """Test configuration with timeout and retries."""

        timeout: int = 30
        retries: int = 3

    class CategoryOperationScenario(m.Value):
        """Scenario for category operations."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Category operation scenario name")]
        category: Annotated[str, Field(description="Category key")]
        entries: Annotated[
            list[str], Field(description="Entries associated with operation")
        ]
        operation: Annotated[str, Field(description="Category operation name")]

    class StatisticsCount(m.Statistics):
        """Statistics with count field."""

        count: int = 0

    class StatisticsItems(m.Statistics):
        """Statistics with items list."""

        items: Annotated[list[str], Field(default_factory=list)]

    class StatisticsMixed(m.Statistics):
        """Statistics with count, items, and name."""

        count: int = 0
        items: Annotated[list[str], Field(default_factory=list)]
        name: str = ""

    class StatisticsOptional(m.Statistics):
        """Statistics with optional count and name."""

        count: int | None = None
        name: str | None = None

    class ResultsProcessed(m.Results):
        """Results with processed count."""

        processed: int = 0

    class ResultsErrors(m.Results):
        """Results with errors list."""

        errors: Annotated[list[str], Field(default_factory=list)]

    class ResultsMetadata(m.Results):
        """Results with metadata dictionary."""

        metadata: Annotated[dict[str, str], Field(default_factory=dict)]

    class ResultsMixed(m.Results):
        """Results with processed, errors, and status."""

        processed: int = 0
        errors: Annotated[list[str], Field(default_factory=list)]
        status: str = ""

    class ResultsOptional(m.Results):
        """Results with optional processed and status."""

        processed: int | None = None
        status: str | None = None

    class OptionsFixture(m.Options):
        """Options with verbose and color flags."""

        verbose: bool = False
        color: bool = True

    CATEGORY_OPERATIONS: ClassVar[list[CategoryOperationScenario]] = [
        CategoryOperationScenario(
            name="add_new",
            category="users",
            entries=["user1", "user2"],
            operation="add",
        ),
        CategoryOperationScenario(
            name="add_existing",
            category="users",
            entries=["user3"],
            operation="add",
        ),
        CategoryOperationScenario(
            name="set_replace",
            category="users",
            entries=["user4"],
            operation="set",
        ),
        CategoryOperationScenario(
            name="remove",
            category="users",
            entries=[],
            operation="remove",
        ),
    ]

    def test_categories_initialization(self) -> None:
        categories = m.Categories(categories={})
        tm.that(categories.categories, eq={})
        tm.that(len(categories), eq=0)

    def test_categories_get_empty(self) -> None:
        categories = m.Categories(categories={})
        tm.that(categories.get("nonexistent"), eq=[])

    @pytest.mark.parametrize(
        "scenario",
        CATEGORY_OPERATIONS,
        ids=lambda scenario: scenario.name,
    )
    def test_categories_operations(self, scenario: CategoryOperationScenario) -> None:
        categories = m.Categories(categories={})
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
        categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        categories.add_entries("users", ["user2", "user3"])
        tm.that(categories.get("users"), eq=["user1", "user2", "user3"])

    def test_categories_has_category(self) -> None:
        categories = m.Categories(categories={})
        tm.that("users" in categories.categories, eq=False)
        categories.add_entries("users", ["user1"])
        tm.that(categories.has_category("users"), eq=True)

    def test_categories_remove_category_nonexistent(self) -> None:
        categories = m.Categories(categories={})
        categories.remove_category("nonexistent")

    def test_categories_category_names(self) -> None:
        categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        categories.add_entries("groups", ["group1"])
        names = list(categories.category_names)
        tm.that(all(name in names for name in ["users", "groups"]), eq=True)
        tm.that(len(names), eq=2)

    def test_categories_total_entries(self) -> None:
        categories = m.Categories(categories={})
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        tm.that(categories.total_entries, eq=3)

    def test_categories_summary(self) -> None:
        categories = m.Categories(categories={})
        categories.add_entries("users", ["user1", "user2"])
        categories.add_entries("groups", ["group1"])
        summary: dict[str, int] = {
            name: len(entries) for name, entries in categories.categories.items()
        }
        tm.that(summary["users"], eq=2)
        tm.that(summary["groups"], eq=1)

    def test_categories_dict_like_operations(self) -> None:
        categories = m.Categories(categories={})
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
        categories = m.Categories(categories={})
        tm.that(categories.get("nonexistent", ["default"]), eq=["default"])
        tm.that(categories.get("nonexistent"), eq=[])

    def test_categories_model_validate(self) -> None:
        data = {"categories": {"users": ["user1"], "groups": ["group1"]}}
        categories = m.Categories.model_validate(data)
        tm.that(categories.get("users"), eq=["user1"])
        tm.that(categories.get("groups"), eq=["group1"])

    def test_categories_to_mapping(self) -> None:
        categories = m.Categories(categories={})
        categories.add_entries("users", ["user1"])
        tm.that(categories.to_mapping(), eq={"users": ["user1"]})

    def test_statistics_aggregate_empty(self) -> None:
        tm.that(self.StatisticsCount.aggregate([]), eq={})

    def test_statistics_aggregate_numbers(self) -> None:
        stats1 = self.StatisticsCount(count=10)
        stats2 = self.StatisticsCount(count=20)
        result = self.StatisticsCount.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(result), eq=True)
        tm.that(result["count"], eq=30)

    def test_statistics_aggregate_lists(self) -> None:
        stats1 = self.StatisticsItems(items=["a", "b"])
        stats2 = self.StatisticsItems(items=["c"])
        result = self.StatisticsItems.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(result), eq=True)
        tm.that(result["items"], eq=["a", "b", "c"])

    def test_statistics_aggregate_mixed(self) -> None:
        stats1 = self.StatisticsMixed(count=10, items=["a"], name="first")
        stats2 = self.StatisticsMixed(count=20, items=["b"], name="second")
        result = self.StatisticsMixed.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(result), eq=True)
        tm.that(result["count"], eq=30)
        tm.that(result["items"], eq=["a", "b"])
        tm.that(result["name"], eq="second")

    def test_statistics_aggregate_none_values(self) -> None:
        stats1 = self.StatisticsOptional(count=10, name="first")
        stats2 = self.StatisticsOptional(count=None, name=None)
        result = self.StatisticsOptional.aggregate([stats1, stats2])
        tm.that(FlextRuntime.is_dict_like(result), eq=True)
        tm.that(result["count"], eq=10)
        tm.that(result["name"], eq="first")

    def test_config_merge(self) -> None:
        config1 = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        config2 = self.ConfigFixture.model_validate({"timeout": 60})
        merged = config1.merge(config2)
        tm.that(merged.timeout, eq=60)
        tm.that(merged.retries, eq=3)

    def test_config_from_dict(self) -> None:
        config_data = t.ConfigMap(root={"timeout": 60})
        config = self.ConfigFixture.from_mapping(config_data)
        tm.that(config.timeout, eq=60)

    def test_config_to_dict(self) -> None:
        config = self.ConfigFixture.model_validate({"timeout": 60})
        tm.that(config.to_mapping()["timeout"], eq=60)

    def test_config_with_updates(self) -> None:
        config = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        updated = config.with_updates(timeout=60)
        tm.that(updated.timeout, eq=60)
        tm.that(updated.retries, eq=3)
        tm.that(config.timeout, eq=30)

    def test_config_diff(self) -> None:
        config1 = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        config2 = self.ConfigFixture.model_validate({"timeout": 60, "retries": 3})
        diff = config1.diff(config2)
        tm.that("timeout" in diff, eq=True)
        tm.that(diff["timeout"], eq=(30, 60))
        tm.that("retries" in diff, eq=False)

    def test_config_diff_all_different(self) -> None:
        config1 = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        config2 = self.ConfigFixture.model_validate({"timeout": 60, "retries": 5})
        diff = config1.diff(config2)
        tm.that(len(diff), eq=2)
        tm.that(diff["timeout"], eq=(30, 60))
        tm.that(diff["retries"], eq=(3, 5))

    def test_config_eq(self) -> None:
        config1 = self.ConfigFixture.model_validate({"timeout": 30})
        config2 = self.ConfigFixture.model_validate({"timeout": 30})
        config3 = self.ConfigFixture.model_validate({"timeout": 60})
        tm.that(config1 == config2, eq=True)
        tm.that(config1 != config3, eq=True)
        tm.that(config1 != "not a config", eq=True)

    def test_results_aggregate_empty(self) -> None:
        tm.that(self.ResultsProcessed.aggregate([]), eq={})

    def test_results_aggregate_numbers(self) -> None:
        result1 = self.ResultsProcessed(processed=10)
        result2 = self.ResultsProcessed(processed=20)
        aggregated = self.ResultsProcessed.aggregate([result1, result2])
        tm.that(FlextRuntime.is_dict_like(aggregated), eq=True)
        tm.that(aggregated["processed"], eq=30)

    def test_results_aggregate_lists(self) -> None:
        result1 = self.ResultsErrors(errors=["error1"])
        result2 = self.ResultsErrors(errors=["error2"])
        aggregated = self.ResultsErrors.aggregate([result1, result2])
        tm.that(FlextRuntime.is_dict_like(aggregated), eq=True)
        tm.that(aggregated["errors"], eq=["error1", "error2"])

    def test_results_aggregate_dicts(self) -> None:
        result1 = self.ResultsMetadata(metadata={"key1": "value1"})
        result2 = self.ResultsMetadata(metadata={"key2": "value2"})
        aggregated = self.ResultsMetadata.aggregate([result1, result2])
        tm.that(FlextRuntime.is_dict_like(aggregated), eq=True)
        tm.that(aggregated["metadata"], eq={"key1": "value1", "key2": "value2"})

    def test_results_aggregate_mixed(self) -> None:
        result1 = self.ResultsMixed(processed=10, errors=["a"], status="ok")
        result2 = self.ResultsMixed(processed=20, errors=["b"], status="done")
        aggregated = self.ResultsMixed.aggregate([result1, result2])
        tm.that(FlextRuntime.is_dict_like(aggregated), eq=True)
        tm.that(aggregated["processed"], eq=30)
        tm.that(aggregated["errors"], eq=["a", "b"])
        tm.that(aggregated["status"], eq="done")

    def test_results_aggregate_none_values(self) -> None:
        result1 = self.ResultsOptional(processed=10, status="ok")
        result2 = self.ResultsOptional(processed=None, status=None)
        aggregated = self.ResultsOptional.aggregate([result1, result2])
        tm.that(FlextRuntime.is_dict_like(aggregated), eq=True)
        tm.that(aggregated["processed"], eq=10)
        tm.that(aggregated["status"], eq="ok")

    def test_options_merge(self) -> None:
        options1 = self.OptionsFixture(verbose=False, color=True)
        options2 = self.OptionsFixture(verbose=True)
        merged = options1.merge(options2)
        tm.that(merged.verbose is True, eq=True)
        tm.that(merged.color is True, eq=True)

    def test_options_merge_all_fields(self) -> None:
        options1 = self.OptionsFixture(verbose=False, color=True)
        options2 = self.OptionsFixture(verbose=True, color=False)
        merged = options1.merge(options2)
        tm.that(merged.verbose is True, eq=True)
        tm.that(merged.color is False, eq=True)


__all__ = ["TestFlextModelsCollectionsCoverage100"]
