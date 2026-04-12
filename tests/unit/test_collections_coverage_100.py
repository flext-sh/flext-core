from __future__ import annotations

from pydantic import Field

from flext_tests import tm
from tests import m, t, u


class TestFlextModelsCollectionsCoverage100:
    class ConfigFixture(m.Config):
        """Test configuration with timeout and retries."""

        timeout: int = 30
        retries: int = 3

    class StatisticsCount(m.Statistics):
        """Statistics with count field."""

        count: int = 0

    class StatisticsItems(m.Statistics):
        """Statistics with items list."""

        items: t.StrSequence = Field(default_factory=list)

    class StatisticsMixed(m.Statistics):
        """Statistics with count, items, and name."""

        count: int = 0
        items: t.StrSequence = Field(default_factory=list)
        name: str = ""

    class StatisticsOptional(m.Statistics):
        """Statistics with optional count and name."""

        count: int | None = None
        name: str | None = None

    class OptionsFixture(m.Options):
        """Options with verbose and color flags."""

        verbose: bool = False
        color: bool = True

    def test_statistics_aggregate_empty(self) -> None:
        tm.that(self.StatisticsCount.aggregate([]), eq={})

    def test_statistics_aggregate_numbers(self) -> None:
        stats1 = self.StatisticsCount(count=10)
        stats2 = self.StatisticsCount(count=20)
        result = self.StatisticsCount.aggregate([stats1, stats2])
        tm.that(u.dict_like(result), eq=True)
        tm.that(result["count"], eq=30)

    def test_statistics_aggregate_lists(self) -> None:
        stats1 = self.StatisticsItems(items=["a", "b"])
        stats2 = self.StatisticsItems(items=["c"])
        result = self.StatisticsItems.aggregate([stats1, stats2])
        tm.that(u.dict_like(result), eq=True)
        tm.that(result["items"], eq=["a", "b", "c"])

    def test_statistics_aggregate_mixed(self) -> None:
        stats1 = self.StatisticsMixed(count=10, items=["a"], name="first")
        stats2 = self.StatisticsMixed(count=20, items=["b"], name="second")
        result = self.StatisticsMixed.aggregate([stats1, stats2])
        tm.that(u.dict_like(result), eq=True)
        tm.that(result["count"], eq=30)
        tm.that(result["items"], eq=["a", "b"])
        tm.that(result["name"], eq="second")

    def test_statistics_aggregate_none_values(self) -> None:
        stats1 = self.StatisticsOptional(count=10, name="first")
        stats2 = self.StatisticsOptional(count=None, name=None)
        result = self.StatisticsOptional.aggregate([stats1, stats2])
        tm.that(u.dict_like(result), eq=True)
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
        settings = self.ConfigFixture.from_mapping(config_data)
        tm.that(settings.timeout, eq=60)

    def test_config_to_dict(self) -> None:
        settings = self.ConfigFixture.model_validate({"timeout": 60})
        tm.that(settings.to_mapping()["timeout"], eq=60)

    def test_config_with_updates(self) -> None:
        settings = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        updated = settings.with_updates(timeout=60)
        tm.that(updated.timeout, eq=60)
        tm.that(updated.retries, eq=3)
        tm.that(settings.timeout, eq=30)

    def test_config_diff(self) -> None:
        config1 = self.ConfigFixture.model_validate({"timeout": 30, "retries": 3})
        config2 = self.ConfigFixture.model_validate({"timeout": 60, "retries": 3})
        diff = config1.diff(config2)
        tm.that(diff, has="timeout")
        tm.that(diff["timeout"], eq=(30, 60))
        tm.that("retries" not in diff, eq=True)

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
        tm.that(config1, eq=config2)
        tm.that(config1, ne=config3)
        tm.that(config1, ne="not a settings")

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


__all__: list[str] = ["TestFlextModelsCollectionsCoverage100"]
