"""Collection models for categorized data.

This module provides generic collection classes for categorizing items.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, computed_field

from flext_core._models.entity import FlextModelsEntity
from flext_core.runtime import FlextRuntime


class FlextModelsCollections:
    """Collection models container class."""

    class Categories[T](FlextModelsEntity.ArbitraryTypesModel):
        """Generic categorized collection with dynamic categories.

        Provides type-safe storage for items organized by category names.
        Uses PEP 695 type parameter syntax for Python 3.12+.

        Example:
            categories = Categories[Entry]()
            categories.add_entries("users", [user1, user2])
            categories.add_entries("groups", [group1])

            # Access
            users = categories.get_entries("users")
            total = categories.total_entries
            names = categories.category_names

        """

        model_config = ConfigDict(
            strict=True,
            validate_default=True,
            validate_assignment=True,
        )

        categories: dict[str, list[T]] = Field(
            default_factory=dict,
            description="Map of category name to list of items",
        )

        def get_entries(self, category: str) -> list[T]:
            """Get entries for a category, returns empty list if not found."""
            return self.categories.get(category, [])

        def add_entries(self, category: str, entries: list[T]) -> None:
            """Add entries to a category."""
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].extend(entries)

        def set_entries(self, category: str, entries: list[T]) -> None:
            """Set (replace) entries for a category."""
            self.categories[category] = entries

        def has_category(self, category: str) -> bool:
            """Check if category exists."""
            return category in self.categories

        def remove_category(self, category: str) -> None:
            """Remove a category."""
            if category in self.categories:
                del self.categories[category]

        @computed_field
        def category_names(self) -> list[str]:
            """Get list of all category names."""
            return list(self.categories.keys())

        @computed_field
        def total_entries(self) -> int:
            """Get total number of entries across all categories."""
            return sum(len(entries) for entries in self.categories.values())

        @computed_field
        def summary(self) -> dict[str, int]:
            """Get summary with entry count per category."""
            return {name: len(entries) for name, entries in self.categories.items()}

        def items(self) -> list[tuple[str, list[T]]]:
            """Iterate over categories."""
            return list(self.categories.items())

        def keys(self) -> list[str]:
            """Get category names."""
            return list(self.categories.keys())

        def values(self) -> list[list[T]]:
            """Get all entry lists."""
            return list(self.categories.values())

        def __getitem__(self, category: str) -> list[T]:
            """Dict-like access: categories['users']."""
            return self.categories[category]

        def __setitem__(self, category: str, entries: list[T]) -> None:
            """Dict-like assignment: categories['users'] = [...]."""
            self.categories[category] = entries

        def __contains__(self, category: str) -> bool:
            """Support 'in' operator: 'users' in categories."""
            return category in self.categories

        def __len__(self) -> int:
            """Number of categories."""
            return len(self.categories)

        def get(self, category: str, default: list[T] | None = None) -> list[T]:
            """Dict-like get with default."""
            return self.categories.get(category, default if default is not None else [])

        @classmethod
        def from_dict(
            cls,
            data: dict[str, list[T]],
        ) -> FlextModelsCollections.Categories[T]:
            """Create from existing dict."""
            return cls(categories=data)

        def to_dict(self) -> dict[str, list[T]]:
            """Convert to dict."""
            return self.categories.copy()

    class Statistics(FlextModelsEntity.Value):
        """Base for statistics models (frozen Value)."""

        @classmethod
        def aggregate(cls, stats_list: list[Self]) -> dict[str, object]:
            """Aggregate multiple statistics instances.

            Combines statistics by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Using max for comparable values
            - Keeping last value for other types

            Example:
                stats1 = Stats(count=10, items=['a'])
                stats2 = Stats(count=20, items=['b'])
                result = Stats.aggregate([stats1, stats2])
                # {'count': 30, 'items': ['a', 'b']}

            """
            if not stats_list:
                return {}

            result: dict[str, object] = {}
            first = stats_list[0].model_dump()

            for key in first:
                values = [s.model_dump().get(key) for s in stats_list]

                # Filter None values
                non_none = [v for v in values if v is not None]
                if not non_none:
                    result[key] = None
                    continue

                first_val = non_none[0]

                # Sum numbers
                if isinstance(first_val, int | float):
                    result[key] = sum(v for v in non_none if isinstance(v, int | float))
                # Concatenate lists
                elif FlextRuntime.is_list_like(first_val):
                    combined: list[object] = []
                    for v in non_none:
                        if FlextRuntime.is_list_like(v):
                            combined.extend(v)
                    result[key] = combined
                # Keep last for other types
                else:
                    result[key] = non_none[-1]

            return result

    class Config(FlextModelsEntity.ArbitraryTypesModel):
        """Base for configuration models - mutable Pydantic v2 model.

        Pydantic v2 models are not hashable by default when not frozen.
        Explicitly set __hash__ = None to make this clear to type checkers.
        """

        def __hash__(self) -> int:
            """Make Config instances unhashable (mutable models should not be hashable).

            Returns:
                Never returns - always raises TypeError

            """
            msg = f"{self.__class__.__name__} instances are not hashable"
            raise TypeError(msg)

        def merge(
            self,
            other: Self,
        ) -> Self:
            """Merge another config into this one (other takes precedence)."""
            merged = self.model_dump()
            merged.update(other.model_dump(exclude_unset=True))
            return self.__class__(**merged)

        @classmethod
        def from_dict(cls, data: dict[str, object]) -> Self:
            """Create config from dict."""
            return cls.model_validate(data)

        def to_dict(self) -> dict[str, object]:
            """Convert to dict."""
            return self.model_dump()

        def with_updates(self, **kwargs: object) -> Self:
            """Return new config with updated fields."""
            data = self.model_dump()
            data.update(kwargs)
            return self.__class__(**data)

        def diff(
            self,
            other: FlextModelsCollections.Config,
        ) -> dict[str, tuple[object, object]]:
            """Get differences between configs for debugging.

            Returns dict mapping field names to (self_value, other_value) tuples
            for fields that differ between the two configs.

            Example:
                config1 = Config(timeout=30, retries=3)
                config2 = Config(timeout=60, retries=3)
                diff = config1.diff(config2)
                # {'timeout': (30, 60)}

            """
            self_data = self.model_dump()
            other_data = other.model_dump()
            differences: dict[str, tuple[object, object]] = {}

            all_keys = set(self_data.keys()) | set(other_data.keys())
            for key in all_keys:
                self_val = self_data.get(key)
                other_val = other_data.get(key)
                if self_val != other_val:
                    differences[key] = (self_val, other_val)

            return differences

        def __eq__(self, other: object) -> bool:
            """Compare configs by value."""
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.model_dump() == other.model_dump()

    class Results(FlextModelsEntity.Value):
        """Base for result models."""

        @classmethod
        def _aggregate_values(cls, values: list[object | None]) -> object:
            """Aggregate a list of values based on type."""
            non_none = [v for v in values if v is not None]
            if not non_none:
                return None

            first_val = non_none[0]

            # Sum numbers
            if isinstance(first_val, int | float):
                return sum(v for v in non_none if isinstance(v, int | float))

            # Concatenate lists
            if FlextRuntime.is_list_like(first_val):
                combined: list[object] = []
                for v in non_none:
                    if FlextRuntime.is_list_like(v):
                        combined.extend(v)
                return combined

            # Merge dicts
            if FlextRuntime.is_dict_like(first_val):
                merged: dict[str, object] = {}
                for v in non_none:
                    if FlextRuntime.is_dict_like(v):
                        merged.update(v)
                return merged

            # Keep last for other types
            return non_none[-1]

        @classmethod
        def aggregate(cls, results: list[Self]) -> dict[str, object]:
            """Aggregate multiple result instances.

            Combines results by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Merging dicts (later values override)
            - Keeping last value for other types

            Example:
                result1 = MyResult(processed=10, errors=['a'])
                result2 = MyResult(processed=20, errors=['b'])
                combined = MyResult.aggregate([result1, result2])
                # {'processed': 30, 'errors': ['a', 'b']}

            """
            if not results:
                return {}

            aggregated: dict[str, object] = {}
            first = results[0].model_dump()

            for key in first:
                values = [r.model_dump().get(key) for r in results]
                aggregated[key] = cls._aggregate_values(values)

            return aggregated

    class Rules(FlextModelsEntity.ArbitraryTypesModel):
        """Base for rules models."""

        model_config = ConfigDict(extra="forbid", validate_assignment=True)

    class Options(FlextModelsEntity.Value):
        """Base for options models."""

        def merge(self, other: Self) -> Self:
            """Merge another options into this one (other takes precedence).

            Creates a new Options instance with values from other overriding self.
            Uses exclude_unset to only override explicitly set values.

            Example:
                defaults = Options(verbose=False, timeout=30)
                overrides = Options(timeout=60)
                merged = defaults.merge(overrides)
                # Options(verbose=False, timeout=60)

            """
            base_data = self.model_dump()
            override_data = other.model_dump(exclude_unset=True)
            base_data.update(override_data)
            return self.__class__(**base_data)

        def with_only(self, *fields: str) -> dict[str, object]:
            """Return dict with only specified fields.

            Useful for extracting subset of options for specific operations.

            Example:
                opts = Options(verbose=True, timeout=30, retries=3)
                subset = opts.with_only('timeout', 'retries')
                # {'timeout': 30, 'retries': 3}

            """
            data = self.model_dump()
            return {k: v for k, v in data.items() if k in fields}
