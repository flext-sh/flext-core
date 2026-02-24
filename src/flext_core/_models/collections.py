"""Collection models for categorized data.

TIER 0.5: Depends only on base.py (Tier 0).
Uses inline type checks to avoid importing runtime.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime
from typing import Self

from pydantic import ConfigDict, Field

from flext_core._models.base import FlextModelFoundation
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsCollections:
    """Collection models container class."""

    class Categories(FlextModelFoundation.ArbitraryTypesModel):
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

        categories: MutableMapping[str, list[t.ConfigMapValue]] = Field(
            default_factory=dict,
            description="Map of category name to list of items",
        )

        @classmethod
        def __class_getitem__(
            cls,
            typevar_values: type | tuple[type, ...],
        ) -> type[FlextModelsCollections.Categories]:
            _ = typevar_values
            return cls

        def __len__(self) -> int:
            """Return total number of entries across all categories.

            Returns:
                int: Total number of entries

            """
            return sum(len(entries) for entries in self.categories.values())

        def get_entries(self, category: str) -> list[t.ConfigMapValue]:
            """Get entries for a category, returns empty list if not found.

            Returns:
                list[T]: List of entries for the category, or empty list if not found.

            """
            return self.categories.get(category, [])

        def has_category(self, category: str) -> bool:
            """Check if a category exists.

            Args:
                category: Category name to check

            Returns:
                bool: True if category exists, False otherwise

            """
            return category in self.categories

        def get(
            self,
            category: str,
            default: list[t.ConfigMapValue] | None = None,
        ) -> list[t.ConfigMapValue]:
            """Get entries for a category with optional default (dict-like interface).

            Args:
                category: Category name
                default: Default value if category not found (None returns empty list)

            Returns:
                list[T]: List of entries for the category, or default/empty
                list if not found.

            """
            if default is None:
                return self.categories.get(category, [])
            return self.categories.get(category, default)

        def add_entries(
            self,
            category: str,
            entries: Sequence[t.ConfigMapValue],
        ) -> None:
            """Add entries to a category.

            Args:
                category: Category name
                entries: Sequence of entries (accepts subtypes due to covariance)

            """
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].extend(entries)

        def set_entries(
            self,
            category: str,
            entries: Sequence[t.ConfigMapValue],
        ) -> None:
            """Set entries for a category (replaces existing).

            Args:
                category: Category name
                entries: Sequence of entries

            """
            self.categories[category] = list(entries)

        def remove_category(self, category: str) -> None:
            """Remove a category and all its entries.

            Args:
                category: Category name to remove

            """
            self.categories.pop(category, None)

        def clear(self) -> None:
            """Clear all categories and entries."""
            self.categories.clear()

        @property
        def total_entries(self) -> int:
            """Get total number of entries across all categories.

            Returns:
                int: Total count of entries

            """
            return sum(len(entries) for entries in self.categories.values())

        @property
        def category_names(self) -> list[str]:
            """Get list of all category names.

            Returns:
                list[str]: List of category names

            """
            return list(self.categories.keys())

        @classmethod
        def from_dict(
            cls,
            data: Mapping[str, Sequence[t.ConfigMapValue]],
        ) -> Self:
            """Create Categories instance from dictionary.

            Args:
                data: Dictionary mapping category names to sequences of items

            Returns:
                Categories instance

            """
            instance = cls()
            for category, entries in data.items():
                instance.add_entries(category, entries)
            return instance

        def to_dict(self) -> Mapping[str, Sequence[t.ConfigMapValue]]:
            """Convert categories to dictionary representation.

            Normalizes list[T] to Sequence[t.GuardInputValue] for type compatibility.
            Uses inline _normalize_to_general_value to avoid circular import.

            Returns:
                CategoryGroupsMapping: Dictionary representation of categories.

            """
            # Normalize list[T] to Sequence[t.GuardInputValue] for type compatibility
            result: MutableMapping[str, Sequence[t.ConfigMapValue]] = {}
            for key, value_list in self.categories.items():
                # Normalize each item in the list to t.GuardInputValue
                # First convert T to PayloadValue, then normalize
                normalized_list: list[t.ConfigMapValue] = []
                for item in value_list:
                    normalized = FlextRuntime.normalize_to_general_value(item)
                    normalized_list.append(normalized)
                result[key] = normalized_list
            return result

    class Statistics(FlextModelFoundation.FrozenValueModel):
        """Base for statistics models (frozen Value)."""

        @classmethod
        def _resolve_aggregate_conflict(
            cls,
            existing: t.ConfigMapValue,
            value: t.ConfigMapValue,
        ) -> t.ConfigMapValue:
            """Resolve conflict when aggregating two statistic values.

            Args:
                existing: Existing value in result dict
                value: New value to aggregate

            Returns:
                Resolved value (sum for numeric, concatenated for lists,
                last for others)

            """
            # Filter out None values for comparison
            non_none = [v for v in [existing, value] if v is not None]
            if not non_none:
                # None is part of t.GuardInputValue, so this is valid
                return None

            first_val = non_none[0]
            # Sum numeric values
            if isinstance(first_val, (int, float)) and not isinstance(first_val, bool):
                numeric_values: list[int | float] = [
                    v
                    for v in non_none
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                return sum(numeric_values)
            # Concatenate lists
            if FlextRuntime.is_list_like(first_val):
                combined: list[str | int | float | bool | datetime | None] = []
                for v in non_none:
                    if FlextRuntime.is_list_like(v) and v.__class__ not in {str, bytes}:
                        combined.extend(
                            item
                            for item in v
                            if item is None
                            or isinstance(
                                item,
                                (str, int, float, bool, datetime),
                            )
                        )
                return combined
            # Keep last for other types
            return non_none[-1]

        @classmethod
        def from_dict(
            cls,
            data: t.ConfigMap,
        ) -> Self:
            """Create Statistics instance from dictionary.

            Args:
                data: Dictionary with statistics data

            Returns:
                Statistics instance

            """
            return cls.model_validate(data.root)

        @classmethod
        def aggregate(
            cls,
            stats_list: list[Self],
        ) -> Mapping[str, t.ConfigMapValue]:
            """Aggregate multiple statistics instances (ConfigurationMapping pattern).

            Combines statistics by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Using max for comparable values
            - Keeping last value for other types

            Returns:
                t.GuardInputValue: Aggregated statistics dictionary.

            Example:
                stats1 = Stats(count=10, items=['a'])
                stats2 = Stats(count=20, items=['b'])
                result = Stats.aggregate([stats1, stats2])
                # {'count': 30, 'items': ['a', 'b']}

            """
            if not stats_list:
                return {}

            # Start with first stats as base
            result: MutableMapping[str, t.ConfigMapValue] = {}
            for stats in stats_list:
                stats_dict = stats.model_dump()
                for key, value in stats_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_aggregate_conflict(
                            result[key],
                            value,
                        )

            # Normalize result dict to PayloadValue's dict type
            normalized_result: MutableMapping[str, t.ConfigMapValue] = {}
            for key, value in result.items():
                # Filter to types matching PayloadValue
                if value is None or value.__class__ in {
                    str,
                    int,
                    float,
                    bool,
                    datetime,
                }:
                    normalized_result[key] = value
                elif isinstance(value, list):
                    filtered = [
                        item
                        for item in value
                        if item is None
                        or isinstance(
                            item,
                            (str, int, float, bool, datetime),
                        )
                    ]
                    normalized_result[key] = filtered
            return normalized_result

    class Rules(FlextModelFoundation.ArbitraryTypesModel):
        """Base for rules models (mutable)."""

        @classmethod
        def merge(
            cls,
            *rules: Self,
        ) -> Self:
            """Merge multiple rules instances.

            Args:
                *rules: Rules instances to merge

            Returns:
                Merged rules instance

            """
            merged_data: MutableMapping[str, t.ConfigMapValue] = {}
            for rule in rules:
                merged_data.update(rule.model_dump())
            return cls(**merged_data)

    class Results(FlextModelFoundation.ArbitraryTypesModel):
        """Base for results models (mutable)."""

        @classmethod
        def _sum_numeric_values(
            cls,
            non_none: list[t.ConfigMapValue],
        ) -> int | float | None:
            """Sum numeric values excluding booleans.

            Args:
                non_none: List of non-None values

            Returns:
                Sum of numeric values, or None if no numeric values found

            """
            numeric_values: list[int | float] = [
                v
                for v in non_none
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return sum(numeric_values) if numeric_values else None

        @classmethod
        def _concatenate_lists(
            cls,
            non_none: list[t.ConfigMapValue],
        ) -> list[str | int | float | bool | datetime | None]:
            """Concatenate list-like values.

            Args:
                non_none: List of non-None values

            Returns:
                Combined list matching PayloadValue's list type

            """
            combined: list[str | int | float | bool | datetime | None] = []
            for v in non_none:
                if FlextRuntime.is_list_like(v) and v.__class__ not in {str, bytes}:
                    combined.extend(
                        item
                        for item in v
                        if item is None
                        or isinstance(
                            item,
                            (str, int, float, bool, datetime),
                        )
                    )
            return combined

        @classmethod
        def _merge_dicts(
            cls,
            non_none: list[t.ConfigMapValue],
        ) -> Mapping[str, t.ConfigMapValue]:
            """Merge dict-like values.

            Args:
                non_none: List of non-None values

            Returns:
                Merged dictionary matching PayloadValue's dict type

            """
            merged: MutableMapping[str, t.ConfigMapValue] = {}
            for v in non_none:
                if FlextRuntime.is_dict_like(v):
                    # Explicit check for Mapping - normalize each value
                    for key, val in v.items():
                        # Only add values that match the dict value types
                        if val is None or val.__class__ in {
                            str,
                            int,
                            float,
                            bool,
                            datetime,
                        }:
                            merged[str(key)] = val
                        elif isinstance(val, list):
                            # Filter list items
                            filtered = [
                                item
                                for item in val
                                if item is None
                                or isinstance(
                                    item,
                                    (str, int, float, bool, datetime),
                                )
                            ]
                            merged[str(key)] = filtered
            return merged

        @classmethod
        def _resolve_aggregate_conflict(
            cls,
            existing: t.ConfigMapValue,
            value: t.ConfigMapValue,
        ) -> t.ConfigMapValue:
            """Resolve conflict when aggregating two result values.

            Args:
                existing: Existing value in result dict
                value: New value to aggregate

            Returns:
                Resolved value (sum for numeric, concatenated for lists,
                merged for dicts, last for others)

            """
            # Filter out None values for comparison
            # Explicitly type as list to match helper method signatures
            non_none: list[t.ConfigMapValue] = [
                v for v in [existing, value] if v is not None
            ]
            if not non_none:
                # None is part of t.GuardInputValue, so this is valid
                return None

            first_val = non_none[0]
            # Check for bool first - bool is a subclass of int in Python
            # but we don't want to sum boolean values
            if first_val.__class__ is bool:
                return non_none[-1]
            # Sum numeric values (but not bool)
            if first_val.__class__ in {int, float}:
                numeric_sum = cls._sum_numeric_values(non_none)
                return numeric_sum if numeric_sum is not None else non_none[-1]
            # Concatenate lists
            if FlextRuntime.is_list_like(first_val):
                return cls._concatenate_lists(non_none)
            # Merge dicts
            if FlextRuntime.is_dict_like(first_val):
                return cls._merge_dicts(non_none)
            # Keep last for other types
            return non_none[-1]

        @classmethod
        def combine(
            cls,
            *results: Self,
        ) -> Self:
            """Combine multiple results instances.

            Args:
                *results: Results instances to combine

            Returns:
                Combined results instance

            """
            combined_data: MutableMapping[str, t.ConfigMapValue] = {}
            for result in results:
                combined_data.update(result.model_dump())
            return cls(**combined_data)

        @classmethod
        def aggregate(cls, results_list: list[Self]) -> Mapping[str, t.ConfigMapValue]:
            """Aggregate multiple results instances (ConfigurationMapping pattern).

            Combines results by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Merging dicts
            - Keeping last value for other types

            Returns:
                t.GuardInputValue: Aggregated results dictionary.

            Example:
                result1 = Results(processed=10, errors=['a'])
                result2 = Results(processed=20, errors=['b'])
                aggregated = Results.aggregate([result1, result2])
                # {'processed': 30, 'errors': ['a', 'b']}

            """
            if not results_list:
                return {}

            # Start with first result as base
            result: MutableMapping[str, t.ConfigMapValue] = {}
            for res in results_list:
                res_dict = res.model_dump()
                for key, value in res_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_aggregate_conflict(
                            result[key],
                            value,
                        )

            # Return result directly - PayloadValue allows Mapping[str, PayloadValue]
            return result

    class Options(FlextModelFoundation.ArbitraryTypesModel):
        """Base for options models (mutable)."""

        @classmethod
        def _resolve_merge_conflict(
            cls,
            existing: t.ConfigMapValue,
            value: t.ConfigMapValue,
        ) -> t.ConfigMapValue:
            """Resolve conflict when merging two option values.

            Args:
                existing: Existing value in result dict
                value: New value to merge

            Returns:
                Resolved value (sum for numeric, concatenated for lists,
                last for others including booleans)

            """
            # Filter out None values for comparison
            non_none = [v for v in [existing, value] if v is not None]
            if not non_none:
                # None is part of t.GuardInputValue, so this is valid
                return None

            first_val = non_none[0]
            # Keep last for booleans (don't sum them - True + True = 2 which is invalid)
            if first_val.__class__ is bool:
                return non_none[-1]
            # Sum numeric values (int, float only, not bool)
            if first_val.__class__ in {int, float}:
                numeric_values: list[int | float] = [
                    v
                    for v in non_none
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_values:
                    return sum(numeric_values)
            # Concatenate lists
            if FlextRuntime.is_list_like(first_val):
                combined: list[t.ConfigMapValue] = []
                for v in non_none:
                    if FlextRuntime.is_list_like(v) and v.__class__ not in {str, bytes}:
                        # Explicit check for Sequence (excluding str)
                        for item in v:
                            normalized = FlextRuntime.normalize_to_general_value(item)
                            combined.append(normalized)
                return combined
            # Keep last for other types
            return non_none[-1]

        def merge(self, *options: Self) -> Self:
            """Merge this instance with other options instances (instance method).

            Convenience method that delegates to merge_options classmethod.

            Args:
                *options: Options instances to merge with self

            Returns:
                Merged options instance

            """
            return self.__class__.merge_options(self, *options)

        @classmethod
        def merge_options(
            cls,
            *options: Self,
        ) -> Self:
            """Merge multiple options instances with conflict resolution.

            Business Rule: Merges options with last-wins strategy for conflicts.
            For numeric values, sums them. For lists, concatenates them.
            For other types, keeps the last value.

            Args:
                *options: Options instances to merge

            Returns:
                Merged options instance

            """
            if not options:
                return cls()

            # Start with first options as base
            result: MutableMapping[str, t.ConfigMapValue] = {}
            for opt in options:
                opt_dict = opt.model_dump()
                for key, value in opt_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_merge_conflict(result[key], value)

            # Normalize result dict to t.GuardInputValue and create instance
            normalized_result: MutableMapping[str, t.ConfigMapValue] = {}
            for key, value in result.items():
                normalized_result[key] = FlextRuntime.normalize_to_general_value(value)
            # Create new instance from normalized dict
            return cls(**normalized_result)

    class Config(FlextModelFoundation.ArbitraryTypesModel):
        """Base for configuration models - mutable Pydantic v2 model.

        Pydantic v2 models are not hashable by default when not frozen.
        Type checkers understand that non-frozen models are not hashable.
        This is intentional - Config models are mutable and should not be hashable.

        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="forbid",
            validate_assignment=True,
        )

        def __hash__(self) -> int:
            """Raise TypeError to indicate this class is not hashable.

            Config models are mutable and should not be used as dict keys or set elements.
            """
            msg = f"{self.__class__.__name__} objects are not hashable"
            raise TypeError(msg)

        @classmethod
        def from_mapping(
            cls,
            mapping: t.ConfigMap,
        ) -> Self:
            """Create Config instance from mapping.

            Args:
                mapping: Mapping with configuration data

            Returns:
                Config instance

            """
            # Pydantic v2: Use model_validate for type-safe creation from mapping
            # Convert Mapping to dict for model_validate
            mapping_dict = dict(mapping)
            return cls.model_validate(mapping_dict)

        def to_mapping(self) -> t.ConfigMap:
            """Convert Config to mapping.

            Returns:
                ConfigurationMapping: Mapping representation

            """
            return t.ConfigMap.model_validate(self.model_dump())

        @classmethod
        def from_dict(
            cls,
            data: t.ConfigMap,
        ) -> Self:
            """Create Config instance from dictionary.

            Args:
                data: Dictionary with configuration data

            Returns:
                Config instance

            """
            # Pydantic v2: Use model_validate for type-safe creation from dict
            # Convert Mapping to dict for model_validate
            data_dict = dict(data)
            return cls.model_validate(data_dict)

        def merge(self, other: Self) -> Self:
            """Merge this config with another config.

            Args:
                other: Another config instance to merge with

            Returns:
                Merged config instance (other values override self)

            """
            self_dict = self.model_dump()
            other_dict = other.model_dump()
            merged_dict = {**self_dict, **other_dict}
            return self.__class__(**merged_dict)

        def diff(
            self,
            other: Self,
        ) -> Mapping[str, tuple[t.GuardInputValue, t.GuardInputValue]]:
            """Compute differences between this config and another.

            Args:
                other: Another config instance to compare with

            Returns:
                Dictionary mapping field names to (self_value, other_value) tuples
                for fields that differ

            """
            self_dict = self.model_dump()
            other_dict = other.model_dump()
            differences: MutableMapping[
                str,
                tuple[t.GuardInputValue, t.GuardInputValue],
            ] = {}
            all_keys = set(self_dict.keys()) | set(other_dict.keys())
            for key in all_keys:
                self_val = self_dict.get(key)
                other_val = other_dict.get(key)
                if self_val != other_val:
                    differences[key] = (self_val, other_val)
            return differences

        def with_updates(self, **updates: t.GuardInputValue) -> Self:
            """Create a new config instance with updated values.

            Args:
                **updates: Field updates to apply

            Returns:
                New config instance with updates applied

            """
            current_dict = self.model_dump()
            updated_dict = {**current_dict, **updates}
            return self.__class__(**updated_dict)

        def __eq__(self, other: object) -> bool:
            """Compare configs by value.

            Args:
                other: Object to compare with

            Returns:
                True if configs are equal by value, False otherwise

            """
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.model_dump() == other.model_dump()

    class ParseOptions(FlextModelFoundation.ArbitraryTypesModel):
        """Options for string parsing operations."""

        strip: bool = Field(
            default=True,
            description="Strip whitespace from components",
        )
        remove_empty: bool = Field(
            default=True,
            description="Remove empty components from result",
        )
        validator: Callable[[str], bool] | None = Field(
            default=None,
            description="Optional validator function for components",
        )

    class PatternApplicationParams(FlextModelFoundation.ArbitraryTypesModel):
        """Parameters for regex pattern application."""

        text: str = Field(description="Text to apply pattern to")
        pattern: str = Field(description="Regex pattern")
        replacement: str = Field(description="Replacement string")
        flags: int = Field(default=0, description="Regex flags")
        pattern_index: int = Field(description="Index of pattern in pipeline")
        total_patterns: int = Field(description="Total number of patterns")


__all__ = ["FlextModelsCollections"]
