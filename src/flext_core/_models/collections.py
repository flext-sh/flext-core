"""Collection models for categorized data.

TIER 0.5: Depende apenas de base.py (Tier 0).
Usa verificações inline de tipo para evitar import de runtime.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Self, TypeGuard

from pydantic import ConfigDict, Field

from flext_core._models.base import FlextModelsBase

if TYPE_CHECKING:
    from flext_core.typings import t


# Tier 0.5 - Helper functions inline (avoid FlextRuntime import)
def _is_list_like(value: object) -> TypeGuard[Sequence[t.GeneralValueType]]:
    """Check if value is list-like (but not string/bytes) with type narrowing.

    Args:
        value: Any object to check

    Returns:
        TypeGuard[Sequence[t.GeneralValueType]]: True if value is Sequence
        and not str/bytes.

    """
    # Inline type check to avoid circular import with guards.py -> runtime.py
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_dict_like(
    value: object,
) -> TypeGuard[Mapping[str, t.GeneralValueType]]:
    """Check if value is dict-like with type narrowing.

    Args:
        value: Any object to check

    Returns:
        TypeGuard[Mapping[str, t.GeneralValueType]]: True if value is Mapping.

    """
    # Inline type check to avoid circular import with guards.py -> runtime.py
    return isinstance(value, Mapping)


def _normalize_to_general_value(val: object) -> t.GeneralValueType:
    """Normalize any value to t.GeneralValueType recursively (inline implementation).

    This is an inline implementation of FlextRuntime.normalize_to_general_value
    to avoid circular import. Collections.py is Tier 0.5 and cannot import runtime.py.

    Args:
        val: Any value to normalize

    Returns:
        Normalized value compatible with GeneralValueType

    """
    # Handle primitives
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    # Handle dict-like values recursively
    if _is_dict_like(val):
        result: t.Types.ConfigurationDict = {}
        dict_v = dict(val.items()) if hasattr(val, "items") else dict(val)
        for k, v in dict_v.items():
            if isinstance(k, str):
                result[k] = _normalize_to_general_value(v)
        return result
    # Handle list-like values recursively
    if _is_list_like(val):
        return [_normalize_to_general_value(item) for item in val]
    # For arbitrary objects, convert to string representation
    return str(val)


class FlextModelsCollections:
    """Collection models container class."""

    class Categories[T](FlextModelsBase.ArbitraryTypesModel):
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
            """Get entries for a category, returns empty list if not found.

            Returns:
                list[T]: List of entries for the category, or empty list if not found.

            """
            return self.categories.get(category, [])

        def add_entries(self, category: str, entries: Sequence[T]) -> None:
            """Add entries to a category.

            Args:
                category: Category name
                entries: Sequence of entries (accepts subtypes due to covariance)

            """
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].extend(entries)

        def set_entries(self, category: str, entries: Sequence[T]) -> None:
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

        def to_dict(self) -> t.Types.StringSequenceGeneralValueDict:
            """Convert categories to dictionary representation.

            Normalizes list[T] to Sequence[GeneralValueType] for type compatibility.
            Uses inline _normalize_to_general_value to avoid circular import.

            Returns:
                CategoryGroupsMapping: Dictionary representation of categories.

            """
            # Normalize list[T] to Sequence[GeneralValueType] for type compatibility
            result: t.Types.StringSequenceGeneralValueDict = {}
            for key, value_list in self.categories.items():
                # Normalize each item in the list to GeneralValueType
                # Use inline helper to avoid circular import with runtime.py
                normalized_list: list[t.GeneralValueType] = [
                    _normalize_to_general_value(item) for item in value_list
                ]
                result[key] = normalized_list
            return result

    class Statistics(FlextModelsBase.FrozenValueModel):
        """Base for statistics models (frozen Value)."""

        @classmethod
        def _resolve_aggregate_conflict(
            cls,
            existing: t.GeneralValueType,
            value: t.GeneralValueType,
        ) -> t.GeneralValueType:
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
                return None  # type: ignore[return-value]

            first_val = non_none[0]
            # Sum numeric values
            if isinstance(first_val, (int, float)):
                numeric_values: list[int | float] = [
                    v for v in non_none if isinstance(v, (int, float))
                ]
                return sum(numeric_values)
            # Concatenate lists
            if _is_list_like(first_val):
                combined: list[t.GeneralValueType] = []
                for v in non_none:
                    if _is_list_like(v):
                        # Normalize each item to GeneralValueType
                        for item in v:
                            normalized = _normalize_to_general_value(item)
                            combined.append(normalized)
                return combined
            # Keep last for other types
            return non_none[-1]

        @classmethod
        def from_dict(
            cls,
            data: Mapping[str, t.GeneralValueType],
        ) -> Self:
            """Create Statistics instance from dictionary.

            Args:
                data: Dictionary with statistics data

            Returns:
                Statistics instance

            """
            return cls(**data)

        @classmethod
        def aggregate(cls, stats_list: list[Self]) -> t.GeneralValueType:
            """Aggregate multiple statistics instances (ConfigurationMapping pattern).

            Combines statistics by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Using max for comparable values
            - Keeping last value for other types

            Returns:
                GeneralValueType: Aggregated statistics dictionary.

            Example:
                stats1 = Stats(count=10, items=['a'])
                stats2 = Stats(count=20, items=['b'])
                result = Stats.aggregate([stats1, stats2])
                # {'count': 30, 'items': ['a', 'b']}

            """
            if not stats_list:
                return {}

            # Start with first stats as base
            result: dict[str, t.GeneralValueType] = {}
            for stats in stats_list:
                stats_dict = stats.model_dump()
                for key, value in stats_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_aggregate_conflict(
                            result[key], value
                        )

            # Normalize result dict to GeneralValueType
            normalized_result: t.Types.ConfigurationDict = {}
            for key, value in result.items():
                normalized_result[key] = _normalize_to_general_value(value)
            return normalized_result

    class Rules(FlextModelsBase.ArbitraryTypesModel):
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
            merged_data: dict[str, t.GeneralValueType] = {}
            for rule in rules:
                merged_data.update(rule.model_dump())
            return cls(**merged_data)

    class Results(FlextModelsBase.ArbitraryTypesModel):
        """Base for results models (mutable)."""

        @classmethod
        def _resolve_aggregate_conflict(
            cls,
            existing: t.GeneralValueType,
            value: t.GeneralValueType,
        ) -> t.GeneralValueType:
            """Resolve conflict when aggregating two result values.

            Args:
                existing: Existing value in result dict
                value: New value to aggregate

            Returns:
                Resolved value (sum for numeric, concatenated for lists,
                merged for dicts, last for others)

            """
            # Filter out None values for comparison
            non_none = [v for v in [existing, value] if v is not None]
            if not non_none:
                return None  # type: ignore[return-value]

            first_val = non_none[0]
            # Sum numeric values
            if isinstance(first_val, (int, float)):
                numeric_values: list[int | float] = [
                    v for v in non_none if isinstance(v, (int, float))
                ]
                return sum(numeric_values)
            # Concatenate lists
            if _is_list_like(first_val):
                combined: list[t.GeneralValueType] = []
                for v in non_none:
                    if _is_list_like(v):
                        # Normalize each item to GeneralValueType
                        for item in v:
                            normalized = _normalize_to_general_value(item)
                            combined.append(normalized)
                return combined
            # Merge dicts
            if _is_dict_like(first_val):
                merged: t.Types.ConfigurationDict = {}
                for v in non_none:
                    if _is_dict_like(v):
                        # Type narrowing: v is Mapping[str, GeneralValueType]
                        dict_v = dict(v.items()) if hasattr(v, "items") else dict(v)
                        merged.update(dict_v)
                return merged
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
            combined_data: dict[str, t.GeneralValueType] = {}
            for result in results:
                combined_data.update(result.model_dump())
            return cls(**combined_data)

        @classmethod
        def aggregate(cls, results_list: list[Self]) -> t.GeneralValueType:
            """Aggregate multiple results instances (ConfigurationMapping pattern).

            Combines results by:
            - Summing numeric values (int, float)
            - Concatenating lists
            - Merging dicts
            - Keeping last value for other types

            Returns:
                GeneralValueType: Aggregated results dictionary.

            Example:
                result1 = Results(processed=10, errors=['a'])
                result2 = Results(processed=20, errors=['b'])
                aggregated = Results.aggregate([result1, result2])
                # {'processed': 30, 'errors': ['a', 'b']}

            """
            if not results_list:
                return {}

            # Start with first result as base
            result: dict[str, t.GeneralValueType] = {}
            for res in results_list:
                res_dict = res.model_dump()
                for key, value in res_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_aggregate_conflict(
                            result[key], value
                        )

            # Normalize result dict to GeneralValueType
            normalized_result: t.Types.ConfigurationDict = {}
            for key, value in result.items():
                normalized_result[key] = _normalize_to_general_value(value)
            return normalized_result

    class Options(FlextModelsBase.ArbitraryTypesModel):
        """Base for options models (mutable)."""

        @classmethod
        def _resolve_merge_conflict(
            cls,
            existing: t.GeneralValueType,
            value: t.GeneralValueType,
        ) -> t.GeneralValueType:
            """Resolve conflict when merging two option values.

            Args:
                existing: Existing value in result dict
                value: New value to merge

            Returns:
                Resolved value (sum for numeric, concatenated for lists,
                last for others)

            """
            # Filter out None values for comparison
            non_none = [v for v in [existing, value] if v is not None]
            if not non_none:
                return None  # type: ignore[return-value]

            first_val = non_none[0]
            # Sum numeric values
            if isinstance(first_val, (int, float)):
                numeric_values: list[int | float] = [
                    v for v in non_none if isinstance(v, (int, float))
                ]
                return sum(numeric_values)
            # Concatenate lists
            if _is_list_like(first_val):
                combined: list[t.GeneralValueType] = []
                for v in non_none:
                    if _is_list_like(v):
                        # Normalize each item to GeneralValueType
                        for item in v:
                            normalized = _normalize_to_general_value(item)
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
            return self.merge_options(self, *options)

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
            result: dict[str, t.GeneralValueType] = {}
            for opt in options:
                opt_dict = opt.model_dump()
                for key, value in opt_dict.items():
                    if key not in result:
                        result[key] = value
                    else:
                        # Conflict resolution - delegate to helper method
                        result[key] = cls._resolve_merge_conflict(result[key], value)

            # Normalize result dict to GeneralValueType and create instance
            normalized_result: t.Types.ConfigurationDict = {}
            for key, value in result.items():
                normalized_result[key] = _normalize_to_general_value(value)
            # Create new instance from normalized dict
            return cls(**normalized_result)

    class Config(FlextModelsBase.ArbitraryTypesModel):
        """Base for configuration models - mutable Pydantic v2 model.

        Pydantic v2 models are not hashable by default when not frozen.
        Explicitly set __hash__ = None to make this clear to type checkers.

        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="forbid",
            validate_assignment=True,
        )

        @classmethod
        def from_mapping(
            cls,
            mapping: Mapping[str, t.GeneralValueType],
        ) -> Self:
            """Create Config instance from mapping.

            Args:
                mapping: Mapping with configuration data

            Returns:
                Config instance

            """
            return cls(**mapping)

        def to_mapping(self) -> t.Types.ConfigurationMapping:
            """Convert Config to mapping.

            Returns:
                ConfigurationMapping: Mapping representation

            """
            return self.model_dump()

    class ParseOptions(FlextModelsBase.ArbitraryTypesModel):
        """Options for string parsing operations."""

        strip: bool = Field(
            default=True, description="Strip whitespace from components"
        )
        remove_empty: bool = Field(
            default=True,
            description="Remove empty components from result",
        )
        validator: Callable[[str], bool] | None = Field(
            default=None,
            description="Optional validator function for components",
        )

    class PatternApplicationParams(FlextModelsBase.ArbitraryTypesModel):
        """Parameters for regex pattern application."""

        text: str = Field(description="Text to apply pattern to")
        pattern: str = Field(description="Regex pattern")
        replacement: str = Field(description="Replacement string")
        flags: int = Field(default=0, description="Regex flags")
        pattern_index: int = Field(description="Index of pattern in pipeline")
        total_patterns: int = Field(description="Total number of patterns")


__all__ = ["FlextModelsCollections"]
