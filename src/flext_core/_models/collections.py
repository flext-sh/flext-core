"""Collection models for categorized data.

TIER 0.5: Depends only on base.py (Tier 0).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Self, override

from pydantic import ConfigDict, Field, computed_field

from flext_core import t
from flext_core._models import FlextModelFoundation


class FlextModelsCollections:
    """Collection models container class."""

    class _MetadataAggregateMixin:
        """Pure-logic mixin for metadata aggregation via MRO.

        Must be combined with a Pydantic BaseModel subclass.
        Provides shared conflict resolution for Statistics, Results, and Options.
        """

        @classmethod
        def _sum_numeric_values(
            cls,
            non_none: Sequence[t.MetadataValue],
        ) -> int | float | None:
            numeric: list[int | float] = [
                v
                for v in non_none
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return sum(numeric) if numeric else None

        @classmethod
        def _concatenate_lists(
            cls,
            non_none: Sequence[t.MetadataValue],
        ) -> list[t.Scalar]:
            combined: list[t.Scalar] = []
            for v in non_none:
                if isinstance(v, list):
                    combined.extend(
                        item
                        for item in v
                        if isinstance(item, (str, int, float, bool, datetime))
                    )
            return combined

        @classmethod
        def _merge_dicts(
            cls,
            non_none: Sequence[t.MetadataValue],
        ) -> Mapping[str, t.Scalar | Sequence[t.Scalar]]:
            merged: dict[str, t.Scalar | Sequence[t.Scalar]] = {}
            for v in non_none:
                if isinstance(v, Mapping):
                    for key, val in v.items():
                        if isinstance(val, (str, int, float, bool, datetime)):
                            merged[str(key)] = val
                        elif isinstance(val, list):
                            merged[str(key)] = [
                                item
                                for item in val
                                if isinstance(item, (str, int, float, bool, datetime))
                            ]
            return merged

        @classmethod
        def _resolve_conflict(
            cls,
            existing: t.MetadataValue | None,
            value: t.MetadataValue | None,
        ) -> t.MetadataValue | None:
            """Resolve conflict when aggregating two metadata values.

            Strategy: booleans last-wins, numerics sum, lists concatenate,
            mappings merge, all others last-wins.
            """
            non_none: list[t.MetadataValue] = [
                v for v in (existing, value) if v is not None
            ]
            if not non_none:
                return None
            first_val = non_none[0]
            if isinstance(first_val, bool):
                return non_none[-1]
            if isinstance(first_val, (int, float)):
                numeric_sum = cls._sum_numeric_values(non_none)
                return numeric_sum if numeric_sum is not None else non_none[-1]
            if isinstance(first_val, list):
                return cls._concatenate_lists(non_none)
            if isinstance(first_val, Mapping):
                return cls._merge_dicts(non_none)
            return non_none[-1]

    class Categories(FlextModelFoundation.ArbitraryTypesModel):
        """Generic categorized collection with dynamic categories.

        Provides type-safe storage for items organized by category names.
        Uses PEP 695 type parameter syntax for Python 3.12+.

        Example:
            categories = Categories[Entry]()
            categories.add_entries("users", [user1, user2])
            categories.add_entries("groups", [group1])

            # Access
            users = categories.get("users")
            total = categories.total_entries
            names = categories.category_names

        """

        model_config = ConfigDict(
            strict=True, validate_default=True, validate_assignment=True
        )
        categories: dict[str, list[t.MetadataValue]] = Field(
            default_factory=dict,
            description="Map of category name to list of items",
        )

        def __len__(self) -> int:
            return sum(len(entries) for entries in self.categories.values())

        @classmethod
        @override
        def __class_getitem__(
            cls, typevar_values: type | tuple[type, ...]
        ) -> type[FlextModelsCollections.Categories]:
            _ = typevar_values
            return cls

        @computed_field
        @property
        def category_names(self) -> Sequence[str]:
            return list(self.categories.keys())

        @computed_field
        @property
        def total_entries(self) -> int:
            return sum(len(entries) for entries in self.categories.values())

        def add_entries(
            self, category: str, entries: Sequence[t.MetadataValue]
        ) -> None:
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].extend(entries)

        def clear(self) -> None:
            self.categories.clear()

        def get(
            self, category: str, default: Sequence[t.MetadataValue] | None = None
        ) -> Sequence[t.MetadataValue]:
            if default is None:
                return self.categories.get(category, [])
            return self.categories.get(category, default)

        def has_category(self, category: str) -> bool:
            return category in self.categories

        def remove_category(self, category: str) -> None:
            _ = self.categories.pop(category, None)

        def to_mapping(self) -> Mapping[str, Sequence[t.MetadataValue]]:
            return {key: list(entries) for key, entries in self.categories.items()}

    class Statistics(_MetadataAggregateMixin, FlextModelFoundation.FrozenValueModel):
        """Base for statistics models (frozen Value)."""

        @classmethod
        def aggregate(
            cls,
            stats_list: Sequence[Self],
        ) -> Mapping[str, t.MetadataValue]:
            """Aggregate multiple statistics instances.

            Combines statistics by summing numerics, concatenating lists,
            merging mappings, and keeping last value for other types.
            """
            if not stats_list:
                return {}
            result: dict[str, t.MetadataValue | None] = {}
            for stats in stats_list:
                for key, value in stats.model_dump().items():
                    result[key] = cls._resolve_conflict(result.get(key), value)
            return {k: v for k, v in result.items() if v is not None}

        @classmethod
        def from_mapping(cls, data: Mapping[str, t.MetadataValue]) -> Self:
            return cls.model_validate(dict(data))

    class Rules(FlextModelFoundation.ArbitraryTypesModel):
        """Base for rules models (mutable)."""

        @classmethod
        def merge(cls, *rules: Self) -> Self:
            if not rules:
                return cls()
            base = rules[0].model_copy()
            for other in rules[1:]:
                base = base.model_copy(update=other.model_dump())
            return base

    class Results(_MetadataAggregateMixin, FlextModelFoundation.ArbitraryTypesModel):
        """Base for results models (mutable)."""

        @classmethod
        def aggregate(
            cls,
            results_list: Sequence[Self],
        ) -> Mapping[str, t.MetadataValue]:
            """Aggregate multiple results instances.

            Combines results by summing numerics, concatenating lists,
            merging mappings, and keeping last value for other types.
            """
            if not results_list:
                return {}
            result: dict[str, t.MetadataValue | None] = {}
            for res in results_list:
                for key, value in res.model_dump().items():
                    result[key] = cls._resolve_conflict(result.get(key), value)
            return {k: v for k, v in result.items() if v is not None}

        @classmethod
        def combine(cls, *results: Self) -> Self:
            if not results:
                return cls()
            base = results[0].model_copy()
            for other in results[1:]:
                base = base.model_copy(update=other.model_dump())
            return base

    class Options(_MetadataAggregateMixin, FlextModelFoundation.ArbitraryTypesModel):
        """Base for options models (mutable)."""

        @classmethod
        def merge_options(cls, *options: Self) -> Self:
            if not options:
                return cls()
            result: dict[str, t.MetadataValue | None] = {}
            for opt in options:
                for key, value in opt.model_dump().items():
                    result[key] = cls._resolve_conflict(
                        result.get(key),
                        value,
                    )
            return cls.model_validate({
                k: v for k, v in result.items() if v is not None
            })

        def merge(self, *options: Self) -> Self:
            return self.__class__.merge_options(self, *options)

    class Config(FlextModelFoundation.ArbitraryTypesModel):
        """Base for configuration models - mutable Pydantic v2 model.

        Non-frozen models are not hashable by design.
        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
        )

        @override
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.model_dump() == other.model_dump()

        def __hash__(self) -> int:
            msg = f"{self.__class__.__name__} objects are not hashable"
            raise TypeError(msg)

        @classmethod
        def from_mapping(cls, mapping: t.ConfigMap) -> Self:
            return cls.model_validate(dict(mapping))

        def diff(
            self,
            other: Self,
        ) -> Mapping[str, tuple[t.MetadataValue | None, t.MetadataValue | None]]:
            self_dict = self.model_dump()
            other_dict = other.model_dump()
            all_keys = set(self_dict) | set(other_dict)
            return {
                key: (self_dict.get(key), other_dict.get(key))
                for key in all_keys
                if self_dict.get(key) != other_dict.get(key)
            }

        def merge(self, other: Self) -> Self:
            return self.model_copy(update=other.model_dump())

        def to_mapping(self) -> t.ConfigMap:
            return t.ConfigMap(root=self.model_dump())

        def with_updates(self, **updates: t.MetadataValue) -> Self:
            return self.model_copy(update=updates)

    class ParseOptions(FlextModelFoundation.ArbitraryTypesModel):
        """Options for string parsing operations."""

        strip: Annotated[
            bool,
            Field(
                default=True,
                description="Strip whitespace from components",
            ),
        ] = True
        remove_empty: Annotated[
            bool,
            Field(
                default=True,
                description="Remove empty components from result",
            ),
        ] = True
        validator: Annotated[
            Callable[[str], bool] | None,
            Field(
                default=None,
                description="Optional validator function for components",
            ),
        ] = None

    class PatternApplicationParams(FlextModelFoundation.ArbitraryTypesModel):
        """Parameters for regex pattern application."""

        text: Annotated[str, Field(description="Text to apply pattern to")]
        pattern: Annotated[str, Field(description="Regex pattern")]
        replacement: Annotated[str, Field(description="Replacement string")]
        flags: Annotated[int, Field(default=0, description="Regex flags")] = 0
        pattern_index: Annotated[int, Field(description="Index of pattern in pipeline")]
        total_patterns: Annotated[int, Field(description="Total number of patterns")]


__all__ = ["FlextModelsCollections"]
