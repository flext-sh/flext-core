"""Collection models for categorized data.

TIER 0.5: Depends only on base.py (Tier 0).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from typing import Annotated, ClassVar, Self, override

from pydantic import ConfigDict, Field, computed_field

from flext_core import FlextModelFoundation, t


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
        ) -> t.Numeric | None:
            numeric: Sequence[t.Numeric] = [
                v
                for v in non_none
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return sum(numeric) if numeric else None

        @classmethod
        def _concatenate_lists(
            cls,
            non_none: Sequence[t.MetadataValue],
        ) -> t.ScalarList:
            combined: MutableSequence[t.Scalar] = []
            for v in non_none:
                if isinstance(v, list):
                    combined.extend(v)
            return combined

        @classmethod
        def _merge_dicts(
            cls,
            non_none: Sequence[t.MetadataValue],
        ) -> Mapping[str, t.Scalar | t.ScalarList]:
            merged: MutableMapping[str, t.Scalar | t.ScalarList] = {}
            for v in non_none:
                if isinstance(v, Mapping):
                    for key, val in v.items():
                        if isinstance(val, (str, int, float, bool, datetime, list)):
                            merged[str(key)] = val
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
            non_none: Sequence[t.MetadataValue] = [
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

        @classmethod
        def _aggregate_dumped_models(
            cls,
            items: Sequence[
                FlextModelFoundation.ArbitraryTypesModel
                | FlextModelFoundation.FrozenValueModel
            ],
        ) -> Mapping[str, t.MetadataValue]:
            if not items:
                return {}
            aggregated: MutableMapping[str, t.MetadataValue | None] = {}
            for item in items:
                for key, value in item.model_dump().items():
                    aggregated[key] = cls._resolve_conflict(aggregated.get(key), value)
            return {
                key: value for key, value in aggregated.items() if value is not None
            }

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

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            validate_default=True,
            validate_assignment=True,
        )
        categories: Annotated[
            MutableMapping[str, MutableSequence[t.MetadataValue]],
            Field(
                default_factory=dict,
                description="Map of category name to list of items",
            ),
        ]

        def __len__(self) -> int:
            return sum(len(entries) for entries in self.categories.values())

        @classmethod
        @override
        def __class_getitem__(
            cls,
            typevar_values: type | tuple[type, ...],
        ) -> type[FlextModelsCollections.Categories]:
            _ = typevar_values
            return cls

        @computed_field
        @property
        def category_names(self) -> t.StrSequence:
            return list(self.categories.keys())

        @computed_field
        @property
        def total_entries(self) -> int:
            return sum(len(entries) for entries in self.categories.values())

        def add_entries(
            self,
            category: str,
            entries: Sequence[t.MetadataValue],
        ) -> None:
            self.categories.setdefault(category, []).extend(entries)

        def clear(self) -> None:
            self.categories.clear()

        def get(
            self,
            category: str,
            default: Sequence[t.MetadataValue] | None = None,
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
            return cls._aggregate_dumped_models(stats_list)

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
            return cls._aggregate_dumped_models(results_list)

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
            result: MutableMapping[str, t.MetadataValue | None] = {}
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

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            extra="forbid",
            validate_assignment=True,
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

    class CollectionBatchSpec(FlextModelFoundation.ArbitraryTypesModel):
        """Batch processing options for collection operations."""

        size: Annotated[
            t.PositiveInt | None,
            Field(
                default=None,
                title="Batch Size",
                description="Optional batch size hint for compatibility and slicing behavior.",
            ),
        ] = None
        on_error: Annotated[
            str | None,
            Field(
                default=None,
                title="Error Mode",
                description="Error handling mode: fail immediately, collect errors, or skip failed items.",
            ),
        ] = None
        parallel: Annotated[
            bool,
            Field(
                default=False,
                title="Parallel",
                description="Whether parallel processing should be requested by callers.",
            ),
        ] = False
        progress: Annotated[
            Callable[[int, int], None] | None,
            Field(
                default=None,
                title="Progress Callback",
                description="Optional callback receiving processed and total item counts.",
            ),
        ] = None
        progress_interval: Annotated[
            t.PositiveInt,
            Field(
                default=1,
                title="Progress Interval",
                description="How often progress callback is invoked during processing.",
            ),
        ] = 1
        pre_validate: Annotated[
            Callable[[t.ValueOrModel], bool] | None,
            Field(
                default=None,
                title="Pre Validate",
                description="Optional predicate to filter items before operation execution.",
            ),
        ] = None
        flatten: Annotated[
            bool,
            Field(
                default=False,
                title="Flatten",
                description="Whether list-like operation results should be flattened into a single output list.",
            ),
        ] = False

    class GuardCheckSpec(FlextModelFoundation.ArbitraryTypesModel):
        """Specification for guard conditions used in collection filters."""

        eq: Annotated[
            t.NormalizedValue | None,
            Field(
                default=None,
                title="Equals",
                description="Require the value to equal this value.",
            ),
        ] = None
        ne: Annotated[
            t.NormalizedValue | None,
            Field(
                default=None,
                title="Not Equals",
                description="Require the value to differ from this value.",
            ),
        ] = None
        gt: Annotated[
            float | None,
            Field(
                default=None,
                title="Greater Than",
                description="Require numeric or length-derived value to be greater than this value.",
            ),
        ] = None
        gte: Annotated[
            float | None,
            Field(
                default=None,
                title="Greater Than Or Equal",
                description="Require numeric or length-derived value to be greater than or equal to this value.",
            ),
        ] = None
        lt: Annotated[
            float | None,
            Field(
                default=None,
                title="Less Than",
                description="Require numeric or length-derived value to be less than this value.",
            ),
        ] = None
        lte: Annotated[
            float | None,
            Field(
                default=None,
                title="Less Than Or Equal",
                description="Require numeric or length-derived value to be less than or equal to this value.",
            ),
        ] = None
        is_: Annotated[
            type | None,
            Field(
                default=None,
                title="Is Type",
                description="Require the value to be an instance of this type.",
            ),
        ] = None
        not_: Annotated[
            type | None,
            Field(
                default=None,
                title="Not Type",
                description="Require the value to not be an instance of this type.",
            ),
        ] = None
        in_: Annotated[
            t.ContainerList | None,
            Field(
                default=None,
                title="In Values",
                description="Require the value to be present in this sequence.",
            ),
        ] = None
        not_in: Annotated[
            t.ContainerList | None,
            Field(
                default=None,
                title="Not In Values",
                description="Require the value to not be present in this sequence.",
            ),
        ] = None
        none: Annotated[
            bool | None,
            Field(
                default=None,
                title="None Constraint",
                description="When True, require None. When False, require non-None.",
            ),
        ] = None
        empty: Annotated[
            bool | None,
            Field(
                default=None,
                title="Empty Constraint",
                description="When True, require empty value; when False, require non-empty.",
            ),
        ] = None
        match: Annotated[
            str | None,
            Field(
                default=None,
                title="Regex Match",
                description="Require string value to match this regular expression.",
            ),
        ] = None
        contains: Annotated[
            t.NormalizedValue | None,
            Field(
                default=None,
                title="Contains",
                description="Require string or iterable value to contain this item.",
            ),
        ] = None
        starts: Annotated[
            str | None,
            Field(
                default=None,
                title="Starts With",
                description="Require string value to start with this prefix.",
            ),
        ] = None
        ends: Annotated[
            str | None,
            Field(
                default=None,
                title="Ends With",
                description="Require string value to end with this suffix.",
            ),
        ] = None

    class PatternApplicationParams(FlextModelFoundation.ArbitraryTypesModel):
        """Parameters for regex pattern application."""

        text: Annotated[str, Field(description="Text to apply pattern to")]
        pattern: Annotated[t.NonEmptyStr, Field(description="Regex pattern")]
        replacement: Annotated[str, Field(description="Replacement string")]
        flags: Annotated[
            t.NonNegativeInt,
            Field(default=0, description="Regex flags"),
        ] = 0
        pattern_index: Annotated[
            t.NonNegativeInt,
            Field(description="Index of pattern in pipeline"),
        ]
        total_patterns: Annotated[
            t.PositiveInt,
            Field(description="Total number of patterns"),
        ]


__all__ = ["FlextModelsCollections"]
