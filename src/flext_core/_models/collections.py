"""Collection models for categorized data.

TIER 0.5: Depends only on base.py (Tier 0).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Annotated, ClassVar, Self, override

from pydantic import ConfigDict, Field

from flext_core import FlextModelsBase, t
from flext_core._utilities.domain import FlextUtilitiesDomain


class FlextModelsCollections:
    """Collection models container class."""

    class Statistics(FlextModelsBase.FrozenValueModel):
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
            return FlextUtilitiesDomain.aggregate_dumped_models(stats_list)

        @classmethod
        def from_mapping(cls, data: Mapping[str, t.MetadataValue]) -> Self:
            return cls.model_validate(dict(data))

    class Rules(FlextModelsBase.ArbitraryTypesModel):
        """Base for rules models (mutable)."""

        @classmethod
        def merge(cls, *rules: Self) -> Self:
            if not rules:
                return cls()
            base = rules[0].model_copy()
            for other in rules[1:]:
                base = base.model_copy(update=other.model_dump())
            return base

    class Options(FlextModelsBase.ArbitraryTypesModel):
        """Base for options models (mutable)."""

        @classmethod
        def merge_options(cls, *options: Self) -> Self:
            if not options:
                return cls()
            result: MutableMapping[str, t.MetadataValue | None] = {}
            for opt in options:
                for key, value in opt.model_dump().items():
                    result[key] = FlextUtilitiesDomain.aggregate_metadata_values(
                        result.get(key),
                        value,
                    )
            return cls.model_validate({
                k: v for k, v in result.items() if v is not None
            })

        def merge(self, *options: Self) -> Self:
            return self.__class__.merge_options(self, *options)

    class Config(FlextModelsBase.ArbitraryTypesModel):
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

    class GuardCheckSpec(FlextModelsBase.ArbitraryTypesModel):
        """Specification for guard conditions used in collection filters."""

        eq: Annotated[
            t.RecursiveContainer | None,
            Field(
                default=None,
                title="Equals",
                description="Require the value to equal this value.",
            ),
        ] = None
        ne: Annotated[
            t.RecursiveContainer | None,
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
            t.RecursiveContainer | None,
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


__all__ = ["FlextModelsCollections"]
