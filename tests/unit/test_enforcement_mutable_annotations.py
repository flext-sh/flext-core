"""Public factory classification preserves unresolved collection item names.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSet, Set as AbstractSet
from typing import Annotated, ForwardRef

import pytest

from tests import m, t, tm, u


class TestsFlextCoreMutableAnnotations:
    """Match actual mutable interfaces, never spelling or deferred item types."""

    @staticmethod
    @pytest.mark.parametrize(
        ("annotation", "factory", "expected"),
        [
            ("t.MutableSequenceOf['Owner.Child']", list, True),
            ("t.MutableMappingKV[str, 'Owner.Child']", dict, True),
            ("MutableSet['Owner.Child']", set, True),
            ("Annotated[t.MutableSequenceOf['Owner.Child'], 'metadata']", list, True),
            ("t.SequenceOf['Owner.Child']", list, False),
            ("t.MappingKV[str, 'Owner.Child']", dict, False),
            ("AbstractSet['Owner.Child']", set, False),
            ("t.MutableSequenceOf['Owner.Child']", dict, False),
            ("t.MutableMappingKV[str, 'Owner.Child']", list, False),
            ("MutableSet['Owner.Child']", list, False),
            ("unavailable.MutableSequenceOf['Owner.Child']", list, False),
            ("list['Owner.Child']", list, False),
        ],
    )
    def test_forward_collection_factory_identity(
        annotation: str,
        factory: type,
        expected: bool,
    ) -> None:
        """The public helper uses the declaring model's live namespace."""

        class Owner:
            class State(m.ArbitraryTypesModel):
                entries: Annotated[
                    t.MutableSequenceOf["Owner.Child"],
                    m.Field(description="Mutable deferred native entries."),
                ] = m.Field(default_factory=list["Owner.Child"], validate_default=True)

            class Child(m.ArbitraryTypesModel):
                value: Annotated[str, m.Field(description="Owned child value.")]

        tm.that(
            u.allows_mutable_default_factory(
                ForwardRef(annotation), factory, owner=Owner.State
            ),
            eq=expected,
        )
        tm.that(
            u.allows_mutable_default_factory(annotation, factory, owner=Owner.State),
            eq=expected,
        )
        tm.that(
            u.allows_mutable_default_factory(AbstractSet[str], set, owner=Owner.State),
            eq=False,
        )

    @staticmethod
    def test_native_pydantic_fields_keep_mutable_forward_factories() -> None:
        """Every real model construction runs its unchanged public enforcement."""

        class Owner:
            class State(m.ArbitraryTypesModel):
                entries: Annotated[
                    t.MutableSequenceOf["Owner.Child"],
                    m.Field(description="Mutable deferred native entries."),
                ] = m.Field(default_factory=list["Owner.Child"], validate_default=True)
                indexed: Annotated[
                    t.MutableMappingKV[str, "Owner.Child"],
                    m.Field(description="Mutable deferred native mapping."),
                ] = m.Field(
                    default_factory=dict[str, "Owner.Child"], validate_default=True
                )
                selected: Annotated[
                    MutableSet["Owner.Child"],
                    m.Field(description="Mutable deferred native set."),
                ] = m.Field(default_factory=set["Owner.Child"], validate_default=True)

            class Child(m.ArbitraryTypesModel):
                value: Annotated[str, m.Field(description="Owned child value.")]

        state = Owner.State()
        tm.that(len(state.entries), eq=0)
        tm.that(len(state.indexed), eq=0)
        tm.that(len(state.selected), eq=0)
        for name, info in Owner.State.model_fields.items():
            tm.that(
                u.allows_mutable_default_factory(
                    info.annotation, info.default_factory, owner=Owner.State
                ),
                eq=True,
                msg=f"{name}: {info.annotation!r}; factory={info.default_factory!r}",
            )
