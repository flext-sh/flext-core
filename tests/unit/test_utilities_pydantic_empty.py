"""Behavior contract for ``u.empty``: the canonical empty-collection default.

The oracle is pydantic itself: every default must equal, in value and in
container type, what pydantic re-validates from the model's own JSON dump,
so a default never diverges from a real empty input of the same contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableSet, Set as AbstractSet
from types import ModuleType

import pytest
from flext_tests import tm
from typing_extensions import TypeForm

from tests import m, t, u


class TestsFlextUtilitiesPydanticEmpty:
    """Tests for ``u.empty``."""

    class _Item(m.ArbitraryTypesModel):
        name: str = u.Field(default="item", description="Item name.")

    class _Defaults(m.ArbitraryTypesModel):
        json_mapping: t.JsonMapping = u.Field(
            default_factory=u.empty(TypeForm(t.JsonMapping)),
            description="Read-only JSON mapping contract.",
        )
        mutable_json_mapping: t.MutableJsonMapping = u.Field(
            default_factory=u.empty(TypeForm(t.MutableJsonMapping)),
            description="Mutable JSON mapping contract.",
        )
        str_sequence: t.StrSequence = u.Field(
            default_factory=u.empty(TypeForm(t.StrSequence)),
            description="Read-only string sequence contract.",
        )
        items: t.MutableSequenceOf[TestsFlextUtilitiesPydanticEmpty._Item] = u.Field(
            default_factory=u.empty(
                TypeForm(t.MutableSequenceOf["TestsFlextUtilitiesPydanticEmpty._Item"]),
            ),
            description="Mutable sequence of a forward-referenced model.",
        )
        frozen_tags: AbstractSet[str] = u.Field(
            default_factory=u.empty(TypeForm(AbstractSet[str])),
            description="Read-only set contract.",
        )
        mutable_tags: MutableSet[str] = u.Field(
            default_factory=u.empty(TypeForm(MutableSet[str])),
            description="Mutable set contract.",
        )
        modules: t.MutableMappingKV[str, ModuleType] = u.Field(
            default_factory=u.empty(TypeForm(t.MutableMappingKV[str, ModuleType])),
            description="Mapping of an arbitrary, JSON-unrepresentable element.",
        )

    _JSON_FIELDS: frozenset[str] = frozenset({
        "json_mapping",
        "mutable_json_mapping",
        "str_sequence",
        "items",
        "frozen_tags",
        "mutable_tags",
    })

    @staticmethod
    def test_defaults_equal_pydantic_revalidated_values() -> None:
        """Each default equals pydantic's own value for the same empty input."""
        defaults = TestsFlextUtilitiesPydanticEmpty._Defaults()
        fields = TestsFlextUtilitiesPydanticEmpty._JSON_FIELDS
        revalidated = TestsFlextUtilitiesPydanticEmpty._Defaults.model_validate_json(
            defaults.model_dump_json(include=set(fields)),
        )

        for name in sorted(fields):
            default_value = getattr(defaults, name)
            reference_value = getattr(revalidated, name)
            tm.that(default_value, eq=reference_value)
            tm.that(type(default_value) is type(reference_value), eq=True)

    @staticmethod
    def test_mutable_defaults_are_fresh_per_instance() -> None:
        """Mutating one instance's default never leaks into another."""
        first = TestsFlextUtilitiesPydanticEmpty._Defaults()
        second = TestsFlextUtilitiesPydanticEmpty._Defaults()

        first.mutable_json_mapping["key"] = "value"
        first.items.append(TestsFlextUtilitiesPydanticEmpty._Item())
        first.mutable_tags.add("tag")
        first.modules["module"] = pytest

        tm.that(second.mutable_json_mapping, eq={})
        tm.that(second.items, eq=[])
        tm.that(second.mutable_tags, eq=set[str]())
        tm.that(second.modules, eq={})

    @staticmethod
    def test_defaults_survive_deep_copy() -> None:
        """A deep copy of a defaulted model is equal to the original."""
        defaults = TestsFlextUtilitiesPydanticEmpty._Defaults()

        tm.that(defaults.model_copy(deep=True), eq=defaults)

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(u.empty(TypeForm(int)), id="scalar"),
            pytest.param(u.empty(TypeForm(str)), id="text"),
            pytest.param(u.empty(TypeForm(_Item)), id="model"),
        ],
    )
    @staticmethod
    def test_non_collection_contract_fails_loud(
        factory: Callable[[], int | str | TestsFlextUtilitiesPydanticEmpty._Item],
    ) -> None:
        """A contract with no empty collection value raises ``TypeError``."""
        with pytest.raises(TypeError, match="requires a collection contract"):
            factory()
