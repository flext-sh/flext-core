"""Behavior contract for ``u.empty``: the canonical empty-collection default.

The oracle is pydantic itself: every default must equal, in value and in
container type, what pydantic re-validates from the model's own JSON dump,
so a default never diverges from a real empty input of the same contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from types import ModuleType
from typing import Annotated

import pytest
from typing_extensions import TypeForm

from tests import m, t, tm, u


class TestsFlextUtilitiesPydanticEmpty:
    """Tests for ``u.empty``."""

    type _NonEmptyValues[ValueT] = Annotated[list[ValueT], m.Field(min_length=1)]
    type _NonEmptyAlias = _NonEmptyValues[int]
    type _AnnotatedList = Annotated[list[int], m.Field(min_length=1)]
    type _AnnotatedMapping = Annotated[Mapping[str, int], m.Field(min_length=1)]
    type _AnnotatedSet = Annotated[AbstractSet[int], m.Field(min_length=1)]

    class _Item(m.ArbitraryTypesModel):
        name: str = m.Field(default="item", description="Item name.")

    class _Defaults(m.ArbitraryTypesModel):
        json_mapping: t.JsonMapping = m.Field(
            default_factory=u.empty(t.TypeForm(t.JsonMapping)),
            description="Read-only JSON mapping contract.",
        )
        mutable_json_mapping: t.MutableJsonMapping = m.Field(
            default_factory=u.empty(t.TypeForm(t.MutableJsonMapping)),
            description="Mutable JSON mapping contract.",
        )
        str_sequence: Sequence[str] = m.Field(
            default_factory=u.empty(t.TypeForm(Sequence[str])),
            description="Read-only string sequence contract.",
        )
        items: MutableSequence[TestsFlextUtilitiesPydanticEmpty._Item] = m.Field(
            default_factory=u.empty(
                contract_factory=lambda: TypeForm(
                    MutableSequence[TestsFlextUtilitiesPydanticEmpty._Item]
                ),
            ),
            validate_default=True,
            description="Mutable sequence of a forward-referenced model.",
        )
        frozen_tags: AbstractSet[str] = m.Field(
            default_factory=u.empty(t.TypeForm(AbstractSet[str])),
            description="Read-only set contract.",
        )
        mutable_tags: MutableSet[str] = m.Field(
            default_factory=u.empty(t.TypeForm(MutableSet[str])),
            description="Mutable set contract.",
        )
        modules: MutableMapping[str, ModuleType] = m.Field(
            default_factory=u.empty(t.TypeForm(MutableMapping[str, ModuleType])),
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
            tm.ok(type(default_value) is type(reference_value))

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

    @staticmethod
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                u.empty(t.TypeForm(_AnnotatedList)),
                id="annotated-list",
            ),
            pytest.param(
                u.empty(
                    t.TypeForm(_AnnotatedMapping),
                ),
                id="annotated-mapping",
            ),
            pytest.param(
                u.empty(t.TypeForm(_AnnotatedSet)),
                id="annotated-set",
            ),
            pytest.param(u.empty(t.TypeForm(tuple[int, str])), id="fixed-tuple"),
            pytest.param(u.empty(t.TypeForm(_NonEmptyAlias)), id="nested-alias"),
            pytest.param(u.empty(t.TypeForm(_NonEmptyValues[int])), id="generic-alias"),
        ],
    )
    def test_full_contract_rejects_invalid_empty_input(
        factory: Callable[
            [], Sequence[int | str] | Mapping[str, int] | AbstractSet[int]
        ],
    ) -> None:
        """Constraints and fixed arity survive outer collection classification."""
        with pytest.raises(m.ValidationError):
            factory()

    @staticmethod
    def test_contract_validation_is_lazy_and_repeated_for_fresh_values() -> None:
        """Factory creation does no validation; each call validates a new value."""
        observed: list[list[int]] = []

        def record(value: list[int]) -> list[int]:
            observed.append(value)
            return value

        contract: TypeForm[list[int]] = Annotated[list[int], u.AfterValidator(record)]
        factory = u.empty(contract)
        tm.that(observed, eq=[])

        first = factory()
        second = factory()
        first.append(1)

        tm.that(second, eq=[])
        tm.that(len(observed), eq=2)
        tm.ok(observed[0] is first)
        tm.ok(observed[1] is second)

    @staticmethod
    def test_resolved_model_element_contract_preserves_native_model_config() -> None:
        """A collection adapter does not override its element model's own config."""
        factory = u.empty(
            t.TypeForm(list[TestsFlextUtilitiesPydanticEmpty._Item]),
        )
        tm.that(factory(), eq=[])
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydanticEmpty._Item.model_validate({"unexpected": True})

    @staticmethod
    def test_contract_factory_resolves_after_declaration() -> None:
        """A supplied type can be bound after declaring the default factory."""
        contract: TypeForm[MutableSequence[int]]
        factory = u.empty(contract_factory=lambda: contract)
        contract = TypeForm(MutableSequence[int])

        first = factory()
        first.append(1)

        tm.that(first, eq=[1])
        tm.that(factory(), eq=[])

    @staticmethod
    def test_contract_factory_keeps_full_collection_constraints() -> None:
        """Deferred type resolution never erases a non-empty collection bound."""
        factory = u.empty(
            contract_factory=lambda: TypeForm(
                TestsFlextUtilitiesPydanticEmpty._NonEmptyAlias
            ),
        )

        with pytest.raises(m.ValidationError):
            factory()

    @staticmethod
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(u.empty(t.TypeForm(int)), id="scalar"),
            pytest.param(u.empty(t.TypeForm(str)), id="text"),
            pytest.param(
                u.empty(contract_factory=lambda: TypeForm(int)),
                id="deferred-scalar",
            ),
            pytest.param(u.empty(t.TypeForm(_Item)), id="model"),
        ],
    )
    def test_non_collection_contract_fails_loud(
        factory: Callable[[], int | str | TestsFlextUtilitiesPydanticEmpty._Item],
    ) -> None:
        """A contract with no empty collection value raises ``TypeError``."""
        with pytest.raises(TypeError, match="requires a collection contract"):
            factory()
