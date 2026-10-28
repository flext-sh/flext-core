"""Behavioral contract tests for the Tier-0 base model layer.

Exercises the public contract of ``m.BaseModel`` (plain Pydantic base) and
``m.Value`` (immutable, compared-by-value DDD value object) through their
public API only: construction, validation, serialization, equality, hashing
and immutability. No private attributes or internals are touched.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

import pytest

from tests import c, m, t, tm


class TestsFlextCoreBase:
    """Behavioral contract for the base model and value-object surfaces."""

    class Sample(m.BaseModel):
        """Plain base model with a required field and a defaulted field."""

        name: str
        count: int = 3

    class SampleValue(m.Value):
        """Value object with two descriptive fields for equality/hash tests."""

        amount: Annotated[
            int,
            m.Field(description="Numeric amount of the value object."),
        ]
        label: Annotated[str, m.Field(description="Human-readable label of the value.")]

    def test_base_model_dump_returns_declared_field_values(self) -> None:
        model = self.Sample(name="alpha", count=7)

        dumped = model.model_dump()

        tm.that(dumped, eq={"name": "alpha", c.Aggregation.COUNT: 7})

    def test_base_model_applies_declared_default(self) -> None:
        model = self.Sample(name="beta")

        tm.that(model.count, eq=3)

    def test_base_model_round_trips_through_json(self) -> None:
        original = self.Sample(name="gamma", count=11)

        restored = self.Sample.model_validate_json(original.model_dump_json())

        tm.that(restored, eq=original)

    def test_base_model_validate_from_mapping(self) -> None:
        model = self.Sample.model_validate({"name": "delta", c.Aggregation.COUNT: 2})

        tm.that(model.name, eq="delta")
        tm.that(model.count, eq=2)

    def test_base_model_exposes_declared_fields(self) -> None:
        tm.that(set(self.Sample.model_fields), eq={"name", c.Aggregation.COUNT})

    @pytest.mark.parametrize(
        "payload",
        [
            {c.Aggregation.COUNT: 1},
            {"name": "x", c.Aggregation.COUNT: "not-an-int"},
            {"name": None, c.Aggregation.COUNT: 1},
        ],
    )
    def test_base_model_rejects_invalid_payloads(
        self,
        payload: t.JsonMapping,
    ) -> None:
        with pytest.raises(m.ValidationError):
            self.Sample.model_validate(payload)

    def test_value_objects_are_equal_when_all_fields_match(self) -> None:
        left = self.SampleValue(amount=5, label="usd")
        right = self.SampleValue(amount=5, label="usd")

        tm.that(left, eq=right)

    @pytest.mark.parametrize(("amount", "label"), [(6, "usd"), (5, "eur")])
    def test_value_objects_differ_when_any_field_differs(
        self,
        amount: int,
        label: str,
    ) -> None:
        base = self.SampleValue(amount=5, label="usd")

        tm.that(base, ne=self.SampleValue(amount=amount, label=label))

    def test_equal_value_objects_share_hash_and_distinct_values_do_not(self) -> None:
        first = self.SampleValue(amount=9, label="gbp")
        second = self.SampleValue(amount=9, label="gbp")
        distinct = self.SampleValue(amount=10, label="gbp")

        tm.that(first, eq=second)
        tm.that(first, ne=distinct)
        tm.that(hash(first), eq=hash(second))
        tm.that(len({hash(first), hash(second), hash(distinct)}), eq=2)

    def test_value_object_is_not_equal_to_foreign_type(self) -> None:
        value = self.SampleValue(amount=1, label="jpy")

        tm.that(value, ne=object())
        tm.that(value, ne="jpy")

    def test_value_object_is_immutable(self) -> None:
        value = self.SampleValue(amount=2, label="chf")

        tm.rejects_assignment(value, "amount", 3, expected=m.ValidationError)

    @pytest.mark.parametrize("amount", [c.ParserBooleanToken.ONE, 1.5])
    def test_value_object_strictly_validates_field_types(
        self,
        amount: str | float,
    ) -> None:
        with pytest.raises(m.ValidationError):
            self.SampleValue.model_validate({"amount": amount, "label": "strict"})
