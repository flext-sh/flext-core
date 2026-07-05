"""Behavioral contract tests for the Tier-0 base model layer.

Exercises the public contract of ``m.BaseModel`` (plain Pydantic base) and
``m.Value`` (immutable, compared-by-value DDD value object) through their
public API only: construction, validation, serialization, equality, hashing
and immutability. No private attributes or internals are touched.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import ValidationError

from tests.models import m


class Sample(m.BaseModel):
    """Plain base model with a required field and a defaulted field."""

    name: str
    count: int = 3


class SampleValue(m.Value):
    """Value object with two descriptive fields for equality/hash tests."""

    amount: Annotated[int, m.Field(description="Numeric amount of the value object.")]
    label: Annotated[str, m.Field(description="Human-readable label of the value.")]


class TestsFlextCoreBase:
    """Behavioral contract for the base model and value-object surfaces."""

    def test_base_model_dump_returns_declared_field_values(self) -> None:
        model = Sample(name="alpha", count=7)

        dumped = model.model_dump()

        assert dumped == {"name": "alpha", "count": 7}

    def test_base_model_applies_declared_default(self) -> None:
        model = Sample(name="beta")

        assert model.count == 3

    def test_base_model_round_trips_through_json(self) -> None:
        original = Sample(name="gamma", count=11)

        restored = Sample.model_validate_json(original.model_dump_json())

        assert restored == original

    def test_base_model_validate_from_mapping(self) -> None:
        model = Sample.model_validate({"name": "delta", "count": 2})

        assert model.name == "delta"
        assert model.count == 2

    def test_base_model_exposes_declared_fields(self) -> None:
        assert set(Sample.model_fields) == {"name", "count"}

    @pytest.mark.parametrize(
        "payload",
        [
            {"count": 1},
            {"name": "x", "count": "not-an-int"},
            {"name": None, "count": 1},
        ],
    )
    def test_base_model_rejects_invalid_payloads(
        self, payload: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            Sample.model_validate(payload)

    def test_value_objects_are_equal_when_all_fields_match(self) -> None:
        left = SampleValue(amount=5, label="usd")
        right = SampleValue(amount=5, label="usd")

        assert left == right

    @pytest.mark.parametrize(
        ("amount", "label"),
        [(6, "usd"), (5, "eur")],
    )
    def test_value_objects_differ_when_any_field_differs(
        self, amount: int, label: str
    ) -> None:
        base = SampleValue(amount=5, label="usd")

        assert base != SampleValue(amount=amount, label=label)

    def test_equal_value_objects_share_hash_and_deduplicate_in_set(self) -> None:
        first = SampleValue(amount=9, label="gbp")
        second = SampleValue(amount=9, label="gbp")
        distinct = SampleValue(amount=10, label="gbp")

        assert hash(first) == hash(second)
        assert len({first, second, distinct}) == 2

    def test_value_object_is_not_equal_to_foreign_type(self) -> None:
        value = SampleValue(amount=1, label="jpy")

        assert value != object()
        assert value != "jpy"

    def test_value_object_is_immutable(self) -> None:
        value = SampleValue(amount=2, label="chf")

        with pytest.raises(ValidationError):
            value.amount = 3

    @pytest.mark.parametrize(
        "amount",
        ["1", 1.5],
    )
    def test_value_object_strictly_validates_field_types(
        self, amount: str | float
    ) -> None:
        with pytest.raises(ValidationError):
            SampleValue(amount=amount, label="strict")
