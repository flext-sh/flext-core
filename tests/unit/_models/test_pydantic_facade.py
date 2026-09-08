"""Behavioral contract tests for advanced pydantic declarations on ``m``.

Exercises the public models facade contract for constraint, discrimination,
fail-fast, instance, coercion and serializer annotations. Every symbol is
consumed through ``m`` exactly as fleet consumers do: observable validation
and serialization behavior only, never pydantic internals.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pytest

from tests.models import m


class _VectorInput(m.BaseModel):
    """Natively validated input shape for the ValidateAs hook."""

    x: int
    y: int


class _Vector:
    """Custom type populated from a natively validated model."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def magnitude_squared(self) -> int:
        """Squared length of the vector."""
        return self.x * self.x + self.y * self.y


class _Cat(m.BaseModel):
    """Discriminated union member for the cat tag."""

    kind: Literal["cat"]
    meow: str


class _Dog(m.BaseModel):
    """Discriminated union member for the dog tag."""

    kind: Literal["dog"]
    bark: str


class _Pet(m.BaseModel):
    """Field resolved to a union member through the discriminator tag."""

    animal: Annotated[_Cat | _Dog, m.Discriminator("kind")]


class _Item(m.BaseModel):
    """Base item serialized through a base-typed field."""

    name: str


class _DetailedItem(_Item):
    """Subclass field preserved at serialization time via SerializeAsAny."""

    detail: str


class _Box(m.BaseModel):
    """Container whose base-typed field serializes runtime subclasses."""

    item: Annotated[_Item, m.SerializeAsAny()]


class _Greeter:
    """Plain runtime class guarded by InstanceOf."""

    def hello(self) -> str:
        return "hello"


class _GreetingCard(m.BaseModel):
    """Model whose payload must be a Greeter instance."""

    payload: m.InstanceOf[_Greeter]


class _Constrained(m.BaseModel):
    """Model field constrained through StringConstraints."""

    code: Annotated[str, m.StringConstraints(min_length=3, pattern=r"^[a-z]+$")]


class TestsFlextCorePydanticDeclarations:
    """Behavioral contract for the advanced pydantic facade exports."""

    def test_string_constraints_accept_valid_and_reject_invalid_values(self) -> None:
        assert _Constrained(code="abc").code == "abc"

        with pytest.raises(m.ValidationError):
            _Constrained(code="ab")

        with pytest.raises(m.ValidationError):
            _Constrained(code="ABC")

    def test_discriminator_resolves_union_member_from_tag(self) -> None:
        pet = _Pet.model_validate({"animal": {"kind": "dog", "bark": "woof"}})

        assert isinstance(pet.animal, _Dog)
        assert pet.animal.bark == "woof"

        with pytest.raises(m.ValidationError):
            _Pet.model_validate({"animal": {"kind": "cow", "moo": "moo"}})

    def test_serialize_as_any_keeps_subclass_fields_in_dump(self) -> None:
        box = _Box(item=_DetailedItem(name="flext", detail="advanced"))

        assert box.model_dump() == {"item": {"name": "flext", "detail": "advanced"}}

    def test_fail_fast_reports_only_the_first_list_error(self) -> None:
        adapter = m.TypeAdapter(Annotated[list[int], m.FailFast()])

        assert adapter.validate_python([1, 2]) == [1, 2]

        with pytest.raises(m.ValidationError) as exc_info:
            adapter.validate_python(["first", "second"])

        assert len(exc_info.value.errors()) == 1

    def test_instance_of_accepts_instance_and_rejects_foreign_object(self) -> None:
        greeter = _Greeter()
        card = _GreetingCard(payload=greeter)

        assert card.payload is greeter

        with pytest.raises(m.ValidationError):
            _GreetingCard(payload=object())

    def test_validate_as_builds_custom_type_from_native_model(self) -> None:
        adapter = m.TypeAdapter(
            Annotated[
                _Vector,
                m.ValidateAs(
                    _VectorInput, lambda validated: _Vector(validated.x, validated.y)
                ),
            ]
        )

        vector = adapter.validate_python({"x": 1, "y": 2})

        assert isinstance(vector, _Vector)
        assert (vector.x, vector.y) == (1, 2)

        with pytest.raises(m.ValidationError):
            adapter.validate_python({"x": "no-int", "y": 2})
