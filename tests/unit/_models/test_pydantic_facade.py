"""Behavioral contract tests for advanced pydantic declarations on ``m``/``u``.

Exercises the public models facade contract for constraint, discrimination,
fail-fast, instance, coercion and serializer annotations. Every symbol is
consumed through ``m``/``u`` exactly as fleet consumers do: observable validation
and serialization behavior only, never pydantic internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

import pytest
from typing_extensions import TypeForm

from tests import m, p, t, u


class TestsFlextCorePydanticDeclarations:
    """Behavioral contract for the advanced pydantic facade exports."""

    class _DetailedPayload(m.Tests.PublicPayload):
        """Subclass-only data must respect the declared serialization boundary."""

        detail: str

    class _ProtocolBox(m.ContractModel):
        """Protocol annotations retain the model owner's native schema."""

        item: Annotated[
            p.Tests.PublicPayload,
            m.GetPydanticSchema(
                lambda _source, handler: handler.generate_schema(m.Tests.PublicPayload),
            ),
        ]

    @staticmethod
    @pytest.mark.parametrize("json_input", [False, True])
    def test_protocol_field_boundary(json_input: bool) -> None:
        box = (
            TestsFlextCorePydanticDeclarations._ProtocolBox.model_validate_json(
                '{"item":{"rawName":" alice ","visits":2}}',
            )
            if json_input
            else TestsFlextCorePydanticDeclarations._ProtocolBox.model_validate({
                "item": {"rawName": " alice ", "visits": 2},
            })
        )
        expected = {
            "raw_name": "Alice",
            "visits": "2 visits",
            "label": "Alice:2",
        }
        assert (box.item.raw_name, box.item.visits) == ("Alice", 2)
        assert box.model_dump()["item"] == expected
        assert '"visits":"2 visits"' in box.model_dump_json()
        payload: p.Tests.PublicPayload
        for payload in (
            m.Tests.PublicPayload(raw_name="alice", visits=2),
            TestsFlextCorePydanticDeclarations._DetailedPayload(
                raw_name="alice",
                visits=2,
                detail="private",
            ),
        ):
            box = TestsFlextCorePydanticDeclarations._ProtocolBox(item=payload)
            assert box.item is payload
            assert box.model_dump()["item"] == expected

        def reject() -> None:
            """Validate the malformed payload through the declared input form."""
            if json_input:
                TestsFlextCorePydanticDeclarations._ProtocolBox.model_validate_json(
                    '{"item":{"rawName":"alice","visits":"invalid"}}',
                )
            else:
                TestsFlextCorePydanticDeclarations._ProtocolBox.model_validate(
                    {"item": {"rawName": "alice", "visits": "invalid"}},
                )

        with pytest.raises(m.ValidationError):
            reject()

    @staticmethod
    @pytest.mark.parametrize("mode", ["validation", "serialization"])
    def test_protocol_field_preserves_owner_schema(
        mode: Literal["validation", "serialization"],
    ) -> None:
        model_type = TestsFlextCorePydanticDeclarations._ProtocolBox
        field = model_type.model_fields["item"]
        assert field.annotation is p.Tests.PublicPayload
        assert model_type.model_config.get(
            "frozen", False
        ) == m.ContractModel.model_config.get("frozen", False)
        schema = TestsFlextCorePydanticDeclarations._ProtocolBox.model_json_schema(
            mode=mode,
        )
        assert schema["$defs"][
            "PublicPayload"
        ] == m.Tests.PublicPayload.model_json_schema(
            mode=mode,
        )

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
            """Squared length of the vector.

            Returns:
                The resulting ``int``.

            """
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

        animal: Annotated[
            TestsFlextCorePydanticDeclarations._Cat
            | TestsFlextCorePydanticDeclarations._Dog,
            m.Discriminator("kind"),
        ]

    class _Item(m.BaseModel):
        """Base item serialized through a base-typed field."""

        name: str

    class _DetailedItem(_Item):
        """Subclass field preserved at serialization time via SerializeAsAny."""

        detail: str

    class _Box(m.BaseModel):
        """Container whose base-typed field serializes runtime subclasses."""

        item: m.SerializeAsAny[TestsFlextCorePydanticDeclarations._Item]

    class _Greeter:
        """Plain runtime class guarded by InstanceOf."""

        @staticmethod
        def hello() -> str:
            return "hello"

    class _GreetingCard(m.BaseModel):
        """Model whose payload must be a Greeter instance."""

        payload: m.InstanceOf[TestsFlextCorePydanticDeclarations._Greeter]

    class _Constrained(m.BaseModel):
        """Model field constrained through StringConstraints."""

        code: Annotated[str, m.StringConstraints(min_length=3, pattern=r"^[a-z]+$")]

    class _Credentials(m.BaseModel):
        """Model whose secret field is declared through the typings facade."""

        model_config = m.ConfigDict(frozen=True)

        token: t.SecretStr

    class _SerializedFields(m.BaseModel):
        """Native plain and wrap serializers through both public facades."""

        plain_model: int
        plain_utility: int
        wrapped_model: int
        wrapped_utility: int

        @m.field_serializer("plain_model", mode="plain")
        @staticmethod
        def serialize_plain_model(value: int) -> str:
            """Replace the field's native representation through ``m``.

            Returns:
                The labeled field value.

            """
            return f"model:{value}"

        @u.field_serializer("plain_utility", mode="plain")
        @classmethod
        def serialize_plain_utility(cls, value: int) -> str:
            """Replace the field's native representation through ``u``.

            Returns:
                The labeled field value.

            """
            return f"utility:{value}"

        @m.field_serializer("wrapped_model", mode="wrap")
        @staticmethod
        def serialize_wrapped_model(value: int, handler: Callable[[int], int]) -> int:
            """Extend the native field serializer through ``m``.

            Returns:
                The native serialized value incremented by one.

            """
            return handler(value) + 1

        @u.field_serializer("wrapped_utility", mode="wrap", when_used="json")
        @classmethod
        def serialize_wrapped_utility(
            cls, value: int, handler: Callable[[int], int]
        ) -> int:
            """Extend the native JSON-only field serializer through ``u``.

            Returns:
                The native serialized value incremented by two.

            """
            return handler(value) + 2

    @staticmethod
    def test_native_field_serializers_preserve_plain_and_wrap_behavior() -> None:
        """Real dumps invoke every registered serializer and its native handler."""
        payload = TestsFlextCorePydanticDeclarations._SerializedFields.model_validate({
            "plain_model": 3,
            "plain_utility": 5,
            "wrapped_model": 7,
            "wrapped_utility": 11,
        })

        assert payload.model_dump() == {
            "plain_model": "model:3",
            "plain_utility": "utility:5",
            "wrapped_model": 8,
            "wrapped_utility": 11,
        }
        expected_json = {
            "plain_model": "model:3",
            "plain_utility": "utility:5",
            "wrapped_model": 8,
            "wrapped_utility": 13,
        }
        assert payload.model_dump(mode="json") == expected_json
        assert (
            u.type_adapter(TypeForm(t.JsonDict)).validate_json(
                payload.model_dump_json()
            )
            == expected_json
        )
        assert (payload.plain_model, payload.plain_utility) == (3, 5)
        assert (payload.wrapped_model, payload.wrapped_utility) == (7, 11)

    class _ComputedFields(m.BaseModel):
        """Real computed properties in bare and configured native forms."""

        value: int

        @m.computed_field
        @property
        def directory(self) -> str:
            """Expose a string property to ordinary runtime consumers.

            Returns:
                The directory derived from the model value.

            """
            return f"stream-{self.value}"

        @u.computed_field(alias="scaled", repr=False)
        @property
        def doubled(self) -> int:
            """Expose a configured computed field through the utility facade.

            Returns:
                Twice the model value.

            """
            return self.value * 2

    @staticmethod
    def test_native_computed_fields_preserve_property_types_and_dump_behavior() -> None:
        """Consume real properties and include them in Python and JSON dumps."""
        payload = TestsFlextCorePydanticDeclarations._ComputedFields(value=7)
        directory: str = payload.directory
        doubled: int = payload.doubled
        assert directory.removeprefix("stream-") == str(payload.value)
        assert doubled == payload.value * 2
        expected = {
            "value": payload.value,
            "directory": directory,
            "scaled": doubled,
        }
        assert payload.model_dump(by_alias=True) == expected
        assert payload.model_dump(mode="json", by_alias=True) == expected
        assert (
            u.type_adapter(TypeForm(t.JsonDict)).validate_json(
                payload.model_dump_json(by_alias=True)
            )
            == expected
        )

    @staticmethod
    def test_secret_str_field_hides_value_and_round_trips() -> None:
        round_trip_value = "s3cret"
        credentials = TestsFlextCorePydanticDeclarations._Credentials.model_validate({
            "token": round_trip_value,
        })

        assert isinstance(credentials.token, t.SecretStr)
        assert credentials.token.get_secret_value() == "s3cret"
        assert "s3cret" not in repr(credentials)

        with pytest.raises(m.ValidationError):
            TestsFlextCorePydanticDeclarations._Credentials.model_validate({})

    @staticmethod
    def test_string_constraints_accept_valid_and_reject_invalid_values() -> None:
        assert TestsFlextCorePydanticDeclarations._Constrained(code="abc").code == "abc"

        with pytest.raises(m.ValidationError):
            TestsFlextCorePydanticDeclarations._Constrained(code="ab")

        with pytest.raises(m.ValidationError):
            TestsFlextCorePydanticDeclarations._Constrained(code="ABC")

    @staticmethod
    def test_discriminator_resolves_union_member_from_tag() -> None:
        pet = TestsFlextCorePydanticDeclarations._Pet.model_validate({
            "animal": {"kind": "dog", "bark": "woof"},
        })

        assert isinstance(pet.animal, TestsFlextCorePydanticDeclarations._Dog)
        assert pet.animal.bark == "woof"

        with pytest.raises(m.ValidationError):
            TestsFlextCorePydanticDeclarations._Pet.model_validate({
                "animal": {"kind": "cow", "moo": "moo"},
            })

    @staticmethod
    def test_serialize_as_any_keeps_subclass_fields_in_dump() -> None:
        box = TestsFlextCorePydanticDeclarations._Box(
            item=TestsFlextCorePydanticDeclarations._DetailedItem(
                name="flext",
                detail="advanced",
            ),
        )

        assert box.model_dump() == {"item": {"name": "flext", "detail": "advanced"}}

    @staticmethod
    def test_type_adapter_constructor_validates_python_values() -> None:
        """The public facade constructor is pydantic's TypeAdapter."""
        sample_value = 7
        adapter: m.TypeAdapter[int] = u.type_adapter(int)
        validated = adapter.validate_python(sample_value)

        assert validated == sample_value
        with pytest.raises(m.ValidationError):
            adapter.validate_python("not an integer")

    @staticmethod
    def test_fail_fast_reports_only_the_first_list_error() -> None:
        adapter: m.TypeAdapter[list[int]] = u.type_adapter(
            Annotated[list[int], m.FailFast()],
        )

        assert adapter.validate_python([1, 2]) == [1, 2]

        with pytest.raises(m.ValidationError) as exc_info:
            adapter.validate_python(["first", "second"])

        assert len(exc_info.value.errors()) == 1

    @staticmethod
    def test_instance_of_accepts_instance_and_rejects_foreign_object() -> None:
        greeter = TestsFlextCorePydanticDeclarations._Greeter()
        card = TestsFlextCorePydanticDeclarations._GreetingCard(payload=greeter)

        assert card.payload is greeter

        with pytest.raises(m.ValidationError):
            TestsFlextCorePydanticDeclarations._GreetingCard.model_validate({
                "payload": object(),
            })

    @staticmethod
    def test_validate_as_builds_custom_type_from_native_model() -> None:
        adapter: m.TypeAdapter[TestsFlextCorePydanticDeclarations._Vector] = (
            u.type_adapter(
                Annotated[
                    TestsFlextCorePydanticDeclarations._Vector,
                    m.ValidateAs(
                        TestsFlextCorePydanticDeclarations._VectorInput,
                        lambda validated: TestsFlextCorePydanticDeclarations._Vector(
                            validated.x,
                            validated.y,
                        ),
                    ),
                ],
            )
        )

        vector = adapter.validate_python({"x": 1, "y": 2})

        assert isinstance(vector, TestsFlextCorePydanticDeclarations._Vector)
        assert (vector.x, vector.y) == (1, 2)

        with pytest.raises(m.ValidationError):
            adapter.validate_python({"x": "no-int", "y": 2})
