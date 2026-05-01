"""Pydantic facade coverage tests.

Exercises every class-level alias in FlextUtilitiesPydantic via the public
``u.*`` surface.
"""

from __future__ import annotations

from tests import u


class TestsFlextUtilitiesPydantic:
    def test_field_is_callable(self) -> None:
        assert callable(u.Field)

    def test_private_attr_is_callable(self) -> None:
        assert callable(u.PrivateAttr)

    def test_skip_validation_is_accessible(self) -> None:
        assert u.SkipValidation is not None

    def test_computed_field_is_callable(self) -> None:
        assert callable(u.computed_field)

    def test_field_validator_is_callable(self) -> None:
        assert callable(u.field_validator)

    def test_field_serializer_is_callable(self) -> None:
        assert callable(u.field_serializer)

    def test_model_validator_is_callable(self) -> None:
        assert callable(u.model_validator)

    def test_model_serializer_is_callable(self) -> None:
        assert callable(u.model_serializer)

    def test_after_validator_is_accessible(self) -> None:
        assert u.AfterValidator is not None

    def test_before_validator_is_accessible(self) -> None:
        assert u.BeforeValidator is not None

    def test_plain_validator_is_accessible(self) -> None:
        assert u.PlainValidator is not None

    def test_wrap_validator_is_accessible(self) -> None:
        assert u.WrapValidator is not None

    def test_plain_serializer_is_accessible(self) -> None:
        assert u.PlainSerializer is not None

    def test_wrap_serializer_is_accessible(self) -> None:
        assert u.WrapSerializer is not None

    def test_config_dict_is_accessible(self) -> None:
        assert u.ConfigDict is not None

    def test_field_serialization_info_is_accessible(self) -> None:
        assert u.FieldSerializationInfo is not None

    def test_type_adapter_is_accessible(self) -> None:
        assert u.TypeAdapter is not None

    def test_create_model_is_callable(self) -> None:
        assert callable(u.create_model)

    def test_validate_call_is_callable(self) -> None:
        assert callable(u.validate_call)

    def test_with_config_is_callable(self) -> None:
        assert callable(u.with_config)

    def test_from_json_is_callable(self) -> None:
        assert callable(u.from_json)

    def test_to_json_is_callable(self) -> None:
        assert callable(u.to_json)

    def test_to_jsonable_python_is_callable(self) -> None:
        assert callable(u.to_jsonable_python)

    def test_create_model_produces_working_model(self) -> None:
        dynamic_model = u.create_model("DynamicModel", name=(str, ...))
        inst = dynamic_model(name="hello")
        assert getattr(inst, "name") == "hello"

    def test_to_json_serializes_dict(self) -> None:
        result = u.to_json({"key": "value"})
        assert result == b'{"key":"value"}'

    def test_from_json_deserializes_bytes(self) -> None:
        result = u.from_json(b'{"key":"value"}')
        assert result == {"key": "value"}

    def test_to_jsonable_python_converts_dict(self) -> None:
        result = u.to_jsonable_python({"key": "value"})
        assert result == {"key": "value"}

    def test_type_adapter_validates_list(self) -> None:
        ta = u.TypeAdapter(list[int])
        result = ta.validate_python([1, 2, 3])
        assert result == [1, 2, 3]

    def test_field_creates_field_info(self) -> None:
        fi = u.Field(description="test")
        assert fi is not None

    def test_validate_call_wraps_function(self) -> None:
        @u.validate_call()
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3
