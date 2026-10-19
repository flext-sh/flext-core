"""Behavior contract for the public Pydantic facade exposed via ``u.*``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, ClassVar

import pytest
from typing_extensions import TypeForm

from flext_core import FlextModels, FlextUtilities
from tests import m, p, t, tm, u


def _input_reader() -> Callable[[str], str]:
    return input


def _validated_label(values: t.MappingKV[str, p.AttributeProbe]) -> str:
    """Derive a field default from Pydantic's already validated input.

    Returns:
        The validated label passed to the field factory.

    """
    label = values["label"]
    if not isinstance(label, str):
        msg = "Validated label must be a string"
        raise TypeError(msg)
    return label


class TestsFlextUtilitiesPydantic:
    """Tests for ``FlextUtilitiesPydantic``."""

    @staticmethod
    def test_unset_factories_expose_native_metadata() -> None:
        """An omitted value produces metadata, not a contextual field value."""
        metadata = u.Field(description="Native field metadata")
        tm.that(metadata.annotation is None, eq=True)
        private = u.PrivateAttr()
        tm.that(isinstance(private, p.UnsetPrivateAttribute), eq=True)
        undefined = private.get_default()
        tm.that(private.get_default(call_default_factory=True) is undefined, eq=True)

    class _FieldFactoryContract(m.BaseModel):
        """Required fields and typed factories through both public facades."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(str_strip_whitespace=True)

        label: str = u.Field(min_length=1)
        echo: str = m.Field(default_factory=_validated_label)
        model_values: t.MutableStrMapping = m.Field(
            default_factory=u.empty(TypeForm(t.MutableStrMapping)),
        )
        utility_values: t.MutableStrMapping = u.Field(
            default_factory=u.empty(TypeForm(t.MutableStrMapping)),
        )

    class _RequiredFieldContract(m.BaseModel):
        """Native required declarations remain typed through both facades."""

        model_label: str = m.Field(min_length=1)
        utility_label: str = u.Field(min_length=1)

    @staticmethod
    def test_required_fields_preserve_annotations_and_validation() -> None:
        """Metadata declarations are required, not descriptor-valued defaults."""
        model = TestsFlextUtilitiesPydantic._RequiredFieldContract(
            model_label="model",
            utility_label="utility",
        )
        model_label: str = model.model_label
        utility_label: str = model.utility_label
        assert model_label == "model"
        assert utility_label == "utility"
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._RequiredFieldContract.model_validate({})

    @staticmethod
    def test_field_factories_receive_validated_data_and_keep_required_fields() -> None:
        """Factories see validated fields and allocate independent defaults."""
        first = TestsFlextUtilitiesPydantic._FieldFactoryContract(label="  first  ")
        second = TestsFlextUtilitiesPydantic._FieldFactoryContract(label="second")
        echo: str = first.echo
        assert echo == first.label == "first"
        assert second.echo == second.label

        first.model_values["record"] = "model"
        first.utility_values["record"] = "utility"
        assert second.model_values == {}
        assert second.utility_values == {}
        assert (
            TestsFlextUtilitiesPydantic._FieldFactoryContract.model_validate_json(
                first.model_dump_json(),
            )
            == first
        )
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._FieldFactoryContract.model_validate({})

    class _AnnotatedFactoryContract(m.BaseModel):
        """Assignment metadata preserves annotation-owned default factories."""

        model_values: Annotated[
            t.MutableStrMapping,
            m.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
        ] = m.Field(description="Model collection with an annotation-owned factory.")
        utility_values: Annotated[
            t.MutableStrMapping,
            u.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
        ] = u.Field(description="Utility collection with an annotation-owned factory.")

    @staticmethod
    def test_assignment_metadata_preserves_annotated_factories() -> None:
        """Omitted factories survive composition, isolation and JSON round-trips."""
        first = TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate({})
        second = (
            TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate({})
        )
        assert first.model_values == first.utility_values == {}
        first.model_values["model"] = "first"
        first.utility_values["utility"] = "first"
        assert second.model_values == second.utility_values == {}
        assert (
            TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate_json(
                first.model_dump_json(),
            )
            == first
        )

    class _ClearedAnnotatedFactoryContract(m.BaseModel):
        """Explicit None clears the earlier factory through both facades."""

        model_values: Annotated[
            t.MutableStrMapping,
            m.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
        ] = m.Field(default_factory=None)
        utility_values: Annotated[
            t.MutableStrMapping,
            u.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
        ] = u.Field(default_factory=None)

    @pytest.mark.parametrize("missing_name", ["model_values", "utility_values"])
    @staticmethod
    def test_explicit_none_clears_annotated_factories(missing_name: str) -> None:
        """Cleared factories require supplied values instead of inventing defaults."""
        partial: t.JsonDict = {"model_values": {}, "utility_values": {}}
        del partial[missing_name]
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._ClearedAnnotatedFactoryContract.model_validate(
                partial,
            )
        values: t.JsonDict = {
            "model_values": {"model": "supplied"},
            "utility_values": {"utility": "supplied"},
        }
        payload = (
            TestsFlextUtilitiesPydantic._ClearedAnnotatedFactoryContract.model_validate(
                values,
            )
        )
        assert payload.model_values == {"model": "supplied"}
        assert payload.utility_values == {"utility": "supplied"}

    class _AliasFactoryContract(FlextModels.BaseModel):
        """Literal aliases and factories participate in constructor synthesis."""

        model_label: str = FlextModels.Field(alias="modelLabel")
        utility_label: str = FlextUtilities.Field(alias="utilityLabel")
        model_value: str = FlextModels.Field(default_factory=str)
        utility_value: str = FlextUtilities.Field(default_factory=str)

    @staticmethod
    def test_alias_named_constructor_preserves_factory_defaults() -> None:
        """Alias-keyed payloads construct models without factory-defaulted args."""
        values: t.MutableStrMapping = {"modelLabel": "model", "utilityLabel": "utility"}
        payload = TestsFlextUtilitiesPydantic._AliasFactoryContract(**values)
        assert payload.model_label == "model"
        assert payload.utility_label == "utility"
        assert payload.model_value == payload.utility_value == ""
        assert (
            TestsFlextUtilitiesPydantic._AliasFactoryContract.model_validate_json(
                payload.model_dump_json(by_alias=True),
            )
            == payload
        )

    class _PrivateAttrContract(FlextModels.BaseModel):
        label: str
        _model_values: list[str] = FlextModels.PrivateAttr(
            default_factory=list[str],
        )
        _utility_values: list[str] = u.PrivateAttr(
            default_factory=u.empty(TypeForm(list[str])),
        )
        _label_copy: str = FlextModels.PrivateAttr(default_factory=_validated_label)
        _reader: Callable[[str], str] = u.PrivateAttr(default_factory=_input_reader)

        def record_model(self, value: str) -> None:
            """Append to the core-model private history."""
            self._model_values.append(value)

        def record_utility(self, value: str) -> None:
            """Append to the utility private history."""
            self._utility_values.append(value)

        @property
        def recorded_model_values(self) -> list[str]:
            """Snapshot of the core-model private history."""
            return list(self._model_values)

        @property
        def recorded_utility_values(self) -> list[str]:
            """Snapshot of the utility private history."""
            return list(self._utility_values)

        @property
        def label_echo(self) -> str:
            """Copy of the validated label captured at init time."""
            return self._label_copy

        @property
        def reads_standard_input(self) -> bool:
            """Whether the default reader resolved to the builtin input."""
            return self._reader is input

    @pytest.mark.parametrize(
        ("raw_name", "expected_name"),
        [
            ("  ada lovelace ", "Ada Lovelace"),
            ("GRACE HOPPER", "Grace Hopper"),
            ("alan", "Alan"),
        ],
    )
    @staticmethod
    def test_field_validator_normalizes_aliased_name(
        raw_name: str,
        expected_name: str,
    ) -> None:
        """Test field validator normalizes aliased name."""
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": raw_name,
            "visits": "3",
        })

        assert payload.raw_name == expected_name

    @staticmethod
    def test_serialization_applies_alias_serializer_and_computed_field() -> None:
        """Test serialization applies alias serializer and computed field."""
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": "  ada lovelace ",
            "visits": "3",
        })

        payload_dump = payload.model_dump(mode="json", by_alias=True)

        assert payload_dump["rawName"] == "Ada Lovelace"
        assert payload_dump["visits"] == "3 visits"
        assert payload_dump["label"] == "Ada Lovelace:3"

    @staticmethod
    def test_computed_field_kwargs_overload_renders_alias() -> None:
        """Test the kwargs-only computed_field overload renders alias and value."""
        payload = m.Tests.KwargsComputedPayload.model_validate({"raw": "ada"})

        payload_dump = payload.model_dump(mode="json", by_alias=True)

        assert payload_dump["upperLabel"] == "ADA"
        assert payload.model_dump(mode="json")["label"] == "ADA"

    @staticmethod
    def test_plain_model_serializer_replaces_the_dump_shape() -> None:
        """Test the bare plain model serializer replaces the dump shape."""
        payload = m.Tests.PlainSerializedPayload.model_validate({"name": "ada"})

        tm.that(payload.model_dump(), eq=payload.name)
        tm.that(payload.model_dump(mode="json"), eq=payload.name)

    @staticmethod
    def test_public_facade_supports_json_roundtrip() -> None:
        """Test public facade supports json roundtrip."""
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": "  ada lovelace ",
            "visits": "3",
        })
        payload_dump = payload.model_dump()
        payload_json = u.to_json(payload.model_dump())
        payload_dict = u.from_json(payload_json)
        payload_jsonable = u.to_jsonable_python(payload)

        assert payload_dict == payload_dump
        assert payload_jsonable == payload.model_dump(mode="json", by_alias=True)

    @staticmethod
    def test_validate_call_rejects_invalid_argument_values() -> None:
        """Test validate call rejects invalid argument values."""

        @u.validate_call
        def double_positive(value: t.PositiveInt) -> int:
            doubled: int = value * 2
            return doubled

        assert double_positive(4) == 8
        with pytest.raises(m.ValidationError):
            double_positive(-1)

    @staticmethod
    def test_validate_call_factory_preserves_config_and_return_validation() -> None:
        """The keyword-only overload configures arguments and validates returns."""

        @u.validate_call(
            config=m.ConfigDict(str_strip_whitespace=True),
            validate_return=True,
        )
        def positive_checksum(value: str) -> t.PositiveInt:
            return sum(ord(character) for character in value)

        checksum: int = positive_checksum(value="  value  ")
        assert checksum == sum(ord(character) for character in "value")
        with pytest.raises(m.ValidationError):
            positive_checksum(value="  ")

    @staticmethod
    def test_public_facade_resolves_runtime_bootstrap_options_from_json() -> None:
        """Test public facade resolves runtime bootstrap options from json."""
        runtime_options = m.RuntimeBootstrapOptions.model_validate_json(
            u.to_json({"settings_overrides": {"dry_run": True}}),
        )

        resolved = u.resolve_runtime_options(runtime_options)

        assert resolved.settings_overrides == {"dry_run": True}
        assert resolved.settings is None
        assert resolved.context is None
        assert resolved.settings_type is runtime_options.settings_type
        assert resolved.container is runtime_options.container
        assert resolved.dispatcher is runtime_options.dispatcher

    @staticmethod
    def test_private_attr_factories_preserve_pydantic_instance_semantics() -> None:
        """Test private attr factories preserve pydantic instance semantics."""
        first = TestsFlextUtilitiesPydantic._PrivateAttrContract(label="first")
        second = TestsFlextUtilitiesPydantic._PrivateAttrContract(label="second")

        first.record_model("model")
        first.record_utility("utility")

        assert first.recorded_model_values == ["model"]
        assert second.recorded_model_values == []
        assert first.recorded_utility_values == ["utility"]
        assert second.recorded_utility_values == []
        assert first.label_echo == "first"
        assert second.label_echo == "second"
        assert first.reads_standard_input
        assert second.reads_standard_input
        assert first.model_dump() == {"label": "first"}
