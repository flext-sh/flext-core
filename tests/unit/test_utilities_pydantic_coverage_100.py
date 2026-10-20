"""Behavior contract for the public Pydantic facade exposed via ``u.*``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, ClassVar

import pytest
from typing_extensions import TypeForm

from tests import m, t, tm, u


class TestsFlextUtilitiesPydantic:
    """Tests for ``FlextUtilitiesPydantic``."""

    @staticmethod
    def test_unset_factories_expose_native_metadata() -> None:
        """An omitted value produces metadata, not a contextual field value."""
        metadata = m.Field(description="Native field metadata")
        tm.ok(metadata.annotation is None)
        private = m.PrivateAttr()
        undefined = private.get_default()
        tm.ok(private.get_default(call_default_factory=True) is undefined)

    class _FieldFactoryContract(m.BaseModel):
        """Required fields and typed factories through both public facades."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(str_strip_whitespace=True)

        label: str = m.Field(min_length=1)
        echo: str = m.Field(default_factory=u.Tests.validated_label)
        model_values: t.MutableStrMapping = m.Field(
            default_factory=u.empty(TypeForm(t.MutableStrMapping)),
        )
        utility_values: t.MutableStrMapping = m.Field(
            default_factory=u.empty(TypeForm(t.MutableStrMapping)),
        )

    class _RequiredFieldContract(m.BaseModel):
        """Native required declarations remain typed through both facades."""

        model_label: str = m.Field(min_length=1)
        utility_label: str = m.Field(min_length=1)

    class _FieldDefaultsContract(m.BaseModel):
        """Native default overloads remain available through the composed facade."""

        label: str = m.Field(default="native", min_length=1)
        attempts: int = m.Field(3, ge=0)
        optional: str | None = m.Field(default=None)

    @staticmethod
    def test_field_defaults_preserve_native_types_and_validation() -> None:
        """Default overloads preserve identity, defaults and schema constraints."""
        tm.ok(m.Field is u.Field)
        model = TestsFlextUtilitiesPydantic._FieldDefaultsContract.model_validate({})
        label: str = model.label
        attempts: int = model.attempts
        optional: str | None = model.optional
        tm.that(label, eq="native")
        tm.that(attempts, eq=3)
        tm.ok(optional is None)
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._FieldDefaultsContract.model_validate({
                "label": "",
                "attempts": -1,
            })

    @staticmethod
    def test_required_fields_preserve_annotations_and_validation() -> None:
        """Metadata declarations are required, not descriptor-valued defaults."""
        model = TestsFlextUtilitiesPydantic._RequiredFieldContract(
            model_label="model",
            utility_label="utility",
        )
        model_label: str = model.model_label
        utility_label: str = model.utility_label
        tm.that(model_label == "model", truth=True)
        tm.that(utility_label == "utility", truth=True)
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._RequiredFieldContract.model_validate({})

    @staticmethod
    def test_field_factories_receive_validated_data_and_keep_required_fields() -> None:
        """Factories see validated fields and allocate independent defaults."""
        first = TestsFlextUtilitiesPydantic._FieldFactoryContract(label="  first  ")
        second = TestsFlextUtilitiesPydantic._FieldFactoryContract(label="second")
        echo: str = first.echo
        tm.that(echo == first.label == "first", truth=True)
        tm.that(second.echo == second.label, truth=True)

        first.model_values["record"] = "model"
        first.utility_values["record"] = "utility"
        tm.that(second.model_values == {}, truth=True)
        tm.that(second.utility_values == {}, truth=True)
        tm.that(
            TestsFlextUtilitiesPydantic._FieldFactoryContract.model_validate_json(
                first.model_dump_json(),
            )
            == first,
            truth=True,
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
            m.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
        ] = m.Field(description="Utility collection with an annotation-owned factory.")

    @staticmethod
    def test_assignment_metadata_preserves_annotated_factories() -> None:
        """Omitted factories survive composition, isolation and JSON round-trips."""
        first = TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate({})
        second = (
            TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate({})
        )
        tm.that(first.model_values == first.utility_values == {}, truth=True)
        first.model_values["model"] = "first"
        first.utility_values["utility"] = "first"
        tm.that(second.model_values == second.utility_values == {}, truth=True)
        tm.that(
            TestsFlextUtilitiesPydantic._AnnotatedFactoryContract.model_validate_json(
                first.model_dump_json(),
            )
            == first,
            truth=True,
        )

    class _RequiredCollectionContract(m.BaseModel):
        """Required collections declare their types without default factories."""

        model_values: Annotated[
            t.MutableStrMapping,
            m.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
            m.FieldInfo.from_field(default_factory=None),
        ]
        utility_values: Annotated[
            t.MutableStrMapping,
            m.Field(default_factory=u.empty(TypeForm(t.MutableStrMapping))),
            m.FieldInfo.from_field(default_factory=None),
        ]

    @pytest.mark.parametrize("missing_name", ["model_values", "utility_values"])
    @staticmethod
    def test_required_collections_reject_missing_values(missing_name: str) -> None:
        """Required fields reject omission and retain the supplied values."""
        partial: t.JsonDict = {"model_values": {}, "utility_values": {}}
        del partial[missing_name]
        with pytest.raises(m.ValidationError):
            TestsFlextUtilitiesPydantic._RequiredCollectionContract.model_validate(
                partial,
            )
        values: t.JsonDict = {
            "model_values": {"model": "supplied"},
            "utility_values": {"utility": "supplied"},
        }
        payload = (
            TestsFlextUtilitiesPydantic._RequiredCollectionContract.model_validate(
                values,
            )
        )
        tm.that(payload.model_values == {"model": "supplied"}, truth=True)
        tm.that(payload.utility_values == {"utility": "supplied"}, truth=True)

    class _AliasFactoryContract(m.BaseModel):
        """Literal aliases and factories participate in constructor synthesis."""

        model_label: str = m.Field(alias="modelLabel")
        utility_label: str = m.Field(alias="utilityLabel")
        model_value: str = m.Field(default_factory=str)
        utility_value: str = m.Field(default_factory=str)

    @staticmethod
    def test_alias_named_constructor_preserves_factory_defaults() -> None:
        """Alias-keyed payloads construct models without factory-defaulted args."""
        values: t.MutableStrMapping = {"modelLabel": "model", "utilityLabel": "utility"}
        payload = TestsFlextUtilitiesPydantic._AliasFactoryContract(**values)
        tm.that(payload.model_label == "model", truth=True)
        tm.that(payload.utility_label == "utility", truth=True)
        tm.that(payload.model_value == payload.utility_value == "", truth=True)
        tm.that(
            TestsFlextUtilitiesPydantic._AliasFactoryContract.model_validate_json(
                payload.model_dump_json(by_alias=True),
            )
            == payload,
            truth=True,
        )

    class _PrivateAttrContract(m.BaseModel):
        label: str
        _model_values: list[str] = m.PrivateAttr(
            default_factory=list[str],
        )
        _utility_values: list[str] = m.PrivateAttr(
            default_factory=u.empty(TypeForm(list[str])),
        )
        _label_copy: str = m.PrivateAttr(default_factory=u.Tests.validated_label)
        _reader: Callable[[str], str] = m.PrivateAttr(
            default_factory=u.Tests.input_reader,
        )

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

        tm.that(payload.raw_name == expected_name, truth=True)

    @staticmethod
    def test_serialization_applies_alias_serializer_and_computed_field() -> None:
        """Test serialization applies alias serializer and computed field."""
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": "  ada lovelace ",
            "visits": "3",
        })

        payload_dump = payload.model_dump(mode="json", by_alias=True)

        tm.that(payload_dump["rawName"] == "Ada Lovelace", truth=True)
        tm.that(payload_dump["visits"] == "3 visits", truth=True)
        tm.that(payload_dump["label"] == "Ada Lovelace:3", truth=True)

    @staticmethod
    def test_computed_field_kwargs_overload_renders_alias() -> None:
        """Test the kwargs-only computed_field overload renders alias and value."""
        payload = m.Tests.KwargsComputedPayload.model_validate({"raw": "ada"})

        payload_dump = payload.model_dump(mode="json", by_alias=True)

        tm.that(payload_dump["upperLabel"] == "ADA", truth=True)
        tm.that(payload.model_dump(mode="json")["label"] == "ADA", truth=True)

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

        tm.that(payload_dict == payload_dump, truth=True)
        tm.that(
            payload_jsonable == payload.model_dump(mode="json", by_alias=True),
            truth=True,
        )

    @staticmethod
    def test_validate_call_rejects_invalid_argument_values() -> None:
        """Test validate call rejects invalid argument values."""

        @u.validate_call
        def double_positive(value: t.PositiveInt) -> int:
            doubled: int = value * 2
            return doubled

        tm.that(double_positive(4) == 8, truth=True)
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
        tm.that(checksum == sum(ord(character) for character in "value"), truth=True)
        with pytest.raises(m.ValidationError):
            positive_checksum(value="  ")

    @staticmethod
    def test_public_facade_resolves_runtime_bootstrap_options_from_json() -> None:
        """Test public facade resolves runtime bootstrap options from json."""
        runtime_options = m.RuntimeBootstrapOptions.model_validate_json(
            u.to_json({"settings_overrides": {"dry_run": True}}),
        )

        resolved = u.resolve_runtime_options(runtime_options)

        tm.that(resolved.settings_overrides == {"dry_run": True}, truth=True)
        tm.that(resolved.settings is None, truth=True)
        tm.that(resolved.context is None, truth=True)
        tm.that(resolved.settings_type is runtime_options.settings_type, truth=True)
        tm.that(resolved.container is runtime_options.container, truth=True)
        tm.that(resolved.dispatcher is runtime_options.dispatcher, truth=True)

    @staticmethod
    def test_private_attr_factories_preserve_pydantic_instance_semantics() -> None:
        """Test private attr factories preserve pydantic instance semantics."""
        first = TestsFlextUtilitiesPydantic._PrivateAttrContract(label="first")
        second = TestsFlextUtilitiesPydantic._PrivateAttrContract(label="second")

        first.record_model("model")
        first.record_utility("utility")

        tm.that(first.recorded_model_values == ["model"], truth=True)
        tm.that(second.recorded_model_values == [], truth=True)
        tm.that(first.recorded_utility_values == ["utility"], truth=True)
        tm.that(second.recorded_utility_values == [], truth=True)
        tm.that(first.label_echo == "first", truth=True)
        tm.that(second.label_echo == "second", truth=True)
        tm.that(first.reads_standard_input, truth=True)
        tm.that(second.reads_standard_input, truth=True)
        tm.that(first.model_dump() == {"label": "first"}, truth=True)
