"""Behavior contract for the public Pydantic facade exposed via ``u.*``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from operator import itemgetter

import pytest
from typing_extensions import TypeForm

from flext_core import FlextModels, u
from tests import m, t


def _input_reader() -> Callable[[str], str]:
    return input


class TestsFlextUtilitiesPydantic:
    """Tests for ``FlextUtilitiesPydantic``."""

    class _PrivateAttrContract(FlextModels.BaseModel):
        label: str
        _model_values: list[str] = FlextModels.PrivateAttr(
            default_factory=FlextModels.empty(TypeForm(list[str])),
        )
        _utility_values: list[str] = u.PrivateAttr(
            default_factory=u.empty(TypeForm(list[str])),
        )
        _label_copy: str = FlextModels.PrivateAttr(default_factory=itemgetter("label"))
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

        assert payload.model_dump() == "ada"
        assert payload.model_dump(mode="json") == "ada"

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
    def test_public_facade_resolves_runtime_bootstrap_options_from_json() -> None:
        """Test public facade resolves runtime bootstrap options from json."""
        runtime_options = m.RuntimeBootstrapOptions.model_validate_json(
            u.to_json({"settings_overrides": {"dry_run": True}}),
        )

        @u.validate_call
        def resolve_options(
            options: m.RuntimeBootstrapOptions,
        ) -> m.RuntimeBootstrapOptions:
            return u.resolve_runtime_options(options)

        resolved = resolve_options(runtime_options)

        assert resolved.settings_overrides == {"dry_run": True}
        assert resolved.settings is None
        assert resolved.context is None
        assert resolved.model_dump(mode="json") == {
            "settings_overrides": {"dry_run": True},
        }

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
