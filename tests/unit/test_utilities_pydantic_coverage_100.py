"""Behavior contract for the public Pydantic facade exposed via ``u.*``."""

from __future__ import annotations

import pytest

from flext_core import u
from tests.models import m
from tests.typings import t


class TestsFlextUtilitiesPydantic:
    @pytest.mark.parametrize(
        ("raw_name", "expected_name"),
        [
            ("  ada lovelace ", "Ada Lovelace"),
            ("GRACE HOPPER", "Grace Hopper"),
            ("alan", "Alan"),
        ],
    )
    def test_field_validator_normalizes_aliased_name(
        self,
        raw_name: str,
        expected_name: str,
    ) -> None:
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": raw_name,
            "visits": "3",
        })

        assert payload.raw_name == expected_name

    def test_serialization_applies_alias_serializer_and_computed_field(
        self,
    ) -> None:
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": "  ada lovelace ",
            "visits": "3",
        })

        payload_dump = payload.model_dump(mode="json", by_alias=True)

        assert payload_dump["rawName"] == "Ada Lovelace"
        assert payload_dump["visits"] == "3 visits"
        assert payload_dump["label"] == "Ada Lovelace:3"

    def test_public_facade_supports_dynamic_models_and_json_roundtrip(self) -> None:
        dynamic_model = u.create_model(
            "DynamicPayload",
            name=(str, ...),
            count=(int, ...),
            tags=(list[str], u.Field(default_factory=list)),
        )
        adapter = u.TypeAdapter(dynamic_model)

        payload = adapter.validate_python({
            "name": "queue",
            "count": "2",
            "tags": ["cli"],
        })
        payload_dump = payload.model_dump()
        payload_json = u.to_json(payload.model_dump())
        payload_dict = u.from_json(payload_json)
        payload_jsonable = u.to_jsonable_python(payload)

        assert payload_dump == {"name": "queue", "count": 2, "tags": ["cli"]}
        assert payload_dict == {"name": "queue", "count": 2, "tags": ["cli"]}
        assert payload_jsonable == payload_dict

    def test_validate_call_rejects_invalid_argument_types(self) -> None:
        @u.validate_call()
        def to_upper(value: str) -> str:
            return value.upper()

        assert to_upper("cli") == "CLI"
        with pytest.raises(m.ValidationError):
            # Why: negative test feeds a deliberately wrong-typed argument to
            # prove ``u.validate_call`` enforces the annotation at runtime.
            to_upper(object())  # type: ignore[arg-type]

    def test_public_facade_resolves_runtime_bootstrap_options_from_json(
        self,
    ) -> None:
        runtime_options = m.RuntimeBootstrapOptions.model_validate_json(
            u.to_json({
                "subproject": "source-runtime",
                "wire_packages": ["flext.core.runtime", "tests.runtime"],
                "settings_overrides": {"dry_run": True},
            }),
        )

        @u.validate_call()
        def build_runtime_options(
            options: m.RuntimeBootstrapOptions,
            override_subproject: str,
            override_packages: t.StrSequence,
        ) -> m.RuntimeBootstrapOptions:
            return u.resolve_runtime_options(
                options,
                subproject=override_subproject.strip().replace("_", "-"),
                wire_packages=override_packages,
            )

        resolved = build_runtime_options(
            runtime_options,
            " cli_runtime ",
            ("flext.cli.runtime", "flext.cli.jobs"),
        )

        assert runtime_options.subproject == "source-runtime"
        assert list(runtime_options.wire_packages or ()) == [
            "flext.core.runtime",
            "tests.runtime",
        ]
        assert runtime_options.settings_overrides == {"dry_run": True}
        assert resolved.subproject == "cli-runtime"
        assert list(resolved.wire_packages or ()) == [
            "flext.cli.runtime",
            "flext.cli.jobs",
        ]
        assert resolved.settings_overrides == {"dry_run": True}
