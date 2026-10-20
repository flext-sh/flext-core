"""Public native-serialization, metadata and composition boundary contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypeForm

from tests import c, e, m, r, t, tm, u

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreNormalizationBoundaries:
    """Validate domain boundaries rather than preserving a retired facade."""

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [None, 42, "text", b"text", Path("relative/path"), [1, None], {"key": None}],
    )
    def test_json_serialization_matches_native_sdk(
        value: t.Scalar | Path | t.JsonValue,
    ) -> None:
        """Serialization follows the native SDK's JSON admission contract."""
        expected = u.json_value_adapter().validate_python(
            u.to_jsonable_python(value), strict=True
        )
        tm.that(u.normalize_to_json_value(value), eq=expected)

    @staticmethod
    def test_json_null_is_preserved_at_both_boundaries() -> None:
        """Both boundaries preserve explicit null values."""
        tm.that(u.normalize_to_json_value(None) is None, eq=True)
        tm.that(u.normalize_to_container(None) is None, eq=True)
        tm.that(u.normalize_to_container([None]), eq=[None])

    @staticmethod
    def test_json_serialization_rejects_unknown_input() -> None:
        """Unsupported native values fail at JSON admission."""
        with pytest.raises(ValueError, match="Unable to serialize unknown type"):
            u.normalize_to_json_value(type)

    @staticmethod
    @pytest.mark.parametrize("tuple_items", [False, True])
    def test_json_payload_preserves_mapping_inside_nested_sequences(
        *,
        tuple_items: bool,
    ) -> None:
        """Mapping payloads must not materialize the abstract model union arm."""
        items: list[t.MetadataMapping] = [{"locations": []}]
        parsed = m.ConfigMap.model_validate({
            "issues": tuple(items) if tuple_items else items,
        })
        issues = parsed["issues"]
        if not isinstance(issues, (tuple, list)):
            pytest.fail("Issue sequence did not retain its native collection shape")
        tm.that(isinstance(issues[0], Mapping), eq=True)
        expected: t.JsonDict = {"issues": [{"locations": []}]}
        tm.that(u.normalize_to_json_mapping(parsed), eq=expected)

    @staticmethod
    def test_json_payload_preserves_nested_model_identity() -> None:
        """Existing model values retain identity and serialize their real fields."""
        model = m.Metadata(attributes={"source": "publication"})
        parsed = m.ConfigMap.model_validate({"items": [model]})
        items = parsed["items"]
        if not isinstance(items, (tuple, list)):
            pytest.fail("Model sequence did not retain its native collection shape")
        tm.that(items[0] is model, eq=True)
        tm.that(
            u.normalize_to_json_mapping(parsed),
            eq={"items": [model.model_dump(mode="json")]},
        )

    @staticmethod
    @pytest.mark.parametrize("nested", [False, True])
    def test_json_payload_rejects_non_string_mapping_keys(*, nested: bool) -> None:
        """Validation rejects invalid keys before native JSON can stringify them."""
        invalid = {1: "not a string key"}
        with pytest.raises(m.ValidationError):
            m.ConfigMap.model_validate({"issues": [invalid]} if nested else invalid)

    @staticmethod
    def test_json_payload_rejects_unsupported_nested_values() -> None:
        """Unknown values cannot be accepted as empty abstract model instances."""
        with pytest.raises(m.ValidationError):
            m.ConfigMap.model_validate({
                "issues": [{"invalid": ModuleType("invalid")}],
            })

    @staticmethod
    def test_native_adapter_is_constructible_through_the_model_provider() -> None:
        """The public SDK provider constructs an adapter, not a type-only alias."""
        adapter = u.type_adapter(TypeForm(t.MutableMappingKV[str, int]))
        tm.that(adapter.validate_json(b'{"value": 1}'), eq={"value": 1})
        with pytest.raises(m.ValidationError, match="int_parsing"):
            adapter.validate_json(b'{"value": "not an integer"}')

    @staticmethod
    def test_native_adapter_preserves_explicit_validation_policy() -> None:
        """The native provider does not relax the caller's declared policy."""
        adapter = u.type_adapter(TypeForm(int), config=m.ConfigDict(strict=True))
        tm.that(adapter.validate_python(1), eq=1)
        with pytest.raises(m.ValidationError, match="int_type"):
            adapter.validate_python("1")

    @staticmethod
    def test_container_preserves_models_and_unwraps_root_payloads() -> None:
        """Models retain identity while root payloads normalize recursively."""
        model = m.Metadata(attributes={"source": "native"})
        tm.that(u.normalize_to_container(model) is model, eq=True)
        tm.that(
            u.normalize_to_container(m.ConfigMap(root={"key": Path("relative")})),
            eq={"key": str(Path("relative"))},
        )

    @staticmethod
    def test_metadata_field_owns_attribute_validation() -> None:
        """Metadata validates attributes through its declared field boundary."""
        path = Path("metadata/source")
        metadata = m.Metadata.model_validate({
            "attributes": {"path": path, "absent": None, "nested": [None]}
        })
        tm.that(metadata.attributes, eq={"path": str(path), "nested": [None]})
        with pytest.raises(m.ValidationError, match="_private"):
            m.Metadata.model_validate({"attributes": {"_private": 1}})

    @staticmethod
    def test_metadata_accepts_readonly_native_mapping_defaults() -> None:
        """Readonly attribute inputs use the same validated JSON admission."""
        source = MappingProxyType({"active": True})
        metadata = m.Metadata(attributes=source)
        tm.that(metadata.attributes, eq=u.normalize_to_json_mapping(source))
        tm.that(source, eq={"active": True})
        first, second = m.Metadata(), m.Metadata()
        tm.that(first.attributes, eq=second.attributes)
        tm.that(first.attributes is second.attributes, eq=False)

    @staticmethod
    def test_registration_uses_native_payload_and_metadata_contracts() -> None:
        """Registration admits native payloads and validates metadata."""
        metadata = m.Metadata(attributes={"source": "registration"})
        scalar = m.ServiceRegistration(
            name="scalar", service="payload", metadata=metadata
        )
        tm.that(scalar.metadata is metadata, eq=True)
        mapping = m.ServiceRegistration.model_validate({
            "name": "mapping",
            "service": {"nested": [None]},
            "metadata": {"tag": "value"},
        })
        if not isinstance(mapping.service, m.ConfigMap):
            pytest.fail("Mapping service must follow the ConfigMap contract")
        if not isinstance(mapping.metadata, m.Metadata):
            pytest.fail("Registration attributes must follow the Metadata contract")
        tm.that(mapping.service.model_dump(mode="json"), eq={"nested": [None]})
        tm.that(mapping.metadata.attributes, eq={"tag": "value"})
        with pytest.raises(m.ValidationError):
            m.ServiceRegistration.model_validate({
                "name": "invalid",
                "service": ModuleType("unsupported"),
            })

    @staticmethod
    def test_registration_guard_preserves_model_and_bound_callable() -> None:
        """Registration values pass the guard boundary without losing identity."""
        model = m.ConfigMap(root={"plugin": "guard"})
        services: tuple[p.RegisterableService, ...] = (model, model.model_dump)
        for service in services:
            value: p.GuardInput = service
            tm.that(u.normalize_registerable_service(value) is service, eq=True)

    @staticmethod
    def test_timestamp_validator_preserves_aware_timezone() -> None:
        """Naive timestamps gain UTC while aware timestamps retain identity."""
        naive = datetime(2025, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        aware = naive.replace(tzinfo=timezone(timedelta(hours=3)))
        normalized = m.TimestampedModel.model_validate({"created_at": naive})
        preserved = m.TimestampedModel.model_validate({"created_at": aware})
        tm.that(normalized.created_at.tzinfo is UTC, eq=True)
        tm.that(preserved.created_at is aware, eq=True)

    @staticmethod
    def test_invalid_pagination_does_not_become_a_default() -> None:
        """Invalid explicit pagination is rejected rather than defaulted."""
        with pytest.raises(m.ValidationError):
            m.Query.model_validate({"pagination": {"page": "invalid"}})

    @staticmethod
    def test_result_and_exception_accept_native_model_mappings() -> None:
        """Results and exceptions admit model-owned mapping payloads."""
        payload = m.ConfigMap(root={"path": Path("error/source"), "nullable": None})
        result = r[int].fail("failure", error_data=payload)
        tm.that(result.error_data, eq=payload.model_dump(mode="json"))
        error = e.BaseError(
            "failure", options=m.ExceptionInitOptions(context=payload, auto_log=False)
        )
        tm.that(error.metadata.attributes, eq={"path": str(Path("error/source"))})

    @staticmethod
    def test_scope_preserves_injected_composition(clean_container: p.Container) -> None:
        """Child scopes inherit composition without leaking local bindings."""
        tm.that(
            clean_container.bind("shared-service", "inherited") is clean_container,
            eq=True,
        )
        scope = clean_container.scope()
        tm.that(tm.ok(scope.resolve("shared-service")), eq="inherited")
        tm.that(scope.bind("scoped-service", "isolated") is scope, eq=True)
        tm.that(clean_container.resolve("scoped-service").failure, eq=True)

    @staticmethod
    @pytest.mark.parametrize("trace", [False, True])
    def test_trace_does_not_change_explicit_threshold(*, trace: bool) -> None:
        """Tracing never overrides an explicitly elected logging threshold."""
        selected = c.LogLevel.WARNING
        tm.that(
            u.resolve_effective_log_level(
                trace=trace, debug=True, log_level=selected, log_level_explicit=True
            ),
            eq=selected,
        )
        tm.that(
            u.resolve_effective_log_level(
                trace=trace, debug=True, log_level=selected, log_level_explicit=False
            ),
            eq=c.LogLevel.DEBUG,
        )
