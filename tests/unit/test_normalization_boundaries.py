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
        tm.ok(u.normalize_to_json_value(None) is None)
        tm.ok(u.normalize_to_container(None) is None)
        tm.that(u.normalize_to_container([None]), eq=[None])

    @staticmethod
    def test_json_serialization_rejects_unknown_input() -> None:
        """Unsupported native values fail at JSON admission."""
        with pytest.raises(ValueError, match="Unable to serialize unknown type"):
            u.normalize_to_json_value(type)

    @staticmethod
    def test_json_payload_preserves_mapping_inside_nested_sequences() -> None:
        """Mapping payloads must not materialize the abstract model union arm."""
        items: list[t.MetadataMapping] = [{"locations": []}]
        parsed = m.ConfigMap.model_validate({"issues": items})
        issues = parsed["issues"]
        if not isinstance(issues, list):
            pytest.fail("Issue sequence did not retain its native collection shape")
        tm.ok(isinstance(issues[0], Mapping))

    @staticmethod
    @pytest.mark.parametrize("depth", [1, 3])
    def test_json_payload_accepts_native_json_at_recursive_collection_depths(
        depth: int,
    ) -> None:
        """JsonValue owns nested mappings and sequences without a recursive alias."""
        payload: t.JsonValue = {"source": "recursive"}
        for _ in range(depth):
            payload = {"nested": [payload]}
        parsed = m.ConfigMap.model_validate({"payload": payload})
        tm.that(u.normalize_to_json_mapping(parsed), eq={"payload": payload})

    @staticmethod
    @pytest.mark.parametrize("nested", [False, True])
    def test_json_payload_rejects_non_string_mapping_keys(*, nested: bool) -> None:
        """Validation rejects invalid keys before native JSON can stringify them."""
        invalid = {1: "not a string key"}
        with pytest.raises(m.ValidationError):
            m.ConfigMap.model_validate({"issues": [invalid]} if nested else invalid)

    @staticmethod
    @pytest.mark.parametrize("payload", [{1: "invalid-key"}, {"nested": {1: None}}])
    def test_json_normalization_rejects_invalid_mapping_keys(
        payload: p.GuardInput,
    ) -> None:
        """Both JSON boundaries reject keys before they can be stringified."""
        with pytest.raises(m.ValidationError):
            u.normalize_to_json_value(payload)
        with pytest.raises(m.ValidationError):
            u.normalize_to_json_mapping(payload)

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
        tm.ok(u.normalize_to_container(model) is model)
        serialized = m.Tests.PlainSerializedPayload(name="before")
        normalized = u.normalize_to_container(serialized)
        tm.ok(normalized is serialized)
        serialized.name = "after"
        tm.that(u.normalize_to_json_value(normalized), eq="after")
        tm.that(
            u.normalize_to_container(m.ConfigMap(root={"key": str(Path("relative"))})),
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
        tm.fail(first.attributes is second.attributes)

    @staticmethod
    def test_registration_uses_native_payload_and_metadata_contracts() -> None:
        """Registration admits native payloads and validates metadata."""
        metadata = m.Metadata(attributes={"source": "registration"})
        scalar = m.ServiceRegistration(
            name="scalar", service="payload", metadata=metadata
        )
        tm.ok(scalar.metadata is metadata)
        mapping = m.ServiceRegistration.model_validate({
            "name": "mapping",
            "service": {"nested": [None]},
            "metadata": {"tag": "value"},
        })
        tm.that(mapping.service, eq={"nested": [None]})
        tm.that(
            mapping.metadata,
            is_=m.Metadata,
            attr_eq={"attributes": {"tag": "value"}},
        )
        with pytest.raises(m.ValidationError):
            m.ServiceRegistration.model_validate({
                "name": "invalid",
                "service": ModuleType("unsupported"),
            })

    @staticmethod
    def test_registration_preserves_model_and_bound_callable() -> None:
        """The registration model retains validated service references."""
        model = m.ConfigMap(root={"plugin": "guard"})
        services: tuple[p.RegisterableService, ...] = (model, model.model_dump)
        for service in services:
            registration = m.ServiceRegistration(name="identity", service=service)
            tm.that(registration.service is service, truth=True)

    @staticmethod
    def test_factory_preserves_native_collection_identity(
        clean_container: p.Container,
    ) -> None:
        """Factory results retain their native payload and reference on resolve."""
        payload: t.JsonMapping = {"nested": [None]}

        def produce() -> t.JsonMapping:
            return payload

        clean_container.factory("identity", produce)
        tm.that(
            tm.ok(clean_container.resolve("identity")),
            where=lambda value: value is payload,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "payload",
        [
            {1: "invalid-key"},
            {"nested": [ModuleType("unsupported")]},
            [ModuleType("unsupported")],
        ],
    )
    @pytest.mark.parametrize("registration_kind", ["factories", "resources"])
    def test_factory_and_resource_reject_invalid_payloads(
        clean_container: p.Container,
        payload: p.GuardInput,
        registration_kind: str,
    ) -> None:
        """Untrusted callable outputs cross the canonical admission model."""
        spec = m.ServiceRegistrationSpec.model_validate({
            registration_kind: {"invalid-payload": lambda: payload},
        })
        scope = clean_container.scope(registration=spec)
        tm.fail(scope.resolve("invalid-payload"))


class TestsFlextCoreRuntimeNormalization:
    """Timestamp, pagination, composition, and runtime policy boundaries."""

    @staticmethod
    def test_timestamp_validator_preserves_aware_timezone() -> None:
        """Naive timestamps gain UTC while aware timestamps retain identity."""
        naive = datetime(2025, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        aware = naive.replace(tzinfo=timezone(timedelta(hours=3)))
        normalized = m.TimestampedModel.model_validate({"created_at": naive})
        preserved = m.TimestampedModel.model_validate({"created_at": aware})
        tm.ok(normalized.created_at.tzinfo is UTC)
        tm.ok(preserved.created_at is aware)

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
        tm.ok(scope.bind("scoped-service", "isolated") is scope)
        tm.ok(clean_container.resolve("scoped-service").failure)

    @staticmethod
    def test_debug_does_not_change_explicit_threshold() -> None:
        """Debug never overrides an explicitly elected logging threshold."""
        selected = c.LogLevel.WARNING
        tm.that(
            u.resolve_effective_log_level(
                debug=True, log_level=selected, log_level_explicit=True
            ),
            eq=selected,
        )
        tm.that(
            u.resolve_effective_log_level(
                debug=True, log_level=selected, log_level_explicit=False
            ),
            eq=c.LogLevel.DEBUG,
        )
