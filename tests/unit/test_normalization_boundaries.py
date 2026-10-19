"""Public native-serialization, metadata and composition boundary contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from tests import c, e, m, r, tm, u

if TYPE_CHECKING:
    from tests import p, t


class TestsNormalizationBoundaries:
    """Validate domain boundaries rather than preserving a retired facade."""

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [None, 42, "text", b"text", Path("relative/path"), [1, None], {"key": None}],
    )
    def test_json_serialization_matches_native_sdk(
        value: t.Scalar | Path | t.JsonValue,
    ) -> None:
        expected = u.json_value_adapter().validate_python(
            u.to_jsonable_python(value), strict=True
        )
        tm.that(u.normalize_to_json_value(value), eq=expected)
        tm.that(u.normalize_to_metadata(value), eq=expected)

    @staticmethod
    def test_json_null_and_container_blank_are_distinct() -> None:
        assert u.normalize_to_json_value(None) is None
        assert u.normalize_to_metadata(None) is None
        tm.that(u.normalize_to_container(None), eq="")
        tm.that(u.normalize_to_container([None]), eq=[None])

    @staticmethod
    def test_json_serialization_rejects_unknown_input() -> None:
        with pytest.raises(ValueError):
            u.normalize_to_json_value(type)
        with pytest.raises(ValueError):
            u.normalize_to_metadata(type)

    @staticmethod
    def test_container_preserves_models_and_unwraps_root_payloads() -> None:
        model = m.Metadata(attributes={"source": "native"})
        assert u.normalize_to_container(model) is model
        tm.that(
            u.normalize_to_container(m.ConfigMap(root={"key": Path("relative")})),
            eq={"key": str(Path("relative"))},
        )

    @staticmethod
    def test_metadata_field_owns_attribute_validation() -> None:
        path = Path("metadata/source")
        metadata = m.Metadata.model_validate({
            "attributes": {"path": path, "absent": None, "nested": [None]}
        })
        tm.that(metadata.attributes, eq={"path": str(path), "nested": [None]})
        with pytest.raises(m.ValidationError, match="_private"):
            m.Metadata.model_validate({"attributes": {"_private": 1}})

    @staticmethod
    def test_registration_uses_native_payload_and_metadata_contracts() -> None:
        metadata = m.Metadata(attributes={"source": "registration"})
        scalar = m.ServiceRegistration(
            name="scalar", service="payload", metadata=metadata
        )
        assert scalar.metadata is metadata
        mapping = m.ServiceRegistration.model_validate({
            "name": "mapping",
            "service": {"nested": [None]},
            "metadata": {"tag": "value"},
        })
        assert isinstance(mapping.service, m.ConfigMap)
        assert isinstance(mapping.metadata, m.Metadata)
        tm.that(mapping.service.model_dump(mode="json"), eq={"nested": [None]})
        tm.that(mapping.metadata.attributes, eq={"tag": "value"})
        with pytest.raises(m.ValidationError):
            m.ServiceRegistration.model_validate({
                "name": "invalid",
                "service": ModuleType("unsupported"),
            })

    @staticmethod
    def test_timestamp_validator_preserves_aware_timezone() -> None:
        naive = datetime(2025, 1, 1)
        aware = naive.replace(tzinfo=timezone(timedelta(hours=3)))
        normalized = m.TimestampedModel.model_validate({"created_at": naive})
        preserved = m.TimestampedModel.model_validate({"created_at": aware})
        assert normalized.created_at.tzinfo is UTC
        assert preserved.created_at is aware

    @staticmethod
    def test_invalid_pagination_does_not_become_a_default() -> None:
        with pytest.raises(m.ValidationError):
            m.Query.model_validate({"pagination": {"page": "invalid"}})

    @staticmethod
    def test_result_and_exception_accept_native_model_mappings() -> None:
        payload = m.ConfigMap(root={"path": Path("error/source"), "nullable": None})
        result = r[int].fail("failure", error_data=payload)
        tm.that(result.error_data, eq=payload.model_dump(mode="json"))
        error = e.BaseError(
            "failure", options=m.ExceptionInitOptions(context=payload, auto_log=False)
        )
        tm.that(error.metadata.attributes, eq={"path": str(Path("error/source"))})

    @staticmethod
    def test_scope_preserves_injected_composition(clean_container: p.Container) -> None:
        tm.ok(clean_container.bind("shared-service", "inherited"))
        scope = clean_container.scope()
        tm.that(tm.ok(scope.resolve("shared-service")), eq="inherited")
        tm.ok(scope.bind("scoped-service", "isolated"))
        assert clean_container.resolve("scoped-service").is_failure

    @staticmethod
    @pytest.mark.parametrize("trace", [False, True])
    def test_trace_does_not_change_explicit_threshold(*, trace: bool) -> None:
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
