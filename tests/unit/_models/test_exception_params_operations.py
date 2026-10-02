"""Behavioral tests for the exception-parameter model public contract.

Exercises only the public surface of the ``m.*ErrorParams`` models: field
values, serialization, roundtrip, ``extra="forbid"`` / strict validation, the
``connection_target`` computed value, and inherited fields. No private
attribute access, no patching, no collaborator spying.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.unit._models._exception_params_support import (
    _ALL_PARAMS_IDS,
    _ALL_PARAMS_MODELS,
)


class TestsFlextCoreExceptionParamsOperations:
    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    @staticmethod
    def test_no_arg_construction_yields_all_none_fields(
        model_cls: type[m.ParamsModel],
    ) -> None:
        instance = model_cls()
        for value in instance.model_dump().values():
            tm.that(value, none=True)

    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    @staticmethod
    def test_unknown_field_is_rejected(model_cls: type[m.ParamsModel]) -> None:
        with pytest.raises(c.ValidationError):
            model_cls.model_validate({"bogus_field": "nope"})

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [("str", "int"), ("list", "dict"), ("BaseModel", "NoneType")],
        ids=["str-int", "list-dict", "model-none"],
    )
    @staticmethod
    def test_type_error_params_exposes_expected_and_actual(
        expected: str,
        actual: str,
    ) -> None:
        params = m.TypeErrorParams(expected_type=expected, actual_type=actual)
        tm.that(params.expected_type, eq=expected)
        tm.that(params.actual_type, eq=actual)

    @staticmethod
    def test_operation_error_params_serialize_field_values() -> None:
        params = m.OperationErrorParams(operation="save_state", reason="disk_full")
        data = params.model_dump()
        tm.that(data["operation"], eq="save_state")
        tm.that(data["reason"], eq="disk_full")

    @staticmethod
    def test_attribute_access_error_params_preserve_mapping_context() -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="token",
            attribute_context={"owner": "session"},
        )
        tm.that(params.attribute_name, eq="token")
        tm.that(params.attribute_context, eq={"owner": "session"})

    @pytest.mark.parametrize(
        ("host", "port", "expected_target"),
        [
            ("db.internal", 5432, "db.internal:5432"),
            ("db.internal", None, "db.internal"),
            (None, None, "unknown"),
        ],
        ids=["host-port", "host-only", "neither"],
    )
    @staticmethod
    def test_connection_target_formats_host_and_port(
        host: str | None,
        port: int | None,
        expected_target: str,
    ) -> None:
        params = m.ConnectionErrorParams(host=host, port=port)
        tm.that(params.connection_target, eq=expected_target)

    @pytest.mark.parametrize(
        ("model_cls", "payload"),
        [
            (m.ValidationErrorParams, {"field": 123}),
            (m.ConnectionErrorParams, {"host": "h", "port": "5432"}),
            (m.TypeErrorParams, {"expected_type": 1}),
        ],
        ids=["field-int", "port-str", "expected-type-int"],
    )
    @staticmethod
    def test_strict_typing_rejects_wrong_type(
        model_cls: type[m.ParamsModel],
        payload: dict[str, object],
    ) -> None:
        with pytest.raises(c.ValidationError):
            model_cls.model_validate(payload)

    @staticmethod
    def test_assignment_revalidates_field_type() -> None:
        params = m.ValidationErrorParams(field="email")
        wrong_value: object = 123
        tm.rejects_assignment(params, "field", wrong_value, expected=c.ValidationError)

    @pytest.mark.parametrize(
        "params",
        [
            m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5),
            m.AuthorizationErrorParams(
                user_id="u-1",
                resource="docs:secret",
                permission="read",
            ),
            m.RateLimitErrorParams(limit=1000, window_seconds=3600, retry_after=2.5),
        ],
        ids=["connection", "authorization", "rate-limit"],
    )
    @staticmethod
    def test_model_dump_roundtrip_preserves_values(params: m.ParamsModel) -> None:
        rebuilt = type(params).model_validate(params.model_dump())
        tm.that(rebuilt.model_dump(), eq=params.model_dump())
