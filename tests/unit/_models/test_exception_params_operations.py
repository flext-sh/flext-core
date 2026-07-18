"""Behavioral tests for the exception-parameter model public contract.

Exercises only the public surface of the ``m.*ErrorParams`` models: field
values, serialization, roundtrip, ``extra="forbid"`` / strict validation, the
``connection_target`` computed value, and inherited fields. No private
attribute access, no patching, no collaborator spying.
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests import c, p
from tests import m
from tests.unit._models._exception_params_support import (
    _ALL_PARAMS_IDS,
    _ALL_PARAMS_MODELS,
)


class TestsFlextCoreExceptionParamsOperations:
    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    def test_no_arg_construction_yields_all_none_fields(
        self, model_cls: type[p.ParamsModel]
    ) -> None:
        instance = model_cls()
        for value in instance.model_dump().values():
            tm.that(value, none=True)

    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    def test_unknown_field_is_rejected(self, model_cls: type[p.ParamsModel]) -> None:
        with pytest.raises(c.ValidationError):
            model_cls.model_validate({"bogus_field": "nope"})

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [("str", "int"), ("list", "dict"), ("BaseModel", "NoneType")],
        ids=["str-int", "list-dict", "model-none"],
    )
    def test_type_error_params_exposes_expected_and_actual(
        self, expected: str, actual: str
    ) -> None:
        params = m.TypeErrorParams(expected_type=expected, actual_type=actual)
        tm.that(params.expected_type, eq=expected)
        tm.that(params.actual_type, eq=actual)

    def test_operation_error_params_serialize_field_values(self) -> None:
        params = m.OperationErrorParams(operation="save_state", reason="disk_full")
        data = params.model_dump()
        tm.that(data["operation"], eq="save_state")
        tm.that(data["reason"], eq="disk_full")

    def test_attribute_access_error_params_preserve_mapping_context(self) -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="token", attribute_context={"owner": "session"}
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
    def test_connection_target_formats_host_and_port(
        self, host: str | None, port: int | None, expected_target: str
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
    def test_strict_typing_rejects_wrong_type(
        self, model_cls: type[p.ParamsModel], payload: dict[str, object]
    ) -> None:
        with pytest.raises(c.ValidationError):
            model_cls.model_validate(payload)

    def test_assignment_revalidates_field_type(self) -> None:
        params = m.ValidationErrorParams(field="email")
        wrong_value: object = 123
        with pytest.raises(c.ValidationError):
            setattr(params, "field", wrong_value)

    @pytest.mark.parametrize(
        "params",
        [
            m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5),
            m.AuthorizationErrorParams(
                user_id="u-1", resource="docs:secret", permission="read"
            ),
            m.RateLimitErrorParams(limit=1000, window_seconds=3600, retry_after=2.5),
        ],
        ids=["connection", "authorization", "rate-limit"],
    )
    def test_model_dump_roundtrip_preserves_values(self, params: p.ParamsModel) -> None:
        rebuilt = type(params).model_validate(params.model_dump())
        tm.that(rebuilt.model_dump(), eq=params.model_dump())
