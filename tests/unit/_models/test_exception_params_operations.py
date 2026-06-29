"""Exception parameter type, operation, attribute, and cross-cutting tests."""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.unit._models._exception_params_support import (
    _ALL_PARAMS_IDS,
    _ALL_PARAMS_MODELS,
)


class TestsFlextModelsExceptionParamsOperations:
    def test_type_error_params_defaults(self) -> None:
        params = m.TypeErrorParams()
        tm.that(params.expected_type, none=True)
        tm.that(params.actual_type, none=True)

    def test_type_error_params_with_values(self) -> None:
        params = m.TypeErrorParams(expected_type="str", actual_type="int")
        tm.that(params.expected_type, eq="str")
        tm.that(params.actual_type, eq="int")

    @pytest.mark.parametrize(
        ("expected", "actual"),
        [
            ("str", "int"),
            ("list", "dict"),
            ("BaseModel", "NoneType"),
        ],
        ids=["str-int", "list-dict", "model-none"],
    )
    def test_type_error_params_type_pairs(self, expected: str, actual: str) -> None:
        params = m.TypeErrorParams(expected_type=expected, actual_type=actual)
        tm.that(params.expected_type, eq=expected)
        tm.that(params.actual_type, eq=actual)

    def test_operation_error_params_defaults(self) -> None:
        params = m.OperationErrorParams()
        tm.that(params.operation, none=True)
        tm.that(params.reason, none=True)

    def test_operation_error_params_with_values(self) -> None:
        params = m.OperationErrorParams(
            operation="publish_events", reason="transient_backend_error"
        )
        tm.that(params.operation, eq="publish_events")
        tm.that(params.reason, eq="transient_backend_error")

    def test_operation_error_params_serialization(self) -> None:
        params = m.OperationErrorParams(operation="save_state", reason="disk_full")
        data = params.model_dump()
        tm.that(data["operation"], eq="save_state")
        tm.that(data["reason"], eq="disk_full")

    def test_attribute_access_error_params_defaults(self) -> None:
        params = m.AttributeAccessErrorParams()
        tm.that(params.attribute_name, none=True)
        tm.that(params.attribute_context, none=True)

    def test_attribute_access_error_params_with_values(self) -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="token", attribute_context={"owner": "session"}
        )
        tm.that(params.attribute_name, eq="token")
        tm.that(params.attribute_context, eq={"owner": "session"})

    def test_attribute_access_error_params_serialization(self) -> None:
        params = m.AttributeAccessErrorParams(
            attribute_name="settings", attribute_context="runtime"
        )
        data = params.model_dump()
        tm.that(data["attribute_name"], eq="settings")
        tm.that(data["attribute_context"], eq="runtime")

    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    def test_all_params_reject_extra_fields(
        self,
        model_cls: type[m.ParamsModel],
    ) -> None:
        with pytest.raises(c.ValidationError):
            model_cls.model_validate({"bogus_field": "nope"})

    @pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_IDS)
    def test_all_params_instantiate_with_no_args(self, model_cls: type) -> None:
        instance = model_cls()
        data = instance.model_dump()
        for value in data.values():
            tm.that(value, none=True)

    def test_validate_assignment_enforced(self) -> None:
        """Assigning a wrong type to a strict str field raises."""
        params = m.ValidationErrorParams(field="email")
        with pytest.raises(c.ValidationError):
            setattr(params, "field", 123)

    def test_roundtrip_connection_error_params(self) -> None:
        original = m.ConnectionErrorParams(host="db.internal", port=5432, timeout=5)
        rebuilt = m.ConnectionErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.host, eq=original.host)
        tm.that(rebuilt.port, eq=original.port)
        tm.that(rebuilt.timeout, eq=original.timeout)

    def test_roundtrip_authorization_error_params(self) -> None:
        original = m.AuthorizationErrorParams(
            user_id="u-1", resource="docs:secret", permission="read"
        )
        rebuilt = m.AuthorizationErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.user_id, eq=original.user_id)
        tm.that(rebuilt.resource, eq=original.resource)
        tm.that(rebuilt.permission, eq=original.permission)

    def test_roundtrip_rate_limit_error_params(self) -> None:
        original = m.RateLimitErrorParams(
            limit=1000, window_seconds=3600, retry_after=2.5
        )
        rebuilt = m.RateLimitErrorParams.model_validate(original.model_dump())
        tm.that(rebuilt.limit, eq=original.limit)
        tm.that(rebuilt.window_seconds, eq=original.window_seconds)
        tm.that(rebuilt.retry_after, eq=original.retry_after)
