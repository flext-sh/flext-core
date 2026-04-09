"""Test all patterns documented in FLEXT_SERVICE_ARCHITECTURE.md.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import override

import pytest

from tests import e, p, r, s, t, u


class TestDocumentedPatterns:
    """Unified documented-pattern tests with one top-level class."""

    @pytest.mark.parametrize("case", u.Core.Tests.success_cases())
    def test_v1_explicit_success(self, case: tuple[str, str]) -> None:
        user_id, description = case
        service_case = self.ServiceTestCase(user_id=user_id, description=description)
        service = service_case.create_user_service()
        result = service.execute()
        _ = u.Core.Tests.assert_success(result)
        user = result.value
        assert isinstance(user, self.User)
        assert user.unique_id == service_case.user_id
        assert user.name == f"User {service_case.user_id}"

    @pytest.mark.parametrize("case", u.Core.Tests.failure_cases())
    def test_v1_explicit_failure(self, case: tuple[str, str, str]) -> None:
        user_id, expected_error, description = case
        service_case = self.ServiceTestCase(
            user_id=user_id,
            expected_success=False,
            expected_error=expected_error,
            description=description,
        )
        service = service_case.create_user_service()
        result = service.execute()
        _ = u.Core.Tests.assert_failure(result)
        error_msg = result.error
        assert error_msg is not None
        expected = service_case.expected_error
        assert expected is not None
        assert expected in error_msg.lower()

    @pytest.mark.parametrize("case", u.Core.Tests.success_cases())
    def test_v1_explicit_with_if_check(self, case: tuple[str, str]) -> None:
        user_id, description = case
        service_case = self.ServiceTestCase(user_id=user_id, description=description)
        result = service_case.create_user_service().execute()
        if result.is_success:
            user = result.value
            assert isinstance(user, self.User)
            assert user.unique_id == service_case.user_id
        else:
            pytest.fail("Should succeed")

    @pytest.mark.parametrize("case", u.Core.Tests.success_cases())
    def test_v2_property_success(self, case: tuple[str, str]) -> None:
        user_id, description = case
        service_case = self.ServiceTestCase(user_id=user_id, description=description)
        result_value = service_case.create_user_service().result
        assert isinstance(result_value, self.User)
        user = result_value
        assert user.unique_id == service_case.user_id
        assert user.name == f"User {service_case.user_id}"

    @pytest.mark.parametrize("case", u.Core.Tests.failure_cases())
    def test_v2_property_failure_raises(self, case: tuple[str, str, str]) -> None:
        user_id, expected_error, description = case
        service_case = self.ServiceTestCase(
            user_id=user_id,
            expected_success=False,
            expected_error=expected_error,
            description=description,
        )
        with pytest.raises(e.BaseError) as exc_info:
            service_case.create_user_service().result
        error_str = str(exc_info.value).lower()
        assert service_case.expected_error is not None
        assert service_case.expected_error in error_str

    @pytest.mark.parametrize("case", u.Core.Tests.success_cases())
    def test_v2_property_execute_still_available(self, case: tuple[str, str]) -> None:
        user_id, description = case
        service_case = self.ServiceTestCase(user_id=user_id, description=description)
        result = service_case.create_user_service().execute()
        _ = u.Core.Tests.assert_success(result)
        user = result.value
        assert isinstance(user, self.User)
        assert user.unique_id == service_case.user_id

    @pytest.mark.parametrize("case", u.Core.Tests.railway_success_cases())
    def test_v1_railway_complex_pipeline(
        self,
        case: tuple[t.StrSequence, t.StrSequence, int, str],
    ) -> None:
        user_ids, operations, expected_pipeline_length, description = case
        railway_case = self.RailwayTestCase(
            user_ids=user_ids,
            operations=operations,
            expected_pipeline_length=expected_pipeline_length,
            description=description,
        )
        result = railway_case.execute_v1_pipeline()
        _ = u.Core.Tests.assert_success(result)
        if "get_status" in railway_case.operations:
            assert result.value == "sent"
        elif "get_email" in railway_case.operations:
            unwrapped = result.value
            email: str = str(unwrapped) if not isinstance(unwrapped, str) else unwrapped
            assert isinstance(email, str)
            assert "@" in email
        else:
            assert isinstance(result.value, self.User)

    @pytest.mark.parametrize("case", u.Core.Tests.railway_success_cases())
    def test_v2_property_can_use_execute_for_railway(
        self,
        case: tuple[t.StrSequence, t.StrSequence, int, str],
    ) -> None:
        _ = case
        user_result_raw = self.make(self.GetUserService, user_id="123").result
        assert isinstance(user_result_raw, self.User)
        user_result = user_result_raw
        assert user_result.unique_id == "123"
        result = (
            self
            .make(self.GetUserService, user_id="123")
            .execute()
            .map(lambda u: u.email)
        )
        _ = u.Core.Tests.assert_success(result)
        assert result.value == "user123@example.com"

    @pytest.mark.parametrize("case", u.Core.Tests.railway_success_cases())
    def test_v2_property_railway_chaining(
        self,
        case: tuple[t.StrSequence, t.StrSequence, int, str],
    ) -> None:
        _ = case
        pipeline = (
            self
            .make(self.GetUserService, user_id="456")
            .execute()
            .flat_map(
                lambda user: self.make(
                    self.SendEmailService,
                    to=user.email,
                    subject="Hello",
                ).execute(),
            )
            .map(lambda response: response.message_id)
        )
        assert pipeline.is_success
        message_id: str = str(pipeline.value)
        assert message_id.startswith("msg-")

    def test_monadic_map(self) -> None:
        result = (
            self
            .make(self.GetUserService, user_id="123")
            .execute()
            .map(lambda user: user.name.upper())
        )
        assert result.value == "USER 123"

    def test_monadic_flat_map(self) -> None:
        pipeline = (
            self
            .make(self.GetUserService, user_id="123")
            .execute()
            .flat_map(lambda user: r[str].ok(user.email))
            .flat_map(
                lambda email: self.make(
                    self.SendEmailService,
                    to=email,
                    subject="Test",
                ).execute(),
            )
        )
        assert pipeline.is_success

    def test_monadic_filter(self) -> None:
        result = (
            self
            .make(self.ValidationService, value=50)
            .execute()
            .filter(self.value_lt_100)
        )
        _ = u.Core.Tests.assert_success(result)

    def test_monadic_complex_pipeline(self) -> None:
        pipeline = (
            self
            .make(self.GetUserService, user_id="123")
            .execute()
            .map(lambda user: user.email)
            .filter(lambda email: "@" in email)
            .flat_map(
                lambda email: self.make(
                    self.SendEmailService,
                    to=email,
                    subject="Test",
                ).execute(),
            )
            .map(lambda response: response.status)
        )
        assert pipeline.is_success
        assert pipeline.value == "sent"

    def test_error_handling_try_except_v2_property(self) -> None:
        try:
            user_result_raw = self.make(self.GetUserService, user_id="123").result
            assert isinstance(user_result_raw, self.User)
            user_result = user_result_raw
            assert user_result.unique_id == "123"
        except e.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_try_except_v2_property_failure(self) -> None:
        with pytest.raises(e.BaseError) as exc_info:
            self.make(self.GetUserService, user_id="invalid").result
        assert "not found" in str(exc_info.value).lower()

    def test_error_handling_graceful_degradation(self) -> None:
        try:
            user_result_raw = self.make(self.GetUserService, user_id="123").result
            assert isinstance(user_result_raw, self.User)
            user_result = user_result_raw
            email = user_result.email
        except e.BaseError:
            email = "fallback@example.com"
        assert email == "user123@example.com"

    def test_infrastructure_config_automatic(self) -> None:
        service = self.make(self.GetUserService, user_id="123")
        assert service.config is not None
        assert isinstance(service.config, p.Settings)

    def test_infrastructure_logger_automatic(self) -> None:
        service = self.make(self.GetUserService, user_id="123")
        assert service.logger is not None
        assert isinstance(service.logger, p.Logger)

    def test_infrastructure_container_automatic(self) -> None:
        service = self.make(self.GetUserService, user_id="123")
        assert service.container is not None
        assert isinstance(service.container, p.Container)

    def test_infrastructure_lazy_initialization(self) -> None:
        service = self.make(self.GetUserService, user_id="123")
        config1 = service.config
        config2 = service.config
        assert config1 is config2

    @pytest.mark.parametrize(
        ("operation", "value", "expected"),
        u.Core.Tests.multi_operation_cases(),
    )
    def test_multiple_operations(
        self,
        operation: str,
        value: int,
        expected: t.ConfigMap,
    ) -> None:
        result: t.ConfigMap = self.make(
            self.MultiOperationService,
            operation=operation,
            value=value,
        ).result
        assert result["operation"] == expected["operation"]
        assert result["result"] == expected["result"]

    def test_multiple_operations_invalid(self) -> None:
        with pytest.raises(e.BaseError) as exc_info:
            self.make(self.MultiOperationService, operation="invalid", value=5).result
        assert "Unknown operation" in str(exc_info.value)

    def test_multiple_operations_with_railway(self) -> None:
        pipeline = (
            self
            .make(self.MultiOperationService, operation="double", value=5)
            .execute()
            .map(operator.itemgetter("result"))
            .flat_map(
                lambda result: self.make(
                    self.MultiOperationService,
                    operation="square",
                    value=result,
                ).execute(),
            )
            .map(operator.itemgetter("result"))
        )
        assert pipeline.is_success
        assert pipeline.value == 100

    def test_v1_v2_property_interoperability(self) -> None:
        v1_result = self.make(self.GetUserService, user_id="123").execute()
        assert v1_result.is_success
        v2_user_raw = self.make(self.GetUserService, user_id="456").result
        assert isinstance(v2_user_raw, self.User)
        v2_user_result = v2_user_raw
        assert v2_user_result.unique_id == "456"
        assert isinstance(v1_result.value, self.User)
        assert isinstance(v2_user_result, self.User)

    def test_railway_pattern_works_in_all_versions(self) -> None:
        v1_pipeline = (
            self
            .make(self.GetUserService, user_id="123")
            .execute()
            .map(lambda u: u.email)
        )
        assert v1_pipeline.is_success
        v2_pipeline = (
            self
            .make(self.GetUserService, user_id="456")
            .execute()
            .map(lambda u: u.email)
        )
        assert v2_pipeline.is_success

        class CustomService(s[TestDocumentedPatterns.User]):
            user_id: str = ""

            @override
            def execute(self) -> r[TestDocumentedPatterns.User]:
                return r[TestDocumentedPatterns.User].ok(
                    TestDocumentedPatterns.User(
                        unique_id=self.user_id,
                        name="Test",
                        email="test@example.com",
                    ),
                )

        custom_pipeline = (
            self.make(CustomService, user_id="789").execute().map(lambda u: u.email)
        )
        assert custom_pipeline.is_success

    def test_complete_real_world_scenario(self) -> None:
        user_raw = self.make(self.GetUserService, user_id="123").result
        assert isinstance(user_raw, self.User)
        user = user_raw
        email_result = (
            self
            .make(self.SendEmailService, to=user.email, subject="Welcome")
            .execute()
            .filter(lambda response: response.status == "sent")
            .map(lambda response: response.message_id)
        )
        assert email_result.is_success
        message_id: str = str(email_result.value)
        assert message_id.startswith("msg-")
        calc_result: t.ConfigMap = self.make(
            self.MultiOperationService,
            operation="double",
            value=10,
        ).result
        assert calc_result["result"] == 20
