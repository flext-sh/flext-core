"""Behavioral integration tests for the ``s`` service contract.

Exercises the OBSERVABLE public surface of the test service implementations:
``r[T]`` outcomes (success/failure, value, error), public model state, public
properties, dependency-injection resolve round-trips, and ``r[T]`` combinators.
No private attributes, no monkeypatching, no internal-collaborator spying.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


import pytest
from flext_tests import tm

from tests import u

from .service_lifecycle_cases import (
    TestsFlextFlextServiceLifecycleCases as _ServiceLifecycleCases,
)

from tests import p

_FETCH_CALL_COUNT = 2


@pytest.mark.integration
@pytest.mark.usefixtures("clean_container")
class TestsFlextCoreService(_ServiceLifecycleCases):
    """Behavioral contract of the user/notification/lifecycle services.

    Inherits the lifecycle behavioral cases and the service fixture accessors
    (``UserQueryService``, ``NotificationService``, ``UserServiceEntity`` ...).
    """

    # ------------------------------------------------------------------ #
    # UserQueryService.execute — availability contract
    # ------------------------------------------------------------------ #
    def test_user_service_execute_reports_available(self) -> None:
        """execute() succeeds with True when the service is available."""
        service = self.UserQueryService()
        result = service.execute()
        tm.that(u.Tests.assert_success(result, expected_value=True), eq=True)

    def test_user_service_execute_reports_unavailable_in_failure_mode(self) -> None:
        """execute() fails with the unavailable error under failure mode."""
        service = self.UserQueryService()
        service.configure_failure_mode(should_fail=True)
        result = service.execute()
        tm.that(result.failure, eq=True)
        _ = u.Tests.assert_failure(result, "User service unavailable")

    # ------------------------------------------------------------------ #
    # UserQueryService.fetch_user — value derivation contract
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "user_id",
        ["test_user_123", "abc", "user-42"],
    )
    def test_fetch_user_derives_default_entity(
        self,
        user_id: str,
    ) -> None:
        """fetch_user() derives a default entity from the requested id."""
        service = self.UserQueryService()
        entity = u.Tests.assert_success(service.fetch_user(user_id))
        tm.that(entity.unique_id, eq=user_id)
        tm.that(entity.name, eq=f"User {user_id}")
        tm.that(entity.email, eq=f"user{user_id}@example.com")
        tm.that(entity.active, eq=True)

    def test_fetch_user_returns_applied_custom_entity(self) -> None:
        """fetch_user() returns previously applied custom user data verbatim."""
        service = self.UserQueryService()
        user_id = "custom_user"
        custom = self.UserServiceEntity(
            unique_id=user_id,
            name="Custom User",
            email="custom@example.com",
            active=True,
        )
        service.apply_user_data(user_id, custom)
        entity = u.Tests.assert_success(service.fetch_user(user_id))
        tm.that(entity.model_dump(), eq=custom.model_dump())

    def test_fetch_user_fails_in_failure_mode(self) -> None:
        """fetch_user() surfaces a failure result under failure mode."""
        service = self.UserQueryService()
        service.configure_failure_mode(should_fail=True)
        result = service.fetch_user("anyone")
        tm.that(result.failure, eq=True)
        _ = u.Tests.assert_failure(result, "User service unavailable")

    def test_fetch_user_counts_each_call(self) -> None:
        """call_count reflects the number of fetch_user invocations."""
        service = self.UserQueryService()
        tm.that(service.call_count, eq=0)
        _ = service.fetch_user("a")
        _ = service.fetch_user("b")
        tm.that(service.call_count, eq=_FETCH_CALL_COUNT)

    def test_fetch_user_result_supports_combinators(self) -> None:
        """The r[T] returned by fetch_user chains via map/flat_map."""
        service = self.UserQueryService()
        email = (
            service
            .fetch_user("chained")
            .map(lambda entity: entity.email)
            .unwrap_or("missing")
        )
        tm.that(email, eq="userchained@example.com")

    # ------------------------------------------------------------------ #
    # NotificationService — send/execute contract
    # ------------------------------------------------------------------ #
    def test_notification_execute_reports_sent(self) -> None:
        """execute() succeeds with the 'sent' status."""
        service = self.NotificationService()
        result = service.execute()
        tm.that(u.Tests.assert_success(result, expected_value="sent"), eq="sent")

    def test_notification_send_records_recipient(self) -> None:
        """send() succeeds and records the recipient in the public log."""
        service = self.NotificationService()
        email = "test@example.com"
        result = service.send(email)
        tm.that(u.Tests.assert_success(result, expected_value="sent"), eq="sent")
        tm.that(service.sent_notifications, has=email)
        tm.that(service.call_count, eq=1)

    def test_notification_send_fails_in_failure_mode(self) -> None:
        """send() fails and does not record the recipient under failure mode."""
        service = self.NotificationService()
        service.configure_failure_mode(should_fail=True)
        result = service.send("test@example.com")
        tm.that(result.failure, eq=True)
        _ = u.Tests.assert_failure(result, "Notification service unavailable")
        tm.that(service.sent_notifications, lacks="test@example.com")

    # ------------------------------------------------------------------ #
    # Dependency injection — resolve round-trip contract
    # ------------------------------------------------------------------ #
    def test_container_resolves_bound_services_functionally(
        self,
        clean_container: p.Container,
    ) -> None:
        """Bound services resolve back and remain fully functional."""
        user_service = self.UserQueryService()
        notification_service = self.NotificationService()
        user_id = "test_user_123"
        user_service.apply_user_data(
            user_id,
            self.UserServiceEntity(
                unique_id=user_id,
                name=f"User {user_id}",
                email=f"user{user_id}@example.com",
                active=True,
            ),
        )
        _ = clean_container.bind("user_service", user_service)
        _ = clean_container.bind("notification_service", notification_service)

        resolved_user = u.Tests.assert_success(
            clean_container.resolve("user_service", type_cls=self.UserQueryService),
        )
        resolved_notification = u.Tests.assert_success(
            clean_container.resolve(
                "notification_service",
                type_cls=self.NotificationService,
            ),
        )

        entity = u.Tests.assert_success(resolved_user.fetch_user(user_id))
        _ = u.Tests.assert_success(
            resolved_notification.send(entity.email),
            expected_value="sent",
        )
        tm.that(resolved_notification.sent_notifications, has=entity.email)

    # ------------------------------------------------------------------ #
    # External-service integration — boundary contract
    # ------------------------------------------------------------------ #
    def test_external_service_processes_user_email(
        self,
        mock_external_service: u.Tests.FunctionalExternalService,
    ) -> None:
        """A fetched user email flows through the external service boundary."""
        service = self.UserQueryService()
        entity = u.Tests.assert_success(service.fetch_user("test_user"))
        processed = u.Tests.assert_success(
            mock_external_service.process(entity.email),
        )
        tm.that(processed, eq=f"processed_{entity.email}")
        tm.that(mock_external_service.processed_items, has=processed)
        tm.that(mock_external_service.get_call_count(), eq=1)
