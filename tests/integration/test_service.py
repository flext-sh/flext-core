"""Integration tests for s implementations.

Tests real s implementations with proper dependency injection,
service composition, and lifecycle management patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests.protocols import p
from tests.utilities import u

from .service_lifecycle_cases import FlextServiceLifecycleCases


class TestsFlextServiceIntegration(FlextServiceLifecycleCases):
    pytestmark = [pytest.mark.integration]

    @pytest.mark.integration
    def test_user_service_execution(self, clean_container: p.Container) -> None:
        """Test user service execution with s.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        result = user_service.execute()
        _ = u.Tests.assert_success(result)
        assert result.value is True

    @pytest.mark.integration
    def test_user_service_fetch_user(self, clean_container: p.Container) -> None:
        """Test user service fetch_user method.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        user_id = "test_user_123"
        result = user_service.fetch_user(user_id)
        _ = u.Tests.assert_success(result)
        assert result.value is not None
        assert result.value.unique_id == user_id
        assert result.value.name == f"User {user_id}"
        assert result.value.email == f"user{user_id}@example.com"
        assert user_service.call_count == 1

    @pytest.mark.integration
    def test_user_service_with_custom_data(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test user service with custom user data.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        user_id = "custom_user"
        custom_user = self.UserServiceEntity(
            unique_id=user_id,
            name="Custom User",
            email="custom@example.com",
            active=True,
        )
        user_service.apply_user_data(user_id, custom_user)
        result = user_service.fetch_user(user_id)
        _ = u.Tests.assert_success(result)
        assert result.value is not None
        assert result.value.unique_id == user_id
        assert result.value.name == "Custom User"
        assert result.value.email == "custom@example.com"

    @pytest.mark.integration
    def test_user_service_failure_mode(self, clean_container: p.Container) -> None:
        """Test user service failure mode.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        user_service.configure_failure_mode(should_fail=True)
        result = user_service.execute()
        _ = u.Tests.assert_failure(result)
        assert result.error == "User service unavailable"

    @pytest.mark.integration
    def test_notification_service_execution(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test notification service execution.

        Args:
            clean_container: Isolated container fixture.

        """
        notification_service = self.NotificationService()
        result = notification_service.execute()
        _ = u.Tests.assert_success(result)
        assert result.value == "sent"

    @pytest.mark.integration
    def test_notification_service_send(self, clean_container: p.Container) -> None:
        """Test notification service send method.

        Args:
            clean_container: Isolated container fixture.

        """
        notification_service = self.NotificationService()
        email = "test@example.com"
        result = notification_service.send(email)
        _ = u.Tests.assert_success(result)
        assert result.value == "sent"
        assert email in notification_service.sent_notifications
        assert notification_service.call_count == 1

    @pytest.mark.integration
    def test_notification_service_failure_mode(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test notification service failure mode.

        Args:
            clean_container: Isolated container fixture.

        """
        notification_service = self.NotificationService()
        notification_service.configure_failure_mode(should_fail=True)
        result = notification_service.send("test@example.com")
        _ = u.Tests.assert_failure(result)
        assert result.error == "Notification service unavailable"

    @pytest.mark.integration
    def test_service_dependency_injection(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test dependency injection patterns with real services.

        Demonstrates proper dependency injection testing with real s
        implementations, service composition, and result validation.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        notification_service = self.NotificationService()
        user_id = "test_user_123"
        user_entity = self.UserServiceEntity(
            unique_id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
            active=True,
        )
        user_service.apply_user_data(user_id, user_entity)
        _ = clean_container.bind("user_service", user_service)
        _ = clean_container.bind("notification_service", notification_service)
        user_service_result = clean_container.resolve(
            "user_service",
            type_cls=self.UserQueryService,
        )
        notification_service_result = clean_container.resolve(
            "notification_service",
            type_cls=self.NotificationService,
        )
        assert user_service_result.success
        assert notification_service_result.success
        retrieved_user_service = user_service_result.value
        retrieved_notification_service = notification_service_result.value
        assert isinstance(retrieved_user_service, self.UserQueryService)
        assert isinstance(retrieved_notification_service, self.NotificationService)
        user_result = retrieved_user_service.fetch_user(user_id)
        assert user_result.success is True
        user_entity = user_result.value
        assert user_entity is not None
        notification_result = retrieved_notification_service.send(user_entity.email)
        assert notification_result.success is True
        assert user_entity.email in retrieved_notification_service.sent_notifications

    @pytest.mark.integration
    def test_service_with_external_service(
        self,
        clean_container: p.Container,
        mock_external_service: u.Tests.FunctionalExternalService,
    ) -> None:
        """Test service integration with external service.

        Args:
            clean_container: Isolated container fixture.
            mock_external_service: External service fixture.

        """
        user_service = self.UserQueryService()
        user_id = "test_user"
        user_result = user_service.fetch_user(user_id)
        assert user_result.success is True
        user_entity = user_result.value
        assert user_entity is not None
        external_result = mock_external_service.process(user_entity.email)
        assert external_result.success is True
        expected_processed = f"processed_{user_entity.email}"
        assert expected_processed in mock_external_service.processed_items
        assert mock_external_service.get_call_count() == 1
