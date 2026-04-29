"""Integration tests for s implementations.

Tests real s implementations with proper dependency injection,
service composition, and lifecycle management patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    MutableMapping,
    MutableSequence,
)
from pathlib import Path
from typing import Annotated, override

import pytest
from flext_tests.base import s

from tests import m, p, r, t, u


class TestsFlextServiceIntegration:
    class UserServiceEntity(m.BaseModel):
        """Test user entity model using dataclass."""

        unique_id: Annotated[str, m.Field(description="Unique user identifier")]
        name: Annotated[str, m.Field(description="User display name")]
        email: Annotated[str, m.Field(description="User email address")]
        active: Annotated[bool, m.Field(description="Whether user is active")] = True

    class UserQueryService(s[bool]):
        """Real user query service using s.

        Business Rule: execute() returns bool to indicate success/failure status.
        This pattern is used for services that don't return domain entities but
        rather status indicators. The bool value indicates whether the operation
        completed successfully.

        Implications for Audit:
        - bool return type simplifies status checking
        - Use get_user() for actual entity retrieval
        - This pattern separates status from data retrieval
        """

        _users: MutableMapping[str, TestsFlextServiceIntegration.UserServiceEntity] = (
            m.PrivateAttr(
                default_factory=lambda: dict[
                    str, TestsFlextServiceIntegration.UserServiceEntity
                ](),
            )
        )
        _should_fail: bool = m.PrivateAttr(default_factory=lambda: False)
        _call_count: int = m.PrivateAttr(default_factory=lambda: 0)

        @override
        def execute(self) -> p.Result[bool]:
            """Execute user query service.

            Business Rule: Returns bool to indicate service availability and readiness.
            True indicates the service is ready and operational, False would indicate
            a failure state. This pattern separates status checking from data retrieval.

            Returns:
                r[bool]: True if service is ready, failure otherwise.

            """
            if self._should_fail:
                return r[bool].fail("User service unavailable")
            return r[bool].ok(True)

        def get_user(
            self, user_id: str
        ) -> p.Result[TestsFlextServiceIntegration.UserServiceEntity]:
            """Get user by ID.

            Args:
                user_id: User identifier.

            Returns:
                r[UserServiceEntity]: User entity or failure.

            """
            self._call_count += 1
            if self._should_fail:
                return r[TestsFlextServiceIntegration.UserServiceEntity].fail(
                    "User service unavailable"
                )
            if user_id in self._users:
                return r[TestsFlextServiceIntegration.UserServiceEntity].ok(
                    self._users[user_id]
                )
            default_user = TestsFlextServiceIntegration.UserServiceEntity(
                unique_id=user_id,
                name=f"User {user_id}",
                email=f"user{user_id}@example.com",
                active=True,
            )
            return r[TestsFlextServiceIntegration.UserServiceEntity].ok(default_user)

        def set_user_data(
            self,
            user_id: str,
            user: TestsFlextServiceIntegration.UserServiceEntity,
        ) -> None:
            """Set user data for testing.

            Args:
                user_id: User identifier.
                user: User entity to store.

            """
            self._users[user_id] = user

        def set_failure_mode(self, should_fail: bool) -> None:
            """Set failure mode for testing.

            Args:
                should_fail: Whether service should fail.

            """
            self._should_fail = should_fail

        @property
        def call_count(self) -> int:
            """Get call count."""
            return self._call_count

    class NotificationService(s[str]):
        """Real notification service using s."""

        _sent_notifications: MutableSequence[str] = m.PrivateAttr(
            default_factory=lambda: list[str](),
        )
        _call_count: int = m.PrivateAttr(default_factory=lambda: 0)
        _should_fail: bool = m.PrivateAttr(default_factory=lambda: False)

        @override
        def execute(self) -> p.Result[str]:
            """Execute notification service."""
            if self._should_fail:
                return r[str].fail("Notification service unavailable")
            return r[str].ok("sent")

        def send(self, email: str) -> p.Result[str]:
            """Send notification.

            Args:
                email: Email address to send notification to.

            Returns:
                r[str]: Success confirmation or failure.

            """
            self._call_count += 1
            if self._should_fail:
                return r[str].fail("Notification service unavailable")
            self._sent_notifications.append(email)
            return r[str].ok("sent")

        def set_failure_mode(self, should_fail: bool) -> None:
            """Set failure mode for testing.

            Business Rule: Uses setattr to modify PrivateAttr fields
            in frozen Pydantic models. This is the correct pattern for mutable
            state in frozen models per Pydantic v2 advanced usage.

            Args:
                should_fail: Whether service should fail.

            """
            self._should_fail = should_fail

        @property
        def sent_notifications(self) -> t.StrSequence:
            """Get sent notifications."""
            return list(self._sent_notifications)

        @property
        def call_count(self) -> int:
            """Get call count."""
            return self._call_count

    class ServiceConfig(m.Value):
        """Service configuration model with required fields."""

        name: str
        version: str
        temp_dir: str | None = None

    @staticmethod
    def _build_service_config(
        *,
        name: str,
        version: str,
        temp_dir: str,
    ) -> TestsFlextServiceIntegration.ServiceConfig:
        return TestsFlextServiceIntegration.ServiceConfig(
            name=name,
            version=version,
            temp_dir=temp_dir,
        )

    class LifecycleService(s[str]):
        """Real lifecycle service using s with settings model."""

        _initialized: bool = m.PrivateAttr(default_factory=lambda: False)
        _service_config: TestsFlextServiceIntegration.ServiceConfig | None = (
            m.PrivateAttr(default_factory=lambda: None)
        )
        _shutdown_called: bool = m.PrivateAttr(default_factory=lambda: False)
        _should_fail_init: bool = m.PrivateAttr(default_factory=lambda: False)
        _should_fail_shutdown: bool = m.PrivateAttr(default_factory=lambda: False)

        @override
        def execute(self) -> p.Result[str]:
            """Execute lifecycle service."""
            if self._initialized:
                return r[str].ok("initialized")
            return r[str].ok("ready")

        def initialize(
            self, settings: TestsFlextServiceIntegration.ServiceConfig
        ) -> p.Result[str]:
            """Initialize service with settings model.

            Args:
                settings: Configuration model.

            Returns:
                r[str]: Success or failure.

            """
            if self._should_fail_init:
                return r[str].fail("Initialization failed")
            self._initialized = True
            self._service_config = settings
            return r[str].ok("initialized")

        def health_check(self) -> bool:
            """Check service health.

            Returns:
                bool: True if healthy, False otherwise.

            """
            return self._initialized and (not self._shutdown_called)

        def shutdown(self) -> p.Result[str]:
            """Shutdown service.

            Returns:
                r[str]: Success or failure.

            """
            if self._should_fail_shutdown:
                return r[str].fail("Shutdown failed")
            self._shutdown_called = True
            return r[str].ok("shutdown")

        def set_failure_mode(
            self,
            *,
            fail_init: bool = False,
            fail_shutdown: bool = False,
        ) -> None:
            self._should_fail_init = fail_init
            self._should_fail_shutdown = fail_shutdown

        @property
        def initialized(self) -> bool:
            """Get initialization status."""
            return self._initialized

        @property
        def service_config(
            self,
        ) -> TestsFlextServiceIntegration.ServiceConfig | None:
            """Get service configuration."""
            return self._service_config

        @property
        def shutdown_called(self) -> bool:
            """Get shutdown status."""
            return self._shutdown_called

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
    def test_user_service_get_user(self, clean_container: p.Container) -> None:
        """Test user service get_user method.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        user_id = "test_user_123"
        result = user_service.get_user(user_id)
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
        user_service.set_user_data(user_id, custom_user)
        result = user_service.get_user(user_id)
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
        user_service.set_failure_mode(should_fail=True)
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
        notification_service.set_failure_mode(should_fail=True)
        result = notification_service.send("test@example.com")
        _ = u.Tests.assert_failure(result)
        assert result.error == "Notification service unavailable"

    @pytest.mark.integration
    def test_lifecycle_service_execution(self, clean_container: p.Container) -> None:
        """Test lifecycle service execution.

        Args:
            clean_container: Isolated container fixture.

        """
        lifecycle_service = self.LifecycleService()
        result = lifecycle_service.execute()
        _ = u.Tests.assert_success(result)
        assert result.value == "ready"
        assert lifecycle_service.initialized is False

    @pytest.mark.integration
    def test_lifecycle_service_initialization(
        self,
        clean_container: p.Container,
        temp_directory: Path,
    ) -> None:
        """Test lifecycle service initialization with settings.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )
        result = lifecycle_service.initialize(service_config)
        _ = u.Tests.assert_success(result)
        assert result.value == "initialized"
        assert lifecycle_service.initialized is True
        assert lifecycle_service.service_config is not None
        assert lifecycle_service.service_config.name == "test_service"

    @pytest.mark.integration
    def test_lifecycle_service_health_check(
        self,
        clean_container: p.Container,
        temp_directory: Path,
    ) -> None:
        """Test lifecycle service health check.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )
        health_before = lifecycle_service.health_check()
        _ = lifecycle_service.initialize(service_config)
        health_after = lifecycle_service.health_check()
        assert health_before is False
        assert health_after is True

    @pytest.mark.integration
    def test_lifecycle_service_shutdown(
        self,
        clean_container: p.Container,
        temp_directory: Path,
    ) -> None:
        """Test lifecycle service shutdown.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )
        _ = lifecycle_service.initialize(service_config)
        result = lifecycle_service.shutdown()
        _ = u.Tests.assert_success(result)
        assert result.value == "shutdown"
        assert lifecycle_service.shutdown_called is True
        assert lifecycle_service.health_check() is False

    @pytest.mark.integration
    def test_lifecycle_service_failure_modes(
        self,
        clean_container: p.Container,
        temp_directory: Path,
    ) -> None:
        """Test lifecycle service failure modes.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )
        lifecycle_service.set_failure_mode(fail_init=True)
        init_result = lifecycle_service.initialize(service_config)
        assert init_result.success is False
        assert init_result.error == "Initialization failed"
        lifecycle_service.set_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        assert shutdown_result.success is False
        assert shutdown_result.error == "Shutdown failed"

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
        user_service.set_user_data(user_id, user_entity)
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
        user_result = retrieved_user_service.get_user(user_id)
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
        user_result = user_service.get_user(user_id)
        assert user_result.success is True
        user_entity = user_result.value
        assert user_entity is not None
        external_result = mock_external_service.process(user_entity.email)
        assert external_result.success is True
        expected_processed = f"processed_{user_entity.email}"
        assert expected_processed in mock_external_service.processed_items
        assert mock_external_service.get_call_count() == 1
