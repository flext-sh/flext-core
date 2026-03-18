"""Integration tests for FlextService implementations.

Tests real FlextService implementations with proper dependency injection,
service composition, and lifecycle management patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, override

import pytest
from pydantic import BaseModel, Field, PrivateAttr

from flext_core import FlextService, m, p, r

from ..conftest import FunctionalExternalService
from ..test_utils import assertion_helpers


class TestService:
    class UserServiceEntity(BaseModel):
        """Test user entity model using dataclass."""

        unique_id: Annotated[str, Field(description="Unique user identifier")]
        name: Annotated[str, Field(description="User display name")]
        email: Annotated[str, Field(description="User email address")]
        active: Annotated[
            bool, Field(default=True, description="Whether user is active")
        ] = True

    class UserQueryService(FlextService[bool]):
        """Real user query service using FlextService.

        Business Rule: execute() returns bool to indicate success/failure status.
        This pattern is used for services that don't return domain entities but
        rather status indicators. The bool value indicates whether the operation
        completed successfully.

        Implications for Audit:
        - bool return type simplifies status checking
        - Use get_user() for actual entity retrieval
        - This pattern separates status from data retrieval
        """

        _users: dict[str, TestService.UserServiceEntity] = PrivateAttr(
            default_factory=dict
        )
        _should_fail: bool = PrivateAttr(default=False)
        _call_count: int = PrivateAttr(default_factory=lambda: 0)

        @override
        def execute(self) -> r[bool]:
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

        def get_user(self, user_id: str) -> r[TestService.UserServiceEntity]:
            """Get user by ID.

            Args:
                user_id: User identifier.

            Returns:
                r[UserServiceEntity]: User entity or failure.

            """
            self._call_count += 1
            if self._should_fail:
                return r[TestService.UserServiceEntity].fail("User service unavailable")
            if user_id in self._users:
                return r[TestService.UserServiceEntity].ok(self._users[user_id])
            default_user = TestService.UserServiceEntity(
                unique_id=user_id,
                name=f"User {user_id}",
                email=f"user{user_id}@example.com",
                active=True,
            )
            return r[TestService.UserServiceEntity].ok(default_user)

        def set_user_data(
            self,
            user_id: str,
            user: TestService.UserServiceEntity,
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

    class NotificationService(FlextService[str]):
        """Real notification service using FlextService."""

        _sent_notifications: list[str] = PrivateAttr(default_factory=list)
        _call_count: int = PrivateAttr(default_factory=lambda: 0)
        _should_fail: bool = PrivateAttr(default=False)

        @override
        def execute(self) -> r[str]:
            """Execute notification service."""
            if self._should_fail:
                return r[str].fail("Notification service unavailable")
            return r[str].ok("sent")

        def send(self, email: str) -> r[str]:
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
        def sent_notifications(self) -> list[str]:
            """Get sent notifications."""
            return self._sent_notifications.copy()

        @property
        def call_count(self) -> int:
            """Get call count."""
            return self._call_count

    class ServiceConfig(m.Config):
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
    ) -> TestService.ServiceConfig:
        return TestService.ServiceConfig(
            name=name,
            version=version,
            temp_dir=temp_dir,
        )

    class LifecycleService(FlextService[str]):
        """Real lifecycle service using FlextService with config model."""

        _initialized: bool = PrivateAttr(default=False)
        _service_config: TestService.ServiceConfig | None = PrivateAttr(default=None)
        _shutdown_called: bool = PrivateAttr(default=False)
        _should_fail_init: bool = PrivateAttr(default=False)
        _should_fail_shutdown: bool = PrivateAttr(default=False)

        @override
        def execute(self) -> r[str]:
            """Execute lifecycle service."""
            if self._initialized:
                return r[str].ok("initialized")
            return r[str].ok("ready")

        def initialize(self, config: TestService.ServiceConfig) -> r[str]:
            """Initialize service with config model.

            Args:
                config: Configuration model.

            Returns:
                r[str]: Success or failure.

            """
            if self._should_fail_init:
                return r[str].fail("Initialization failed")
            self._initialized = True
            self._service_config = config
            return r[str].ok("initialized")

        def health_check(self) -> bool:
            """Check service health.

            Returns:
                bool: True if healthy, False otherwise.

            """
            return self._initialized and (not self._shutdown_called)

        def shutdown(self) -> r[str]:
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
            """Set failure modes for testing.

            Business Rule: Uses object.__setattr__ to modify PrivateAttr fields
            in frozen Pydantic models. This is the correct pattern for mutable
            state in frozen models per Pydantic v2 advanced usage.

            Args:
                fail_init: Whether initialization should fail.
                fail_shutdown: Whether shutdown should fail.

            """
            self._should_fail_init = fail_init
            self._should_fail_shutdown = fail_shutdown

        @property
        def initialized(self) -> bool:
            """Get initialization status."""
            return self._initialized

        @property
        def service_config(self) -> TestService.ServiceConfig | None:
            """Get service configuration."""
            return self._service_config

        @property
        def shutdown_called(self) -> bool:
            """Get shutdown status."""
            return self._shutdown_called

    pytestmark = [pytest.mark.integration]

    @pytest.mark.integration
    def test_user_service_execution(self, clean_container: p.Container) -> None:
        """Test user service execution with FlextService.

        Args:
            clean_container: Isolated container fixture.

        """
        user_service = self.UserQueryService()
        result = user_service.execute()
        _ = assertion_helpers.assert_flext_result_success(result)
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
        _ = assertion_helpers.assert_flext_result_success(result)
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
        _ = assertion_helpers.assert_flext_result_success(result)
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
        _ = assertion_helpers.assert_flext_result_failure(result)
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
        _ = assertion_helpers.assert_flext_result_success(result)
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
        _ = assertion_helpers.assert_flext_result_success(result)
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
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "Notification service unavailable"

    @pytest.mark.integration
    def test_lifecycle_service_execution(self, clean_container: p.Container) -> None:
        """Test lifecycle service execution.

        Args:
            clean_container: Isolated container fixture.

        """
        lifecycle_service = self.LifecycleService()
        result = lifecycle_service.execute()
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "ready"
        assert lifecycle_service.initialized is False

    @pytest.mark.integration
    def test_lifecycle_service_initialization(
        self,
        clean_container: p.Container,
        temp_directory: str,
    ) -> None:
        """Test lifecycle service initialization with config.

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
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "initialized"
        assert lifecycle_service.initialized is True
        assert lifecycle_service.service_config is not None
        assert lifecycle_service.service_config.name == "test_service"

    @pytest.mark.integration
    def test_lifecycle_service_health_check(
        self,
        clean_container: p.Container,
        temp_directory: str,
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
        temp_directory: str,
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
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "shutdown"
        assert lifecycle_service.shutdown_called is True
        assert lifecycle_service.health_check() is False

    @pytest.mark.integration
    def test_lifecycle_service_failure_modes(
        self,
        clean_container: p.Container,
        temp_directory: str,
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
        assert init_result.is_success is False
        assert init_result.error == "Initialization failed"
        lifecycle_service.set_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        assert shutdown_result.is_success is False
        assert shutdown_result.error == "Shutdown failed"

    @pytest.mark.integration
    def test_service_dependency_injection(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test dependency injection patterns with real services.

        Demonstrates proper dependency injection testing with real FlextService
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
        _ = clean_container.register("user_service", user_service)
        _ = clean_container.register("notification_service", notification_service)
        user_service_result = clean_container.get(
            "user_service",
            type_cls=self.UserQueryService,
        )
        notification_service_result = clean_container.get(
            "notification_service",
            type_cls=self.NotificationService,
        )
        assert user_service_result.is_success
        assert notification_service_result.is_success
        retrieved_user_service = user_service_result.value
        retrieved_notification_service = notification_service_result.value
        assert isinstance(retrieved_user_service, self.UserQueryService)
        assert isinstance(retrieved_notification_service, self.NotificationService)
        user_result = retrieved_user_service.get_user(user_id)
        assert user_result.is_success is True
        user_entity = user_result.value
        assert user_entity is not None
        notification_result = retrieved_notification_service.send(user_entity.email)
        assert notification_result.is_success is True
        assert user_entity.email in retrieved_notification_service.sent_notifications

    @pytest.mark.integration
    def test_service_with_external_service(
        self,
        clean_container: p.Container,
        mock_external_service: FunctionalExternalService,
    ) -> None:
        """Test service integration with external service.

        Args:
            clean_container: Isolated container fixture.
            mock_external_service: External service fixture.

        """
        user_service = self.UserQueryService()
        user_id = "test_user"
        user_result = user_service.get_user(user_id)
        assert user_result.is_success is True
        user_entity = user_result.value
        assert user_entity is not None
        external_result = mock_external_service.process(user_entity.email)
        assert external_result.is_success is True
        expected_processed = f"processed_{user_entity.email}"
        assert expected_processed in mock_external_service.processed_items
        assert mock_external_service.get_call_count() == 1
