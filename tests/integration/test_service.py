"""Integration tests for FlextService implementations.

Tests real FlextService implementations with proper dependency injection,
service composition, and lifecycle management patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from pydantic import PrivateAttr

from flext_core import FlextContainer, FlextResult, FlextService
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.entity import FlextModelsEntity
from flext_core.typings import FlextTypes
from tests.integration.test_integration import FunctionalExternalService


class UserServiceEntity(FlextModelsEntity.Core):
    """Test user entity model using FlextModels.Entity."""

    name: str
    email: str
    active: bool = True


class UserQueryService(FlextService[UserServiceEntity | None]):
    """Real user query service using FlextService."""

    _users: dict[str, UserServiceEntity] = PrivateAttr(default_factory=dict)
    _should_fail: bool = PrivateAttr(default=False)
    _call_count: int = PrivateAttr(default_factory=lambda: 0)

    def __init__(
        self,
        **data: FlextTypes.GeneralValueType,
    ) -> None:
        """Initialize user query service."""
        super().__init__(**data)

    def execute(self) -> FlextResult[bool]:
        """Execute user query service.

        Returns:
            FlextResult[bool]: True on success, False or failure otherwise.

        """
        if self._should_fail:
            return FlextResult[bool].fail("User service unavailable")
        return FlextResult[bool].ok(True)

    def get_user(self, user_id: str) -> FlextResult[UserServiceEntity]:
        """Get user by ID.

        Args:
            user_id: User identifier.

        Returns:
            FlextResult[UserServiceEntity]: User entity or failure.

        """
        self._call_count += 1

        if self._should_fail:
            return FlextResult[UserServiceEntity].fail("User service unavailable")

        if user_id in self._users:
            return FlextResult[UserServiceEntity].ok(self._users[user_id])
        # Create default user entity
        default_user = UserServiceEntity(
            unique_id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
            active=True,
        )
        return FlextResult[UserServiceEntity].ok(default_user)

    def set_user_data(self, user_id: str, user: UserServiceEntity) -> None:
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

    def __init__(
        self,
        **data: FlextTypes.GeneralValueType,
    ) -> None:
        """Initialize notification service."""
        super().__init__(**data)

    def execute(self) -> FlextResult[str]:
        """Execute notification service."""
        if self._should_fail:
            return FlextResult[str].fail("Notification service unavailable")
        return FlextResult[str].ok("sent")

    def send(self, email: str) -> FlextResult[str]:
        """Send notification.

        Args:
            email: Email address to send notification to.

        Returns:
            FlextResult[str]: Success confirmation or failure.

        """
        self._call_count += 1
        if self._should_fail:
            return FlextResult[str].fail("Notification service unavailable")
        self._sent_notifications.append(email)
        return FlextResult[str].ok("sent")

    def set_failure_mode(self, should_fail: bool) -> None:
        """Set failure mode for testing.

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


# Use the actual class, not the type alias
class ServiceConfig(FlextModelsCollections.Config):
    """Service configuration model with required fields."""

    name: str
    version: str
    temp_dir: str | None = None


class LifecycleService(FlextService[str]):
    """Real lifecycle service using FlextService with config model."""

    _initialized: bool = PrivateAttr(default=False)
    _service_config: ServiceConfig | None = PrivateAttr(default=None)
    _shutdown_called: bool = PrivateAttr(default=False)
    _should_fail_init: bool = PrivateAttr(default=False)
    _should_fail_shutdown: bool = PrivateAttr(default=False)

    def __init__(
        self,
        **data: FlextTypes.GeneralValueType,
    ) -> None:
        """Initialize lifecycle service."""
        super().__init__(**data)

    def execute(self) -> FlextResult[str]:
        """Execute lifecycle service."""
        if self._initialized:
            return FlextResult[str].ok("initialized")
        return FlextResult[str].ok("ready")

    def initialize(self, config: ServiceConfig) -> FlextResult[str]:
        """Initialize service with config model.

        Args:
            config: Configuration model.

        Returns:
            FlextResult[str]: Success or failure.

        """
        if self._should_fail_init:
            return FlextResult[str].fail("Initialization failed")
        self._initialized = True
        self._service_config = config
        return FlextResult[str].ok("initialized")

    def health_check(self) -> bool:
        """Check service health.

        Returns:
            bool: True if healthy, False otherwise.

        """
        return self._initialized and not self._shutdown_called

    def shutdown(self) -> FlextResult[str]:
        """Shutdown service.

        Returns:
            FlextResult[str]: Success or failure.

        """
        if self._should_fail_shutdown:
            return FlextResult[str].fail("Shutdown failed")
        self._shutdown_called = True
        return FlextResult[str].ok("shutdown")

    def set_failure_mode(
        self,
        *,
        fail_init: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        """Set failure modes for testing.

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
    def service_config(self) -> ServiceConfig | None:
        """Get service configuration."""
        return self._service_config

    @property
    def shutdown_called(self) -> bool:
        """Get shutdown status."""
        return self._shutdown_called


pytestmark = [pytest.mark.integration]


class TestFlextServiceIntegration:
    """Integration tests for FlextService implementations."""

    @pytest.mark.integration
    def test_user_service_execution(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test user service execution with FlextService.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        user_service = UserQueryService()

        # Act
        result = user_service.execute()

        # Assert
        assert result.is_success is True
        assert result.value is True  # execute() returns True on success

    @pytest.mark.integration
    def test_user_service_get_user(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test user service get_user method.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        user_service = UserQueryService()
        user_id = "test_user_123"

        # Act
        result = user_service.get_user(user_id)

        # Assert
        assert result.is_success is True
        assert result.value is not None
        assert result.value.unique_id == user_id
        assert result.value.name == f"User {user_id}"
        assert result.value.email == f"user{user_id}@example.com"
        assert user_service.call_count == 1

    @pytest.mark.integration
    def test_user_service_with_custom_data(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test user service with custom user data.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        user_service = UserQueryService()
        user_id = "custom_user"
        custom_user = UserServiceEntity(
            unique_id=user_id,
            name="Custom User",
            email="custom@example.com",
            active=True,
        )
        user_service.set_user_data(user_id, custom_user)

        # Act
        result = user_service.get_user(user_id)

        # Assert
        assert result.is_success is True
        assert result.value is not None
        assert result.value.unique_id == user_id
        assert result.value.name == "Custom User"
        assert result.value.email == "custom@example.com"

    @pytest.mark.integration
    def test_user_service_failure_mode(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test user service failure mode.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        user_service = UserQueryService()
        user_service.set_failure_mode(should_fail=True)

        # Act
        result = user_service.execute()

        # Assert
        assert result.is_success is False
        assert result.error == "User service unavailable"

    @pytest.mark.integration
    def test_notification_service_execution(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test notification service execution.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        notification_service = NotificationService()

        # Act
        result = notification_service.execute()

        # Assert
        assert result.is_success is True
        assert result.value == "sent"

    @pytest.mark.integration
    def test_notification_service_send(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test notification service send method.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        notification_service = NotificationService()
        email = "test@example.com"

        # Act
        result = notification_service.send(email)

        # Assert
        assert result.is_success is True
        assert result.value == "sent"
        assert email in notification_service.sent_notifications
        assert notification_service.call_count == 1

    @pytest.mark.integration
    def test_notification_service_failure_mode(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test notification service failure mode.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        notification_service = NotificationService()
        notification_service.set_failure_mode(should_fail=True)

        # Act
        result = notification_service.send("test@example.com")

        # Assert
        assert result.is_success is False
        assert result.error == "Notification service unavailable"

    @pytest.mark.integration
    def test_lifecycle_service_execution(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test lifecycle service execution.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange
        lifecycle_service = LifecycleService()

        # Act
        result = lifecycle_service.execute()

        # Assert
        assert result.is_success is True
        assert result.value == "ready"
        assert lifecycle_service.initialized is False

    @pytest.mark.integration
    def test_lifecycle_service_initialization(
        self,
        clean_container: FlextContainer,
        temp_directory: str,
    ) -> None:
        """Test lifecycle service initialization with config.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        # Arrange
        lifecycle_service = LifecycleService()
        service_config = ServiceConfig(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )

        # Act
        result = lifecycle_service.initialize(service_config)

        # Assert
        assert result.is_success is True
        assert result.value == "initialized"
        assert lifecycle_service.initialized is True
        assert lifecycle_service.service_config is not None
        assert lifecycle_service.service_config.name == "test_service"

    @pytest.mark.integration
    def test_lifecycle_service_health_check(
        self,
        clean_container: FlextContainer,
        temp_directory: str,
    ) -> None:
        """Test lifecycle service health check.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        # Arrange
        lifecycle_service = LifecycleService()
        service_config = ServiceConfig(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )

        # Act - Before initialization
        health_before = lifecycle_service.health_check()

        # Initialize
        _ = lifecycle_service.initialize(service_config)

        # Act - After initialization
        health_after = lifecycle_service.health_check()

        # Assert
        assert health_before is False
        assert health_after is True

    @pytest.mark.integration
    def test_lifecycle_service_shutdown(
        self,
        clean_container: FlextContainer,
        temp_directory: str,
    ) -> None:
        """Test lifecycle service shutdown.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        # Arrange
        lifecycle_service = LifecycleService()
        service_config = ServiceConfig(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )
        _ = lifecycle_service.initialize(service_config)

        # Act
        result = lifecycle_service.shutdown()

        # Assert
        assert result.is_success is True
        assert result.value == "shutdown"
        assert lifecycle_service.shutdown_called is True
        assert lifecycle_service.health_check() is False

    @pytest.mark.integration
    def test_lifecycle_service_failure_modes(
        self,
        clean_container: FlextContainer,
        temp_directory: str,
    ) -> None:
        """Test lifecycle service failure modes.

        Args:
            clean_container: Isolated container fixture.
            temp_directory: Temporary directory fixture.

        """
        # Arrange
        lifecycle_service = LifecycleService()
        service_config = ServiceConfig(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_directory),
        )

        # Test initialization failure
        lifecycle_service.set_failure_mode(fail_init=True)
        init_result = lifecycle_service.initialize(service_config)
        assert init_result.is_success is False
        assert init_result.error == "Initialization failed"

        # Test shutdown failure
        lifecycle_service.set_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        assert shutdown_result.is_success is False
        assert shutdown_result.error == "Shutdown failed"

    @pytest.mark.integration
    def test_service_dependency_injection(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test dependency injection patterns with real services.

        Demonstrates proper dependency injection testing with real FlextService
        implementations, service composition, and result validation.

        Args:
            clean_container: Isolated container fixture.

        """
        # Arrange - Create real services using FlextService
        user_service = UserQueryService()
        notification_service = NotificationService()

        # Configure service behavior with real test data
        user_id = "test_user_123"
        user_entity = UserServiceEntity(
            unique_id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
            active=True,
        )
        user_service.set_user_data(user_id, user_entity)

        # Register services in container
        _ = clean_container.with_service("user_service", user_service)
        _ = clean_container.with_service("notification_service", notification_service)

        # Act - Retrieve and use services
        user_service_result = clean_container.get_typed(
            "user_service",
            UserQueryService,
        )
        notification_service_result = clean_container.get_typed(
            "notification_service",
            NotificationService,
        )

        assert user_service_result.is_success
        assert notification_service_result.is_success
        retrieved_user_service = user_service_result.value
        retrieved_notification_service = notification_service_result.value
        assert isinstance(retrieved_user_service, UserQueryService)
        assert isinstance(retrieved_notification_service, NotificationService)

        # Get user entity first
        user_result = retrieved_user_service.get_user(user_id)
        assert user_result.is_success is True

        # Send notification using entity email
        user_entity = user_result.value
        assert user_entity is not None
        notification_result = retrieved_notification_service.send(user_entity.email)

        # Assert
        assert notification_result.is_success is True
        assert user_entity.email in retrieved_notification_service.sent_notifications

    @pytest.mark.integration
    def test_service_with_external_service(
        self,
        clean_container: FlextContainer,
        mock_external_service: FunctionalExternalService,
    ) -> None:
        """Test service integration with external service.

        Args:
            clean_container: Isolated container fixture.
            mock_external_service: External service fixture.

        """
        # Arrange
        user_service = UserQueryService()
        user_id = "test_user"

        # Act - Process user data through external service
        user_result = user_service.get_user(user_id)
        assert user_result.is_success is True

        user_entity = user_result.value
        assert user_entity is not None

        # Process through external service
        external_result = mock_external_service.process(user_entity.email)

        # Assert
        assert external_result.is_success is True
        assert user_entity.email in mock_external_service.processed_items
        assert mock_external_service.get_call_count() == 1
