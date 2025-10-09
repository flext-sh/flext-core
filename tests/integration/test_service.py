"""Service integration testing patterns for FLEXT Core.

Demonstrates enterprise-grade integration testing patterns with proper mocking,
service isolation, and performance validation aligned with modern pytest best practices.

Architecture:
    Integration Testing → Service Patterns → Mock Integration → Performance Validation

    This module showcases:
    - Mock-based service integration patterns
    - FlextResult pipeline integration testing
    - Performance characteristics validation
    - Error propagation across service boundaries
    - Type safety in service integration workflows

Testing Patterns Demonstrated:
    - AAA (Arrange-Act-Assert) structure
    - Proper fixture usage and dependency injection
    - Mock configuration and verification
    - Performance threshold validation
    - Error scenario testing with proper isolation


Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import time
from pathlib import Path
from typing import cast

import pytest

from flext_core import FlextContainer, FlextResult, FlextTypes


class FunctionalExternalService:
    """Functional external service for integration testing - real implementation."""

    def __init__(self) -> None:
        """Initialize functional external service."""
        super().__init__()
        self.call_count = 0
        self.processed_items: FlextTypes.List = []
        self.should_fail = False
        self.failure_message = "Service unavailable"

    def process(self, data: object = None) -> FlextResult[str]:
        """Functional processing method - validates real behavior.

        Returns:
            FlextResult[str]: Success with processed data or failure with error message.

        """
        self.call_count += 1
        self.processed_items.append(data)

        if self.should_fail:
            return FlextResult[str].fail(self.failure_message)

        return FlextResult[str].ok("processed")

    def set_failure_mode(
        self,
        *,
        should_fail: bool,
        message: str = "Service unavailable",
    ) -> None:
        """Configure service to fail for testing error scenarios."""
        self.should_fail = should_fail
        self.failure_message = message


class FunctionalUserService:
    """Functional user service for integration testing - real implementation."""

    def __init__(self) -> None:
        """Initialize functional user service."""
        super().__init__()
        self.users: FlextTypes.NestedDict = {}
        self.call_count = 0
        self.should_fail = False

    def get_user(
        self,
        user_id: str,
    ) -> FlextResult[FlextTypes.Dict]:
        """Get user by ID - functional implementation.

        Returns:
            FlextResult[FlextTypes.Dict]: Success with user data or failure with error message.

        """
        self.call_count += 1

        if self.should_fail:
            return FlextResult[FlextTypes.Dict].fail(
                "User service unavailable",
            )

        if user_id in self.users:
            return FlextResult[FlextTypes.Dict].ok(
                self.users[user_id],
            )

        # Default user data for testing
        default_user: FlextTypes.Dict = {
            "id": user_id,
            "email": f"user{user_id}@example.com",
            "name": f"User {user_id}",
            "active": True,
        }
        return FlextResult[FlextTypes.Dict].ok(
            default_user,
        )

    def set_user_data(
        self,
        user_id: str,
        user_data: FlextTypes.Dict,
    ) -> None:
        """Set user data for testing."""
        self.users[user_id] = user_data

    def set_failure_mode(self, *, should_fail: bool) -> None:
        """Configure service to fail for testing error scenarios."""
        self.should_fail = should_fail


class FunctionalNotificationService:
    """Functional notification service for integration testing - real implementation."""

    def __init__(self) -> None:
        """Initialize functional notification service."""
        super().__init__()
        self.sent_notifications: FlextTypes.StringList = []
        self.call_count = 0
        self.should_fail = False

    def send(self, email: str) -> FlextResult[str]:
        """Send notification - functional implementation.

        Returns:
            FlextResult[str]: Success with confirmation or failure with error message.

        """
        self.call_count += 1

        if self.should_fail:
            return FlextResult[str].fail("Notification service unavailable")

        self.sent_notifications.append(email)
        return FlextResult[str].ok("sent")

    def set_failure_mode(self, *, should_fail: bool) -> None:
        """Configure service to fail for testing error scenarios."""
        self.should_fail = should_fail


class FunctionalLifecycleService:
    """Functional lifecycle service for integration testing - real implementation."""

    def __init__(self) -> None:
        """Initialize functional lifecycle service."""
        super().__init__()
        self.initialized = False
        self.config: FlextTypes.Dict | None = None
        self.shutdown_called = False
        self.should_fail_init = False
        self.should_fail_shutdown = False

    def initialize(self, config: FlextTypes.Dict) -> FlextResult[str]:
        """Initialize service - functional implementation.

        Returns:
            FlextResult[str]: Success with confirmation or failure with error message.

        """
        if self.should_fail_init:
            return FlextResult[str].fail("Initialization failed")

        self.initialized = True
        self.config = config
        return FlextResult[str].ok("initialized")

    def is_healthy(self) -> bool:
        """Check service health - functional implementation.

        Returns:
            bool: True if service is healthy, False otherwise.

        """
        return self.initialized and not self.shutdown_called

    def shutdown(self) -> FlextResult[str]:
        """Shutdown service - functional implementation.

        Returns:
            FlextResult[str]: Success with confirmation or failure with error message.

        """
        if self.should_fail_shutdown:
            return FlextResult[str].fail("Shutdown failed")

        self.shutdown_called = True
        return FlextResult[str].ok("shutdown")

    def set_failure_modes(
        self,
        *,
        fail_init: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        """Configure service to fail for testing error scenarios."""
        self.should_fail_init = fail_init
        self.should_fail_shutdown = fail_shutdown


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Functional external service for integration testing.

    Returns:
        FunctionalExternalService: A configured external service instance.

    """
    return FunctionalExternalService()


@pytest.mark.integration
class TestServiceIntegrationPatterns:
    """Demonstrates modern service integration testing patterns."""

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.usefixtures("performance_threshold")
    def test_service_pipeline_performance(
        self,
        mock_external_service: FunctionalExternalService,
        performance_threshold: dict[str, float],
        benchmark_data: FlextTypes.Dict | dict[str, dict[str, dict[str, list[int]]]],
    ) -> None:
        """Test service pipeline meets performance requirements.

        Validates that integrated service pipelines execute within
        acceptable performance thresholds for enterprise workloads.

        Args:
            configured_container: Pre-configured container with services
            mock_external_service: Functional external service
            performance_threshold: Performance threshold configuration
            benchmark_data: Benchmark data sets

        """
        # Arrange
        # Type is already validated by parameter type annotation
        if "large_dataset" not in benchmark_data:
            pytest.skip("large_dataset not found in benchmark_data")
        large_dataset = benchmark_data["large_dataset"]

        def process_pipeline(
            data: list[int]
            | FlextTypes.StringDict
            | dict[str, dict[str, dict[str, list[int]]]],
        ) -> FlextResult[str]:
            # Simulate service pipeline with realistic operations
            result = FlextResult[object].ok(data)
            return result.flat_map(mock_external_service.process).map(
                lambda _r: f"pipeline_result_{len(str(data))}",
            )

        # Act - Measure pipeline performance
        start_time = time.perf_counter()
        # Type check large_dataset before passing to process_pipeline
        if isinstance(large_dataset, (list, dict)) or hasattr(
            large_dataset,
            "__iter__",
        ):
            # Cast to expected type to satisfy Pyrefly
            result = process_pipeline(
                cast(
                    "list[int] | dict[str, dict[str, dict[str, list[int]]]]",
                    large_dataset,
                ),
            )
        else:
            pytest.skip("large_dataset is not a valid type for process_pipeline")
        execution_time = time.perf_counter() - start_time

        # Assert - Performance and functionality
        assert result.is_success is True
        assert execution_time < performance_threshold["serialization"]
        assert result.value is not None
        assert "pipeline_result" in result.value
        # Functional validation - check actual service call
        assert mock_external_service.call_count == 1
        assert large_dataset in mock_external_service.processed_items

    @pytest.mark.integration
    @pytest.mark.error_path
    @pytest.mark.usefixtures("error_context")
    def test_service_error_propagation(
        self,
        mock_external_service: FunctionalExternalService,
        error_context: dict[str, str | None],
    ) -> None:
        """Test error propagation across service boundaries.

        Validates that errors propagate correctly through service
        integration layers while maintaining proper error context.

        Args:
            configured_container: Pre-configured container with services
            mock_external_service: Functional external service
            error_context: Error context fixture

        """
        # Arrange - Configure functional service to simulate failure
        error_message = f"Service error: {error_context['error_code']}"
        mock_external_service.set_failure_mode(should_fail=True, message=error_message)

        def failing_pipeline(data: str) -> FlextResult[str]:
            return (
                FlextResult[str]
                .ok(data)
                .flat_map(mock_external_service.process)
                .map(lambda _r: f"processed_{_r}")
            )

        # Act - Execute failing pipeline
        result = failing_pipeline("test_data")

        # Assert - Error propagation
        assert result.is_failure is True
        assert result.error is not None
        assert error_message in (result.error or "")
        # Functional validation - check actual service was called
        assert mock_external_service.call_count == 1
        assert "test_data" in mock_external_service.processed_items

    @pytest.mark.integration
    @pytest.mark.architecture
    def test_dependency_injection_with_mocks(
        self,
        clean_container: FlextContainer,
        test_user_data: FlextTypes.Dict | FlextTypes.StringList,
    ) -> None:
        """Test dependency injection patterns with functional services.

        Demonstrates proper dependency injection testing with functional services,
        service composition, and result validation.

        Args:
            clean_container: Isolated container fixture
            test_user_data: User data fixture

        """
        # Arrange - Create functional services
        user_service = FunctionalUserService()
        notification_service = FunctionalNotificationService()

        # Configure functional service behavior with real test data
        if isinstance(test_user_data, dict):
            test_user_dict: FlextTypes.Dict = test_user_data
            user_id = str(test_user_dict["id"])
            user_service.set_user_data(user_id, test_user_dict)
        else:
            # Handle case where test_user_data is not a dict
            user_id = "default_user"
            default_user_data: FlextTypes.Dict = {
                "id": user_id,
                "email": f"user{user_id}@example.com",
                "name": f"User {user_id}",
                "active": True,
            }
            user_service.set_user_data(user_id, default_user_data)

        # Register services in container
        clean_container.register("user_service", user_service)
        clean_container.register("notification_service", notification_service)

        def user_notification_workflow(workflow_user_id: str) -> FlextResult[str]:
            # Simulate service composition workflow
            user_service_result = clean_container.get("user_service")
            notification_service_result = clean_container.get("notification_service")

            if (
                not user_service_result.is_success
                or not notification_service_result.is_success
            ):
                return FlextResult[str].fail("Service unavailable")

            retrieved_user_service = cast(
                "FunctionalUserService",
                user_service_result.value,
            )
            retrieved_notification_service = cast(
                "FunctionalNotificationService",
                notification_service_result.value,
            )

            # Get user data first
            user_result = retrieved_user_service.get_user(workflow_user_id)
            if not user_result.is_success:
                return FlextResult[str].fail("User not found")

            # Send notification
            return retrieved_notification_service.send(
                str(user_result.value["email"]),
            )

        # Act - Execute workflow
        result = user_notification_workflow(user_id)

        # Assert - Workflow success and service interaction
        assert result.is_success is True
        assert result.value == "sent"

        # Functional validation - check actual service calls
        assert user_service.call_count == 1
        assert notification_service.call_count == 1

        # Get email from the user data that was set
        if isinstance(test_user_data, dict) and "email" in test_user_data:
            expected_email = str(test_user_data["email"])
        else:
            expected_email = f"user{user_id}@example.com"

        assert expected_email in notification_service.sent_notifications

    @pytest.mark.integration
    @pytest.mark.boundary
    def test_container_service_lifecycle(
        self,
        clean_container: FlextContainer,
        temp_directory: Path,
    ) -> None:
        """Test complete service lifecycle with container management.

        Validates service registration, retrieval, lifecycle management,
        and cleanup patterns in dependency injection workflows.

        Args:
            clean_container: Isolated container fixture
            temp_directory: Temporary directory fixture

        """
        # Arrange - Create lifecycle-aware functional service
        lifecycle_service = FunctionalLifecycleService()

        service_config = {
            "name": "test_service",
            "version": "1.0.0",
            "temp_dir": str(temp_directory),
        }

        # Act - Register service with configuration
        registration_result = clean_container.register(
            "lifecycle_service",
            lifecycle_service,
        )
        config_result = clean_container.register("service_config", service_config)

        # Assert - Registration success
        assert registration_result.is_success is True
        assert config_result.is_success is True

        # Act - Initialize service
        service_result = clean_container.get("lifecycle_service")
        config_fetch_result = clean_container.get("service_config")

        # Assert - Service retrieval
        assert service_result.is_success is True
        assert config_fetch_result.is_success is True

        service = cast(
            "FunctionalLifecycleService",
            service_result.value,
        )  # This is our FunctionalLifecycleService
        config = cast("FlextTypes.Dict", config_fetch_result.value)

        # Act - Test service lifecycle
        init_result = service.initialize(config)
        health_status = service.is_healthy()
        shutdown_result = service.shutdown()

        # Assert - Lifecycle operations
        assert init_result.is_success is True
        assert health_status is True
        assert shutdown_result.is_success is True

        # Functional validation - check actual service state
        assert service.initialized is True
        assert service.config == config
        assert service.shutdown_called is True

        # Act - Clear container
        clean_container.clear()

        # Assert - Container is empty after clear
        empty_result = clean_container.get("lifecycle_service")
        assert empty_result.is_failure is True
