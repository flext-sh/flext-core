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
"""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flext_core import FlextContainer, FlextResult


@pytest.mark.integration
class TestServiceIntegrationPatterns:
    """Demonstrates modern service integration testing patterns."""

    @pytest.mark.integration
    @pytest.mark.performance
    def test_service_pipeline_performance(
        self,
        configured_container: FlextContainer,  # noqa: ARG002 - provided by fixture for clarity
        mock_external_service: MagicMock,
        performance_threshold: dict[str, float],
        benchmark_data: dict[
            str,
            list[int] | dict[str, str] | dict[str, dict[str, dict[str, list[int]]]],
        ],
    ) -> None:
        """Test service pipeline meets performance requirements.

        Validates that integrated service pipelines execute within
        acceptable performance thresholds for enterprise workloads.

        Args:
            configured_container: Pre-configured container with services
            mock_external_service: Mock external service
            performance_threshold: Performance threshold configuration
            benchmark_data: Benchmark data sets

        """
        # Arrange
        large_dataset = benchmark_data["large_dataset"]
        mock_external_service.process.return_value = FlextResult.ok("processed")

        def process_pipeline(
            data: list[int]
            | dict[str, str]
            | dict[str, dict[str, dict[str, list[int]]]],
        ) -> FlextResult[str]:
            # Simulate service pipeline with realistic operations
            result = FlextResult.ok(data)
            return result.flat_map(mock_external_service.process).map(
                lambda _r: f"pipeline_result_{len(str(data))}",
            )

        # Act - Measure pipeline performance
        start_time = time.perf_counter()
        result = process_pipeline(large_dataset)
        execution_time = time.perf_counter() - start_time

        # Assert - Performance and functionality
        assert result.success is True
        assert execution_time < performance_threshold["serialization"]
        assert result.data is not None
        assert "pipeline_result" in result.data
        mock_external_service.process.assert_called_once_with(large_dataset)

    @pytest.mark.integration
    @pytest.mark.error_path
    def test_service_error_propagation(
        self,
        configured_container: FlextContainer,  # noqa: ARG002 - provided by fixture for clarity
        mock_external_service: MagicMock,
        error_context: dict[str, str | None],
    ) -> None:
        """Test error propagation across service boundaries.

        Validates that errors propagate correctly through service
        integration layers while maintaining proper error context.

        Args:
            configured_container: Pre-configured container with services
            mock_external_service: Mock external service
            error_context: Error context fixture

        """
        # Arrange - Configure mock to simulate service failure
        error_message = f"Service error: {error_context['error_code']}"
        mock_external_service.process.return_value = FlextResult.fail(error_message)

        def failing_pipeline(data: str) -> FlextResult[str]:
            return (
                FlextResult.ok(data)
                .flat_map(mock_external_service.process)
                .map(lambda _r: f"processed_{_r}")
            )

        # Act - Execute failing pipeline
        result = failing_pipeline("test_data")

        # Assert - Error propagation
        assert result.is_failure is True
        assert result.error is not None
        assert error_message in result.error
        mock_external_service.process.assert_called_once_with("test_data")

    @pytest.mark.integration
    @pytest.mark.architecture
    def test_dependency_injection_with_mocks(
        self,
        clean_container: FlextContainer,
        test_user_data: dict[str, str | int | bool | list[str]],
    ) -> None:
        """Test dependency injection patterns with mock services.

        Demonstrates proper dependency injection testing with mock services,
        service composition, and result validation.

        Args:
            clean_container: Isolated container fixture
            test_user_data: User data fixture

        """
        # Arrange - Create mock services
        mock_user_service = MagicMock()
        mock_notification_service = MagicMock()

        # Configure mock behavior
        mock_user_service.get_user.return_value = FlextResult.ok(test_user_data)
        mock_notification_service.send.return_value = FlextResult.ok("sent")

        # Register services in container
        clean_container.register("user_service", mock_user_service)
        clean_container.register("notification_service", mock_notification_service)

        def user_notification_workflow(user_id: str) -> FlextResult[str]:
            # Simulate service composition workflow
            user_service_result = clean_container.get("user_service")
            notification_service_result = clean_container.get("notification_service")

            if (
                not user_service_result.success
                or not notification_service_result.success
            ):
                return FlextResult.fail("Service unavailable")

            user_service = user_service_result.data
            notification_service = notification_service_result.data

            # Get user data first
            user_result = user_service.get_user(user_id)
            if not user_result.success:
                return FlextResult.fail("User not found")

            # Send notification
            return notification_service.send(user_result.data["email"])

        # Act - Execute workflow
        user_id = str(test_user_data["id"])
        result = user_notification_workflow(user_id)

        # Assert - Workflow success and service interaction
        assert result.success is True
        assert result.data == "sent"

        # Verify service calls
        mock_user_service.get_user.assert_called_once_with(user_id)
        mock_notification_service.send.assert_called_once_with(
            str(test_user_data["email"]),
        )

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
        # Arrange - Create lifecycle-aware mock service
        mock_service = MagicMock()
        mock_service.initialize.return_value = FlextResult.ok("initialized")
        mock_service.is_healthy.return_value = True
        mock_service.shutdown.return_value = FlextResult.ok("shutdown")

        service_config = {
            "name": "test_service",
            "version": "1.0.0",
            "temp_dir": str(temp_directory),
        }

        # Act - Register service with configuration
        registration_result = clean_container.register(
            "lifecycle_service",
            mock_service,
        )
        config_result = clean_container.register("service_config", service_config)

        # Assert - Registration success
        assert registration_result.success is True
        assert config_result.success is True

        # Act - Initialize service
        service_result = clean_container.get("lifecycle_service")
        config_fetch_result = clean_container.get("service_config")

        # Assert - Service retrieval
        assert service_result.success is True
        assert config_fetch_result.success is True

        service = service_result.data  # This is our MagicMock
        config = config_fetch_result.data

        # Act - Test service lifecycle
        init_result = service.initialize(config)
        health_status = service.is_healthy()
        shutdown_result = service.shutdown()

        # Assert - Lifecycle operations
        assert init_result.success is True
        assert health_status is True
        assert shutdown_result.success is True

        # Verify lifecycle method calls
        service.initialize.assert_called_once_with(config)
        service.is_healthy.assert_called_once()
        service.shutdown.assert_called_once()

        # Act - Clear container
        clean_container.clear()

        # Assert - Container is empty after clear
        empty_result = clean_container.get("lifecycle_service")
        assert empty_result.is_failure is True
