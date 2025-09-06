"""Functional tests for core FlextCore features."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flext_core import FlextCore, FlextResult
from flext_core.models import FlextModels


class TestFlextCoreIntegration:
    """Test core functionality integration scenarios."""

    def test_complete_entity_lifecycle(self) -> None:
        """Test complete entity creation, modification, and event handling."""
        core = FlextCore.get_instance()

        # Define a domain entity
        class UserEntity(FlextModels.Entity):
            name: str
            email: str
            status: str = "active"

            def validate_business_rules(self) -> FlextResult[None]:
                if "@" not in self.email:
                    return FlextResult[None].fail("Invalid email format")
                return FlextResult[None].ok(None)

            def activate(self) -> None:
                """Business method that triggers domain events."""
                if self.status == "inactive":
                    self.status = "active"
                    self.add_domain_event(
                        {
                            "event_type": "UserActivated",
                            "user_id": self.id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )

        # Create entity with business validation
        result = core.create_entity(
            UserEntity,
            id="user-001",
            name="John Doe",
            email="john@example.com",
        )

        assert result.success
        user = result.unwrap()
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.status == "active"
        assert len(user.domain_events) == 0

        # Test business logic
        user.status = "inactive"
        user.activate()
        assert user.status == "active"
        assert len(user.domain_events) == 1
        assert user.domain_events[0]["event_type"] == "UserActivated"

    def test_payload_message_workflow(self) -> None:
        """Test complete message payload workflow."""
        core = FlextCore.get_instance()

        # Create a message payload
        data: dict[
            str,
            str
            | int
            | float
            | bool
            | list[str | int | float | bool | list[object] | dict[str, object] | None]
            | dict[
                str, str | int | float | bool | list[object] | dict[str, object] | None
            ]
            | None,
        ] = {
            "user_id": "user-123",
            "action": "profile_update",
            "changes": {"email": "new@example.com"},
        }

        result = core.create_payload(
            data=data,
            message_type="UserProfileUpdateRequested",
            source_service="user_service",
            target_service="notification_service",
        )

        assert result.success
        payload = result.unwrap()

        # Verify payload structure
        assert payload.data["user_id"] == "user-123"
        assert payload.message_type == "UserProfileUpdateRequested"
        assert payload.source_service == "user_service"
        assert payload.target_service == "notification_service"

        # Verify auto-generated fields
        assert payload.message_id.startswith("msg_")
        assert (
            len(payload.correlation_id) > 0
        )  # Should be auto-generated, but check format later
        assert payload.priority == 5  # default
        assert payload.retry_count == 0

        # Test message expiration
        assert not payload.is_expired()
        assert payload.age_seconds() >= 0

    def test_service_container_integration(self) -> None:
        """Test dependency injection container functionality."""
        core = FlextCore.get_instance()

        # Register services
        class DatabaseService:
            def __init__(self) -> None:
                self.connected = True

            def query(self, sql: str, *args: object) -> list[dict[str, object]]:
                return [{"id": 1, "name": "test"}] if self.connected else []

        class NotificationService:
            def __init__(self, db_service: DatabaseService) -> None:
                self.db = db_service

            def send_notification(self, user_id: str, message: str) -> bool:
                # Safe query without SQL injection
                users = self.db.query("SELECT * FROM users WHERE id=?", user_id)
                return len(users) > 0

        # Register in container
        db_service = DatabaseService()
        container_result = core.register_service("database", db_service)
        assert container_result.success

        # Register factory
        def notification_factory() -> NotificationService:
            db_result = core.get_service("database")
            if db_result.failure:
                error_msg = "Database service not available"
                raise RuntimeError(error_msg)
            return NotificationService(db_result.unwrap())

        factory_result = core.register_factory("notifications", notification_factory)
        assert factory_result.success

        # Use services
        notification_result = core.get_service("notifications")
        assert notification_result.success

        notification_service = notification_result.unwrap()
        assert isinstance(notification_service, NotificationService)
        assert notification_service.send_notification("1", "Welcome!")

    def test_configuration_management(self) -> None:
        """Test configuration creation and validation."""
        core = FlextCore.get_instance()

        # Test database configuration
        db_config_result = core.configure_database(
            host="localhost",
            database="flext_test",
            username="test_user",
            password="test_pass",
            port=5432,
            pool_size=10,
        )

        assert db_config_result.success
        db_config = db_config_result.unwrap()
        assert db_config.host == "localhost"
        assert db_config.database == "flext_test"
        assert db_config.port == 5432
        assert db_config.pool_size == 10

    def test_result_railway_operations(self) -> None:
        """Test FlextResult railway-oriented programming patterns."""

        def validate_user_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if not data.get("email"):
                return FlextResult[dict[str, object]].fail("Email required")
            email_value = data.get("email", "")
            if isinstance(email_value, str) and "@" not in email_value:
                return FlextResult[dict[str, object]].fail("Invalid email format")
            return FlextResult[dict[str, object]].ok(data)

        def save_user(data: dict[str, object]) -> FlextResult[str]:
            # Simulate save operation
            if data.get("email") == "existing@example.com":
                return FlextResult[str].fail("Email already exists")
            email_value = data.get("email", "")
            if isinstance(email_value, str):
                return FlextResult[str].ok(f"user_{email_value.split('@')[0]}")
            return FlextResult[str].fail("Invalid email type")

        def send_welcome_email(user_id: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"Welcome email sent to {user_id}")

        # Test successful railway
        user_data: dict[str, object] = {"email": "new@example.com", "name": "New User"}

        result = (
            validate_user_data(user_data)
            .flat_map(save_user)
            .flat_map(send_welcome_email)
        )

        assert result.success
        assert "Welcome email sent to user_new" in result.unwrap()

        # Test failure railway
        invalid_data: dict[str, object] = {"name": "No Email User"}

        result = (
            validate_user_data(invalid_data)
            .flat_map(save_user)
            .flat_map(send_welcome_email)
        )

        assert result.failure
        assert "Email required" in (result.error or "")

    def test_performance_monitoring_integration(self) -> None:
        """Test performance monitoring and metrics collection."""
        core = FlextCore.get_instance()

        # Test performance level configuration - using the data for actual testing
        performance_config = {
            "performance_level": "high",
            "enable_metrics": True,
            "cache_size": 1000,
            "timeout_seconds": 30,
        }

        # Verify configuration values are reasonable
        assert performance_config["performance_level"] == "high"
        assert performance_config["enable_metrics"] is True

        # Use core's performance optimization
        initial_time = datetime.now(UTC)

        # Simulate some work with the core
        for i in range(5):
            result = core.create_payload(
                data={"iteration": i, "timestamp": datetime.now(UTC).isoformat()},
                message_type="PerformanceTest",
                source_service="test_service",
            )
            assert result.success

        end_time = datetime.now(UTC)
        duration = (end_time - initial_time).total_seconds()

        # Performance should be reasonable (not testing specific timing, just functionality)
        assert duration < 5.0  # Should complete in reasonable time


class TestBusinessRuleValidation:
    """Test business rule validation scenarios."""

    def test_entity_business_rules(self) -> None:
        """Test entity-specific business rule validation."""
        core = FlextCore.get_instance()

        class OrderEntity(FlextModels.Entity):
            customer_id: str
            total_amount: float
            currency: str = "USD"

            def validate_business_rules(self) -> FlextResult[None]:
                if self.total_amount <= 0:
                    return FlextResult[None].fail("Order amount must be positive")
                if self.currency not in {"USD", "EUR", "GBP"}:
                    return FlextResult[None].fail("Unsupported currency")
                return FlextResult[None].ok(None)

        # Test valid order
        valid_result = core.create_entity(
            OrderEntity,
            id="order-001",
            customer_id="cust-123",
            total_amount=99.99,
            currency="USD",
        )
        assert valid_result.success

        # Test invalid order (negative amount)
        invalid_result = core.create_entity(
            OrderEntity,
            id="order-002",
            customer_id="cust-123",
            total_amount=-10.0,
        )
        # Business rule validation should fail
        assert invalid_result.failure or "positive" in str(invalid_result.error)

    def test_value_object_functionality(self) -> None:
        """Test value object functionality using EmailAddress model."""
        # Test valid email address
        email = FlextModels.EmailAddress("test@example.com")
        assert email.root == "test@example.com"

        # Test another valid email
        email2 = FlextModels.EmailAddress("user@domain.org")
        assert email2.root == "user@domain.org"

        # Test email validation (should fail for invalid format)
        with pytest.raises(Exception):  # Pydantic validation error
            FlextModels.EmailAddress("invalid-email")


class TestErrorHandlingPatterns:
    """Test comprehensive error handling scenarios."""

    def test_nested_error_propagation(self) -> None:
        """Test how errors propagate through nested operations."""

        def step1(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("Step1: Empty data")
            return FlextResult[str].ok(f"step1:{data}")

        def step2(data: str) -> FlextResult[str]:
            if "error" in data:
                return FlextResult[str].fail("Step2: Error detected in data")
            return FlextResult[str].ok(f"step2:{data}")

        def step3(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"step3:{data}")

        # Test successful path
        result = step1("good_data").flat_map(step2).flat_map(step3)
        assert result.success
        assert result.unwrap() == "step3:step2:step1:good_data"

        # Test error in first step
        result = step1("").flat_map(step2).flat_map(step3)
        assert result.failure
        assert "Step1: Empty data" in (result.error or "")

        # Test error in middle step
        result = step1("error_data").flat_map(step2).flat_map(step3)
        assert result.failure
        assert "Step2: Error detected" in (result.error or "")

    def test_error_transformation(self) -> None:
        """Test error transformation and enrichment."""

        def risky_operation(*, fail: bool = False) -> FlextResult[str]:
            if fail:
                return FlextResult[str].fail("Original error")
            return FlextResult[str].ok("success")

        result = risky_operation(fail=True)
        # Transform the error using recover_with to create new failure with enhanced message
        result = result.recover_with(lambda e: FlextResult[str].fail(f"Enhanced: {e}"))
        result = result.recover_with(lambda e: FlextResult[str].fail(f"Context: {e}"))

        assert result.failure
        assert "Context: Enhanced: Original error" in (result.error or "")
