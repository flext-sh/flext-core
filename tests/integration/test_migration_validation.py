"""Integration tests for flext-core API functionality.

This test suite validates core API functionality and integration patterns
to ensure correct behavior across the ecosystem.

Tests verify:
- Core API access patterns work correctly
- r value access patterns work
- HTTP primitives work correctly
- No regressions across API surface
- Type safety maintained

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import override

from pydantic import BaseModel as PydanticBaseModel, PrivateAttr

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextSettings,
)
from tests import (
    c,
    e,
    h,
    m,
    p,
    r,
    s,
    t,
    u,
    x,
)


class TestMigrationValidation:
    def test_flext_result_value_access_pattern(self) -> None:
        """Verify .value access pattern works correctly."""

        def process_user(user_id: str) -> r[t.StrMapping]:
            if not user_id:
                return r[t.StrMapping].fail("User ID required")
            user_data: t.StrMapping = {"id": user_id, "name": "Alice"}
            return r[t.StrMapping].ok(user_data)

        result = process_user("user_123")
        assert result.success
        assert result.value == {"id": "user_123", "name": "Alice"}
        assert result.value["id"] == "user_123"
        assert result.value["name"] == "Alice"

    def test_flext_result_error_handling(self) -> None:
        """Verify error handling patterns continue working."""

        def validate_email(email: str) -> r[str]:
            if "@" not in email:
                return r[str].fail("Invalid email format")
            return r[str].ok(email)

        failure_result = validate_email("invalid")
        assert failure_result.failure
        assert failure_result.error and "Invalid email format" in failure_result.error
        success_result = validate_email("user@example.com")
        assert success_result.success
        assert success_result.value == "user@example.com"

    def test_container_global_instance(self) -> None:
        """Verify FlextContainer() continues working."""
        container = FlextContainer()
        assert container is not None
        container2 = FlextContainer()
        assert container is container2

    def test_container_registration_and_resolution(self) -> None:
        """Verify service registration and resolution."""
        container = FlextContainer()

        class TestService(PydanticBaseModel):
            name: str = "test"

        test_service = TestService()
        registration_result = container.register("test_migration_service", test_service)
        assert registration_result is container
        resolution_result = container.get("test_migration_service")
        assert resolution_result.success
        service = resolution_result.value
        assert isinstance(service, TestService)
        assert service.name == "test"

    def test_service_base_class_extension(self) -> None:
        """Verify s extension pattern continues working."""

        class UserService(s[None]):
            """User service extending s."""

            _logger: p.Logger = PrivateAttr(
                default_factory=lambda: u.fetch_logger(__name__),
            )

            @override
            def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
                super().model_post_init(__context)

            @override
            def execute(self, **_kwargs: t.Scalar) -> r[None]:
                """Execute method required by s abstract class."""
                return r[None].ok(None)

            def create_user(
                self,
                username: str,
                email: str,
            ) -> r[t.StrMapping]:
                """Create user with validation."""
                if not username or not email:
                    return r[t.StrMapping].fail("Username and email required")
                self._logger.info("Creating user", username=username)
                user_data = {"username": username, "email": email}
                return r[t.StrMapping].ok(user_data)

        service = UserService()
        result = service.create_user("alice", "alice@example.com")
        assert result.success
        assert result.value["username"] == "alice"

    def test_logger_structured_logging(self) -> None:
        """Verify the public logging DSL continues working."""
        logger = u.fetch_logger(__name__)
        assert logger is not None
        logger.info("Test message", test_key="test_value")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_all_stable_apis_accessible(self) -> None:
        """Verify all guaranteed stable APIs from API_STABILITY.md are accessible."""
        assert r is not None
        assert FlextContainer is not None
        assert m is not None
        assert s is not None
        assert callable(u.create_module_logger)
        assert callable(u.build_dispatcher)
        assert callable(u.build_registry)
        assert FlextSettings is not None
        assert c is not None
        assert FlextContext is not None
        assert e is not None
        assert h is not None
        assert x is not None
        assert p is not None
        assert t is not None
        assert u is not None

        assert isinstance(u.build_dispatcher(), p.Dispatcher)
        assert isinstance(u.build_registry(), p.Registry)

    def test_flext_result_all_methods(self) -> None:
        """Verify all r methods work correctly."""
        success = r[str].ok("test_value")
        assert success.success
        assert not success.failure
        assert success.error is None
        assert success.value == "test_value"
        assert success.value == "test_value"
        assert success.unwrap_or("default") == "test_value"
        failure: r[str] = r[str].fail("test_error")
        assert not failure.success
        assert failure.failure
        assert failure.error == "test_error"
        assert failure.unwrap_or("default") == "default"
        mapped = success.map(lambda x: x.upper())
        assert mapped.success
        assert mapped.value == "TEST_VALUE"

    def test_core_apis_work_correctly(self) -> None:
        """Verify core API patterns work correctly."""
        result = r[str].ok("test")
        assert result.value == "test"
        mapped = result.map(str.upper)
        assert mapped.value == "TEST"
        container = FlextContainer()
        assert container is not None

    def test_application_functionality_works(self) -> None:
        """Verify application functionality works correctly."""

        class ApplicationExample:
            """Example application using r and logging."""

            def __init__(self) -> None:
                super().__init__()
                self.logger = u.fetch_logger(__name__)
                self.container = FlextContainer()

            def process_data(
                self,
                data: t.StrMapping,
            ) -> r[t.RecursiveContainerMapping]:
                """Typical data processing method."""
                if not data:
                    return r[t.RecursiveContainerMapping].fail("Data required")
                self.logger.info("Processing data", size=len(data))
                processed: t.RecursiveContainerMapping = {
                    "original": str(data),
                    "processed": True,
                }
                return r[t.RecursiveContainerMapping].ok(processed)

        app = ApplicationExample()
        result = app.process_data({"key": "value"})
        assert result.success
        assert result.value["processed"] is True

    def test_all_core_apis_functional(self) -> None:
        """Verify all core APIs remain functional."""
        result = r[str].ok("test")
        assert result.success
        assert result.value == "test"
        container = FlextContainer()
        assert container is not None
        logger = u.fetch_logger(__name__)
        logger.info("Test")
        assert True
