"""Integration service fixtures and lifecycle cases for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar, override

from typing_extensions import TypeForm

from tests import c, m, p, r, s, tm, u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextUserServiceEntity(m.BaseModel):
    """Test user entity model."""

    unique_id: Annotated[str, m.Field(description="Unique user identifier")]
    name: Annotated[str, m.Field(description="User display name")]
    email: Annotated[str, m.Field(description="User email address")]
    active: Annotated[bool, m.Field(description="Whether user is active")] = True


class TestsFlextUserQueryService(s[bool]):
    """Real user query service using ``s``."""

    _users: MutableMapping[str, TestsFlextUserServiceEntity] = m.PrivateAttr(
        default_factory=u.empty(
            TypeForm(MutableMapping[str, TestsFlextUserServiceEntity]),
        ),
    )
    _should_fail: bool = m.PrivateAttr(default_factory=lambda: False)
    _call_count: int = m.PrivateAttr(default_factory=lambda: 0)

    @override
    def execute(self) -> p.Result[bool]:
        """Return the service availability.

        Returns:
            The service availability.

        """
        if self._should_fail:
            return r[bool].fail("User service unavailable")
        return r[bool].ok(value=True)

    def fetch_user(self, user_id: str) -> p.Result[TestsFlextUserServiceEntity]:
        """Fetch user by ID.

        Returns:
            The resulting ``p.Result[TestsFlextUserServiceEntity]``.

        """
        self._call_count += 1
        if self._should_fail:
            return r[TestsFlextUserServiceEntity].fail("User service unavailable")
        if user_id in self._users:
            return r[TestsFlextUserServiceEntity].ok(self._users[user_id])
        default_user = TestsFlextUserServiceEntity(
            unique_id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
            active=True,
        )
        return r[TestsFlextUserServiceEntity].ok(default_user)

    def apply_user_data(self, user_id: str, user: TestsFlextUserServiceEntity) -> None:
        """Apply user data for testing."""
        self._users[user_id] = user

    def configure_failure_mode(self, *, should_fail: bool) -> None:
        """Configure failure mode for testing."""
        self._should_fail = should_fail

    @property
    def call_count(self) -> int:
        """The call count."""
        return self._call_count


class TestsFlextNotificationService(s[str]):
    """Real notification service using ``s``."""

    _sent_notifications: MutableSequence[str] = m.PrivateAttr(
        default_factory=u.empty(TypeForm(MutableSequence[str])),
    )
    _call_count: int = m.PrivateAttr(default_factory=lambda: 0)
    _should_fail: bool = m.PrivateAttr(default_factory=lambda: False)

    @override
    def execute(self) -> p.Result[str]:
        """Execute notification service.

        Returns:
            The resulting ``p.Result[str]``.

        """
        if self._should_fail:
            return r[str].fail("Notification service unavailable")
        return r[str].ok("sent")

    def send(self, email: str) -> p.Result[str]:
        """Send notification.

        Returns:
            The resulting ``p.Result[str]``.

        """
        self._call_count += 1
        if self._should_fail:
            return r[str].fail("Notification service unavailable")
        self._sent_notifications.append(email)
        return r[str].ok("sent")

    def configure_failure_mode(self, *, should_fail: bool) -> None:
        """Configure failure mode for testing."""
        self._should_fail = should_fail

    @property
    def sent_notifications(self) -> Sequence[str]:
        """The sent notifications."""
        return list(self._sent_notifications)

    @property
    def call_count(self) -> int:
        """The call count."""
        return self._call_count


class TestsFlextServiceConfig(m.Value):
    """Service configuration model with required fields."""

    name: Annotated[str, m.Field(description="Service name")]
    version: Annotated[str, m.Field(description="Service version")]
    temp_dir: Annotated[str | None, m.Field(description="Temporary directory path")] = (
        None
    )


class TestsFlextLifecycleService(s[str]):
    """Real lifecycle service using ``s`` with settings model."""

    _initialized: bool = m.PrivateAttr(default_factory=lambda: False)
    _service_config: TestsFlextServiceConfig | None = m.PrivateAttr(
        default_factory=lambda: None,
    )
    _shutdown_called: bool = m.PrivateAttr(default_factory=lambda: False)
    _should_fail_init: bool = m.PrivateAttr(default_factory=lambda: False)
    _should_fail_shutdown: bool = m.PrivateAttr(default_factory=lambda: False)

    @override
    def execute(self) -> p.Result[str]:
        """Execute lifecycle service.

        Returns:
            The resulting ``p.Result[str]``.

        """
        if self._initialized:
            return r[str].ok("initialized")
        return r[str].ok("ready")

    def initialize(self, settings: TestsFlextServiceConfig) -> p.Result[str]:
        """Initialize service with settings model.

        Returns:
            The resulting ``p.Result[str]``.

        """
        if self._should_fail_init:
            return r[str].fail("Initialization failed")
        self._initialized = True
        self._service_config = settings
        return r[str].ok("initialized")

    def health_check(self) -> bool:
        """Check service health.

        Returns:
            The resulting ``bool``.

        """
        return self._initialized and (not self._shutdown_called)

    def shutdown(self) -> p.Result[str]:
        """Shutdown service.

        Returns:
            The resulting ``p.Result[str]``.

        """
        if self._should_fail_shutdown:
            return r[str].fail("Shutdown failed")
        self._shutdown_called = True
        return r[str].ok("shutdown")

    def configure_failure_mode(
        self,
        *,
        fail_init: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        """Configure initialization and shutdown failure behavior."""
        self._should_fail_init = fail_init
        self._should_fail_shutdown = fail_shutdown

    @property
    def initialized(self) -> bool:
        """The initialization status."""
        return self._initialized

    @property
    def service_config(self) -> TestsFlextServiceConfig | None:
        """The service configuration."""
        return self._service_config

    @property
    def shutdown_called(self) -> bool:
        """The shutdown status."""
        return self._shutdown_called


class TestsFlextFlextServiceFixtures:
    """Expose previous nested service names through inheritance."""

    UserServiceEntity: ClassVar[type[TestsFlextUserServiceEntity]] = (
        TestsFlextUserServiceEntity
    )
    UserQueryService: ClassVar[type[TestsFlextUserQueryService]] = (
        TestsFlextUserQueryService
    )
    NotificationService: ClassVar[type[TestsFlextNotificationService]] = (
        TestsFlextNotificationService
    )
    ServiceConfig: ClassVar[type[TestsFlextServiceConfig]] = TestsFlextServiceConfig
    LifecycleService: ClassVar[type[TestsFlextLifecycleService]] = (
        TestsFlextLifecycleService
    )

    @staticmethod
    def _build_service_config(
        *,
        name: str,
        version: str,
        temp_dir: str,
    ) -> TestsFlextServiceConfig:
        return TestsFlextServiceConfig(name=name, version=version, temp_dir=temp_dir)


@c.Tests.MARK.usefixtures("clean_container")
class TestsFlextFlextServiceLifecycleCases(TestsFlextFlextServiceFixtures):
    """Exercise lifecycle service behavior with an isolated container."""

    @c.Tests.MARK.integration
    def test_lifecycle_service_execution(self) -> None:
        """Test lifecycle service execution."""
        lifecycle_service = self.LifecycleService()
        result = lifecycle_service.execute()
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="ready")
        tm.fail(lifecycle_service.initialized)

    @c.Tests.MARK.integration
    def test_lifecycle_service_initialization(self, temp_dir: Path) -> None:
        """Test lifecycle service initialization with settings."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        result = lifecycle_service.initialize(service_config)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="initialized")
        tm.ok(lifecycle_service.initialized)
        effective_config = tm.not_none(lifecycle_service.service_config)
        tm.that(effective_config.name, eq="test_service")

    @c.Tests.MARK.integration
    def test_lifecycle_service_health_check(self, temp_dir: Path) -> None:
        """Test lifecycle service health check."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        health_before = lifecycle_service.health_check()
        _ = lifecycle_service.initialize(service_config)
        health_after = lifecycle_service.health_check()
        tm.fail(health_before)
        tm.ok(health_after)

    @c.Tests.MARK.integration
    def test_lifecycle_service_shutdown(self, temp_dir: Path) -> None:
        """Test lifecycle service shutdown."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        _ = lifecycle_service.initialize(service_config)
        result = lifecycle_service.shutdown()
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="shutdown")
        tm.ok(lifecycle_service.shutdown_called)
        tm.fail(lifecycle_service.health_check())

    @c.Tests.MARK.integration
    def test_lifecycle_service_failure_modes(self, temp_dir: Path) -> None:
        """Test lifecycle service failure modes."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        lifecycle_service.configure_failure_mode(fail_init=True)
        init_result = lifecycle_service.initialize(service_config)
        tm.fail(init_result.success)
        tm.that(init_result.error, eq="Initialization failed")
        lifecycle_service.configure_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        tm.fail(shutdown_result.success)
        tm.that(shutdown_result.error, eq="Shutdown failed")


__all__: list[str] = ["TestsFlextFlextServiceLifecycleCases"]
