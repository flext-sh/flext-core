"""Service integration fixtures kept outside the collected test module."""

from __future__ import annotations

from collections.abc import (
    MutableMapping,
    MutableSequence,
)
from typing import Annotated, ClassVar, override

from flext_tests import r

from tests.base import s
from tests.models import m
from tests.protocols import p
from tests.typings import t


class TestsFlextUserServiceEntity(m.BaseModel):
    """Test user entity model."""

    unique_id: Annotated[str, m.Field(description="Unique user identifier")]
    name: Annotated[str, m.Field(description="User display name")]
    email: Annotated[str, m.Field(description="User email address")]
    active: Annotated[bool, m.Field(description="Whether user is active")] = True


class TestsFlextUserQueryService(s[bool]):
    """Real user query service using ``s``."""

    _users: MutableMapping[str, TestsFlextUserServiceEntity] = m.PrivateAttr(
        default_factory=lambda: dict[str, TestsFlextUserServiceEntity](),
    )
    _should_fail: bool = m.PrivateAttr(default_factory=lambda: False)
    _call_count: int = m.PrivateAttr(default_factory=lambda: 0)

    @override
    def execute(self) -> p.Result[bool]:
        """Return service availability."""
        if self._should_fail:
            return r[bool].fail("User service unavailable")
        return r[bool].ok(True)

    def fetch_user(self, user_id: str) -> p.Result[TestsFlextUserServiceEntity]:
        """Fetch user by ID."""
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

    def configure_failure_mode(self, should_fail: bool) -> None:
        """Configure failure mode for testing."""
        self._should_fail = should_fail

    @property
    def call_count(self) -> int:
        """Get call count."""
        return self._call_count


class TestsFlextNotificationService(s[str]):
    """Real notification service using ``s``."""

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
        """Send notification."""
        self._call_count += 1
        if self._should_fail:
            return r[str].fail("Notification service unavailable")
        self._sent_notifications.append(email)
        return r[str].ok("sent")

    def configure_failure_mode(self, should_fail: bool) -> None:
        """Configure failure mode for testing."""
        self._should_fail = should_fail

    @property
    def sent_notifications(self) -> t.StrSequence:
        """Get sent notifications."""
        return list(self._sent_notifications)

    @property
    def call_count(self) -> int:
        """Get call count."""
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
        default_factory=lambda: None
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

    def initialize(self, settings: TestsFlextServiceConfig) -> p.Result[str]:
        """Initialize service with settings model."""
        if self._should_fail_init:
            return r[str].fail("Initialization failed")
        self._initialized = True
        self._service_config = settings
        return r[str].ok("initialized")

    def health_check(self) -> bool:
        """Check service health."""
        return self._initialized and (not self._shutdown_called)

    def shutdown(self) -> p.Result[str]:
        """Shutdown service."""
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
        self._should_fail_init = fail_init
        self._should_fail_shutdown = fail_shutdown

    @property
    def initialized(self) -> bool:
        """Get initialization status."""
        return self._initialized

    @property
    def service_config(self) -> TestsFlextServiceConfig | None:
        """Get service configuration."""
        return self._service_config

    @property
    def shutdown_called(self) -> bool:
        """Get shutdown status."""
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
