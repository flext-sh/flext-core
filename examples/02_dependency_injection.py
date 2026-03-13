"""FlextContainer dependency injection demonstration.

Shows complete DI patterns with type-safe service registration and resolution.
Uses railway-oriented error handling and SOLID principles.

**Expected Output:**
- Container creation and configuration
- Service registration (singleton, transient, scoped)
- Dependency resolution with type safety
- Logger injection and resolution
- Configuration injection patterns
- Service composition examples

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import override

from flext_core import FlextContainer, FlextSettings, c, m, r, s, u


class DatabaseService(m.ArbitraryTypesModel):
    """Database service using centralized types and railway pattern."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def connect(self) -> r[bool]:
        """Connect to database with validation."""
        if self.status == c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].ok(value=True)
        url = str(self.config.get("url", ""))
        if not url:
            return r[bool].fail(c.Errors.CONFIGURATION_ERROR)
        timeout_text = str(self.config.get("timeout", 0))
        timeout = int(timeout_text) if timeout_text.isdigit() else 0
        timeout_validation = u.validate_positive(timeout)
        if timeout_validation.is_failure:
            return r[bool].fail(c.Errors.VALIDATION_ERROR)
        self.status = c.Cqrs.CommonStatus.ACTIVE
        return r[bool].ok(value=True)

    def query(self, sql: str) -> r[m.ConfigMap]:
        """Execute query with comprehensive validation using u."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[m.ConfigMap].fail(c.Errors.CONNECTION_ERROR)
        sql_keywords: tuple[str, ...] = (
            c.Cqrs.Action.GET,
            c.Cqrs.Action.CREATE,
            c.Cqrs.Action.UPDATE,
            c.Cqrs.Action.DELETE,
        )
        sql_pattern = f"\\b({'|'.join(sql_keywords)})\\b"
        if not u.validate_pattern(sql, sql_pattern).is_success:
            return r[m.ConfigMap].fail(c.Errors.VALIDATION_ERROR)
        result: m.ConfigMap = m.ConfigMap(
            root={
                "id": u.generate("ulid"),
                "name": "Alice",
                "email": "alice@example.com",
            }
        )
        return r[m.ConfigMap].ok(result)


class CacheService(m.ArbitraryTypesModel):
    """Cache service using centralized types."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def get(self, key: str) -> r[str | int]:
        """Get value from cache using railway pattern."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[str | int].fail(c.Errors.CONNECTION_ERROR)
        return u.validate_length(key, max_length=c.Validation.MAX_NAME_LENGTH).flat_map(
            lambda _: (
                r[str | int].fail(c.Errors.NOT_FOUND_ERROR)
                if key == "missing"
                else r[str | int].ok("cached_value")
            )
        )

    def set(self, key: str, value: str | int) -> r[bool]:
        """Set value in cache using railway pattern."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].fail(c.Errors.CONNECTION_ERROR)
        return (
            u
            .validate_length(key, max_length=c.Validation.MAX_NAME_LENGTH)
            .flat_map(
                lambda _: u.validate_length(
                    str(value), max_length=c.Validation.MAX_NAME_LENGTH
                )
            )
            .map(lambda _: True)
        )


class EmailService(m.ArbitraryTypesModel):
    """Email service using centralized types."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        """Send email with railway pattern validation."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].fail(c.Errors.CONNECTION_ERROR)
        validations = [
            u.validate_pattern(to, c.Platform.PATTERN_EMAIL, "email"),
            u.validate_length(
                subject, min_length=1, max_length=c.Validation.MAX_NAME_LENGTH
            ),
            u.validate_length(
                body, min_length=1, max_length=c.Defaults.MAX_MESSAGE_LENGTH
            ),
        ]
        return r.traverse(validations, lambda r: r).map(lambda _: True)


class DependencyInjectionService(s[m.ConfigMap]):
    """Service demonstrating FlextContainer dependency injection patterns."""

    @staticmethod
    def _demonstrate_advanced_patterns(container: FlextContainer) -> None:
        """Show advanced DI patterns."""
        print("\n=== Advanced DI Patterns ===")
        service_names = ["database", "cache", "email"]
        services_count = 0
        for name in service_names:
            result = container.get(name)
            if result.is_success:
                services_count += 1
        print(f"✅ Auto-wired services: {services_count}")
        print(f"✅ Singleton: {FlextContainer() is FlextContainer()}")
        original_count = len(container.list_services())
        container.clear_all()
        print(
            f"✅ Container cleared: {original_count} → {len(container.list_services())}"
        )
        missing_result = container.get("non_existent")
        typed_db_result = container.get("database", type_cls=DatabaseService)
        invalid_query = (
            typed_db_result.value.query("INVALID QUERY")
            if typed_db_result.is_success
            else r[m.ConfigMap].fail("database service unavailable")
        )
        print(
            f"❌ Errors: Missing={missing_result.is_failure}, Invalid={invalid_query.is_failure}"
        )

    @staticmethod
    def _demonstrate_resolution(container: FlextContainer) -> None:
        """Show dependency resolution patterns."""
        print("\n=== Dependency Resolution ===")
        database_result = container.get("database", type_cls=DatabaseService)
        cache_result = container.get("cache", type_cls=CacheService)
        email_result = container.get("email", type_cls=EmailService)
        db_check = database_result.flat_map(lambda svc: svc.connect())
        cache_check = cache_result.flat_map(
            lambda svc: svc.set("test_key", "test_value")
        )
        email_check = email_result.flat_map(
            lambda svc: svc.send("test@example.com", "Test", "Hello")
        )
        print(f"✅ database: {db_check.is_success}")
        print(f"✅ cache: {cache_check.is_success}")
        print(f"✅ email: {email_check.is_success}")

    @staticmethod
    def _setup_container() -> FlextContainer:
        """Setup container with services."""
        container = FlextContainer()
        db_config: m.ConfigMap = m.ConfigMap(
            root={
                "driver": "sqlite",
                "url": "sqlite:///:memory:",
                "timeout": c.Network.DEFAULT_TIMEOUT,
            }
        )
        db_service = DatabaseService(config=db_config)
        db_service.status = c.Cqrs.CommonStatus.ACTIVE
        cache_config: m.ConfigMap = m.ConfigMap(
            root={"backend": "memory", "ttl": c.Defaults.DEFAULT_CACHE_TTL}
        )
        cache_service = CacheService(config=cache_config)
        cache_service.status = c.Cqrs.CommonStatus.ACTIVE
        email_config: m.ConfigMap = m.ConfigMap(
            root={"host": "smtp.example.com", "port": 587}
        )
        email_service = EmailService(config=email_config)
        email_service.status = c.Cqrs.CommonStatus.ACTIVE
        _ = container.register("database", db_service)
        _ = container.register("cache", cache_service)
        _ = container.register("email", email_service)
        return container

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Execute DI demonstrations."""
        self.logger.info("Starting dependency injection demonstration")
        container = self._setup_container()
        self._demonstrate_registration(container)
        self._demonstrate_resolution(container)
        self._demonstrate_advanced_patterns(container)
        result_data: m.ConfigMap = m.ConfigMap(
            root={
                "patterns_demonstrated": 5,
                "services_registered": ["database", "cache", "email"],
                "di_patterns": [
                    "service_registration",
                    "dependency_resolution",
                    "auto_wiring",
                    "lifecycle_management",
                    "error_handling",
                ],
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        self.logger.info("Dependency injection demonstration completed")
        return r[m.ConfigMap].ok(result_data)

    def _demonstrate_registration(self, container: FlextContainer) -> None:
        """Show service registration patterns."""
        self.logger.info("=== Service Registration ===")
        services = [("database", "Database"), ("cache", "Cache"), ("email", "Email")]
        for service_type, name in services:
            has_service = container.has_service(service_type)
            self.logger.info("✅ %s registered: %s", name, has_service)
        self.logger.info(f"📋 Services: {container.list_services()}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT CONTAINER DEPENDENCY INJECTION DEMONSTRATION")
    print("Advanced Python 3.13+ patterns with centralized types")
    print("=" * 60)
    service = DependencyInjectionService()
    result = service.execute()
    if result.is_success:
        data = result.value
        print(f"✅ Completed {data['patterns_demonstrated']} DI patterns")
    else:
        print(f"❌ Failed: {result.error}")
    print("\n=== Global Config Pattern ===")
    global_config = FlextSettings.get_global()
    another_ref = FlextSettings.get_global()
    print(f"✅ Global singleton: {global_config is another_ref}")
    print("=" * 60)
    print("🎯 Advanced Patterns: PEP 695 types, collections.abc, StrEnum")
    print("🎯 Railway Pattern: Type-safe error handling throughout")
    print("🎯 DRY/SRP: Centralized constants, no code duplication")
    print("🎯 SOLID: Single responsibility, dependency injection")
    print("=" * 60)


if __name__ == "__main__":
    main()
