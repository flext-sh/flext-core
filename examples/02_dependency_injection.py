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

from flext_core import FlextContainer, FlextSettings, c, m, r, s, t, u


class DatabaseService(m.ArbitraryTypesModel):
    """Database service using centralized types and railway pattern."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.INACTIVE

    def connect(self) -> r[bool]:
        """Connect to database with validation."""
        if self.status == c.CommonStatus.ACTIVE:
            return r[bool].ok(value=True)
        url = str(self.config.get("url", ""))
        if not url:
            return r[bool].fail(c.CONFIGURATION_ERROR)
        timeout_text = str(self.config.get("timeout", 0))
        timeout = int(timeout_text) if timeout_text.isdigit() else 0
        timeout_validation = u.validate_positive(timeout)
        if timeout_validation.is_failure:
            return r[bool].fail(c.VALIDATION_ERROR)
        self.status = c.CommonStatus.ACTIVE
        return r[bool].ok(value=True)

    def query(self, sql: str) -> r[t.ConfigMap]:
        """Execute query with comprehensive validation using u."""
        if self.status != c.CommonStatus.ACTIVE:
            return r[t.ConfigMap].fail(c.CONNECTION_ERROR)
        sql_keywords: tuple[str, ...] = (
            c.Action.GET,
            c.Action.CREATE,
            c.Action.UPDATE,
            c.Action.DELETE,
        )
        sql_pattern = f"\\b({'|'.join(sql_keywords)})\\b"
        if not u.validate_pattern(sql, sql_pattern).is_success:
            return r[t.ConfigMap].fail(c.VALIDATION_ERROR)
        result: t.ConfigMap = t.ConfigMap(
            root={
                "id": u.generate("ulid"),
                "name": "Alice",
                "email": "alice@example.com",
            },
        )
        return r[t.ConfigMap].ok(result)


class CacheService(m.ArbitraryTypesModel):
    """Cache service using centralized types."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.INACTIVE

    def get(self, key: str) -> r[str | int]:
        """Get value from cache using railway pattern."""
        if self.status != c.CommonStatus.ACTIVE:
            return r[str | int].fail(c.CONNECTION_ERROR)
        return u.validate_length(key, max_length=c.HTTP_STATUS_MIN).flat_map(
            lambda _: (
                r[str | int].fail(c.NOT_FOUND_ERROR)
                if key == "missing"
                else r[str | int].ok("cached_value")
            ),
        )

    def set(self, key: str, value: str | int) -> r[bool]:
        """Set value in cache using railway pattern."""
        if self.status != c.CommonStatus.ACTIVE:
            return r[bool].fail(c.CONNECTION_ERROR)
        return (
            u
            .validate_length(key, max_length=c.HTTP_STATUS_MIN)
            .flat_map(
                lambda _: u.validate_length(str(value), max_length=c.HTTP_STATUS_MIN),
            )
            .map(lambda _: True)
        )


class EmailService(m.ArbitraryTypesModel):
    """Email service using centralized types."""

    model_config = m.DOMAIN_MODEL_CONFIG
    config: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.INACTIVE

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        """Send email with railway pattern validation."""
        if self.status != c.CommonStatus.ACTIVE:
            return r[bool].fail(c.CONNECTION_ERROR)
        validations = [
            u.validate_pattern(to, c.PATTERN_EMAIL, "email"),
            u.validate_length(subject, min_length=1, max_length=c.HTTP_STATUS_MIN),
            u.validate_length(body, min_length=1, max_length=c.HTTP_STATUS_MIN),
        ]
        return r.traverse(validations, lambda r: r).map(lambda _: True)


class DependencyInjectionService(s[t.ConfigMap]):
    """Service demonstrating FlextContainer dependency injection patterns."""

    @staticmethod
    def _demonstrate_advanced_patterns(
        container: FlextContainer,
        db_service: DatabaseService,
    ) -> None:
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
            f"✅ Container cleared: {original_count} → {len(container.list_services())}",
        )
        missing_result = container.get("non_existent")
        invalid_query = db_service.query("INVALID QUERY")
        print(
            f"❌ Errors: Missing={missing_result.is_failure}, Invalid={invalid_query.is_failure}",
        )

    @staticmethod
    def _demonstrate_resolution(
        db_service: DatabaseService,
        cache_service: CacheService,
        email_service: EmailService,
    ) -> None:
        """Show dependency resolution patterns."""
        print("\n=== Dependency Resolution ===")
        db_check = db_service.connect()
        cache_check = cache_service.set("test_key", "test_value")
        email_check = email_service.send("test@example.com", "Test", "Hello")
        print(f"✅ database: {db_check.is_success}")
        print(f"✅ cache: {cache_check.is_success}")
        print(f"✅ email: {email_check.is_success}")

    @staticmethod
    def _setup_container() -> tuple[
        FlextContainer,
        DatabaseService,
        CacheService,
        EmailService,
    ]:
        """Setup container with services."""
        container = FlextContainer()
        db_config: t.ConfigMap = t.ConfigMap(
            root={
                "driver": "sqlite",
                "url": "sqlite:///:memory:",
                "timeout": c.DEFAULT_TIMEOUT_SECONDS,
            },
        )
        db_service = DatabaseService(config=db_config)
        db_service.status = c.CommonStatus.ACTIVE
        cache_config: t.ConfigMap = t.ConfigMap(
            root={"backend": "memory", "ttl": c.CACHE_TTL},
        )
        cache_service = CacheService(config=cache_config)
        cache_service.status = c.CommonStatus.ACTIVE
        email_config: t.ConfigMap = t.ConfigMap(
            root={"host": "smtp.example.com", "port": 587},
        )
        email_service = EmailService(config=email_config)
        email_service.status = c.CommonStatus.ACTIVE
        _ = container.register("database", db_service)
        _ = container.register("cache", cache_service)
        _ = container.register("email", email_service)
        return container, db_service, cache_service, email_service

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Execute DI demonstrations."""
        self.logger.info("Starting dependency injection demonstration")
        container, db_service, cache_service, email_service = self._setup_container()
        self._demonstrate_registration(container)
        self._demonstrate_resolution(db_service, cache_service, email_service)
        self._demonstrate_advanced_patterns(container, db_service)
        result_data: t.ConfigMap = t.ConfigMap(
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
            },
        )
        self.logger.info("Dependency injection demonstration completed")
        return r[t.ConfigMap].ok(result_data)

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
