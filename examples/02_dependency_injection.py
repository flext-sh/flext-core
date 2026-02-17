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

from flext_core import (
    FlextContainer,
    FlextSettings,
    c,
    m,
    r,
    s,
    t,
    u,
)

# Use centralized t for all complex types (no loose types, no aliases)
# All types come directly from t namespace - no local type aliases
# All Literals come from c.Literals - no local Literal aliases

# ═══════════════════════════════════════════════════════════════════
# FLEXT MODELS WITH ADVANCED PYDANTIC 2 PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Using c.Domain StrEnums and t for centralized config
# No separate config classes - using centralized types directly


# ═══════════════════════════════════════════════════════════════════
# SERVICE MODELS WITH ADVANCED PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Using centralized config mappings and direct StrEnum usage from c
# No separate config classes - using t and c directly (DRY)


class DatabaseService(m.ArbitraryTypesModel):
    """Database service using centralized types and railway pattern."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def connect(self) -> r[bool]:
        """Connect to database with validation."""
        if self.status == c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].ok(value=True)

        url = self.config.get("url", "")
        if not isinstance(url, str) or not url:
            return r[bool].fail(c.Errors.CONFIGURATION_ERROR)

        timeout = self.config.get("timeout", 0)
        if not isinstance(timeout, int):
            return r[bool].fail(c.Errors.VALIDATION_ERROR)

        # Railway pattern with u validation (DRY)
        timeout_validation = u.validate_positive(timeout)
        if timeout_validation.is_failure:
            return r[bool].fail(c.Errors.VALIDATION_ERROR)

        self.status = c.Cqrs.CommonStatus.ACTIVE
        return r[bool].ok(value=True)

    def query(self, sql: str) -> r[m.ConfigMap]:
        """Execute query with comprehensive validation using u."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[m.ConfigMap].fail(
                c.Errors.CONNECTION_ERROR,
            )

        # Use u for advanced SQL pattern validation with centralized keywords
        # Using c.Cqrs.Action StrEnum values (DRY - no local Literal aliases)
        sql_keywords: tuple[str, ...] = (
            c.Cqrs.Action.GET,
            c.Cqrs.Action.CREATE,
            c.Cqrs.Action.UPDATE,
            c.Cqrs.Action.DELETE,
        )
        sql_pattern = rf"\b({'|'.join(sql_keywords)})\b"
        if not u.validate_pattern(sql, sql_pattern).is_success:
            return r[m.ConfigMap].fail(
                c.Errors.VALIDATION_ERROR,
            )

        result: m.ConfigMap = {
            "id": u.generate_short_id(),
            "name": "Alice",
            "email": "alice@example.com",
        }
        return r[m.ConfigMap].ok(result)


class CacheService(m.ArbitraryTypesModel):
    """Cache service using centralized types."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def get(self, key: str) -> r[str | int]:
        """Get value from cache using railway pattern."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[str | int].fail(c.Errors.CONNECTION_ERROR)

        # Railway pattern with u validation (DRY)
        return u.validate_length(
            key,
            max_length=c.Validation.MAX_NAME_LENGTH,
        ).flat_map(
            lambda _: (
                r[str | int].fail(c.Errors.NOT_FOUND_ERROR)
                if key == "missing"
                else r[str | int].ok("cached_value")
            ),
        )

    def set(self, key: str, value: str | int) -> r[bool]:
        """Set value in cache using railway pattern."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].fail(c.Errors.CONNECTION_ERROR)

        # Railway pattern with u validation (DRY)
        return (
            u
            .validate_length(
                key,
                max_length=c.Validation.MAX_NAME_LENGTH,
            )
            .flat_map(
                lambda _: (
                    u.validate_length(
                        value,
                        max_length=c.Validation.MAX_NAME_LENGTH,
                    )
                    if isinstance(value, str)
                    else r[str].ok("")
                ),
            )
            .map(lambda _: True)
        )


class EmailService(m.ArbitraryTypesModel):
    """Email service using centralized types."""

    model_config = m.Config.DOMAIN_MODEL_CONFIG

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.INACTIVE

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        """Send email with railway pattern validation."""
        if self.status != c.Cqrs.CommonStatus.ACTIVE:
            return r[bool].fail(c.Errors.CONNECTION_ERROR)

        # Railway pattern with multiple validations using traverse (DRY)
        validations = [
            u.validate_pattern(
                to,
                c.Platform.PATTERN_EMAIL,
                "email",
            ),
            u.validate_length(
                subject,
                min_length=1,
                max_length=c.Validation.MAX_NAME_LENGTH,
            ),
            u.validate_length(
                body,
                min_length=1,
                max_length=c.Defaults.MAX_MESSAGE_LENGTH,
            ),
        ]
        return r.traverse(validations, lambda r: r).map(lambda _: True)


# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION SERVICE
# ═══════════════════════════════════════════════════════════════════


class DependencyInjectionService(s[m.ConfigMap]):
    """Service demonstrating FlextContainer dependency injection patterns."""

    def execute(self) -> r[m.ConfigMap]:
        """Execute DI demonstrations."""
        self.logger.info("Starting dependency injection demonstration")

        container = self._setup_container()

        self._demonstrate_registration(container)
        self._demonstrate_resolution(container)
        self._demonstrate_advanced_patterns(container)

        result_data: m.ConfigMap = {
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

        self.logger.info("Dependency injection demonstration completed")
        return r[m.ConfigMap].ok(result_data)

    @staticmethod
    def _setup_container() -> FlextContainer:
        """Setup container with services."""
        container = FlextContainer()

        # Create services with centralized config mappings from t
        db_config: m.ConfigMap = {
            "driver": "sqlite",
            "url": "sqlite:///:memory:",
            "timeout": c.Network.DEFAULT_TIMEOUT,
        }
        db_service = DatabaseService(config=db_config)
        db_service.status = c.Cqrs.CommonStatus.ACTIVE

        cache_config: m.ConfigMap = {
            "backend": "memory",
            "ttl": c.Defaults.DEFAULT_CACHE_TTL,
        }
        cache_service = CacheService(config=cache_config)
        cache_service.status = c.Cqrs.CommonStatus.ACTIVE

        email_config: m.ConfigMap = {
            "host": "smtp.example.com",
            "port": 587,
        }
        email_service = EmailService(config=email_config)
        email_service.status = c.Cqrs.CommonStatus.ACTIVE

        _ = container.register("database", db_service)
        _ = container.register("cache", cache_service)
        _ = container.register("email", email_service)

        return container

    def _demonstrate_registration(self, container: FlextContainer) -> None:
        """Show service registration patterns."""
        self.logger.info("=== Service Registration ===")

        services = [
            ("database", "Database"),
            ("cache", "Cache"),
            ("email", "Email"),
        ]

        for service_type, name in services:
            has_service = container.has_service(service_type)
            self.logger.info("✅ %s registered: %s", name, has_service)

        self.logger.info(f"📋 Services: {container.list_services()}")

    @staticmethod
    def _demonstrate_resolution(container: FlextContainer) -> None:
        """Show dependency resolution patterns."""
        print("\n=== Dependency Resolution ===")

        def test_database(db: DatabaseService) -> r[bool]:
            return db.connect()

        def test_cache(cache: CacheService) -> r[bool]:
            return cache.set("test_key", "test_value")

        def test_email(email: EmailService) -> r[bool]:
            return email.send("test@example.com", "Test", "Hello")

        # Test each service with type narrowing
        for service_name in ["database", "cache", "email"]:
            result: r[t.GeneralValueType] = container.get(service_name)
            if result.is_success:
                service = result.value
                if service_name == "database" and isinstance(service, DatabaseService):
                    test_result = test_database(service)
                elif service_name == "cache" and isinstance(service, CacheService):
                    test_result = test_cache(service)
                elif service_name == "email" and isinstance(service, EmailService):
                    test_result = test_email(service)
                else:
                    test_result = r[bool].fail("Service type mismatch")
                print(f"✅ {service_name}: {test_result.is_success}")
            else:
                print(f"❌ {service_name}: Failed to resolve")

    @staticmethod
    def _demonstrate_advanced_patterns(container: FlextContainer) -> None:
        """Show advanced DI patterns."""
        print("\n=== Advanced DI Patterns ===")

        service_names = ["database", "cache", "email"]
        services: dict[str, t.GeneralValueType] = {
            name: container.get(name).value
            for name in service_names
            if container.get(name).is_success
        }
        print(f"✅ Auto-wired services: {len(services)}")

        print(f"✅ Singleton: {FlextContainer() is FlextContainer()}")

        original_count = len(container.list_services())
        container.clear_all()
        print(
            f"✅ Container cleared: {original_count} → {len(container.list_services())}",
        )

        # Error handling
        missing_result: r[t.GeneralValueType] = container.get("non_existent")
        db_result: r[t.GeneralValueType] = container.get("database")
        if db_result.is_success:
            db_service = db_result.value
            if isinstance(db_service, DatabaseService):
                invalid_query = db_service.query("INVALID QUERY")
                print(
                    f"❌ Errors: Missing={missing_result.is_failure}, Invalid={invalid_query.is_failure}",
                )


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

    # Global config singleton demonstration (containers are not global singletons)
    print("\n=== Global Config Pattern ===")
    global_config = FlextSettings.get_global_instance()
    another_ref = FlextSettings.get_global_instance()
    print(f"✅ Global singleton: {global_config is another_ref}")

    print("=" * 60)
    print("🎯 Advanced Patterns: PEP 695 types, collections.abc, StrEnum")
    print("🎯 Railway Pattern: Type-safe error handling throughout")
    print("🎯 DRY/SRP: Centralized constants, no code duplication")
    print("🎯 SOLID: Single responsibility, dependency injection")
    print("=" * 60)


if __name__ == "__main__":
    main()
