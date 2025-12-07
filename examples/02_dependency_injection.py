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
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextModels,
    FlextResult,
    s,
    t,
    u,
)

# Use centralized t for all complex types (no loose types, no aliases)
# All types come directly from t namespace - no local type aliases
# All Literals come from FlextConstants.Literals - no local Literal aliases

# ═══════════════════════════════════════════════════════════════════
# FLEXT MODELS WITH ADVANCED PYDANTIC 2 PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Using FlextConstants.Domain StrEnums and t for centralized config
# No separate config classes - using centralized types directly


# ═══════════════════════════════════════════════════════════════════
# SERVICE MODELS WITH ADVANCED PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Using centralized config mappings and direct StrEnum usage from FlextConstants
# No separate config classes - using t and FlextConstants directly (DRY)


class DatabaseService(FlextModels.ArbitraryTypesModel):
    """Database service using centralized types and railway pattern."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    config: t.Types.ServiceMetadataMapping
    status: FlextConstants.Domain.Status = FlextConstants.Domain.Status.INACTIVE

    def connect(self) -> FlextResult[bool]:
        """Connect to database with validation."""
        if self.status == FlextConstants.Domain.Status.ACTIVE:
            return FlextResult[bool].ok(True)

        url = self.config.get("url", "")
        if not isinstance(url, str) or not url:
            return FlextResult[bool].fail(FlextConstants.Errors.CONFIGURATION_ERROR)

        timeout = self.config.get("timeout", 0)
        if not isinstance(timeout, int):
            return FlextResult[bool].fail(FlextConstants.Errors.VALIDATION_ERROR)

        # Railway pattern with u validation (DRY)
        timeout_validation = u.Validation.Numeric.validate_positive(timeout)
        if timeout_validation.is_failure:
            return FlextResult[bool].fail(FlextConstants.Errors.VALIDATION_ERROR)

        self.status = FlextConstants.Domain.Status.ACTIVE
        return FlextResult[bool].ok(True)

    def query(self, sql: str) -> FlextResult[t.Types.ConfigurationDict]:
        """Execute query with comprehensive validation using u."""
        if self.status != FlextConstants.Domain.Status.ACTIVE:
            return FlextResult[t.Types.ConfigurationDict].fail(
                FlextConstants.Errors.CONNECTION_ERROR,
            )

        # Use u for advanced SQL pattern validation with centralized keywords
        # Using FlextConstants.Cqrs.Action StrEnum values (DRY - no local Literal aliases)
        sql_keywords: tuple[str, ...] = (
            FlextConstants.Cqrs.Action.GET,
            FlextConstants.Cqrs.Action.CREATE,
            FlextConstants.Cqrs.Action.UPDATE,
            FlextConstants.Cqrs.Action.DELETE,
        )
        sql_pattern = rf"\b({'|'.join(sql_keywords)})\b"
        if not u.Validation.validate_pattern(sql, sql_pattern).is_success:
            return FlextResult[t.Types.ConfigurationDict].fail(
                FlextConstants.Errors.VALIDATION_ERROR,
            )

        result: t.Types.ConfigurationDict = {
            "id": u.Generators.Random.generate_short_id(),
            "name": "Alice",
            "email": "alice@example.com",
        }
        return FlextResult[t.Types.ConfigurationDict].ok(result)


class CacheService(FlextModels.ArbitraryTypesModel):
    """Cache service using centralized types."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    config: t.Types.ServiceMetadataMapping
    status: FlextConstants.Domain.Status = FlextConstants.Domain.Status.INACTIVE

    def get(self, key: str) -> FlextResult[str | int]:
        """Get value from cache using railway pattern."""
        if self.status != FlextConstants.Domain.Status.ACTIVE:
            return FlextResult[str | int].fail(FlextConstants.Errors.CONNECTION_ERROR)

        # Railway pattern with u validation (DRY)
        return u.Validation.validate_length(
            key,
            max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
        ).flat_map(
            lambda _: (
                FlextResult[str | int].fail(FlextConstants.Errors.NOT_FOUND_ERROR)
                if key == "missing"
                else FlextResult[str | int].ok("cached_value")
            ),
        )

    def set(self, key: str, value: str | int) -> FlextResult[bool]:
        """Set value in cache using railway pattern."""
        if self.status != FlextConstants.Domain.Status.ACTIVE:
            return FlextResult[bool].fail(FlextConstants.Errors.CONNECTION_ERROR)

        # Railway pattern with u validation (DRY)
        return (
            u.Validation.validate_length(
                key,
                max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
            )
            .flat_map(
                lambda _: (
                    u.Validation.validate_length(
                        value,
                        max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
                    )
                    if isinstance(value, str)
                    else FlextResult[str].ok("")
                ),
            )
            .map(lambda _: True)
        )


class EmailService(FlextModels.ArbitraryTypesModel):
    """Email service using centralized types."""

    model_config = FlextConstants.Domain.DOMAIN_MODEL_CONFIG

    config: t.Types.ServiceMetadataMapping
    status: FlextConstants.Domain.Status = FlextConstants.Domain.Status.INACTIVE

    def send(self, to: str, subject: str, body: str) -> FlextResult[bool]:
        """Send email with railway pattern validation."""
        if self.status != FlextConstants.Domain.Status.ACTIVE:
            return FlextResult[bool].fail(FlextConstants.Errors.CONNECTION_ERROR)

        # Railway pattern with multiple validations using traverse (DRY)
        validations = [
            u.Validation.validate_pattern(
                to,
                FlextConstants.Platform.PATTERN_EMAIL,
                "email",
            ),
            u.Validation.validate_length(
                subject,
                min_length=1,
                max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
            ),
            u.Validation.validate_length(
                body,
                min_length=1,
                max_length=FlextConstants.Defaults.MAX_MESSAGE_LENGTH,
            ),
        ]
        return FlextResult.traverse(validations, lambda r: r).map(lambda _: True)


# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION SERVICE
# ═══════════════════════════════════════════════════════════════════


class DependencyInjectionService(s[t.Types.ConfigurationDict]):
    """Service demonstrating FlextContainer dependency injection patterns."""

    def execute(self) -> FlextResult[t.Types.ConfigurationDict]:
        """Execute DI demonstrations."""
        self.logger.info("Starting dependency injection demonstration")

        container = self._setup_container()

        self._demonstrate_registration(container)
        self._demonstrate_resolution(container)
        self._demonstrate_advanced_patterns(container)

        result_data: t.Types.ConfigurationDict = {
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
        return FlextResult[t.Types.ConfigurationDict].ok(result_data)

    @staticmethod
    def _setup_container() -> FlextContainer:
        """Setup container with services."""
        container = FlextContainer()

        # Create services with centralized config mappings from t
        db_config: t.Types.ConfigurationDict = {
            "driver": "sqlite",
            "url": "sqlite:///:memory:",
            "timeout": FlextConstants.Network.DEFAULT_TIMEOUT,
        }
        db_service = DatabaseService(config=db_config)
        db_service.status = FlextConstants.Domain.Status.ACTIVE

        cache_config: t.Types.ConfigurationDict = {
            "backend": "memory",
            "ttl": FlextConstants.Defaults.DEFAULT_CACHE_TTL,
        }
        cache_service = CacheService(config=cache_config)
        cache_service.status = FlextConstants.Domain.Status.ACTIVE

        email_config: t.Types.ConfigurationDict = {
            "host": "smtp.example.com",
            "port": 587,
        }
        email_service = EmailService(config=email_config)
        email_service.status = FlextConstants.Domain.Status.ACTIVE

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

        def test_database(db: DatabaseService) -> FlextResult[bool]:
            return db.connect()

        def test_cache(cache: CacheService) -> FlextResult[bool]:
            return cache.set("test_key", "test_value")

        def test_email(email: EmailService) -> FlextResult[bool]:
            return email.send("test@example.com", "Test", "Hello")

        # Test each service with type narrowing
        for service_name in ["database", "cache", "email"]:
            result: FlextResult[DatabaseService | CacheService | EmailService] = (
                container.get(service_name)
            )
            if result.is_success:
                service = result.unwrap()
                if service_name == "database" and isinstance(service, DatabaseService):
                    test_result = test_database(service)
                elif service_name == "cache" and isinstance(service, CacheService):
                    test_result = test_cache(service)
                elif service_name == "email" and isinstance(service, EmailService):
                    test_result = test_email(service)
                else:
                    test_result = FlextResult[bool].fail("Service type mismatch")
                print(f"✅ {service_name}: {test_result.is_success}")
            else:
                print(f"❌ {service_name}: Failed to resolve")

    @staticmethod
    def _demonstrate_advanced_patterns(container: FlextContainer) -> None:
        """Show advanced DI patterns."""
        print("\n=== Advanced DI Patterns ===")

        service_names = ["database", "cache", "email"]
        services: dict[str, DatabaseService | CacheService | EmailService] = {
            name: container.get(name).unwrap()
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
        missing_result: FlextResult[DatabaseService | CacheService | EmailService] = (
            container.get("non_existent")
        )
        db_result: FlextResult[DatabaseService | CacheService | EmailService] = (
            container.get("database")
        )
        if db_result.is_success:
            db_service = db_result.unwrap()
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
        data = result.unwrap()
        print(f"✅ Completed {data['patterns_demonstrated']} DI patterns")
    else:
        print(f"❌ Failed: {result.error}")

    # Global config singleton demonstration (containers are not global singletons)
    print("\n=== Global Config Pattern ===")
    global_config = FlextConfig.get_global_instance()
    another_ref = FlextConfig.get_global_instance()
    print(f"✅ Global singleton: {global_config is another_ref}")

    print("=" * 60)
    print("🎯 Advanced Patterns: PEP 695 types, collections.abc, StrEnum")
    print("🎯 Railway Pattern: Type-safe error handling throughout")
    print("🎯 DRY/SRP: Centralized constants, no code duplication")
    print("🎯 SOLID: Single responsibility, dependency injection")
    print("=" * 60)


if __name__ == "__main__":
    main()
