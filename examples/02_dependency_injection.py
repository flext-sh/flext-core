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

from flext_core import (
    FlextContainer,
    FlextSettings,
    c,
    m,
    r,
    s,
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


# ═══════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION SERVICE
# ═══════════════════════════════════════════════════════════════════


class DatabaseService(m.Value):
    """Database service model used by DI example."""

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def connect(self) -> r[bool]:
        """Simulate database connection."""
        return r[bool].ok(True)

    def query(self, sql: str) -> r[m.ConfigMap]:
        """Simulate query execution."""
        if "INVALID" in sql:
            return r[m.ConfigMap].fail("invalid query")
        return r[m.ConfigMap].ok(m.ConfigMap(root={"rows": 1}))


class CacheService(m.Value):
    """Cache service model used by DI example."""

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def set(self, key: str, value: str) -> r[bool]:
        """Simulate cache write."""
        if not key:
            return r[bool].fail("missing key")
        if not value:
            return r[bool].fail("missing value")
        return r[bool].ok(True)


class EmailService(m.Value):
    """Email service model used by DI example."""

    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        """Simulate email send."""
        if not to or not subject or not body:
            return r[bool].fail("invalid email payload")
        return r[bool].ok(True)


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
            f"✅ Container cleared: {original_count} → {len(container.list_services())}",
        )

        # Error handling
        missing_result = container.get("non_existent")
        typed_db_result = container.get("database", type_cls=DatabaseService)
        invalid_query = (
            typed_db_result.value.query("INVALID QUERY")
            if typed_db_result.is_success
            else r[m.ConfigMap].fail("database service unavailable")
        )
        print(
            f"❌ Errors: Missing={missing_result.is_failure}, Invalid={invalid_query.is_failure}",
        )

    @staticmethod
    def _demonstrate_resolution(container: FlextContainer) -> None:
        """Show dependency resolution patterns."""
        print("\n=== Dependency Resolution ===")

        database_result = container.get("database", type_cls=DatabaseService)
        cache_result = container.get("cache", type_cls=CacheService)
        email_result = container.get("email", type_cls=EmailService)

        db_check = database_result.flat_map(lambda service: service.connect())
        cache_check = cache_result.flat_map(
            lambda service: service.set("test_key", "test_value")
        )
        email_check = email_result.flat_map(
            lambda service: service.send("test@example.com", "Test", "Hello")
        )

        print(f"✅ database: {db_check.is_success}")
        print(f"✅ cache: {cache_check.is_success}")
        print(f"✅ email: {email_check.is_success}")

    @staticmethod
    def _setup_container() -> FlextContainer:
        """Setup container with services."""
        container = FlextContainer()

        # Create services with centralized config mappings from t
        db_config: m.ConfigMap = m.ConfigMap(
            root={
                "driver": "sqlite",
                "url": "sqlite:///:memory:",
                "timeout": c.Network.DEFAULT_TIMEOUT,
            },
        )
        db_service = DatabaseService(config=db_config)
        db_service.status = c.Cqrs.CommonStatus.ACTIVE

        cache_config: m.ConfigMap = m.ConfigMap(
            root={
                "backend": "memory",
                "ttl": c.Defaults.DEFAULT_CACHE_TTL,
            },
        )
        cache_service = CacheService(config=cache_config)
        cache_service.status = c.Cqrs.CommonStatus.ACTIVE

        email_config: m.ConfigMap = m.ConfigMap(
            root={
                "host": "smtp.example.com",
                "port": 587,
            },
        )
        email_service = EmailService(config=email_config)
        email_service.status = c.Cqrs.CommonStatus.ACTIVE

        container.register("database", db_service)
        container.register("cache", cache_service)
        container.register("email", email_service)

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
            },
        )

        self.logger.info("Dependency injection demonstration completed")
        return r[m.ConfigMap].ok(result_data)

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
