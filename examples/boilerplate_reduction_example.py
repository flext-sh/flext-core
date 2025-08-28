#!/usr/bin/env python3
"""Boilerplate reduction using modern FLEXT patterns.

Demonstrates reducing repetitive code patterns across FLEXT projects
using enhanced service patterns and railway-oriented programming.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import override

from pydantic import BaseModel, Field

from flext_core import FlextConfig, FlextResult

# ==============================================================================
# BEFORE: Traditional service with lots of boilerplate
# ==============================================================================


class TraditionalDatabaseService:
    """Traditional service with lots of boilerplate code."""

    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.is_connected = False

    def connect(self) -> bool:
        """Connect to database - lots of manual error handling."""
        try:
            print(f"Connecting to {self.host}:{self.port}...")

            # Simulate connection validation
            if not self.host or self.port <= 0:
                print("ERROR: Invalid connection parameters")
                return False

            if not self.username or not self.password:
                print("ERROR: Invalid credentials")
                return False

            # Simulate successful connection
            self.is_connected = True
            print("✅ Connection established")
            return True

        except Exception as e:
            print(f"ERROR: Connection failed: {e}")
            self.is_connected = False
            return False

    def execute_query(self, query: str) -> Sequence[Mapping[str, object]] | None:
        """Execute query - manual error handling and validation."""
        try:
            if not self.is_connected:
                print("ERROR: Not connected to database")
                return None

            if not query or not query.strip():
                print("ERROR: Empty query")
                return None

            print(f"Executing query: {query[:50]}...")

            # Simulate query execution
            if "SELECT" not in query.upper():
                print("ERROR: Only SELECT queries allowed")
                return None

            # Simulate results
            results = [
                {"id": 1, "name": "John Doe", "created": datetime.now(UTC)},
                {"id": 2, "name": "Jane Smith", "created": datetime.now(UTC)},
            ]

            print(f"✅ Query executed, {len(results)} rows returned")
            return results

        except Exception as e:
            print(f"ERROR: Query execution failed: {e}")
            return None

    def disconnect(self) -> bool:
        """Disconnect from database."""
        try:
            if self.is_connected:
                print("Disconnecting...")
                self.is_connected = False
                print("✅ Disconnected successfully")
            return True
        except Exception as e:
            print(f"ERROR: Disconnect failed: {e}")
            return False


# ==============================================================================
# AFTER: Modern FLEXT service with reduced boilerplate
# ==============================================================================


class DatabaseConfig(FlextConfig):

    SSH_PORT: int = 22  # Valor mágico substituído por constante
    """Database configuration with validation."""

    host: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    connection_timeout: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate database configuration."""
        if self.host.lower() in {"localhost", "127.0.0.1"} and self.port == self.SSH_PORT:
            return FlextResult[None].fail("SSH port not allowed for database")

        return FlextResult[None].ok(None)


class DatabaseConnection(BaseModel):
    """Database connection state."""

    config: DatabaseConfig
    is_connected: bool = False
    connection_time: datetime | None = None

    def __str__(self) -> str:
        """String representation."""
        status = "Connected" if self.is_connected else "Disconnected"
        return f"DB({self.config.host}:{self.config.port}) - {status}"


class ModernDatabaseService:
    """Modern FLEXT service with reduced boilerplate."""

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize service with validated configuration."""
        self.connection = DatabaseConnection(config=config)

    def connect(self) -> FlextResult[None]:
        """Connect to database using railway-oriented programming."""
        if self.connection.is_connected:
            return FlextResult[None].ok(None)

        try:
            print(
                f"Connecting to {self.connection.config.host}:{self.connection.config.port}..."
            )

            # Simulate connection
            self.connection.is_connected = True
            self.connection.connection_time = datetime.now(UTC)

            print("✅ Connection established")
            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Connection failed: {e}")

    def execute_query(self, query: str) -> FlextResult[Sequence[Mapping[str, object]]]:
        """Execute query with comprehensive error handling."""
        # Validate connection
        if not self.connection.is_connected:
            return FlextResult[Sequence[Mapping[str, object]]].fail(
                "Not connected to database"
            )

        # Validate query
        if not query or not query.strip():
            return FlextResult[Sequence[Mapping[str, object]]].fail("Empty query")

        query = query.strip()

        if "SELECT" not in query.upper():
            return FlextResult[Sequence[Mapping[str, object]]].fail(
                "Only SELECT queries allowed"
            )

        try:
            print(f"Executing query: {query[:50]}...")

            # Simulate query execution
            results: Sequence[Mapping[str, object]] = [
                {"id": 1, "name": "John Doe", "created": datetime.now(UTC)},
                {"id": 2, "name": "Jane Smith", "created": datetime.now(UTC)},
            ]

            print(f"✅ Query executed, {len(results)} rows returned")
            return FlextResult[Sequence[Mapping[str, object]]].ok(results)

        except Exception as e:
            return FlextResult[Sequence[Mapping[str, object]]].fail(
                f"Query execution failed: {e}"
            )

    def disconnect(self) -> FlextResult[None]:
        """Disconnect from database."""
        try:
            if self.connection.is_connected:
                print("Disconnecting...")
                self.connection.is_connected = False
                self.connection.connection_time = None
                print("✅ Disconnected successfully")

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Disconnect failed: {e}")

    def get_status(self) -> FlextResult[str]:
        """Get connection status."""
        status_info = {
            "connected": self.connection.is_connected,
            "host": self.connection.config.host,
            "port": self.connection.config.port,
            "connection_time": self.connection.connection_time.isoformat()
            if self.connection.connection_time
            else None,
        }

        return FlextResult[str].ok(str(status_info))


# ==============================================================================
# DEMONSTRATION FUNCTIONS
# ==============================================================================


def demonstrate_traditional_approach() -> int:
    """Demonstrate traditional approach with lots of boilerplate."""
    print("\n" + "=" * 60)
    print("📊 Traditional Approach (Lots of Boilerplate)")
    print("=" * 60)

    # Create service with manual validation
    service = TraditionalDatabaseService(
        host="localhost", port=5432, username="REDACTED_LDAP_BIND_PASSWORD", password="password123"  # noqa: S106 (exemplo didático)
    )

    # Manual connection handling
    if not service.connect():
        print("❌ Failed to connect")
        return 1

    # Manual query execution
    results = service.execute_query("SELECT * FROM users LIMIT 2")
    if results is None:
        print("❌ Query failed")
        service.disconnect()
        return 1

    print(f"📋 Retrieved {len(results)} records")
    for record in results:
        print(f"   - {record}")

    # Manual disconnection
    if not service.disconnect():
        print("❌ Failed to disconnect")
        return 1

    print("✅ Traditional approach completed")
    return 0


def demonstrate_modern_approach() -> int:
    """Demonstrate modern FLEXT approach with reduced boilerplate."""
    print("\n" + "=" * 60)
    print("🚀 Modern FLEXT Approach (Reduced Boilerplate)")
    print("=" * 60)

    try:
        # Create configuration with validation
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            username="REDACTED_LDAP_BIND_PASSWORD",
            password="password123",  # noqa: S106 (exemplo didático)
            connection_timeout=30,
        )

        # Validate business rules
        validation_result = config.validate_business_rules()
        if not validation_result.success:
            print(f"❌ Configuration invalid: {validation_result.error}")
            return 1

        # Create service
        service = ModernDatabaseService(config)

        # Connect using railway pattern
        connect_result = service.connect()
        if not connect_result.success:
            print(f"❌ Connection failed: {connect_result.error}")
            return 1

        # Get status
        status_result = service.get_status()
        if status_result.success:
            print(f"📊 Status: {service.connection}")

        # Execute query using railway pattern
        query_result = service.execute_query("SELECT * FROM users LIMIT 2")
        if not query_result.success:
            print(f"❌ Query failed: {query_result.error}")
            service.disconnect()
            return 1

        results = query_result.value
        print(f"📋 Retrieved {len(results)} records")
        for record in results:
            print(f"   - {record}")

        # Disconnect
        disconnect_result = service.disconnect()
        if not disconnect_result.success:
            print(f"⚠️ Disconnect warning: {disconnect_result.error}")

        print("✅ Modern approach completed")
        return 0

    except Exception as e:
        print(f"❌ Modern approach failed: {e}")
        return 1


def demonstrate_error_handling_comparison() -> int:
    """Demonstrate error handling differences."""
    print("\n" + "=" * 60)
    print("🚫 Error Handling Comparison")
    print("=" * 60)

    print("\n🔴 Traditional Error Handling:")
    traditional_service = TraditionalDatabaseService("", 0, "", "")
    if not traditional_service.connect():
        print("   - Manual error checking required")
        print("   - Boolean return values lose error context")
        print("   - No structured error information")

    print("\n🟢 Modern Error Handling:")
    try:
        # Isto irá falhar na validação (exemplo didático, variável não usada removida)
        DatabaseConfig(host="", port=0, username="", password="")
        print("   - This shouldn't print (config validation should fail)")
    except Exception as e:
        print("   ✅ Validation caught at configuration level")
        print(f"   📋 Detailed error: {type(e).__name__}")

    # Valid config but simulated connection error
    config = DatabaseConfig(
        host="nonexistent.server.com", port=5432, username="test", password="test"  # noqa: S106 (exemplo didático)
    )

    service = ModernDatabaseService(config)

    # This would fail in real scenario
    query_result = service.execute_query("SELECT * FROM test")
    if not query_result.success:
        print(f"   ✅ Railway pattern propagates errors: {query_result.error}")

    return 0


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main() -> int:
    """Main demonstration function."""
    print("🎯 Boilerplate Reduction Demo")
    print("Comparing traditional vs modern FLEXT approaches")

    demonstrations = [
        ("Traditional Approach", demonstrate_traditional_approach),
        ("Modern FLEXT Approach", demonstrate_modern_approach),
        ("Error Handling Comparison", demonstrate_error_handling_comparison),
    ]

    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🎮 Running: {demo_name}")
            result = demo_func()
            if result != 0:
                print(f"❌ {demo_name} failed with exit code {result}")
                return result
        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            return 1

    print("\n🎉 All demonstrations completed successfully!")
    print("\n📈 Key Benefits of Modern Approach:")
    print("   ✅ Automatic validation with Pydantic")
    print("   ✅ Railway-oriented error handling")
    print("   ✅ Type safety with FlextResult")
    print("   ✅ Structured configuration")
    print("   ✅ Reduced boilerplate code")
    print("   ✅ Better error messages and context")

    return 0


if __name__ == "__main__":
    sys.exit(main())
