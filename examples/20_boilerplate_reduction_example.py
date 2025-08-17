"""Boilerplate reduction using FlextDomainService.

Demonstrates reducing repetitive code patterns across FLEXT projects
using enhanced domain service patterns.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import cast

from flext_core import FlextDomainService, FlextResult, TAnyDict, get_logger

logger = get_logger(__name__)


# ==============================================================================
# BEFORE: Traditional service with lots of boilerplate
# ==============================================================================


class TraditionalOracleService:
    """Traditional service with lots of boilerplate code."""

    def __init__(self, host: str, port: int, username: str, password: str) -> None:
      self.host = host
      self.port = port
      self.username = username
      self.password = password
      self._connection = None
      self._initialized = False
      logger.info("Initializing TraditionalOracleService")

    def validate_config(self) -> FlextResult[None]:
      """Validate configuration."""
      try:
          if not self.host:
              return FlextResult.fail("Host is required")
          if not self.username:
              return FlextResult.fail("Username is required")
          if not self.password:
              return FlextResult.fail("Password is required")
          if self.port <= 0:
              return FlextResult.fail("Port must be positive")
          return FlextResult.ok(None)
      except (RuntimeError, ValueError, TypeError) as e:
          logger.exception("Configuration validation failed")
          return FlextResult.fail(f"Validation failed: {e}")

    def execute_query(self, query: str) -> FlextResult[TAnyDict]:
      """Execute database query."""
      try:
          # Validate configuration first
          config_result = self.validate_config()
          if config_result.is_failure:
              return FlextResult.fail(
                  config_result.error or "Configuration validation failed",
              )

          logger.info("Executing query: %s", query)

          # Simulate query execution
          result: TAnyDict = {
              "query": query,
              "timestamp": datetime.now(UTC).isoformat(),
          }

          logger.info("Query executed successfully")
          return FlextResult.ok(result)

      except (RuntimeError, ValueError, TypeError) as e:
          logger.exception("Query execution failed")
          return FlextResult.fail(f"Query failed: {e}")

    def get_service_info(self) -> dict[str, object]:
      """Get service information."""
      return {
          "service_type": "TraditionalOracleService",
          "host": self.host,
          "port": self.port,
          "initialized": self._initialized,
          "config_valid": self.validate_config().success,
      }


# ==============================================================================
# AFTER: Enhanced service using FlextDomainService
# ==============================================================================


class EnhancedOracleService(FlextDomainService[TAnyDict]):
    """Enhanced service using FlextDomainService - much less boilerplate."""

    host: str
    port: int
    username: str
    password: str

    def execute(self) -> FlextResult[TAnyDict]:
      """Execute the domain service operation."""
      result = self.execute_operation(
          "oracle_service_operation",
          self._perform_operation,
      )
      return result.map(lambda x: cast("TAnyDict", x))

    def validate_config(self) -> FlextResult[None]:
      """Validate service configuration - override from base class."""
      if not self.host:
          return FlextResult.fail("Host is required")
      if not self.username:
          return FlextResult.fail("Username is required")
      if not self.password:
          return FlextResult.fail("Password is required")
      if self.port <= 0:
          return FlextResult.fail("Port must be positive")
      return FlextResult.ok(None)

    def execute_query(self, query: str) -> FlextResult[TAnyDict]:
      """Execute database query using enhanced error handling."""
      result = self.execute_operation(
          "execute_query",
          self._execute_query_impl,
          query,
      )
      return result.map(lambda x: cast("TAnyDict", x))

    def _execute_query_impl(self, query: str) -> dict[str, str]:
      """Execute query implementation with proper error handling."""
      # Simulate query execution
      return {"query": query, "timestamp": datetime.now(UTC).isoformat()}

    def _perform_operation(self) -> dict[str, object]:
      """Perform operation and return status information."""
      return {"status": "ready", "host": self.host, "port": self.port}


# ==============================================================================
# COMPARISON: Before vs After
# ==============================================================================


def demonstrate_boilerplate_reduction() -> None:
    """Demonstrate the reduction in boilerplate code."""
    # Traditional approach
    traditional_service = TraditionalOracleService(
      host="localhost",
      port=1521,
      username="oracle",
      password=os.environ.get("ORACLE_PASSWORD", "change-me"),
    )

    # Manual validation and error handling
    config_result = traditional_service.validate_config()
    if config_result.is_failure:
      return

    query_result = traditional_service.execute_query("SELECT * FROM users")
    if query_result.success:
      pass

    enhanced_service = EnhancedOracleService(
      host="localhost",
      port=1521,
      username="oracle",
      password=os.environ.get("ORACLE_PASSWORD", "change-me"),
    )

    # Automatic validation and error handling
    operation_result = enhanced_service.execute()
    if operation_result.success:
      pass

    query_result = enhanced_service.execute_query("SELECT * FROM users")
    if query_result.success:
      pass


# ==============================================================================
# REAL-WORLD EXAMPLE: LDAP Service
# ==============================================================================


class LDAPConnectionService(FlextDomainService[TAnyDict]):
    """Real-world example: LDAP connection service."""

    host: str
    port: int
    bind_dn: str
    password: str
    base_dn: str
    use_ssl: bool = False

    def execute(self) -> FlextResult[TAnyDict]:
      """Execute the domain service operation."""
      result = self.execute_operation(
          "ldap_connection_test",
          self._test_connection,
      )
      return result.map(lambda x: cast("TAnyDict", x))

    def validate_config(self) -> FlextResult[None]:
      """Validate LDAP configuration."""
      if not self.host:
          return FlextResult.fail("LDAP host is required")
      if not self.bind_dn:
          return FlextResult.fail("Bind DN is required")
      if not self.password:
          return FlextResult.fail("Password is required")
      max_port = 65535  # Standard TCP port range
      if self.port <= 0 or self.port > max_port:
          return FlextResult.fail(f"Port must be between 1 and {max_port}")
      return FlextResult.ok(None)

    def search_users(self, filter_expr: str) -> FlextResult[TAnyDict]:
      """Search users in LDAP."""
      result = self.execute_operation(
          "ldap_search",
          self._search_users_impl,
          filter_expr,
      )
      return result.map(lambda x: cast("TAnyDict", x))

    def _test_connection(self) -> dict[str, object]:
      """Test LDAP connection."""
      return {
          "status": "connected",
          "host": self.host,
          "port": self.port,
          "base_dn": self.base_dn,
          "ssl_enabled": self.use_ssl,
      }

    def _search_users_impl(self, filter_expr: str) -> dict[str, object]:
      """Execute LDAP search with filter and return results."""
      return {
          "filter": filter_expr,
          "results": [
              {"cn": "John Doe", "mail": "john@example.com"},
              {"cn": "Jane Smith", "mail": "jane@example.com"},
          ],
          "count": 2,
      }


def demonstrate_ldap_service() -> None:
    """Demonstrate LDAP service using enhanced patterns."""
    ldap_service = LDAPConnectionService(
      host="ldap.example.com",
      port=389,
      bind_dn="cn=admin,dc=example,dc=com",
      password=os.environ.get("LDAP_PASSWORD", "change-me"),
      base_dn="dc=example,dc=com",
      use_ssl=False,
    )

    # Test connection
    connection_result = ldap_service.execute()
    if connection_result.success:
      pass

    # Search users
    search_result = ldap_service.search_users("(objectClass=person)")
    if search_result.success:
      pass


if __name__ == "__main__":
    demonstrate_boilerplate_reduction()
    demonstrate_ldap_service()
