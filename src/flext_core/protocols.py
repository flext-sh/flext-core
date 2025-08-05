"""FLEXT Core Protocols - Semantic Pattern Protocol Contracts (v1.0.0).

This module implements Layer 1 of the FLEXT Pydantic Semantic Pattern as specified in
/home/marlonsc/flext/docs/FLEXT_PYDANTIC_SEMANTIC_PATTERN.md

Protocol contracts for interoperability across the entire FLEXT ecosystem.
These protocols define interfaces that models can implement for consistent
behavior patterns across all 33 projects.

Architecture Layers:
    Layer 0: Foundation (models.py) - FlextModel, FlextValue, FlextEntity, FlextConfig
    Layer 1: Protocols (this module) - ConnectionProtocol, AuthProtocol, ObservabilityProtocol
    Layer 2: Domain Extensions (subprojects) - FlextDataOracle, FlextSingerStream, etc.
    Layer 3: Composite Patterns (subprojects) - FlextPipelineConfig, FlextAppConfig, etc.

Design Principles:
    - Protocol-based contracts instead of deep inheritance
    - Composition over inheritance through protocol implementation
    - Type-safe contracts with comprehensive method signatures
    - Cross-project interoperability through shared protocols
    - Maximum 3 protocols to maintain simplicity

Quality Standards:
    - Python 3.13+ Protocol syntax
    - FlextResult integration for all operations
    - Comprehensive type annotations
    - Abstract method contracts for consistent implementation
    - Cross-language compatibility (Go bridge ready)

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# =============================================================================
# PROTOCOL CONTRACTS - Layer 1 protocols (≤3 protocols)
# =============================================================================


class ConnectionProtocol(Protocol):
    """Protocol for anything that connects to external systems.

    Provides consistent connection interface across all FLEXT projects
    for databases, APIs, message queues, and other external systems.

    Implementations:
        - FlextData.Oracle (flext-target-oracle)
        - FlextData.Postgres (flext-core config)
        - FlextData.Redis (flext-core config)
        - FlextAuth.LDAP (flext-ldap)
        - FlextObs.Metrics (flext-observability)

    Usage:
        class OracleConnection(FlextConfig, ConnectionProtocol):
            host: str
            port: int = 1521

            def test_connection(self) -> FlextResult[bool]:
                # Implementation specific logic
                return FlextResult.ok(True)

            def get_connection_string(self) -> str:
                return f"oracle://{self.host}:{self.port}"
    """

    def test_connection(self) -> FlextResult[bool]:
        """Test connection to external system.

        Returns:
            FlextResult[bool]: Success with True if connection works

        """
        ...

    def get_connection_string(self) -> str:
        """Get connection string for external system.

        Returns:
            str: Connection string (credentials may be masked)

        """
        ...

    def close_connection(self) -> FlextResult[None]:
        """Close connection to external system.

        Returns:
            FlextResult[None]: Success if connection closed properly

        """
        ...


class AuthProtocol(Protocol):
    """Protocol for authentication and authorization systems.

    Provides consistent authentication interface across all FLEXT projects
    for JWT, OAuth, LDAP, and other authentication mechanisms.

    Implementations:
        - FlextAuth.JWT (flext-auth)
        - FlextAuth.OAuth (flext-auth)
        - FlextAuth.LDAP (flext-ldap)
        - FlextAuth.APIKey (flext-api)

    Usage:
        class JWTAuth(FlextConfig, AuthProtocol):
            secret_key: SecretStr
            algorithm: str = "HS256"

            def authenticate(self, credentials: dict[str, object]) -> FlextResult[dict[str, object]]:
                # JWT validation logic
                return FlextResult.ok({"user_id": "123", "role": "REDACTED_LDAP_BIND_PASSWORD"})
    """

    def authenticate(
        self, credentials: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Authenticate user with provided credentials.

        Args:
            credentials: Authentication credentials (token, username/password, etc.)

        Returns:
            FlextResult[dict[str, object]]: Success with user info, failure with error

        """
        ...

    def authorize(
        self, user_info: dict[str, object], resource: str
    ) -> FlextResult[bool]:
        """Authorize user access to resource.

        Args:
            user_info: Authenticated user information
            resource: Resource being accessed

        Returns:
            FlextResult[bool]: Success with True if authorized

        """
        ...

    def refresh_token(self, refresh_token: str) -> FlextResult[dict[str, object]]:
        """Refresh authentication token.

        Args:
            refresh_token: Refresh token for new access token

        Returns:
            FlextResult[dict[str, object]]: Success with new token info

        """
        ...


class ObservabilityProtocol(Protocol):
    """Protocol for observability and monitoring systems.

    Provides consistent observability interface across all FLEXT projects
    for metrics, logging, tracing, and health checks.

    Implementations:
        - FlextObs.Metrics (flext-observability)
        - FlextObs.Trace (flext-observability)
        - FlextObs.Health (flext-observability)
        - FlextObs.Logger (flext-core logging)

    Usage:
        class MetricsCollector(FlextValue, ObservabilityProtocol):
            service_name: str
            metric_name: str

            def record_metric(self, name: str, value: float) -> FlextResult[None]:
                # Metrics recording logic
                return FlextResult.ok(None)
    """

    def record_metric(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> FlextResult[None]:
        """Record metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional metric tags

        Returns:
            FlextResult[None]: Success if metric recorded

        """
        ...

    def start_trace(self, operation_name: str) -> FlextResult[str]:
        """Start distributed trace.

        Args:
            operation_name: Name of operation being traced

        Returns:
            FlextResult[str]: Success with trace ID

        """
        ...

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform health check.

        Returns:
            FlextResult[dict[str, object]]: Success with health status

        """
        ...


# =============================================================================
# EXPORTS - Protocol contracts (≤3 items)
# =============================================================================

__all__ = [
    "AuthProtocol",
    "ConnectionProtocol",
    "ObservabilityProtocol",
]

# Total exports: 3 items (within ≤3 limit)
