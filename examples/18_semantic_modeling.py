#!/usr/bin/env python3
"""Unified semantic patterns for FLEXT ecosystem.

Demonstrates harmonized pattern system with consistent naming,
type safety, and business rule validation across projects.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal, cast
from urllib.parse import urlparse

from pydantic import Field, SecretStr

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextModels,
    FlextResult,
    FlextTypes,
)

# =============================================================================
# LAYER 0: FOUNDATION PATTERNS - Core Pydantic Models
# =============================================================================


class DatabaseConfig(FlextConfig):
    """Example database configuration using unified patterns with FlextTypes.

    NOTE: This is a demonstration pattern only. Production database
    configurations should be in domain-specific libraries (e.g., flext-db-oracle).

    Uses FlextTypes.Core.* for maximum FLEXT integration and type safety.
    """

    host: FlextTypes.Core.String = "localhost"
    port: int = 5432  # PostgreSQL default for generic example
    database_name: FlextTypes.Core.String = "example_db"
    username: FlextTypes.Core.String
    password: SecretStr
    max_connections: int = 10

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified business rule validation pattern using FlextConstants."""
        if not self.database_name:
            return FlextResult[None].fail(FlextConstants.Errors.VALIDATION_ERROR)

        min_port: int = FlextConstants.Network.MIN_PORT or 1
        max_port: int = FlextConstants.Network.MAX_PORT or 65535
        if not (min_port <= self.port <= max_port):
            return FlextResult[None].fail(
                f"Port out of range: {self.port} not between {min_port} and {max_port}",
            )

        if self.max_connections < 1:
            return FlextResult[None].fail(
                f"Invalid max_connections: {self.max_connections} must be at least 1",
            )

        return FlextResult[None].ok(None)


class FlextUserProfile(FlextModels.Value):
    """User profile value object using unified patterns."""

    email: str
    full_name: str
    role: Literal["REDACTED_LDAP_BIND_PASSWORD", "user", "viewer"]
    preferences: dict[str, object] = Field(default_factory=dict)

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified validation with semantic business rules."""
        if "@" not in self.email or "." not in self.email.split("@")[1]:
            return FlextResult[None].fail("Invalid email format")

        min_name_length = 2
        if len(self.full_name.strip()) < min_name_length:
            return FlextResult[None].fail(
                f"Full name must be at least {min_name_length} characters",
            )

        return FlextResult[None].ok(None)


class FlextDataPipeline(FlextModels.Entity):
    """Data pipeline entity using unified patterns."""

    name: str
    source_config: DatabaseConfig
    owner: FlextUserProfile
    status: Literal["active", "inactive", "error"] = "inactive"
    processed_records: int = 0

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified entity validation with domain logic."""
        min_pipeline_name_length = 3
        if len(self.name.strip()) < min_pipeline_name_length:
            return FlextResult[None].fail(
                f"Pipeline name must be at least {min_pipeline_name_length} characters",
            )

        if self.processed_records < 0:
            return FlextResult[None].fail("Processed records cannot be negative")

        # Validate nested objects using unified patterns
        source_result = self.source_config.validate_business_rules()
        if not source_result.is_success:
            return FlextResult[None].fail(
                f"Invalid source config: {source_result.error}"
            )

        owner_result = self.owner.validate_business_rules()
        if not owner_result.is_success:
            return FlextResult[None].fail(f"Invalid owner: {owner_result.error}")

        return FlextResult[None].ok(None)

    def activate(self) -> FlextResult[None]:
        """Business operation with unified error handling."""
        if self.status == "active":
            return FlextResult[None].fail("Pipeline is already active")

        if self.status == "error":
            return FlextResult[None].fail("Cannot activate pipeline in error state")

        self.status = "active"
        self.increment_version()
        self.add_domain_event(
            {
                "type": "pipeline_activated",
                "pipeline_id": self.id,
                "timestamp": "2025-08-05T10:00:00Z",
            },
        )

        return FlextResult[None].ok(None)


# =============================================================================
# LAYER 1: SEMANTIC TYPE SYSTEM - Unified Type Organization
# =============================================================================


# Use unified semantic types
def pipeline_factory() -> FlextDataPipeline:
    """Create a default FlextDataPipeline instance."""
    return FlextDataPipeline(
        id="default",
        name="default_pipeline",
        source_config=DatabaseConfig(
            username="user",
            password=SecretStr("pass"),
            database_name="example_db",
        ),
        owner=FlextUserProfile(
            email="user@example.com",
            full_name="Default User",
            role="user",
        ),
    )


def pipeline_validator(p: FlextDataPipeline) -> bool:
    """Validate a FlextDataPipeline using business rules."""
    return p.validate_business_rules().success


# Simplified type annotations without FlextTypes namespace
DatabaseConnection: str = "oracle://localhost:1521/TESTDB"

UserCredentials: dict[str, str] = {
    "username": "REDACTED_LDAP_BIND_PASSWORD",
    "password": "secret123",
}

LoggerContext: dict[str, str] = {
    "service": "flext-unified-patterns",
    "component": "pipeline-manager",
    "version": "2.0.0",
}


# =============================================================================
# LAYER 2: DOMAIN SERVICES - Unified Service Patterns
# =============================================================================


class FlextPipelineService:
    """Pipeline service using unified patterns."""

    def __init__(self) -> None:
        self._pipelines: dict[str, FlextDataPipeline] = {}

    def create_pipeline(
        self,
        name: str,
        database_config: dict[str, object],
        owner_profile: dict[str, object],
    ) -> FlextResult[FlextDataPipeline]:
        """Create pipeline using unified factory pattern."""

        def _build_config() -> FlextResult[DatabaseConfig]:
            try:
                port_value = database_config.get("port", 5432)
                port_int = port_value if isinstance(port_value, int) else 5432
                instance = DatabaseConfig(
                    host=str(database_config.get("host", "localhost")),
                    port=port_int,
                    database_name=str(
                        database_config.get("database_name", "example_db")
                    ),
                    username=str(database_config.get("username", "user")),
                    password=SecretStr(
                        str(database_config.get("password", "password"))
                    ),
                )
                return FlextResult[DatabaseConfig].ok(instance)
            except Exception as e:
                return FlextResult[DatabaseConfig].fail(str(e))

        def _build_owner() -> FlextResult[FlextUserProfile]:
            try:
                instance = FlextUserProfile(
                    email=str(owner_profile.get("email", "REDACTED_LDAP_BIND_PASSWORD@example.com")),
                    full_name=str(owner_profile.get("full_name", "Pipeline Owner")),
                    role=cast(
                        "Literal['REDACTED_LDAP_BIND_PASSWORD', 'user', 'viewer']",
                        owner_profile.get("role", "REDACTED_LDAP_BIND_PASSWORD"),
                    ),
                    preferences=cast(
                        "dict[str, object]",
                        owner_profile.get("preferences", {}),
                    ),
                )
                return FlextResult[FlextUserProfile].ok(instance)
            except Exception as e:
                return FlextResult[FlextUserProfile].fail(str(e))

        def _build_pipeline(
            cfg: DatabaseConfig,
            owner: FlextUserProfile,
        ) -> FlextResult[FlextDataPipeline]:
            try:
                instance = FlextDataPipeline(
                    id=f"pipeline_{len(self._pipelines) + 1}",
                    name=name,
                    source_config=cfg,
                    owner=owner,
                )
                return FlextResult[FlextDataPipeline].ok(instance)
            except Exception as e:
                return FlextResult[FlextDataPipeline].fail(str(e))

        config_result = _build_config()
        if config_result.is_failure or config_result.value is None:
            return FlextResult[FlextDataPipeline].fail(
                f"Invalid Oracle config: {config_result.error or 'None'}",
            )

        owner_result = _build_owner()
        if owner_result.is_failure or owner_result.value is None:
            return FlextResult[FlextDataPipeline].fail(
                f"Invalid owner profile: {owner_result.error or 'None'}",
            )

        pipeline_result = _build_pipeline(config_result.value, owner_result.value)
        if pipeline_result.is_failure or pipeline_result.value is None:
            return FlextResult[FlextDataPipeline].fail(
                f"Pipeline creation failed: {pipeline_result.error or 'None'}",
            )

        pipeline = pipeline_result.value
        self._pipelines[str(pipeline.id)] = pipeline
        return FlextResult[FlextDataPipeline].ok(pipeline)

    def activate_pipeline(self, pipeline_id: str) -> FlextResult[str]:
        """Activate pipeline with unified error handling."""
        if pipeline_id not in self._pipelines:
            return FlextResult[str].fail(f"Pipeline {pipeline_id} not found")

        pipeline = self._pipelines[pipeline_id]
        activation_result = pipeline.activate()

        if activation_result.is_failure:
            return FlextResult[str].fail(
                f"Activation failed: {activation_result.error}"
            )

        return FlextResult[str].ok(f"Pipeline {pipeline_id} activated successfully")

    def get_pipeline_stats(self) -> dict[str, object]:
        """Get pipeline statistics using unified types."""
        stats: dict[str, object] = {
            "total_pipelines": len(self._pipelines),
            "active_pipelines": sum(
                1 for p in self._pipelines.values() if p.status == "active"
            ),
            "inactive_pipelines": sum(
                1 for p in self._pipelines.values() if p.status == "inactive"
            ),
            "error_pipelines": sum(
                1 for p in self._pipelines.values() if p.status == "error"
            ),
            "total_processed_records": sum(
                p.processed_records for p in self._pipelines.values()
            ),
        }
        return stats


# =============================================================================
# LAYER 3: UTILITIES - Unified Utility Patterns
# =============================================================================


# Use standard library urllib.parse for URL parsing instead of custom implementation


def validate_oracle_connection(connection_string: str) -> FlextResult[dict[str, str]]:
    """Parse and validate Oracle connection strings using standard library."""
    try:
        parsed = urlparse(connection_string)
        if parsed.scheme != "oracle":
            return FlextResult[dict[str, str]].fail(
                "Invalid Oracle connection string format - must use oracle:// scheme"
            )

        result: dict[str, str] = {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 1521),
            "service_name": parsed.path.lstrip("/") or "ORCL",
        }

        return FlextResult[dict[str, str]].ok(result)
    except Exception as e:
        return FlextResult[dict[str, str]].fail(
            f"Connection string parsing failed: {e}"
        )


def format_metric_display(metric: dict[str, object]) -> str:
    """Format metrics for display using unified patterns."""
    lines = ["=== Pipeline Metrics ==="]
    for key, value in metric.items():
        formatted_key = key.replace("_", " ").title()
        lines.append(f"{formatted_key}: {value}")
    return "\n".join(lines)


def safe_transform_data(
    data: dict[str, object],
    transformer: Callable[[dict[str, object]], dict[str, object]],
) -> FlextResult[dict[str, object]]:
    """Safe data transformation with unified error handling."""
    try:
        result = transformer(data)
        return FlextResult[dict[str, object]].ok(result)
    except Exception as e:
        return FlextResult[dict[str, object]].fail(f"Data transformation failed: {e}")


# =============================================================================
# DEMONSTRATION - Complete Working Example
# =============================================================================


async def demonstrate_foundation_models() -> FlextDataPipeline | None:
    """Demonstrate Layer 0: Foundation Models."""
    service = FlextPipelineService()

    # Create database configuration
    database_config = {
        "host": "production-db.company.com",
        "port": 5432,
        "database_name": "production_db",
        "username": "flext_user",
        "password": SecretStr("super_secure_password_123"),
        "max_connections": 20,
    }

    # Create user profile
    owner_profile: dict[str, object] = {
        "email": "data.engineer@company.com",
        "full_name": "Senior Data Engineer",
        "role": "REDACTED_LDAP_BIND_PASSWORD",
        "preferences": {"timezone": "UTC", "notifications": True},
    }

    # Create pipeline using unified patterns
    pipeline_result = service.create_pipeline(
        name="Customer Data ETL Pipeline",
        database_config=database_config,
        owner_profile=owner_profile,
    )

    if pipeline_result.is_failure:
        return None

    return pipeline_result.value


def demonstrate_semantic_types() -> None:
    """Demonstrate Layer 1: Semantic Type System."""
    # Demonstrate type usage
    connection_validation = validate_oracle_connection(
        DatabaseConnection,
    )
    if connection_validation.success:
        pass


def demonstrate_domain_services(
    service: FlextPipelineService,
    pipeline: FlextDataPipeline,
) -> None:
    """Demonstrate Layer 2: Domain Services."""
    # Activate pipeline
    activation_result = service.activate_pipeline(str(pipeline.id))
    if activation_result.success:
        pass

    # Update pipeline metrics
    pipeline.processed_records = 15742


def demonstrate_utilities(service: FlextPipelineService) -> None:
    """Demonstrate Layer 3: Unified Utilities."""
    # Get and display metrics
    stats = service.get_pipeline_stats()
    format_metric_display(stats)

    # Demonstrate data transformation
    sample_data = {"records": 1000, "errors": 5, "success_rate": 0.995}

    def enhance_data(data: dict[str, object]) -> dict[str, object]:
        enhanced = data.copy()
        success_rate = enhanced.get("success_rate")
        if isinstance(success_rate, (int, float)):
            enhanced["success_percentage"] = f"{float(success_rate) * 100:.1f}%"
        errors = enhanced.get("errors", 0)
        max_healthy_errors = 10
        enhanced["status"] = (
            "healthy"
            if isinstance(errors, (int, float)) and errors < max_healthy_errors
            else "warning"
        )
        return enhanced

    transform_result = safe_transform_data(
        cast("dict[str, object]", sample_data),
        enhance_data,
    )
    if transform_result.success:
        for _key, _value in transform_result.value.items():
            pass


def demonstrate_error_handling(service: FlextPipelineService) -> None:
    """Demonstrate error handling patterns."""
    # Create user profile for error demonstration
    owner_profile: dict[str, object] = {
        "email": "data.engineer@company.com",
        "full_name": "Senior Data Engineer",
        "role": "REDACTED_LDAP_BIND_PASSWORD",
        "preferences": {"timezone": "UTC", "notifications": True},
    }

    # Try to create invalid configuration
    invalid_config = {"host": "invalid", "port": -1, "username": "test"}
    invalid_result = service.create_pipeline(
        "Invalid Pipeline",
        invalid_config,
        owner_profile,
    )

    if invalid_result.is_failure:
        pass


def demonstrate_domain_events(pipeline: FlextDataPipeline) -> None:
    """Demonstrate domain events."""
    if hasattr(pipeline, "clear_domain_events"):
        events = pipeline.clear_domain_events()
    else:
        events = []
    for _event in events:
        pass


def print_completion_summary() -> None:
    """Print completion summary."""


async def demonstrate_unified_patterns() -> None:
    """Demonstrate complete unified semantic pattern usage."""
    # Layer 0: Foundation Models
    pipeline = await demonstrate_foundation_models()
    if pipeline is None:
        return

    service = FlextPipelineService()

    # Layer 1: Semantic Types
    demonstrate_semantic_types()

    # Layer 2: Domain Services
    demonstrate_domain_services(service, pipeline)

    # Layer 3: Utilities
    demonstrate_utilities(service)

    # Error handling demonstration
    demonstrate_error_handling(service)

    # Domain events demonstration
    demonstrate_domain_events(pipeline)

    # Print summary
    print_completion_summary()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    """Main execution function."""
    try:
        asyncio.run(demonstrate_unified_patterns())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise


if __name__ == "__main__":
    main()
