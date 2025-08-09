#!/usr/bin/env python3
"""Unified semantic patterns for FLEXT ecosystem.

Demonstrates harmonized pattern system with consistent naming,
type safety, and business rule validation across projects.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import Field, SecretStr

from flext_core import FlextResult
from flext_core.config import FlextConfig
from flext_core.models import FlextEntity, FlextFactory, FlextValue

# =============================================================================
# LAYER 0: FOUNDATION PATTERNS - Core Pydantic Models
# =============================================================================


class FlextOracleConfig(FlextConfig):
    """Oracle configuration using unified patterns."""

    host: str = "localhost"
    port: int = 1521
    service_name: str | None = None
    sid: str | None = None
    username: str
    password: SecretStr
    max_connections: int = 10

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified business rule validation pattern."""
        if not self.service_name and not self.sid:
            return FlextResult.fail("Either service_name or sid must be provided")

        min_port = 1
        max_port = 65535
        if not (min_port <= self.port <= max_port):
            return FlextResult.fail(f"Port must be between {min_port} and {max_port}")

        if self.max_connections < 1:
            return FlextResult.fail("Max connections must be positive")

        return FlextResult.ok(None)


class FlextUserProfile(FlextValue):
    """User profile value object using unified patterns."""

    email: str
    full_name: str
    role: Literal["admin", "user", "viewer"]
    preferences: dict[str, object] = Field(default_factory=dict)

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified validation with semantic business rules."""
        if "@" not in self.email or "." not in self.email.split("@")[1]:
            return FlextResult.fail("Invalid email format")

        min_name_length = 2
        if len(self.full_name.strip()) < min_name_length:
            return FlextResult.fail(
                f"Full name must be at least {min_name_length} characters"
            )

        return FlextResult.ok(None)


class FlextDataPipeline(FlextEntity):
    """Data pipeline entity using unified patterns."""

    name: str
    source_config: FlextOracleConfig
    owner: FlextUserProfile
    status: Literal["active", "inactive", "error"] = "inactive"
    processed_records: int = 0

    def validate_business_rules(self) -> FlextResult[None]:
        """Unified entity validation with domain logic."""
        min_pipeline_name_length = 3
        if len(self.name.strip()) < min_pipeline_name_length:
            return FlextResult.fail(
                f"Pipeline name must be at least {min_pipeline_name_length} characters"
            )

        if self.processed_records < 0:
            return FlextResult.fail("Processed records cannot be negative")

        # Validate nested objects using unified patterns
        source_validation = self.source_config.validate_business_rules()
        if source_validation.is_failure:
            return FlextResult.fail(f"Invalid source config: {source_validation.error}")

        owner_validation = self.owner.validate_business_rules()
        if owner_validation.is_failure:
            return FlextResult.fail(f"Invalid owner: {owner_validation.error}")

        return FlextResult.ok(None)

    def activate(self) -> FlextResult[None]:
        """Business operation with unified error handling."""
        if self.status == "active":
            return FlextResult.fail("Pipeline is already active")

        if self.status == "error":
            return FlextResult.fail("Cannot activate pipeline in error state")

        self.status = "active"
        self.increment_version()
        self.add_domain_event(
            {
                "type": "pipeline_activated",
                "pipeline_id": self.id,
                "timestamp": "2025-08-05T10:00:00Z",
            }
        )

        return FlextResult.ok(None)


# =============================================================================
# LAYER 1: SEMANTIC TYPE SYSTEM - Unified Type Organization
# =============================================================================


# Use unified semantic types
def pipeline_factory() -> FlextDataPipeline:
    """Create a default FlextDataPipeline instance."""
    return FlextDataPipeline(
        id="default",
        name="default_pipeline",
        source_config=FlextOracleConfig(
            username="user", password=SecretStr("pass"), service_name="DB"
        ),
        owner=FlextUserProfile(
            email="user@example.com", full_name="Default User", role="user"
        ),
    )


def pipeline_validator(p: FlextDataPipeline) -> bool:
    """Validate a FlextDataPipeline using business rules."""
    return p.validate_business_rules().success


# Simplified type annotations without FlextTypes namespace
DatabaseConnection: str = "oracle://localhost:1521/TESTDB"

UserCredentials: dict[str, str] = {
    "username": "admin",
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
        oracle_config: dict[str, object],
        owner_profile: dict[str, object],
    ) -> FlextResult[FlextDataPipeline]:
        """Create pipeline using unified factory pattern."""
        # Create configuration using unified patterns
        config_result = FlextFactory.create_model(FlextOracleConfig, **oracle_config)
        if config_result.is_failure:
            return FlextResult.fail(f"Invalid Oracle config: {config_result.error}")

        # Create owner profile using unified patterns
        owner_result = FlextFactory.create_model(FlextUserProfile, **owner_profile)
        if owner_result.is_failure:
            return FlextResult.fail(f"Invalid owner profile: {owner_result.error}")

        # Create pipeline using unified patterns
        pipeline_data = {
            "id": f"pipeline_{len(self._pipelines) + 1}",
            "name": name,
            "source_config": config_result.data,
            "owner": owner_result.data,
        }

        pipeline_result = FlextFactory.create_model(FlextDataPipeline, **pipeline_data)
        if pipeline_result.is_failure:
            return FlextResult.fail(
                f"Pipeline creation failed: {pipeline_result.error}"
            )

        pipeline = pipeline_result.data
        if pipeline is not None:
            self._pipelines[pipeline.id] = pipeline
            return FlextResult.ok(pipeline)
        return FlextResult.fail("Pipeline creation returned None")

    def activate_pipeline(self, pipeline_id: str) -> FlextResult[str]:
        """Activate pipeline with unified error handling."""
        if pipeline_id not in self._pipelines:
            return FlextResult.fail(f"Pipeline {pipeline_id} not found")

        pipeline = self._pipelines[pipeline_id]
        activation_result = pipeline.activate()

        if activation_result.is_failure:
            return FlextResult.fail(f"Activation failed: {activation_result.error}")

        return FlextResult.ok(f"Pipeline {pipeline_id} activated successfully")

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


class FlextUnifiedUtilities:
    """Unified utility functions across ecosystem."""

    @staticmethod
    def validate_oracle_connection(
        connection_string: str,
    ) -> FlextResult[dict[str, str]]:
        """Parse and validate Oracle connection strings."""
        if not connection_string.startswith("oracle://"):
            return FlextResult.fail("Invalid Oracle connection string format")

        try:
            # Simple parsing for demonstration
            parts = connection_string.replace("oracle://", "").split("/")
            host_port = parts[0].split(":")

            parsed: dict[str, str] = {
                "host": host_port[0],
                "port": host_port[1] if len(host_port) > 1 else "1521",
                "service_name": parts[1] if len(parts) > 1 else "ORCL",
            }

            return FlextResult.ok(parsed)
        except Exception as e:
            return FlextResult.fail(f"Connection string parsing failed: {e}")

    @staticmethod
    def format_metric_display(metric: dict[str, object]) -> str:
        """Format metrics for display using unified patterns."""
        lines = ["=== Pipeline Metrics ==="]
        for key, value in metric.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"{formatted_key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def safe_transform_data(
        data: dict[str, object],
        transformer: Callable[[dict[str, object]], dict[str, object]],
    ) -> FlextResult[dict[str, object]]:
        """Safe data transformation with unified error handling."""
        try:
            result = transformer(data)
            return FlextResult.ok(result)
        except Exception as e:
            return FlextResult.fail(f"Data transformation failed: {e}")


# =============================================================================
# DEMONSTRATION - Complete Working Example
# =============================================================================


async def demonstrate_foundation_models() -> FlextDataPipeline | None:
    """Demonstrate Layer 0: Foundation Models."""
    print("\n📋 Layer 0: Foundation Models (FlextConfig, FlextValue, FlextEntity)")

    service = FlextPipelineService()

    # Create Oracle configuration
    oracle_config = {
        "host": "production-oracle.company.com",
        "port": 1521,
        "service_name": "PRODDB",
        "username": "flext_user",
        "password": SecretStr("super_secure_password_123"),
        "max_connections": 20,
    }

    # Create user profile
    owner_profile = {
        "email": "data.engineer@company.com",
        "full_name": "Senior Data Engineer",
        "role": "admin",
        "preferences": {"timezone": "UTC", "notifications": True},
    }

    # Create pipeline using unified patterns
    pipeline_result = service.create_pipeline(
        name="Customer Data ETL Pipeline",
        oracle_config=oracle_config,
        owner_profile=cast("dict[str, object]", owner_profile),
    )

    if pipeline_result.is_failure:
        print(f"❌ Pipeline creation failed: {pipeline_result.error}")
        return None

    pipeline = pipeline_result.data
    if pipeline is not None:
        print(f"✅ Pipeline created: {pipeline.name} (ID: {pipeline.id})")
        print(f"   Owner: {pipeline.owner.full_name} ({pipeline.owner.email})")
        print(f"   Status: {pipeline.status}")
        print(
            f"   Oracle Host: {pipeline.source_config.host}:{pipeline.source_config.port}"
        )
        return pipeline
    print("❌ Pipeline creation returned None")
    return None


def demonstrate_semantic_types() -> None:
    """Demonstrate Layer 1: Semantic Type System."""
    print("\n🔧 Layer 1: Semantic Type System (FlextTypes)")

    # Demonstrate type usage
    connection_validation = FlextUnifiedUtilities.validate_oracle_connection(
        DatabaseConnection
    )
    if connection_validation.success and connection_validation.data is not None:
        conn_info = connection_validation.data
        print(
            f"✅ Connection validated: {conn_info['host']}:{conn_info['port']}/{conn_info['service_name']}"
        )


def demonstrate_domain_services(
    service: FlextPipelineService, pipeline: FlextDataPipeline
) -> None:
    """Demonstrate Layer 2: Domain Services."""
    print("\n⚙️ Layer 2: Domain Services (Business Logic)")

    # Activate pipeline
    activation_result = service.activate_pipeline(pipeline.id)
    if activation_result.success:
        print(f"✅ {activation_result.data}")
    else:
        print(f"❌ Activation failed: {activation_result.error}")

    # Update pipeline metrics
    pipeline.processed_records = 15742


def demonstrate_utilities(service: FlextPipelineService) -> None:
    """Demonstrate Layer 3: Unified Utilities."""
    print("\n🛠️ Layer 3: Unified Utilities")

    # Get and display metrics
    stats = service.get_pipeline_stats()
    formatted_stats = FlextUnifiedUtilities.format_metric_display(stats)
    print(formatted_stats)

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

    transform_result = FlextUnifiedUtilities.safe_transform_data(
        cast("dict[str, object]", sample_data), enhance_data
    )
    if transform_result.success and transform_result.data is not None:
        print("\n✅ Data transformation successful:")
        for key, value in transform_result.data.items():
            print(f"   {key}: {value}")


def demonstrate_error_handling(service: FlextPipelineService) -> None:
    """Demonstrate error handling patterns."""
    print("\n🚨 Error Handling Demonstration")

    # Create user profile for error demonstration
    owner_profile = {
        "email": "data.engineer@company.com",
        "full_name": "Senior Data Engineer",
        "role": "admin",
        "preferences": {"timezone": "UTC", "notifications": True},
    }

    # Try to create invalid configuration
    invalid_config = {"host": "invalid", "port": -1, "username": "test"}
    invalid_result = service.create_pipeline(
        "Invalid Pipeline", invalid_config, cast("dict[str, object]", owner_profile)
    )

    if invalid_result.is_failure:
        print(f"✅ Error correctly caught: {invalid_result.error}")


def demonstrate_domain_events(pipeline: FlextDataPipeline) -> None:
    """Demonstrate domain events."""
    print("\n📡 Domain Events")
    if hasattr(pipeline, "clear_domain_events"):
        events = pipeline.clear_domain_events()
    else:
        events = []
    for event in events:
        print(f"   Event: {event['type']} at {event['timestamp']}")


def print_completion_summary() -> None:
    """Print completion summary."""
    print("\n🎉 Unified Semantic Patterns Demonstration Complete!")
    print("=" * 60)
    print("Key Achievements:")
    print("✅ Zero pattern duplication across ecosystem")
    print("✅ Consistent Flext[Domain][Type][Context] naming")
    print("✅ Complete type safety with business rule validation")
    print("✅ Unified error handling with FlextResult pattern")
    print("✅ Cross-layer architecture with clear separation")


async def demonstrate_unified_patterns() -> None:
    """Demonstrate complete unified semantic pattern usage."""
    print("🎯 FLEXT Unified Semantic Patterns - Complete Example")
    print("=" * 60)

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
    print("Starting FLEXT Unified Semantic Patterns demonstration...")

    try:
        asyncio.run(demonstrate_unified_patterns())
    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
