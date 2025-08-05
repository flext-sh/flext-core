#!/usr/bin/env python3
"""FLEXT Unified Semantic Patterns - Complete Working Example.

This example demonstrates the harmonized FLEXT semantic pattern system
that eliminates duplication and provides consistent architecture across
the entire FLEXT ecosystem.

Key Achievements:
- Unified pattern system (4 separate systems → 1 harmonized system)
- Zero duplication across 33+ projects
- Consistent Flext[Domain][Type][Context] naming
- Complete type safety and business rule validation

Architecture: FLEXT_UNIFIED_SEMANTIC_PATTERNS.md
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Literal

from pydantic import SecretStr

from flext_core import FlextResult
from flext_core.models import FlextConfig, FlextEntity, FlextFactory, FlextValue
from flext_core.semantic_types import FlextTypes


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
        
        if not (1 <= self.port <= 65535):
            return FlextResult.fail("Port must be between 1 and 65535")
        
        if self.max_connections < 1:
            return FlextResult.fail("Max connections must be positive")
        
        return FlextResult.ok(None)


class FlextUserProfile(FlextValue):
    """User profile value object using unified patterns."""
    
    email: str
    full_name: str
    role: Literal["REDACTED_LDAP_BIND_PASSWORD", "user", "viewer"]
    preferences: dict[str, object] = {}
    
    def validate_business_rules(self) -> FlextResult[None]:
        """Unified validation with semantic business rules."""
        if "@" not in self.email or "." not in self.email.split("@")[1]:
            return FlextResult.fail("Invalid email format")
        
        if len(self.full_name.strip()) < 2:
            return FlextResult.fail("Full name must be at least 2 characters")
        
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
        if len(self.name.strip()) < 3:
            return FlextResult.fail("Pipeline name must be at least 3 characters")
        
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
        self.add_domain_event({
            "type": "pipeline_activated",
            "pipeline_id": self.id,
            "timestamp": "2025-08-05T10:00:00Z"
        })
        
        return FlextResult.ok(None)


# =============================================================================
# LAYER 1: SEMANTIC TYPE SYSTEM - Unified Type Organization
# =============================================================================

# Use unified semantic types
PipelineFactory: FlextTypes.Core.Factory[FlextDataPipeline] = lambda: FlextDataPipeline(
    id="default",
    name="default_pipeline",
    source_config=FlextOracleConfig(username="user", password=SecretStr("pass"), service_name="DB"),
    owner=FlextUserProfile(email="user@example.com", full_name="Default User", role="user")
)

PipelineValidator: FlextTypes.Core.Validator[FlextDataPipeline] = lambda p: p.validate_business_rules().success

DatabaseConnection: FlextTypes.Data.Connection = "oracle://localhost:1521/TESTDB"

UserCredentials: FlextTypes.Auth.Credentials = {
    "username": "REDACTED_LDAP_BIND_PASSWORD",
    "password": "secret123"
}

LoggerContext: FlextTypes.Observability.LogContext = {
    "service": "flext-unified-patterns",
    "component": "pipeline-manager",
    "version": "2.0.0"
}


# =============================================================================
# LAYER 2: DOMAIN SERVICES - Unified Service Patterns
# =============================================================================

class FlextPipelineService:
    """Pipeline service using unified patterns."""
    
    def __init__(self):
        self._pipelines: dict[str, FlextDataPipeline] = {}
    
    def create_pipeline(
        self,
        name: str,
        oracle_config: dict[str, object],
        owner_profile: dict[str, object]
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
            "owner": owner_result.data
        }
        
        pipeline_result = FlextFactory.create_model(FlextDataPipeline, **pipeline_data)
        if pipeline_result.is_failure:
            return FlextResult.fail(f"Pipeline creation failed: {pipeline_result.error}")
        
        pipeline = pipeline_result.data
        self._pipelines[pipeline.id] = pipeline
        
        return FlextResult.ok(pipeline)
    
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
        stats: FlextTypes.Observability.Metric = {
            "total_pipelines": len(self._pipelines),
            "active_pipelines": sum(1 for p in self._pipelines.values() if p.status == "active"),
            "inactive_pipelines": sum(1 for p in self._pipelines.values() if p.status == "inactive"),
            "error_pipelines": sum(1 for p in self._pipelines.values() if p.status == "error"),
            "total_processed_records": sum(p.processed_records for p in self._pipelines.values())
        }
        return stats


# =============================================================================
# LAYER 3: UTILITIES - Unified Utility Patterns
# =============================================================================

class FlextUnifiedUtilities:
    """Unified utility functions across ecosystem."""
    
    @staticmethod
    def validate_oracle_connection(connection_string: str) -> FlextResult[dict[str, str]]:
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
                "service_name": parts[1] if len(parts) > 1 else "ORCL"
            }
            
            return FlextResult.ok(parsed)
        except Exception as e:
            return FlextResult.fail(f"Connection string parsing failed: {e}")
    
    @staticmethod
    def format_metric_display(metric: FlextTypes.Observability.Metric) -> str:
        """Format metrics for display using unified patterns."""
        lines = ["=== Pipeline Metrics ==="]
        for key, value in metric.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"{formatted_key}: {value}")
        return "\n".join(lines)
    
    @staticmethod
    def safe_transform_data(
        data: dict[str, object],
        transformer: FlextTypes.Core.Transformer[dict[str, object], dict[str, object]]
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

async def demonstrate_unified_patterns():
    """Demonstrate complete unified semantic pattern usage."""
    
    print("🎯 FLEXT Unified Semantic Patterns - Complete Example")
    print("=" * 60)
    
    # Layer 0: Foundation Models
    print("\n📋 Layer 0: Foundation Models (FlextConfig, FlextValue, FlextEntity)")
    
    service = FlextPipelineService()
    
    # Create Oracle configuration
    oracle_config = {
        "host": "production-oracle.company.com",
        "port": 1521,
        "service_name": "PRODDB",
        "username": "flext_user",
        "password": SecretStr("super_secure_password_123"),
        "max_connections": 20
    }
    
    # Create user profile
    owner_profile = {
        "email": "data.engineer@company.com",
        "full_name": "Senior Data Engineer",
        "role": "REDACTED_LDAP_BIND_PASSWORD",
        "preferences": {"timezone": "UTC", "notifications": True}
    }
    
    # Create pipeline using unified patterns
    pipeline_result = service.create_pipeline(
        name="Customer Data ETL Pipeline",
        oracle_config=oracle_config,
        owner_profile=owner_profile
    )
    
    if pipeline_result.is_failure:
        print(f"❌ Pipeline creation failed: {pipeline_result.error}")
        return
    
    pipeline = pipeline_result.data
    print(f"✅ Pipeline created: {pipeline.name} (ID: {pipeline.id})")
    print(f"   Owner: {pipeline.owner.full_name} ({pipeline.owner.email})")
    print(f"   Status: {pipeline.status}")
    print(f"   Oracle Host: {pipeline.source_config.host}:{pipeline.source_config.port}")
    
    # Layer 1: Semantic Types
    print("\n🔧 Layer 1: Semantic Type System (FlextTypes)")
    
    # Demonstrate type usage
    connection_validation = FlextUnifiedUtilities.validate_oracle_connection(DatabaseConnection)
    if connection_validation.success:
        conn_info = connection_validation.data
        print(f"✅ Connection validated: {conn_info['host']}:{conn_info['port']}/{conn_info['service_name']}")
    
    # Layer 2: Domain Services
    print("\n⚙️ Layer 2: Domain Services (Business Logic)")
    
    # Activate pipeline
    activation_result = service.activate_pipeline(pipeline.id)
    if activation_result.success:
        print(f"✅ {activation_result.data}")
    else:
        print(f"❌ Activation failed: {activation_result.error}")
    
    # Update pipeline metrics
    pipeline.processed_records = 15742
    
    # Layer 3: Utilities
    print("\n🛠️ Layer 3: Unified Utilities")
    
    # Get and display metrics
    stats = service.get_pipeline_stats()
    formatted_stats = FlextUnifiedUtilities.format_metric_display(stats)
    print(formatted_stats)
    
    # Demonstrate data transformation
    sample_data = {"records": 1000, "errors": 5, "success_rate": 0.995}
    
    def enhance_data(data: dict[str, object]) -> dict[str, object]:
        enhanced = data.copy()
        if isinstance(enhanced.get("success_rate"), (int, float)):
            enhanced["success_percentage"] = f"{float(enhanced['success_rate']) * 100:.1f}%"
        enhanced["status"] = "healthy" if enhanced.get("errors", 0) < 10 else "warning"
        return enhanced
    
    transform_result = FlextUnifiedUtilities.safe_transform_data(sample_data, enhance_data)
    if transform_result.success:
        print(f"\n✅ Data transformation successful:")
        for key, value in transform_result.data.items():
            print(f"   {key}: {value}")
    
    # Demonstrate error handling
    print("\n🚨 Error Handling Demonstration")
    
    # Try to create invalid configuration
    invalid_config = {"host": "invalid", "port": -1, "username": "test"}
    invalid_result = service.create_pipeline("Invalid Pipeline", invalid_config, owner_profile)
    
    if invalid_result.is_failure:
        print(f"✅ Error correctly caught: {invalid_result.error}")
    
    # Domain events demonstration
    print("\n📡 Domain Events")
    events = pipeline.clear_domain_events()
    for event in events:
        print(f"   Event: {event['type']} at {event['timestamp']}")
    
    print("\n🎉 Unified Semantic Patterns Demonstration Complete!")
    print("=" * 60)
    print("Key Achievements:")
    print("✅ Zero pattern duplication across ecosystem")
    print("✅ Consistent Flext[Domain][Type][Context] naming")
    print("✅ Complete type safety with business rule validation")
    print("✅ Unified error handling with FlextResult pattern")
    print("✅ Cross-layer architecture with clear separation")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
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