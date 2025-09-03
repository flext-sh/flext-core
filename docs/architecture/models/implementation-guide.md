# FlextModels Implementation Guide

**Version**: 0.9.0  
**Target**: FLEXT Library Developers  
**Complexity**: Advanced  
**Estimated Time**: 2-4 hours per library

## 📋 Overview

This guide provides step-by-step instructions for implementing FlextModels domain modeling patterns in FLEXT ecosystem libraries. It covers domain-driven design principles, entity modeling, value objects, aggregate roots, domain events, and message-based architecture patterns.

## 🎯 Implementation Phases

### Phase 1: Domain Analysis & Design (1 hour)

### Phase 2: Model Implementation (2-3 hours)

### Phase 3: Business Rules & Events (1-2 hours)

### Phase 4: Testing & Integration (1 hour)

---

## 🔍 Phase 1: Domain Analysis & Design

### 1.1 Identify Domain Concepts

**Domain Modeling Types to Consider**:

- **Entities**: Objects with identity that change over time (User, Project, Order)
- **Value Objects**: Immutable objects compared by value (Email, Money, Address)
- **Aggregate Roots**: Consistency boundaries managing related entities
- **Domain Events**: Significant business occurrences
- **Domain Services**: Complex business operations spanning multiple aggregates
- **Specifications**: Business rule objects for complex queries

### 1.2 Current Domain Analysis Template

```python
# Analyze your current domain approach
class CurrentDomainApproach:
    """Document what you currently have"""

    # ❌ Identify scattered domain logic
    def create_user(self, name, email):
        # Business logic mixed with data access
        if not email or "@" not in email:
            return None  # Poor validation
        user = {"name": name, "email": email}
        # No domain events, no business rules
        return user

    # ❌ Identify missing validation
    def update_user_email(self, user_id, email):
        # No domain validation
        # No business rules enforcement
        # No event generation
        pass

    # ❌ Identify anemic models
    class User:
        def __init__(self, name, email):
            self.name = name
            self.email = email
        # No behavior, no business rules, just data
```

### 1.3 Domain Design Checklist

- [ ] **Entities identified**: Objects with identity and lifecycle
- [ ] **Value Objects mapped**: Immutable concepts compared by value
- [ ] **Aggregates defined**: Consistency boundaries identified
- [ ] **Domain Events cataloged**: Business-significant occurrences
- [ ] **Business Rules documented**: Validation and invariants
- [ ] **Message Patterns**: Inter-service communication needs
- [ ] **Factory Requirements**: Complex object creation scenarios

---

## 🏗️ Phase 2: Model Implementation

### 2.1 Library-Specific Models Structure

```python
from flext_core import FlextModels, FlextResult
from datetime import datetime, UTC
from pathlib import Path
from typing import override

class YourLibraryModels(FlextModels):
    """Domain models for your library extending FlextModels."""

    # =========================================================================
    # ENTITY MODELS - Objects with identity and lifecycle
    # =========================================================================

    class YourMainEntity(FlextModels.Entity):
        """Main entity for your domain with full business logic."""

        # Core domain properties
        name: str = Field(
            min_length=1,
            max_length=100,
            description="Entity name"
        )
        description: str | None = Field(
            default=None,
            max_length=500,
            description="Optional description"
        )
        status: str = Field(
            default="active",
            pattern="^(active|inactive|pending)$",
            description="Entity status"
        )

        # Domain-specific properties
        category: str = Field(
            min_length=1,
            description="Entity category"
        )
        configuration: dict[str, object] = Field(
            default_factory=dict,
            description="Entity configuration data"
        )

        # Metadata
        tags: list[str] = Field(
            default_factory=list,
            description="Entity tags for organization"
        )

        @override
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate entity-specific business rules."""
            try:
                # Validate name constraints
                if self.name.lower() in ["REDACTED_LDAP_BIND_PASSWORD", "root", "system"]:
                    return FlextResult[None].fail("Name cannot be a reserved word")

                # Validate status transitions
                if self.status == "inactive" and self.category == "critical":
                    return FlextResult[None].fail("Critical entities cannot be inactive")

                # Validate configuration
                if self.category == "production":
                    required_configs = ["endpoint", "timeout", "max_retries"]
                    missing_configs = [c for c in required_configs if c not in self.configuration]
                    if missing_configs:
                        return FlextResult[None].fail(f"Missing required configs: {', '.join(missing_configs)}")

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Business rule validation failed: {e}")

        def activate(self, activated_by: str) -> FlextResult[None]:
            """Activate entity with business logic."""
            try:
                if self.status == "active":
                    return FlextResult[None].fail("Entity is already active")

                # Validate activation prerequisites
                if not self.configuration:
                    return FlextResult[None].fail("Entity must have configuration before activation")

                # Update status
                old_status = self.status
                self.status = "active"
                self.updated_by = activated_by

                # Raise domain event
                self.add_domain_event({
                    "event_type": "EntityActivated",
                    "aggregate_id": self.id,
                    "entity_name": self.name,
                    "previous_status": old_status,
                    "activated_by": activated_by,
                    "timestamp": datetime.now(UTC).isoformat()
                })

                self.increment_version()
                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Failed to activate entity: {e}")

        def update_configuration(self, new_config: dict[str, object], updated_by: str) -> FlextResult[None]:
            """Update entity configuration with validation."""
            try:
                # Validate configuration structure
                config_validation = self._validate_configuration(new_config)
                if config_validation.is_failure:
                    return config_validation

                # Store old configuration for event
                old_config = self.configuration.copy()

                # Update configuration
                self.configuration.update(new_config)
                self.updated_by = updated_by

                # Raise domain event
                self.add_domain_event({
                    "event_type": "ConfigurationUpdated",
                    "aggregate_id": self.id,
                    "entity_name": self.name,
                    "config_changes": {
                        "added": {k: v for k, v in new_config.items() if k not in old_config},
                        "modified": {k: {"old": old_config.get(k), "new": v}
                                   for k, v in new_config.items()
                                   if k in old_config and old_config[k] != v},
                    },
                    "updated_by": updated_by,
                    "timestamp": datetime.now(UTC).isoformat()
                })

                self.increment_version()
                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Failed to update configuration: {e}")

        def _validate_configuration(self, config: dict[str, object]) -> FlextResult[None]:
            """Validate configuration data."""
            try:
                # Basic validation
                if not isinstance(config, dict):
                    return FlextResult[None].fail("Configuration must be a dictionary")

                # Category-specific validation
                if self.category == "database":
                    required_fields = ["host", "port", "database"]
                    missing_fields = [f for f in required_fields if f not in config]
                    if missing_fields:
                        return FlextResult[None].fail(f"Database config missing: {', '.join(missing_fields)}")

                elif self.category == "api":
                    if "endpoint" in config:
                        endpoint = config["endpoint"]
                        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
                            return FlextResult[None].fail("API endpoint must be a valid HTTP URL")

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Configuration validation failed: {e}")

    # =========================================================================
    # VALUE OBJECTS - Immutable objects compared by value
    # =========================================================================

    class ConnectionString(FlextModels.Value):
        """Connection string value object with validation."""

        protocol: str = Field(
            pattern="^(http|https|tcp|udp|ws|wss)$",
            description="Connection protocol"
        )
        host: str = Field(
            min_length=1,
            description="Host address"
        )
        port: int = Field(
            ge=1,
            le=65535,
            description="Port number"
        )
        path: str = Field(
            default="/",
            description="Connection path"
        )
        username: str | None = Field(
            default=None,
            description="Username for authentication"
        )
        password: str | None = Field(
            default=None,
            description="Password for authentication"
        )

        @computed_field
        @property
        def connection_url(self) -> str:
            """Generate complete connection URL."""
            auth_part = ""
            if self.username:
                auth_part = f"{self.username}"
                if self.password:
                    auth_part += f":{self.password}"
                auth_part += "@"

            return f"{self.protocol}://{auth_part}{self.host}:{self.port}{self.path}"

        @override
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate connection string business rules."""
            try:
                # Validate protocol and port combinations
                if self.protocol in ["http", "ws"] and self.port == 443:
                    return FlextResult[None].fail("HTTP/WS protocols should not use port 443")

                if self.protocol in ["https", "wss"] and self.port == 80:
                    return FlextResult[None].fail("HTTPS/WSS protocols should not use port 80")

                # Validate authentication requirements
                if self.username and not self.password:
                    return FlextResult[None].fail("Username requires password")

                # Validate host format
                if self.host.startswith(("http://", "https://")):
                    return FlextResult[None].fail("Host should not include protocol")

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Connection string validation failed: {e}")

    class ProcessingConfig(FlextModels.Value):
        """Processing configuration value object."""

        batch_size: int = Field(
            default=1000,
            ge=1,
            le=10000,
            description="Processing batch size"
        )
        timeout_seconds: int = Field(
            default=30,
            ge=1,
            le=3600,
            description="Processing timeout"
        )
        max_retries: int = Field(
            default=3,
            ge=0,
            le=10,
            description="Maximum retry attempts"
        )
        parallel_workers: int = Field(
            default=4,
            ge=1,
            le=32,
            description="Number of parallel workers"
        )

        @computed_field
        @property
        def total_capacity(self) -> int:
            """Calculate total processing capacity."""
            return self.batch_size * self.parallel_workers

        @override
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate processing configuration business rules."""
            try:
                # Validate capacity limits
                if self.total_capacity > 50000:
                    return FlextResult[None].fail("Total capacity cannot exceed 50,000 items")

                # Validate timeout for batch size
                if self.batch_size > 5000 and self.timeout_seconds < 60:
                    return FlextResult[None].fail("Large batches require timeout >= 60 seconds")

                # Validate worker constraints
                if self.parallel_workers > 16 and self.batch_size > 2000:
                    return FlextResult[None].fail("High worker count requires smaller batch sizes")

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Processing config validation failed: {e}")

    # =========================================================================
    # AGGREGATE ROOTS - Consistency boundaries with domain events
    # =========================================================================

    class YourServiceAggregate(FlextModels.AggregateRoot):
        """Service aggregate root managing related entities."""

        aggregate_type: str = "YourService"

        # Service properties
        service_name: str = Field(
            min_length=1,
            max_length=100,
            description="Service name"
        )
        service_type: str = Field(
            pattern="^(api|worker|scheduler|processor)$",
            description="Type of service"
        )

        # Service configuration
        connection: ConnectionString = Field(
            description="Service connection configuration"
        )
        processing: ProcessingConfig = Field(
            default_factory=ProcessingConfig,
            description="Processing configuration"
        )

        # Service state
        is_enabled: bool = Field(
            default=True,
            description="Service enabled status"
        )
        health_status: str = Field(
            default="unknown",
            pattern="^(healthy|unhealthy|degraded|unknown)$",
            description="Service health status"
        )

        # Related entities
        entities: list[str] = Field(
            default_factory=list,
            description="IDs of related entities"
        )

        @override
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate service aggregate business rules."""
            try:
                # Validate service configuration consistency
                if self.service_type == "api" and self.connection.protocol not in ["http", "https"]:
                    return FlextResult[None].fail("API services must use HTTP protocols")

                if self.service_type == "worker" and self.processing.parallel_workers > 8:
                    return FlextResult[None].fail("Worker services should not exceed 8 parallel workers")

                # Validate enabled services have healthy connection
                if self.is_enabled and self.health_status == "unhealthy":
                    return FlextResult[None].fail("Enabled services cannot be unhealthy")

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Service aggregate validation failed: {e}")

        def enable_service(self, enabled_by: str) -> FlextResult[None]:
            """Enable service with business logic and events."""
            try:
                if self.is_enabled:
                    return FlextResult[None].fail("Service is already enabled")

                # Validate prerequisites
                connection_validation = self.connection.validate_business_rules()
                if connection_validation.is_failure:
                    return FlextResult[None].fail(f"Invalid connection: {connection_validation.error}")

                processing_validation = self.processing.validate_business_rules()
                if processing_validation.is_failure:
                    return FlextResult[None].fail(f"Invalid processing config: {processing_validation.error}")

                # Enable service
                self.is_enabled = True
                self.health_status = "unknown"  # Will be determined by health checks
                self.updated_by = enabled_by

                # Apply domain event
                enable_event = {
                    "event_type": "ServiceEnabled",
                    "aggregate_id": self.id,
                    "service_name": self.service_name,
                    "service_type": self.service_type,
                    "connection_url": self.connection.connection_url,
                    "enabled_by": enabled_by,
                    "timestamp": datetime.now(UTC).isoformat()
                }

                event_result = self.apply_domain_event(enable_event)
                if event_result.is_failure:
                    return event_result

                self.increment_version()
                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Failed to enable service: {e}")

        def update_health_status(self, new_status: str, health_data: dict[str, object]) -> FlextResult[None]:
            """Update service health status with monitoring data."""
            try:
                if new_status not in ["healthy", "unhealthy", "degraded", "unknown"]:
                    return FlextResult[None].fail(f"Invalid health status: {new_status}")

                old_status = self.health_status
                self.health_status = new_status

                # Raise health change event
                health_event = {
                    "event_type": "HealthStatusChanged",
                    "aggregate_id": self.id,
                    "service_name": self.service_name,
                    "previous_status": old_status,
                    "new_status": new_status,
                    "health_data": health_data,
                    "timestamp": datetime.now(UTC).isoformat()
                }

                event_result = self.apply_domain_event(health_event)
                if event_result.is_failure:
                    return event_result

                # Check if we need to disable service due to health
                if new_status == "unhealthy" and self.is_enabled:
                    # Consider auto-disabling after prolonged unhealthy status
                    unhealthy_duration = health_data.get("unhealthy_duration_minutes", 0)
                    if unhealthy_duration > 30:  # 30 minutes
                        self.is_enabled = False

                        disable_event = {
                            "event_type": "ServiceAutoDisabled",
                            "aggregate_id": self.id,
                            "service_name": self.service_name,
                            "reason": "Prolonged unhealthy status",
                            "unhealthy_duration_minutes": unhealthy_duration,
                            "timestamp": datetime.now(UTC).isoformat()
                        }

                        self.apply_domain_event(disable_event)

                self.increment_version()
                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(f"Failed to update health status: {e}")
```

### 2.2 Factory Methods Implementation

```python
class YourLibraryModels(FlextModels):
    """Extended with factory methods for safe creation."""

    # =========================================================================
    # FACTORY METHODS - Safe creation with validation
    # =========================================================================

    @classmethod
    def create_main_entity(
        cls,
        name: str,
        category: str,
        configuration: dict[str, object] | None = None,
        created_by: str | None = None
    ) -> FlextResult[YourMainEntity]:
        """Create main entity with validation."""
        try:
            # Prepare entity data
            entity_data = {
                "id": f"{category}_{uuid.uuid4().hex[:8]}",
                "name": name,
                "category": category,
                "configuration": configuration or {},
                "status": "pending",
                "created_by": created_by,
                "updated_by": created_by
            }

            # Create entity
            entity = cls.YourMainEntity.model_validate(entity_data)

            # Validate business rules
            validation_result = entity.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[cls.YourMainEntity].fail(validation_result.error)

            # Add creation domain event
            entity.add_domain_event({
                "event_type": "EntityCreated",
                "aggregate_id": entity.id,
                "entity_name": name,
                "category": category,
                "created_by": created_by,
                "timestamp": datetime.now(UTC).isoformat()
            })

            return FlextResult[cls.YourMainEntity].ok(entity)

        except ValidationError as e:
            return FlextResult[cls.YourMainEntity].fail(f"Entity validation failed: {e}")
        except Exception as e:
            return FlextResult[cls.YourMainEntity].fail(f"Entity creation failed: {e}")

    @classmethod
    def create_connection_string(
        cls,
        protocol: str,
        host: str,
        port: int,
        path: str = "/",
        username: str | None = None,
        password: str | None = None
    ) -> FlextResult[ConnectionString]:
        """Create connection string with validation."""
        try:
            connection = cls.ConnectionString(
                protocol=protocol,
                host=host,
                port=port,
                path=path,
                username=username,
                password=password
            )

            # Validate business rules
            validation_result = connection.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[cls.ConnectionString].fail(validation_result.error)

            return FlextResult[cls.ConnectionString].ok(connection)

        except ValidationError as e:
            return FlextResult[cls.ConnectionString].fail(f"Connection string validation failed: {e}")
        except Exception as e:
            return FlextResult[cls.ConnectionString].fail(f"Connection string creation failed: {e}")

    @classmethod
    def create_service_aggregate(
        cls,
        service_name: str,
        service_type: str,
        connection_config: dict[str, object],
        processing_config: dict[str, object] | None = None,
        created_by: str | None = None
    ) -> FlextResult[YourServiceAggregate]:
        """Create service aggregate with full validation."""
        try:
            # Create connection string
            connection_result = cls.create_connection_string(
                protocol=connection_config["protocol"],
                host=connection_config["host"],
                port=connection_config["port"],
                path=connection_config.get("path", "/"),
                username=connection_config.get("username"),
                password=connection_config.get("password")
            )

            if connection_result.is_failure:
                return FlextResult[cls.YourServiceAggregate].fail(connection_result.error)

            # Create processing config
            processing = cls.ProcessingConfig(**(processing_config or {}))
            processing_validation = processing.validate_business_rules()
            if processing_validation.is_failure:
                return FlextResult[cls.YourServiceAggregate].fail(processing_validation.error)

            # Create service aggregate
            service = cls.YourServiceAggregate(
                id=f"service_{uuid.uuid4().hex[:8]}",
                service_name=service_name,
                service_type=service_type,
                connection=connection_result.value,
                processing=processing,
                created_by=created_by,
                updated_by=created_by
            )

            # Validate business rules
            validation_result = service.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[cls.YourServiceAggregate].fail(validation_result.error)

            # Add creation domain event
            service.add_domain_event({
                "event_type": "ServiceAggregateCreated",
                "aggregate_id": service.id,
                "service_name": service_name,
                "service_type": service_type,
                "connection_url": service.connection.connection_url,
                "created_by": created_by,
                "timestamp": datetime.now(UTC).isoformat()
            })

            return FlextResult[cls.YourServiceAggregate].ok(service)

        except ValidationError as e:
            return FlextResult[cls.YourServiceAggregate].fail(f"Service validation failed: {e}")
        except Exception as e:
            return FlextResult[cls.YourServiceAggregate].fail(f"Service creation failed: {e}")
```

---

## ⚙️ Phase 3: Business Rules & Events

### 3.1 Domain Event Handling

```python
class YourLibraryEventHandler:
    """Domain event handler for your library."""

    def __init__(self, library_models: YourLibraryModels):
        self.models = library_models

    def handle_entity_activated(self, event: FlextModels.Event) -> FlextResult[None]:
        """Handle entity activation event."""
        try:
            entity_id = event.aggregate_id
            event_data = event.data

            # Log activation
            logger.info(
                f"Entity activated: {event_data.get('entity_name')} by {event_data.get('activated_by')}"
            )

            # Trigger dependent processes
            if event_data.get("category") == "critical":
                # Send notification for critical entity activation
                notification_result = self._send_critical_activation_notification(event_data)
                if notification_result.is_failure:
                    logger.warning(f"Failed to send notification: {notification_result.error}")

            # Update monitoring systems
            monitoring_result = self._update_monitoring_system(entity_id, "active")
            if monitoring_result.is_failure:
                logger.warning(f"Failed to update monitoring: {monitoring_result.error}")

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Failed to handle entity activation: {e}")

    def handle_service_health_changed(self, event: FlextModels.Event) -> FlextResult[None]:
        """Handle service health status change event."""
        try:
            event_data = event.data
            service_name = event_data.get("service_name")
            new_status = event_data.get("new_status")

            # Alert on health degradation
            if new_status in ["unhealthy", "degraded"]:
                alert_result = self._send_health_alert(service_name, new_status, event_data.get("health_data", {}))
                if alert_result.is_failure:
                    logger.error(f"Failed to send health alert: {alert_result.error}")

            # Update service registry
            registry_result = self._update_service_registry(service_name, new_status)
            if registry_result.is_failure:
                logger.warning(f"Failed to update service registry: {registry_result.error}")

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Failed to handle health status change: {e}")

    def _send_critical_activation_notification(self, event_data: dict[str, object]) -> FlextResult[None]:
        """Send notification for critical entity activation."""
        # Implementation would integrate with notification service
        return FlextResult[None].ok(None)

    def _update_monitoring_system(self, entity_id: str, status: str) -> FlextResult[None]:
        """Update external monitoring system."""
        # Implementation would integrate with monitoring service
        return FlextResult[None].ok(None)

    def _send_health_alert(self, service_name: str, status: str, health_data: dict[str, object]) -> FlextResult[None]:
        """Send health status alert."""
        # Implementation would integrate with alerting service
        return FlextResult[None].ok(None)

    def _update_service_registry(self, service_name: str, status: str) -> FlextResult[None]:
        """Update service registry with health status."""
        # Implementation would integrate with service registry
        return FlextResult[None].ok(None)
```

### 3.2 Business Rule Specifications

```python
class YourLibrarySpecifications:
    """Business rule specifications for complex queries."""

    @staticmethod
    def active_entities_specification() -> callable:
        """Specification for active entities."""
        def specification(entity: YourLibraryModels.YourMainEntity) -> bool:
            return entity.status == "active" and entity.configuration is not None
        return specification

    @staticmethod
    def critical_entities_specification() -> callable:
        """Specification for critical entities."""
        def specification(entity: YourLibraryModels.YourMainEntity) -> bool:
            return (
                entity.category == "critical" and
                entity.status == "active" and
                len(entity.tags) > 0
            )
        return specification

    @staticmethod
    def healthy_services_specification() -> callable:
        """Specification for healthy services."""
        def specification(service: YourLibraryModels.YourServiceAggregate) -> bool:
            return (
                service.is_enabled and
                service.health_status == "healthy" and
                service.connection is not None
            )
        return specification

    @staticmethod
    def services_needing_attention_specification() -> callable:
        """Specification for services needing attention."""
        def specification(service: YourLibraryModels.YourServiceAggregate) -> bool:
            return (
                service.is_enabled and
                service.health_status in ["unhealthy", "degraded", "unknown"]
            ) or (
                not service.is_enabled and
                service.health_status == "healthy"
            )
        return specification
```

---

## 🔗 Phase 4: Testing & Integration

### 4.1 Model Testing Strategy

```python
import pytest
from unittest.mock import Mock, patch

class TestYourLibraryModels:
    """Comprehensive model testing."""

    @pytest.fixture
    def sample_entity_data(self):
        return {
            "name": "test_entity",
            "category": "api",
            "configuration": {
                "endpoint": "https://api.example.com",
                "timeout": 30,
                "max_retries": 3
            }
        }

    @pytest.fixture
    def sample_connection_config(self):
        return {
            "protocol": "https",
            "host": "api.example.com",
            "port": 443,
            "path": "/v1",
            "username": "api_user",
            "password": "secure_password"
        }

    def test_entity_creation_success(self, sample_entity_data):
        """Test successful entity creation."""
        result = YourLibraryModels.create_main_entity(
            name=sample_entity_data["name"],
            category=sample_entity_data["category"],
            configuration=sample_entity_data["configuration"],
            created_by="test_user"
        )

        assert result.success
        entity = result.value
        assert entity.name == "test_entity"
        assert entity.category == "api"
        assert entity.status == "pending"
        assert len(entity.domain_events) == 1
        assert entity.domain_events[0]["event_type"] == "EntityCreated"

    def test_entity_business_rules_validation(self):
        """Test entity business rule validation."""
        # Test reserved name validation
        result = YourLibraryModels.create_main_entity(
            name="REDACTED_LDAP_BIND_PASSWORD",  # Reserved name
            category="api",
            created_by="test_user"
        )

        assert result.is_failure
        assert "reserved word" in result.error

    def test_entity_activation(self, sample_entity_data):
        """Test entity activation with domain events."""
        entity_result = YourLibraryModels.create_main_entity(
            name=sample_entity_data["name"],
            category=sample_entity_data["category"],
            configuration=sample_entity_data["configuration"],
            created_by="test_user"
        )

        assert entity_result.success
        entity = entity_result.value

        # Activate entity
        activation_result = entity.activate("REDACTED_LDAP_BIND_PASSWORD_user")

        assert activation_result.success
        assert entity.status == "active"
        assert len(entity.domain_events) == 2  # Created + Activated
        assert entity.domain_events[1]["event_type"] == "EntityActivated"

    def test_value_object_immutability(self, sample_connection_config):
        """Test value object immutability."""
        connection_result = YourLibraryModels.create_connection_string(
            **sample_connection_config
        )

        assert connection_result.success
        connection = connection_result.value

        # Attempting to modify should raise error
        with pytest.raises(ValidationError):
            connection.host = "modified.example.com"  # Should fail - frozen

    def test_value_object_computed_fields(self, sample_connection_config):
        """Test value object computed fields."""
        connection_result = YourLibraryModels.create_connection_string(
            **sample_connection_config
        )

        assert connection_result.success
        connection = connection_result.value

        expected_url = "https://api_user:secure_password@api.example.com:443/v1"
        assert connection.connection_url == expected_url

    def test_aggregate_root_domain_events(self):
        """Test aggregate root domain event handling."""
        service_result = YourLibraryModels.create_service_aggregate(
            service_name="test_service",
            service_type="api",
            connection_config={
                "protocol": "https",
                "host": "service.example.com",
                "port": 443
            },
            created_by="REDACTED_LDAP_BIND_PASSWORD"
        )

        assert service_result.success
        service = service_result.value

        # Enable service
        enable_result = service.enable_service("operator")

        assert enable_result.success
        assert service.is_enabled
        assert len(service.domain_events) == 2  # Created + Enabled

        # Check event application
        events = service.clear_domain_events()
        assert len(events) == 2
        assert events[1]["event_type"] == "ServiceEnabled"

    def test_business_rule_specifications(self):
        """Test business rule specifications."""
        # Create test entities
        active_entity_result = YourLibraryModels.create_main_entity(
            name="active_test",
            category="api",
            configuration={"endpoint": "https://api.example.com"},
            created_by="test_user"
        )
        active_entity_result.value.activate("REDACTED_LDAP_BIND_PASSWORD")

        inactive_entity_result = YourLibraryModels.create_main_entity(
            name="inactive_test",
            category="api",
            created_by="test_user"
        )

        # Test specifications
        active_spec = YourLibrarySpecifications.active_entities_specification()

        assert active_spec(active_entity_result.value) is True
        assert active_spec(inactive_entity_result.value) is False

    def test_configuration_update_with_events(self, sample_entity_data):
        """Test configuration updates generate proper events."""
        entity_result = YourLibraryModels.create_main_entity(
            **sample_entity_data,
            created_by="test_user"
        )

        entity = entity_result.value

        # Update configuration
        new_config = {"timeout": 60, "new_setting": "test_value"}
        update_result = entity.update_configuration(new_config, "REDACTED_LDAP_BIND_PASSWORD_user")

        assert update_result.success
        assert entity.configuration["timeout"] == 60
        assert entity.configuration["new_setting"] == "test_value"

        # Check domain event
        events = entity.domain_events
        config_event = next(e for e in events if e["event_type"] == "ConfigurationUpdated")
        assert config_event["config_changes"]["modified"]["timeout"]["new"] == 60
        assert config_event["config_changes"]["added"]["new_setting"] == "test_value"
```

### 4.2 Integration Testing Patterns

```python
class TestModelsIntegration:
    """Integration tests for models ecosystem."""

    def test_end_to_end_entity_lifecycle(self):
        """Test complete entity lifecycle with events."""
        # Create entity
        entity_result = YourLibraryModels.create_main_entity(
            name="integration_test",
            category="critical",
            configuration={"endpoint": "https://critical.example.com"},
            created_by="system"
        )

        assert entity_result.success
        entity = entity_result.value

        # Activate entity
        activation_result = entity.activate("REDACTED_LDAP_BIND_PASSWORD")
        assert activation_result.success

        # Update configuration
        config_update_result = entity.update_configuration(
            {"max_connections": 100}, "REDACTED_LDAP_BIND_PASSWORD"
        )
        assert config_update_result.success

        # Validate final state
        assert entity.status == "active"
        assert entity.configuration["max_connections"] == 100
        assert len(entity.domain_events) == 3  # Created, Activated, ConfigUpdated

    def test_service_aggregate_health_management(self):
        """Test service aggregate health status management."""
        # Create service
        service_result = YourLibraryModels.create_service_aggregate(
            service_name="health_test_service",
            service_type="api",
            connection_config={
                "protocol": "https",
                "host": "health.example.com",
                "port": 443
            },
            created_by="REDACTED_LDAP_BIND_PASSWORD"
        )

        service = service_result.value

        # Enable service
        service.enable_service("operator")

        # Update health status
        health_result = service.update_health_status(
            "unhealthy",
            {
                "unhealthy_duration_minutes": 35,
                "last_error": "Connection timeout"
            }
        )

        assert health_result.success
        assert service.health_status == "unhealthy"
        assert not service.is_enabled  # Auto-disabled due to prolonged unhealthy status

        # Check events
        events = service.domain_events
        health_event = next(e for e in events if e["event_type"] == "HealthStatusChanged")
        auto_disable_event = next(e for e in events if e["event_type"] == "ServiceAutoDisabled")

        assert health_event["new_status"] == "unhealthy"
        assert auto_disable_event["reason"] == "Prolonged unhealthy status"
```

---

## ✅ Implementation Checklist

### Pre-Implementation

- [ ] **Domain analysis complete**: Entities, values, aggregates identified
- [ ] **Business rules documented**: Validation logic and invariants mapped
- [ ] **Event requirements identified**: Significant business events catalogued
- [ ] **Message patterns designed**: Inter-service communication needs

### Core Implementation

- [ ] **Library models class created**: Inherits from FlextModels properly
- [ ] **Entity models implemented**: With identity, lifecycle, business logic
- [ ] **Value objects implemented**: Immutable with validation and computed fields
- [ ] **Aggregate roots implemented**: With domain events and consistency boundaries
- [ ] **Business rules implemented**: validate_business_rules() methods

### Advanced Features

- [ ] **Factory methods implemented**: Safe creation with FlextResult
- [ ] **Domain events implemented**: Event generation and handling
- [ ] **Message patterns added**: Payload, Event, Message usage
- [ ] **Specifications implemented**: Complex business rule objects

### Testing & Integration

- [ ] **Unit tests comprehensive**: All models and business logic tested
- [ ] **Integration tests complete**: End-to-end workflows tested
- [ ] **Event handling tested**: Domain event generation and processing
- [ ] **Business rules validated**: All validation scenarios covered
- [ ] **Performance tested**: Model creation and validation performance

---

## 📈 Success Metrics

Track these metrics to measure implementation success:

### Domain Modeling Quality

- **Model Coverage**: 100% domain concepts modeled with FlextModels
- **Business Rule Implementation**: >90% validation rules implemented
- **Event Coverage**: Major business events captured and handled

### Code Quality

- **Type Safety**: 100% type annotations on model fields
- **Immutability**: All value objects properly frozen
- **Validation Coverage**: Comprehensive Pydantic validation

### Developer Experience

- **API Consistency**: Uniform modeling patterns across operations
- **Error Messages**: Clear, actionable validation messages
- **Documentation**: Complete model documentation with examples

---

## 🔗 Next Steps

1. **Start with Domain Analysis**: Map your domain entities, values, and aggregates
2. **Implement Core Models**: Create basic entity and value object structure
3. **Add Business Logic**: Implement validation rules and domain operations
4. **Enhance with Events**: Add domain event generation and handling
5. **Test Thoroughly**: Validate all modeling scenarios and business rules

This implementation guide provides the foundation for successful FlextModels adoption. Adapt the patterns to your specific domain needs while maintaining consistency with FLEXT architectural principles and ensuring robust business rule enforcement throughout your domain models.
