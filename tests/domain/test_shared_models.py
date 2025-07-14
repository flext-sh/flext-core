"""Tests for flext_core.domain.shared_models module."""

from datetime import datetime
from uuid import uuid4

import pytest

from flext_core.domain.shared_models import AuthToken
from flext_core.domain.shared_models import ComponentHealth
from flext_core.domain.shared_models import DatabaseConfig
from flext_core.domain.shared_models import DataRecord
from flext_core.domain.shared_models import DataSchema
from flext_core.domain.shared_models import EntityStatus
from flext_core.domain.shared_models import ErrorDetail
from flext_core.domain.shared_models import ErrorResponse
from flext_core.domain.shared_models import HealthStatus
from flext_core.domain.shared_models import LDAPEntry
from flext_core.domain.shared_models import LDAPScope
from flext_core.domain.shared_models import LogLevel
from flext_core.domain.shared_models import OperationStatus
from flext_core.domain.shared_models import PipelineConfig
from flext_core.domain.shared_models import PipelineRunStatus
from flext_core.domain.shared_models import PluginMetadata
from flext_core.domain.shared_models import PluginType
from flext_core.domain.shared_models import RedisConfig
from flext_core.domain.shared_models import SystemHealth
from flext_core.domain.shared_models import UserInfo


class TestEntityStatus:
    """Test EntityStatus enum."""

    def test_entity_status_values(self) -> None:
        """Test EntityStatus enum values."""
        assert EntityStatus.ACTIVE.value == "active"
        assert EntityStatus.INACTIVE.value == "inactive"
        assert EntityStatus.PENDING.value == "pending"
        assert EntityStatus.DELETED.value == "deleted"

    def test_entity_status_membership(self) -> None:
        """Test EntityStatus membership."""
        assert "active" in EntityStatus
        assert "inactive" in EntityStatus
        assert "invalid" not in EntityStatus


class TestOperationStatus:
    """Test OperationStatus enum."""

    def test_operation_status_values(self) -> None:
        """Test OperationStatus enum values."""
        assert OperationStatus.PENDING.value == "pending"
        assert OperationStatus.RUNNING.value == "running"
        assert OperationStatus.SUCCESS.value == "success"
        assert OperationStatus.FAILED.value == "failed"
        assert OperationStatus.CANCELLED.value == "cancelled"


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"


class TestAuthToken:
    """Test AuthToken value object."""

    def test_auth_token_creation(self) -> None:
        """Test AuthToken can be created."""
        token = AuthToken(
            access_token="test_token",
            expires_in=3600,
        )

        assert token.access_token == "test_token"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.refresh_token is None
        assert token.scope is None

    def test_auth_token_with_optional_fields(self) -> None:
        """Test AuthToken with optional fields."""
        token = AuthToken(
            access_token="test_token",
            token_type="Custom",
            expires_in=7200,
            refresh_token="refresh_token",
            scope="read write",
        )

        assert token.access_token == "test_token"
        assert token.token_type == "Custom"
        assert token.expires_in == 7200
        assert token.refresh_token == "refresh_token"
        assert token.scope == "read write"


class TestUserInfo:
    """Test UserInfo model."""

    def test_user_info_creation(self) -> None:
        """Test UserInfo can be created."""
        user_id = uuid4()
        user = UserInfo(
            id=user_id,
            username="testuser",
            full_name="Test User",
        )

        assert user.id == user_id
        assert user.username == "testuser"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.roles == []
        assert user.permissions == []

    def test_user_info_with_roles_and_permissions(self) -> None:
        """Test UserInfo with roles and permissions."""
        user = UserInfo(
            id=uuid4(),
            username="testuser",
            roles=["REDACTED_LDAP_BIND_PASSWORD", "user"],
            permissions=["read", "write", "delete"],
        )

        assert user.roles == ["REDACTED_LDAP_BIND_PASSWORD", "user"]
        assert user.permissions == ["read", "write", "delete"]


class TestPluginType:
    """Test PluginType enum."""

    def test_plugin_type_values(self) -> None:
        """Test PluginType enum values."""
        assert PluginType.EXTRACTOR.value == "extractor"
        assert PluginType.LOADER.value == "loader"
        assert PluginType.TRANSFORM.value == "transform"
        assert PluginType.ORCHESTRATOR.value == "orchestrator"
        assert PluginType.UTILITY.value == "utility"


class TestPluginMetadata:
    """Test PluginMetadata model."""

    def test_plugin_metadata_creation(self) -> None:
        """Test PluginMetadata can be created."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            author="Test Author",
        )

        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.capabilities == []
        assert metadata.requirements == []
        assert metadata.config_schema is None

    def test_plugin_metadata_with_lists(self) -> None:
        """Test PluginMetadata with capabilities and requirements."""
        metadata = PluginMetadata(
            name="test-plugin",
            capabilities=["extract", "validate"],
            requirements=["requests>=2.25.0"],
            config_schema={"type": "object", "properties": {}},
        )

        assert metadata.capabilities == ["extract", "validate"]
        assert metadata.requirements == ["requests>=2.25.0"]
        assert metadata.config_schema == {"type": "object", "properties": {}}


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self) -> None:
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Test ComponentHealth model."""

    def test_component_health_creation(self) -> None:
        """Test ComponentHealth can be created."""
        health = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
        )

        assert health.name == "database"
        assert health.status == HealthStatus.HEALTHY
        assert health.checks == {}
        assert health.last_check is None

    def test_component_health_with_checks(self) -> None:
        """Test ComponentHealth with checks."""
        now = datetime.utcnow()
        health = ComponentHealth(
            name="redis",
            status=HealthStatus.DEGRADED,
            checks={"ping": True, "memory": "high"},
            last_check=now,
        )

        assert health.name == "redis"
        assert health.status == HealthStatus.DEGRADED
        assert health.checks == {"ping": True, "memory": "high"}
        assert health.last_check == now


class TestSystemHealth:
    """Test SystemHealth model."""

    def test_system_health_creation(self) -> None:
        """Test SystemHealth can be created."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            uptime=3600.0,
        )

        assert health.status == HealthStatus.HEALTHY
        assert health.version == "1.0.0"
        assert health.uptime == 3600.0
        assert health.components == []

    def test_system_health_with_components(self) -> None:
        """Test SystemHealth with components."""
        component = ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[component],
        )

        assert len(health.components) == 1
        assert health.components[0].name == "database"


class TestErrorModels:
    """Test error models."""

    def test_error_detail_creation(self) -> None:
        """Test ErrorDetail can be created."""
        error = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Invalid input",
            details={"field": "username", "value": ""},
        )

        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"
        assert error.details == {"field": "username", "value": ""}

    def test_error_response_creation(self) -> None:
        """Test ErrorResponse can be created."""
        error_detail = ErrorDetail(code="NOT_FOUND", message="Resource not found")
        error_response = ErrorResponse(
            error=error_detail,
            request_id="12345",  # Use the alias field name
        )

        assert error_response.success is False
        assert error_response.error.code == "NOT_FOUND"
        assert (
            error_response.correlation_id == "12345"
        )  # Should be accessible via correlation_id
        assert isinstance(error_response.timestamp, datetime)


class TestDatabaseConfig:
    """Test DatabaseConfig value object."""

    def test_database_config_creation(self) -> None:
        """Test DatabaseConfig can be created."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.pool_size == 20  # Default value
        assert config.max_overflow == 40  # Default value

    def test_database_config_port_validation(self) -> None:
        """Test DatabaseConfig port validation."""
        # Valid port
        config = DatabaseConfig(host="localhost", port=3306)
        assert config.port == 3306

        # Invalid ports
        with pytest.raises(ValueError):
            DatabaseConfig(host="localhost", port=0)

        with pytest.raises(ValueError):
            DatabaseConfig(host="localhost", port=65536)


class TestRedisConfig:
    """Test RedisConfig value object."""

    def test_redis_config_creation(self) -> None:
        """Test RedisConfig can be created with defaults."""
        config = RedisConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.decode_responses is True
        assert config.socket_timeout == 5
        assert config.socket_keepalive is True

    def test_redis_config_with_custom_values(self) -> None:
        """Test RedisConfig with custom values."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            socket_timeout=10,
        )

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.socket_timeout == 10


class TestLDAPScope:
    """Test LDAPScope enum."""

    def test_ldap_scope_values(self) -> None:
        """Test LDAPScope enum values."""
        assert LDAPScope.BASE.value == "base"
        assert LDAPScope.ONE.value == "one"
        assert LDAPScope.SUB.value == "sub"


class TestLDAPEntry:
    """Test LDAPEntry model."""

    def test_ldap_entry_creation(self) -> None:
        """Test LDAPEntry can be created."""
        entry = LDAPEntry(
            dn="cn=test,dc=example,dc=com",
            attributes={"cn": ["test"], "mail": ["test@example.com"]},
        )

        assert entry.dn == "cn=test,dc=example,dc=com"
        assert entry.attributes == {"cn": ["test"], "mail": ["test@example.com"]}

    def test_ldap_entry_dn_validation(self) -> None:
        """Test LDAPEntry DN validation."""
        # Valid DN
        entry = LDAPEntry(
            dn="  cn=test,dc=example,dc=com  ",
            attributes={},
        )
        assert entry.dn == "cn=test,dc=example,dc=com"  # Should be stripped

        # Invalid DNs
        with pytest.raises(ValueError):
            LDAPEntry(dn="", attributes={})

        with pytest.raises(ValueError):
            LDAPEntry(dn="   ", attributes={})


class TestPipelineConfig:
    """Test PipelineConfig model."""

    def test_pipeline_config_creation(self) -> None:
        """Test PipelineConfig can be created."""
        config = PipelineConfig(
            name="test-pipeline",
            description="Test pipeline",
        )

        assert config.name == "test-pipeline"
        assert config.description == "Test pipeline"
        assert config.steps == []
        assert config.schedule is None
        assert config.timeout == 3600
        assert config.retries == 3
        assert config.is_active is True

    def test_pipeline_config_with_steps(self) -> None:
        """Test PipelineConfig with steps."""
        steps = [
            {"type": "extract", "source": "database"},
            {"type": "transform", "operation": "clean"},
        ]
        config = PipelineConfig(
            name="test-pipeline",
            steps=steps,
            schedule="0 2 * * *",
            timeout=7200,
            retries=5,
        )

        assert config.steps == steps
        assert config.schedule == "0 2 * * *"
        assert config.timeout == 7200
        assert config.retries == 5


class TestPipelineRunStatus:
    """Test PipelineRunStatus model."""

    def test_pipeline_run_status_creation(self) -> None:
        """Test PipelineRunStatus can be created."""
        run_id = uuid4()
        pipeline_id = uuid4()

        status = PipelineRunStatus(
            run_id=run_id,
            pipeline_id=pipeline_id,
        )

        assert status.run_id == run_id
        assert status.pipeline_id == pipeline_id
        assert status.error is None
        assert status.metrics == {}

    def test_pipeline_run_status_with_error(self) -> None:
        """Test PipelineRunStatus with error."""
        status = PipelineRunStatus(
            run_id=uuid4(),
            error="Connection timeout",
            metrics={"duration": 120.5, "records": 1000},
        )

        assert status.error == "Connection timeout"
        assert status.metrics == {"duration": 120.5, "records": 1000}


class TestDataSchema:
    """Test DataSchema model."""

    def test_data_schema_creation(self) -> None:
        """Test DataSchema can be created."""
        schema = DataSchema(
            fields={
                "id": {"type": "integer", "primary_key": True},
                "name": {"type": "string", "max_length": 100},
            },
            required=["id", "name"],
            unique=["id"],
            indexes=["name"],
        )

        assert len(schema.fields) == 2
        assert schema.required == ["id", "name"]
        assert schema.unique == ["id"]
        assert schema.indexes == ["name"]


class TestDataRecord:
    """Test DataRecord model."""

    def test_data_record_creation(self) -> None:
        """Test DataRecord can be created."""
        record_id = uuid4()
        record = DataRecord(
            id=record_id,
            data={"name": "Test", "value": 42},
            metadata={"source": "api", "version": "1.0"},
        )

        assert record.id == record_id
        assert record.data == {"name": "Test", "value": 42}
        assert record.metadata == {"source": "api", "version": "1.0"}
        assert isinstance(record.created_at, datetime)
        assert record.updated_at is None

    def test_data_record_defaults(self) -> None:
        """Test DataRecord with defaults."""
        record = DataRecord(data={"test": "value"})

        assert record.id is None
        assert record.data == {"test": "value"}
        assert record.metadata == {}
        assert isinstance(record.created_at, datetime)


class TestModelSerialization:
    """Test model serialization capabilities."""

    def test_model_to_dict(self) -> None:
        """Test model serialization to dict."""
        user = UserInfo(
            id=uuid4(),
            username="testuser",
            roles=["REDACTED_LDAP_BIND_PASSWORD"],
        )

        user_dict = user.model_dump()

        assert isinstance(user_dict, dict)
        assert user_dict["username"] == "testuser"
        assert user_dict["roles"] == ["REDACTED_LDAP_BIND_PASSWORD"]
        assert "id" in user_dict

    def test_value_object_immutability(self) -> None:
        """Test value objects are immutable."""
        token = AuthToken(access_token="test", expires_in=3600)

        # Should not be able to modify after creation
        with pytest.raises(Exception):  # Pydantic validation error
            token.access_token = "modified"


class TestModelValidation:
    """Test model validation features."""

    def test_database_config_validation(self) -> None:
        """Test DatabaseConfig validation."""
        # Port validation
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 1"
        ):
            DatabaseConfig(host="localhost", port=0)

    def test_ldap_entry_validation(self) -> None:
        """Test LDAPEntry validation."""
        # DN validation
        with pytest.raises(ValueError, match="DN cannot be empty"):
            LDAPEntry(dn="", attributes={})

    def test_required_field_validation(self) -> None:
        """Test required field validation."""
        # UserInfo requires id
        with pytest.raises(Exception):  # Pydantic validation error
            UserInfo(username="test")  # type: ignore[call-arg]  # Missing required id field (UUID)
