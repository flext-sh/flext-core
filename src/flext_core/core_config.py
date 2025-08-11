"""FLEXT Core Configuration - Enterprise configuration management system.

Consolidates all configuration patterns following PEP8 naming conventions.
Provides Pydantic-based models, environment loading, validation, and
cross-service serialization for distributed data integration pipelines.

Architecture:
    - Abstract Protocols: Configuration contracts and interfaces
    - Base Classes: Foundation configuration patterns
    - Concrete Models: Production-ready configuration classes
    - Factory Methods: Dynamic configuration creation
    - Compatibility Layer: Legacy support and migration

Usage:
    from flext_core.core_config import FlextConfig, FlextDatabaseConfig

    config = FlextConfig(debug=True, environment="production")
    db_config = FlextDatabaseConfig.from_env()
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from pydantic import BaseModel, Field, validator

try:
    from pydantic_settings import BaseSettings as PydanticBaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings as PydanticBaseSettings

if TYPE_CHECKING:
    from collections.abc import Mapping


# =============================================================================
# CONSTANTS AND DEFAULTS
# =============================================================================

# Default configuration values
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_PAGE_SIZE = 100
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ENVIRONMENT = "development"

# Configuration validation messages
CONFIG_VALIDATION_MESSAGES = {
    "required_field": "This field is required",
    "invalid_type": "Invalid type for this field",
    "out_of_range": "Value is out of allowed range",
    "invalid_format": "Invalid format for this field",
}

# =============================================================================
# ABSTRACT PROTOCOLS - Configuration contracts
# =============================================================================


class FlextConfigValidatorProtocol(Protocol):
    """Protocol for configuration validators."""

    def validate(self, config: Mapping[str, Any]) -> bool:
        """Validate configuration data."""
        ...


class FlextConfigLoaderProtocol(Protocol):
    """Protocol for configuration loaders."""

    def load(self, source: str) -> dict[str, Any]:
        """Load configuration from source."""
        ...


class FlextConfigMergerProtocol(Protocol):
    """Protocol for configuration mergers."""

    def merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge two configuration dictionaries."""
        ...


class FlextConfigSerializerProtocol(Protocol):
    """Protocol for configuration serializers."""

    def serialize(self, config: dict[str, Any]) -> str:
        """Serialize configuration to string."""
        ...

    def deserialize(self, data: str) -> dict[str, Any]:
        """Deserialize configuration from string."""
        ...

# =============================================================================
# ABSTRACT BASE CLASSES - Foundation configuration patterns
# =============================================================================


class FlextAbstractConfig(ABC, BaseModel):
    """Abstract base class for all configuration models."""

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
        frozen = False

    @abstractmethod
    def validate_business_rules(self) -> bool:
        """Validate business rules - must be implemented by subclasses."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlextAbstractConfig:
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> FlextAbstractConfig:
        """Create configuration from JSON string."""
        return cls.parse_raw(json_str)


class FlextAbstractSettings(ABC):
    """Abstract base class for environment-aware settings."""

    @abstractmethod
    def load_from_environment(self) -> dict[str, Any]:
        """Load settings from environment variables."""
        ...

    @abstractmethod
    def get_env_prefix(self) -> str:
        """Get environment variable prefix."""
        ...

# =============================================================================
# CONFIGURATION OPERATIONS AND UTILITIES
# =============================================================================


class FlextConfigOperations:
    """Configuration operations and utilities."""

    @staticmethod
    def merge_configs(
        base: dict[str, Any],
        override: dict[str, Any],
        deep: bool = True,
    ) -> dict[str, Any]:
        """Merge configuration dictionaries."""
        if not deep:
            result = base.copy()
            result.update(override)
            return result

        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = FlextConfigOperations.merge_configs(result[key], value, deep=True)
            else:
                result[key] = value
        return result

    @staticmethod
    def load_from_env(prefix: str = "FLEXT") -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        prefix_with_separator = f"{prefix}_"

        for key, value in os.environ.items():
            if key.startswith(prefix_with_separator):
                config_key = key[len(prefix_with_separator):].lower()

                # Try to parse as JSON first, then fall back to string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed_value = value

                config[config_key] = parsed_value

        return config

    @staticmethod
    def load_from_file(file_path: str | Path) -> dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def save_to_file(config: dict[str, Any], file_path: str | Path) -> bool:
        """Save configuration to JSON file."""
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            return True
        except (OSError, TypeError):
            return False


class FlextConfigValidator:
    """Configuration validation utilities."""

    @staticmethod
    def validate_required_fields(
        config: dict[str, Any],
        required_fields: list[str],
    ) -> list[str]:
        """Validate required fields are present."""
        missing = []
        for field in required_fields:
            if field not in config or config[field] is None:
                missing.append(field)
        return missing

    @staticmethod
    def validate_field_types(
        config: dict[str, Any],
        field_types: dict[str, type],
    ) -> list[str]:
        """Validate field types."""
        errors = []
        for field, expected_type in field_types.items():
            if field in config and not isinstance(config[field], expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
        return errors

    @staticmethod
    def validate_field_ranges(
        config: dict[str, Any],
        field_ranges: dict[str, tuple[Any, Any]],
    ) -> list[str]:
        """Validate field value ranges."""
        errors = []
        for field, (min_val, max_val) in field_ranges.items():
            if field in config:
                value = config[field]
                if value < min_val or value > max_val:
                    errors.append(f"Field '{field}' must be between {min_val} and {max_val}")
        return errors


class FlextConfigBuilder[T: FlextAbstractConfig]:
    """Generic configuration builder for type-safe config construction."""

    def __init__(self, config_class: type[T]) -> None:
        """Initialize builder with configuration class."""
        self.config_class = config_class
        self.data: dict[str, Any] = {}

    def with_field(self, field: str, value: object) -> FlextConfigBuilder[T]:
        """Add field value to configuration."""
        self.data[field] = value
        return self

    def from_env(self, prefix: str | None = None) -> FlextConfigBuilder[T]:
        """Load data from environment variables."""
        if prefix is None:
            prefix = "FLEXT"
        env_data = FlextConfigOperations.load_from_env(prefix)
        self.data.update(env_data)
        return self

    def from_file(self, file_path: str | Path) -> FlextConfigBuilder[T]:
        """Load data from JSON file."""
        file_data = FlextConfigOperations.load_from_file(file_path)
        self.data.update(file_data)
        return self

    def merge_dict(self, data: dict[str, object]) -> FlextConfigBuilder[T]:
        """Merge dictionary data."""
        self.data = FlextConfigOperations.merge_configs(self.data, data)
        return self

    def build(self) -> T:
        """Build configuration instance."""
        return self.config_class(**self.data)


class FlextSettings(FlextAbstractSettings, PydanticBaseSettings):
    """Base Pydantic settings class with environment loading."""

    class Config:
        """Pydantic settings configuration."""

        env_prefix = "FLEXT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def load_from_environment(self) -> dict[str, Any]:
        """Load settings from environment variables."""
        return self.dict()

    def get_env_prefix(self) -> str:
        """Get environment variable prefix."""
        return self.Config.env_prefix

# =============================================================================
# TYPED DICTIONARIES - Type definitions for configuration data
# =============================================================================


class FlextDatabaseConfigDict(TypedDict):
    """Database configuration dictionary type."""

    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    echo: bool


class FlextRedisConfigDict(TypedDict):
    """Redis configuration dictionary type."""

    host: str
    port: int
    password: str
    db: int
    decode_responses: bool
    socket_timeout: int
    socket_connect_timeout: int
    socket_keepalive: bool
    socket_keepalive_options: dict[str, int]
    connection_pool_max_connections: int


class FlextJWTConfigDict(TypedDict):
    """JWT configuration dictionary type."""

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int
    issuer: str
    audience: list[str]


class FlextLDAPConfigDict(TypedDict):
    """LDAP configuration dictionary type."""

    server: str
    port: int
    use_ssl: bool
    use_tls: bool
    bind_dn: str
    bind_password: str
    search_base: str
    search_filter: str
    attributes: list[str]
    timeout: int
    connection_timeout: int


class FlextOracleConfigDict(TypedDict):
    """Oracle configuration dictionary type."""

    host: str
    port: int
    service_name: str
    username: str
    password: str
    schema: str
    pool_min: int
    pool_max: int
    pool_increment: int
    connection_timeout: int
    fetch_arraysize: int
    autocommit: bool


class FlextSingerConfigDict(TypedDict):
    """Singer configuration dictionary type."""

    tap_executable: str
    target_executable: str
    config_file: str
    catalog_file: str
    state_file: str
    properties_file: str
    output_file: str


class FlextObservabilityConfigDict(TypedDict):
    """Observability configuration dictionary type."""

    logging_enabled: bool
    logging_level: str
    logging_format: str
    tracing_enabled: bool
    tracing_service_name: str
    tracing_environment: str
    metrics_enabled: bool
    metrics_port: int
    metrics_path: str
    health_check_enabled: bool
    health_check_port: int
    health_check_path: str

# =============================================================================
# CONCRETE CONFIGURATION MODELS - Production-ready configuration classes
# =============================================================================


class FlextBaseConfigModel(FlextAbstractConfig):
    """Base configuration model with common fields and validation."""

    name: str = Field(default="flext", description="Configuration name")
    version: str = Field(default="1.0.0", description="Configuration version")
    description: str = Field(default="FLEXT configuration", description="Configuration description")

    created_at: float = Field(default_factory=lambda: __import__("time").time())
    updated_at: float = Field(default_factory=lambda: __import__("time").time())

    environment: str = Field(default=DEFAULT_ENVIRONMENT, description="Environment name")
    debug: bool = Field(default=False, description="Debug mode enabled")

    @validator("environment")
    def validate_environment(self, v: str) -> str:  # noqa: N805 - Pydantic validator signature
        """Validate environment value."""
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    def validate_business_rules(self) -> bool:
        """Validate business rules for base configuration."""
        return True

    @classmethod
    def from_env(cls, prefix: str = "FLEXT") -> FlextBaseConfigModel:
        """Create configuration from environment variables."""
        return FlextConfigBuilder(cls).from_env(prefix).build()


class FlextConfig(FlextBaseConfigModel):
    """Main FLEXT configuration class."""

    # Core settings
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, description="Logging level")
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Default timeout in seconds")
    retries: int = Field(default=DEFAULT_RETRIES, description="Default retry count")
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, description="Default page size")

    # Feature flags
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")

    @validator("log_level")
    def validate_log_level(self, v: str) -> str:  # noqa: N805 - Pydantic validator signature
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()

    @validator("timeout", "retries", "page_size")
    def validate_positive_integers(self, v: int) -> int:  # noqa: N805 - Pydantic validator signature
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    def validate_business_rules(self) -> bool:
        """Validate FLEXT-specific business rules."""
        # Custom business logic validation
        return not (self.debug and self.environment == "production")


class FlextDatabaseConfig(FlextBaseConfigModel):
    """Database configuration model."""

    host: str = Field(description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    schema: str = Field(default="public", description="Database schema")

    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")

    # Query settings
    echo: bool = Field(default=False, description="Echo SQL queries")

    @validator("port")
    def validate_port(self, v: int) -> int:  # noqa: N805 - Pydantic validator signature
        """Validate port number."""
        if not 1 <= v <= 65535:
            message = "Port must be between 1 and 65535"
            raise ValueError(message)
        return v

    def validate_business_rules(self) -> bool:
        """Validate database business rules."""
        # Ensure pool settings are reasonable
        return not (self.pool_size > self.max_overflow)

    def get_connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class FlextRedisConfig(FlextBaseConfigModel):
    """Redis configuration model."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str = Field(default="", description="Redis password")
    db: int = Field(default=0, description="Redis database number")

    # Connection settings
    decode_responses: bool = Field(default=True, description="Decode responses to strings")
    socket_timeout: int = Field(default=30, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=30, description="Socket connect timeout in seconds")
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")

    # Pool settings
    connection_pool_max_connections: int = Field(default=50, description="Max pool connections")

    def validate_business_rules(self) -> bool:
        """Validate Redis business rules."""
        return True

    def get_connection_string(self) -> str:
        """Get Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class FlextJWTConfig(FlextBaseConfigModel):
    """JWT configuration model."""

    secret_key: str = Field(description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry in minutes")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry in days")
    issuer: str = Field(default="flext", description="JWT issuer")
    audience: list[str] = Field(default_factory=list, description="JWT audience")

    @validator("algorithm")
    def validate_algorithm(self, v: str) -> str:  # noqa: N805 - Pydantic validator signature
        """Validate JWT algorithm."""
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed:
            message = f"Algorithm must be one of: {allowed}"
            raise ValueError(message)
        return v

    def validate_business_rules(self) -> bool:
        """Validate JWT business rules."""
        MIN_SECRET_LEN = 32
        return not (len(self.secret_key) < MIN_SECRET_LEN)


class FlextOracleConfig(FlextBaseConfigModel):
    """Oracle database configuration model."""

    host: str = Field(description="Oracle host")
    port: int = Field(default=1521, description="Oracle port")
    service_name: str = Field(description="Oracle service name")
    username: str = Field(description="Oracle username")
    password: str = Field(description="Oracle password")
    schema: str = Field(description="Oracle schema")

    # Connection pool settings
    pool_min: int = Field(default=1, description="Minimum pool connections")
    pool_max: int = Field(default=10, description="Maximum pool connections")
    pool_increment: int = Field(default=1, description="Pool increment size")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    fetch_arraysize: int = Field(default=1000, description="Fetch array size")
    autocommit: bool = Field(default=False, description="Enable autocommit")

    def validate_business_rules(self) -> bool:
        """Validate Oracle business rules."""
        return not (self.pool_min > self.pool_max)

    def get_connection_string(self) -> str:
        """Get Oracle connection string."""
        return f"oracle://{self.username}:{self.password}@{self.host}:{self.port}/{self.service_name}"


class FlextLDAPConfig(FlextBaseConfigModel):
    """LDAP configuration model."""

    server: str = Field(description="LDAP server")
    port: int = Field(default=389, description="LDAP port")
    use_ssl: bool = Field(default=False, description="Use SSL")
    use_tls: bool = Field(default=False, description="Use TLS")
    bind_dn: str = Field(description="Bind DN")
    bind_password: str = Field(description="Bind password")
    search_base: str = Field(description="Search base DN")
    search_filter: str = Field(default="(objectClass=*)", description="Search filter")
    attributes: list[str] = Field(default_factory=list, description="Attributes to retrieve")
    timeout: int = Field(default=30, description="LDAP timeout in seconds")

    @validator("port")
    def validate_ldap_port(self, v: int) -> int:  # noqa: N805 - Pydantic validator signature
        """Validate LDAP port."""
        if v not in {389, 636, 3268, 3269} and not (1 <= v <= 65535):
            message = "Invalid LDAP port"
            raise ValueError(message)
        return v

    def validate_business_rules(self) -> bool:
        """Validate LDAP business rules."""
        return not (self.use_ssl and self.use_tls)


class FlextSingerConfig(FlextBaseConfigModel):
    """Singer configuration model."""

    tap_executable: str = Field(description="Tap executable path")
    target_executable: str = Field(description="Target executable path")
    config_file: str = Field(description="Configuration file path")
    catalog_file: str = Field(description="Catalog file path")
    state_file: str = Field(description="State file path")
    properties_file: str = Field(description="Properties file path")
    output_file: str = Field(default="/tmp/singer_output.jsonl", description="Output file path")

    def validate_business_rules(self) -> bool:
        """Validate Singer business rules."""
        # Check if executables exist
        for executable in [self.tap_executable, self.target_executable]:
            if not Path(executable).exists():
                return False
        return True


class FlextObservabilityConfig(FlextBaseConfigModel):
    """Observability configuration model."""

    # Logging configuration
    logging_enabled: bool = Field(default=True, description="Enable logging")
    logging_level: str = Field(default="INFO", description="Logging level")
    logging_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Tracing configuration
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    tracing_service_name: str = Field(default="flext", description="Service name for tracing")
    tracing_environment: str = Field(default="development", description="Environment for tracing")

    # Metrics configuration
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8080, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Health check configuration
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_port: int = Field(default=8081, description="Health check port")
    health_check_path: str = Field(default="/health", description="Health check path")

    def validate_business_rules(self) -> bool:
        """Validate observability business rules."""
        return True


class FlextPerformanceConfig(FlextBaseConfigModel):
    """Performance configuration model."""

    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")

    # Batch processing settings
    batch_size: int = Field(default=1000, description="Default batch size")
    max_workers: int = Field(default=4, description="Maximum worker threads")

    # Timeout settings
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    connection_timeout: int = Field(default=10, description="Connection timeout in seconds")

    def validate_business_rules(self) -> bool:
        """Validate performance business rules."""
        MAX_WORKERS = 20
        return not (self.max_workers > MAX_WORKERS)


class FlextApplicationConfig(FlextBaseConfigModel):
    """Application-level configuration model."""

    app_name: str = Field(default="flext", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(default="FLEXT Data Integration Platform")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")

    # Security settings
    cors_origins: list[str] = Field(default_factory=list, description="CORS origins")
    allowed_hosts: list[str] = Field(default_factory=list, description="Allowed hosts")

    def validate_business_rules(self) -> bool:
        """Validate application business rules."""
        return True


class FlextDataIntegrationConfig(FlextBaseConfigModel):
    """Data integration configuration model."""

    # Pipeline settings
    default_batch_size: int = Field(default=1000, description="Default batch size")
    max_retries: int = Field(default=3, description="Maximum retries")
    retry_delay: int = Field(default=1, description="Retry delay in seconds")

    # Data quality settings
    enable_validation: bool = Field(default=True, description="Enable data validation")
    validation_threshold: float = Field(default=0.95, description="Validation threshold")

    # Monitoring settings
    enable_monitoring: bool = Field(default=True, description="Enable pipeline monitoring")
    metrics_interval: int = Field(default=60, description="Metrics collection interval")

    def validate_business_rules(self) -> bool:
        """Validate data integration business rules."""
        return 0.0 <= self.validation_threshold <= 1.0

# =============================================================================
# CONFIGURATION FACTORY AND UTILITIES
# =============================================================================


class FlextConfigFactory:
    """Factory for creating configuration instances."""

    _config_registry: dict[str, type[FlextAbstractConfig]] = {
        "base": FlextBaseConfigModel,
        "main": FlextConfig,
        "database": FlextDatabaseConfig,
        "redis": FlextRedisConfig,
        "jwt": FlextJWTConfig,
        "oracle": FlextOracleConfig,
        "ldap": FlextLDAPConfig,
        "singer": FlextSingerConfig,
        "observability": FlextObservabilityConfig,
        "performance": FlextPerformanceConfig,
        "application": FlextApplicationConfig,
        "data_integration": FlextDataIntegrationConfig,
    }

    @classmethod
    def create(
        self,
        config_type: str,
        **kwargs: object,
    ) -> FlextAbstractConfig:
        """Create configuration instance by type."""
        if config_type not in self._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")

        config_class = self._config_registry[config_type]
        return config_class(**kwargs)

    @classmethod
    def create_from_env(
        cls,
        config_type: str,
        prefix: str = "FLEXT",
    ) -> FlextAbstractConfig:
        """Create configuration from environment variables."""
        if config_type not in cls._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")

        config_class = cls._config_registry[config_type]
        return FlextConfigBuilder(config_class).from_env(prefix).build()

    @classmethod
    def create_from_file(
        cls,
        config_type: str,
        file_path: str | Path,
    ) -> FlextAbstractConfig:
        """Create configuration from file."""
        if config_type not in cls._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")

        config_class = cls._config_registry[config_type]
        return FlextConfigBuilder(config_class).from_file(file_path).build()

    @classmethod
    def register_config(
        cls,
        name: str,
        config_class: type[FlextAbstractConfig],
    ) -> None:
        """Register new configuration class."""
        cls._config_registry[name] = config_class

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of registered configuration types."""
        return list(cls._config_registry.keys())

# =============================================================================
# CONFIGURATION MANAGER AND ORCHESTRATION
# =============================================================================


class FlextConfigDefaults:
    """Default configuration values and patterns."""

    @staticmethod
    def get_database_defaults() -> dict[str, Any]:
        """Get database configuration defaults."""
        return {
            "host": "localhost",
            "port": 5432,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "echo": False,
        }

    @staticmethod
    def get_redis_defaults() -> dict[str, Any]:
        """Get Redis configuration defaults."""
        return {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "decode_responses": True,
            "connection_pool_max_connections": 50,
        }

    @staticmethod
    def get_observability_defaults() -> dict[str, Any]:
        """Get observability configuration defaults."""
        return {
            "logging_enabled": True,
            "logging_level": "INFO",
            "tracing_enabled": False,
            "metrics_enabled": True,
            "health_check_enabled": True,
        }


class FlextConfigOps:
    """Configuration operations and management utilities."""

    @staticmethod
    def validate_config(
        config: FlextAbstractConfig,
        strict: bool = True,
    ) -> tuple[bool, list[str]]:
        """Validate configuration instance."""
        errors = []

        try:
            # Pydantic validation
            config.dict()
        except ValueError as e:
            errors.append(f"Validation error: {e}")

        # Business rules validation
        try:
            if not config.validate_business_rules():
                errors.append("Business rules validation failed")
        except Exception as e:
            errors.append(f"Business rules error: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def merge_configs(
        *configs: FlextAbstractConfig,
    ) -> dict[str, Any]:
        """Merge multiple configuration instances."""
        merged = {}
        for config in configs:
            config_dict = config.to_dict()
            merged = FlextConfigOperations.merge_configs(merged, config_dict)
        return merged

    @staticmethod
    def export_config(
        config: FlextAbstractConfig,
        format_type: str = "json",
    ) -> str:
        """Export configuration to string format."""
        if format_type == "json":
            return config.to_json()
        if format_type == "dict":
            return str(config.to_dict())
        raise ValueError(f"Unsupported format: {format_type}")


class FlextConfigValidation:
    """Configuration validation utilities and rules."""

    @staticmethod
    def validate_environment_config(config: dict[str, Any]) -> list[str]:
        """Validate environment-specific configuration."""
        errors = []
        environment = config.get("environment", "development")

        if environment == "production":
            # Production-specific validations
            if config.get("debug", False):
                errors.append("Debug mode should not be enabled in production")

            if config.get("log_level", "INFO") == "DEBUG":
                errors.append("Debug logging should not be enabled in production")

        return errors

    @staticmethod
    def validate_security_config(config: dict[str, Any]) -> list[str]:
        """Validate security-related configuration."""
        errors = []

        # Check for default passwords
        password_fields = ["password", "bind_password", "secret_key"]
        for field in password_fields:
            if field in config:
                value = str(config[field])
                if value in {"password", "REDACTED_LDAP_BIND_PASSWORD", "root", "secret"}:
                    errors.append(f"Default password detected in field: {field}")

                if len(value) < 8:
                    errors.append(f"Password too short in field: {field}")

        return errors

    @staticmethod
    def validate_performance_config(config: dict[str, Any]) -> list[str]:
        """Validate performance-related configuration."""
        errors = []

        # Check pool sizes
        pool_size = config.get("pool_size", 10)
        max_overflow = config.get("max_overflow", 20)

        if pool_size > 50:
            errors.append("Pool size too large (max recommended: 50)")

        if max_overflow > pool_size * 2:
            errors.append("Max overflow should not exceed 2x pool size")

        return errors


class FlextConfigManager:
    """High-level configuration management and orchestration."""

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._configs: dict[str, FlextAbstractConfig] = {}
        self._factory = FlextConfigFactory()

    def load_config(
        self,
        name: str,
        config_type: str,
        source: str = "env",
        source_path: str | Path | None = None,
        **kwargs: object,
    ) -> FlextAbstractConfig:
        """Load configuration from various sources."""
        if source == "env":
            config = self._factory.create_from_env(config_type, **kwargs)
        elif source == "file" and source_path:
            config = self._factory.create_from_file(config_type, source_path)
        elif source == "dict":
            config = self._factory.create(config_type, **kwargs)
        else:
            raise ValueError(f"Unknown source type: {source}")

        self._configs[name] = config
        return config

    def get_config(self, name: str) -> FlextAbstractConfig | None:
        """Get configuration by name."""
        return self._configs.get(name)

    def list_configs(self) -> list[str]:
        """List all loaded configuration names."""
        return list(self._configs.keys())

    def validate_all_configs(self) -> dict[str, tuple[bool, list[str]]]:
        """Validate all loaded configurations."""
        results = {}
        for name, config in self._configs.items():
            results[name] = FlextConfigOps.validate_config(config)
        return results

    def export_all_configs(self, format_type: str = "json") -> dict[str, str]:
        """Export all configurations."""
        exports = {}
        for name, config in self._configs.items():
            exports[name] = FlextConfigOps.export_config(config, format_type)
        return exports

# =============================================================================
# COMPATIBILITY LAYER - Legacy support and migration
# =============================================================================


def safe_get_env_var(
    var_name: str,
    default: str | None = None,
    var_type: type = str,
) -> Any:
    """Safely get environment variable with type conversion."""
    value = os.getenv(var_name, default)
    if value is None:
        return None

    try:
        if var_type == bool:
            return str(value).lower() in {"true", "1", "yes", "on"}
        if var_type == int:
            return int(value)
        if var_type == float:
            return float(value)
        if var_type == str:
            return str(value)
        return var_type(value)
    except (ValueError, TypeError):
        return default


def safe_load_json_file(file_path: str | Path) -> dict[str, Any]:
    """Safely load JSON configuration file."""
    try:
        return FlextConfigOperations.load_from_file(file_path)
    except Exception:
        return {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge configuration dictionaries (legacy function)."""
    return FlextConfigOperations.merge_configs(base, override)

# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    # Abstract Base Classes and Protocols
    "FlextAbstractConfig",
    "FlextAbstractSettings",
    "FlextConfigValidatorProtocol",
    "FlextConfigLoaderProtocol",
    "FlextConfigMergerProtocol",
    "FlextConfigSerializerProtocol",
    # Utility Classes
    "FlextConfigOperations",
    "FlextConfigValidator",
    "FlextConfigBuilder",
    "FlextSettings",
    # Typed Dictionaries
    "FlextDatabaseConfigDict",
    "FlextRedisConfigDict",
    "FlextJWTConfigDict",
    "FlextLDAPConfigDict",
    "FlextOracleConfigDict",
    "FlextSingerConfigDict",
    "FlextObservabilityConfigDict",
    # Concrete Configuration Models
    "FlextBaseConfigModel",
    "FlextConfig",
    "FlextDatabaseConfig",
    "FlextRedisConfig",
    "FlextJWTConfig",
    "FlextOracleConfig",
    "FlextLDAPConfig",
    "FlextSingerConfig",
    "FlextObservabilityConfig",
    "FlextPerformanceConfig",
    "FlextApplicationConfig",
    "FlextDataIntegrationConfig",
    # Factory and Management
    "FlextConfigFactory",
    "FlextConfigDefaults",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextConfigManager",
    # Legacy/Compatibility Functions
    "safe_get_env_var",
    "safe_load_json_file",
    "merge_configs",
    # Constants
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_ENVIRONMENT",
    "CONFIG_VALIDATION_MESSAGES",
]
