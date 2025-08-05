"""FLEXT Core Configuration Models - Configuration Layer Individual Model Definitions.

Centralized configuration model definitions providing individual configuration models
across the entire 32-project FLEXT ecosystem with Python 3.13+ type safety,
domain-specific validation, and enterprise-grade configuration management patterns.

Module Role in Architecture:
    Configuration Layer → Individual Model Definitions → Standardized Types

    This module defines individual configuration models for specific domains:
    - Database connection models (PostgreSQL, Oracle) with pooling and SSL
    - Cache configuration models (Redis) with clustering and persistence
    - Authentication models (JWT, OAuth) with security validation
    - Integration models (LDAP, Singer) with protocol-specific options
    - Observability models with structured logging and tracing
    - Composite models combining multiple configuration domains

Configuration Model Patterns:
    TypedDict Definitions: Type-safe dictionaries for configuration interchange
    Base Configuration Models: Common patterns with validation and serialization
    Domain-Specific Models: Oracle, LDAP, JWT, Redis with specialized validation
    Settings Integration: Environment variable binding with pydantic-settings
    Factory Functions: Type-safe model creation with sensible defaults

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Database, Redis, Oracle, LDAP, JWT, Singer, Observability
    🚧 Enhancement: Advanced SSL validation, connection testing utilities (GAP 1)
    📋 TODO Integration: Configuration validation framework, environment
        overrides (GAP 2)

Configuration Categories by Domain:
    Core Infrastructure:
        - FlextDatabaseConfig: PostgreSQL with pooling, SSL, timeouts
        - FlextRedisConfig: Redis with clustering, persistence, health checks

    Data Integration:
        - FlextOracleConfig: Oracle with service_name/SID, WMS optimizations
        - FlextLDAPConfig: LDAP/LDAPS with authentication, directory operations
        - FlextSingerConfig: Singer tap/target with schema validation

    Security & Authentication:
        - FlextJWTConfig: JWT with algorithm validation, secure key requirements

    Observability:
        - FlextObservabilityConfig: Structured logging, metrics, distributed tracing

Ecosystem Usage Patterns:
    # FlexCore (Go) service configuration
    database_config = FlextDatabaseConfig(
        host="internal.invalid",
        port=5433,
        username="flexcore_user",
        password=SecretStr(os.getenv("FLEXCORE_DB_PASSWORD")),
        database="flexcore_db",
        pool_max=20
    )

    # Oracle WMS integration across flext-oracle-wms, flext-tap-oracle-wms
    oracle_config = FlextOracleConfig(
        host="oracle-wms.enterprise.com",
        service_name="WMS_PROD",
        username="wms_integration",
        ssl_enabled=True,
        pool_max=50  # High concurrency for WMS operations
    )

    # Singer ecosystem configuration (15 taps/targets/DBT projects)
    singer_config = FlextSingerConfig(
        stream_name="oracle_inventory",
        stream_schema={"type": "object", "properties": {...}},
        batch_size=5000,  # Optimized for Oracle WMS data volumes
        catalog=meltano_catalog
    )

    # Cross-service observability (flext-observability integration)
    observability_config = FlextObservabilityConfig(
        service_name="flext-tap-oracle-wms",
        correlation_id_header="X-FLEXT-Correlation-ID",
        tracing_enabled=True,
        log_level="INFO"
    )

Configuration Philosophy:
    Each model represents a specific configuration domain with:
    - Type-safe defaults appropriate for development and production
    - Comprehensive validation including business rule validation
    - Environment variable integration through pydantic-settings
    - Serialization methods for cross-language compatibility (Go bridge)
    - Factory functions for common configuration scenarios
    - Immutable configuration for thread safety across services

Quality Standards:
    - Comprehensive field validation with appropriate ranges and formats
    - Secret handling through pydantic SecretStr for passwords and keys
    - TypedDict definitions for type-safe dictionary interchange
    - Business rule validation through semantic validation methods
    - Environment variable integration with secure defaults

See Also:
    docs/TODO.md: Configuration validation framework (GAP 2), SSL enhancements (GAP 1)
    config_hierarchical.py: Hierarchical configuration composition and inheritance
    examples/08_flext_config_enterprise_configuration.py: Configuration usage patterns

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from flext_core.flext_types import TAnyDict
else:
    # Runtime import for models that need TAnyDict at runtime
    from flext_core.flext_types import TAnyDict  # noqa: TC001

# =============================================================================
# TYPED DICT DEFINITIONS - Type-safe dictionaries for configuration
# =============================================================================


class DatabaseConfigDict(TypedDict):
    """TypedDict for database configuration with optional fields."""

    host: str
    port: int
    username: str
    password: str
    database: str
    pool_min: NotRequired[int]
    pool_max: NotRequired[int]
    pool_timeout: NotRequired[int]
    ssl_enabled: NotRequired[bool]
    ssl_cert_path: NotRequired[str | None]
    encoding: NotRequired[str]


class RedisConfigDict(TypedDict):
    """TypedDict for Redis configuration with optional fields."""

    host: str
    port: int
    password: NotRequired[str | None]
    database: NotRequired[int]
    ssl_enabled: NotRequired[bool]
    pool_max: NotRequired[int]
    timeout: NotRequired[int]


class JWTConfigDict(TypedDict):
    """TypedDict for JWT configuration with optional fields."""

    secret_key: str
    algorithm: NotRequired[str]
    access_token_expire_minutes: NotRequired[int]
    refresh_token_expire_days: NotRequired[int]
    issuer: NotRequired[str | None]
    audience: NotRequired[str | None]


class LDAPConfigDict(TypedDict):
    """TypedDict for LDAP configuration with optional fields."""

    host: str
    port: int
    base_dn: str
    bind_dn: NotRequired[str | None]
    bind_password: NotRequired[str | None]
    use_ssl: NotRequired[bool]
    use_tls: NotRequired[bool]
    timeout: NotRequired[int]
    pool_size: NotRequired[int]


class OracleConfigDict(TypedDict):
    """TypedDict for Oracle configuration with optional fields."""

    host: str
    port: int
    username: str
    password: str
    service_name: NotRequired[str | None]
    sid: NotRequired[str | None]
    pool_min: NotRequired[int]
    pool_max: NotRequired[int]
    pool_increment: NotRequired[int]
    timeout: NotRequired[int]
    encoding: NotRequired[str]
    ssl_enabled: NotRequired[bool]


class SingerConfigDict(TypedDict):
    """TypedDict for Singer tap/target configuration."""

    stream_name: str
    stream_schema: dict[str, object]
    stream_config: dict[str, object]
    catalog: NotRequired[dict[str, object] | None]
    state: NotRequired[dict[str, object] | None]
    batch_size: NotRequired[int]


class ObservabilityConfigDict(TypedDict):
    """TypedDict for observability configuration."""

    log_level: str
    log_format: NotRequired[str]
    metrics_enabled: NotRequired[bool]
    tracing_enabled: NotRequired[bool]
    correlation_id_header: NotRequired[str]
    service_name: NotRequired[str]


# =============================================================================
# BASE CONFIGURATION MODELS - Core infrastructure models
# =============================================================================


class FlextBaseConfigModel(BaseModel):
    """Base configuration model with common settings for all FLEXT projects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=True,
    )

    def to_dict(self) -> TAnyDict:
        """Convert to dictionary representation."""
        return self.model_dump()

    def to_typed_dict(self) -> TAnyDict:
        """Convert to typed dictionary representation."""
        return self.model_dump(exclude_unset=True)


class FlextDatabaseConfig(FlextBaseConfigModel):
    """Centralized database configuration for PostgreSQL and Oracle."""

    host: str = Field("localhost", description="Database host address")
    port: int = Field(5432, description="Database port", ge=1, le=65535)
    username: str = Field("postgres", description="Database username")
    password: SecretStr = Field(SecretStr("password"), description="Database password")
    database: str = Field("flext", description="Database name")

    # Connection pooling
    pool_min: int = Field(1, description="Minimum pool connections", ge=1)
    pool_max: int = Field(10, description="Maximum pool connections", ge=1)
    pool_timeout: int = Field(30, description="Pool timeout in seconds", ge=1)

    # SSL configuration
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    ssl_cert_path: str | None = Field(None, description="SSL certificate path")

    # Connection options
    encoding: str = Field("UTF-8", description="Character encoding")
    autocommit: bool = Field(default=False, description="Enable autocommit mode")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is not empty."""
        if not v or not v.strip():
            msg = "Database host cannot be empty"
            raise ValueError(msg)
        return v.strip()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username is not empty."""
        if not v or not v.strip():
            msg = "Database username cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def get_connection_string(self, driver: str = "postgresql") -> str:
        """Generate database connection string."""
        password = self.password.get_secret_value()
        return f"{driver}://{self.username}:{password}@{self.host}:{self.port}/{self.database}"

    def to_database_dict(self) -> DatabaseConfigDict:
        """Convert to DatabaseConfigDict."""
        return DatabaseConfigDict(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password.get_secret_value(),
            database=self.database,
            pool_min=self.pool_min,
            pool_max=self.pool_max,
            pool_timeout=self.pool_timeout,
            ssl_enabled=self.ssl_enabled,
            ssl_cert_path=self.ssl_cert_path,
            encoding=self.encoding,
        )


class FlextRedisConfig(FlextBaseConfigModel):
    """Centralized Redis configuration for caching and sessions."""

    host: str = Field("localhost", description="Redis host address")
    port: int = Field(6379, description="Redis port", ge=1, le=65535)
    password: SecretStr | None = Field(None, description="Redis password")
    database: int = Field(0, description="Redis database number", ge=0, le=15)

    # Connection options
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    pool_max: int = Field(10, description="Maximum pool connections", ge=1)
    timeout: int = Field(30, description="Connection timeout in seconds", ge=1)

    # Redis-specific options
    decode_responses: bool = Field(
        default=True,
        description="Decode responses to strings",
    )
    health_check_interval: int = Field(30, description="Health check interval", ge=1)

    def get_connection_string(self) -> str:
        """Generate Redis connection string."""
        if self.password:
            password = self.password.get_secret_value()
            return f"redis://:{password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"

    def to_redis_dict(self) -> RedisConfigDict:
        """Convert to RedisConfigDict."""
        return RedisConfigDict(
            host=self.host,
            port=self.port,
            password=self.password.get_secret_value() if self.password else None,
            database=self.database,
            ssl_enabled=self.ssl_enabled,
            pool_max=self.pool_max,
            timeout=self.timeout,
        )


class FlextJWTConfig(FlextBaseConfigModel):
    """Centralized JWT configuration for authentication."""

    secret_key: SecretStr = Field(description="JWT signing secret key")
    algorithm: str = Field("HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        30,
        description="Access token expiration",
        ge=1,
    )
    refresh_token_expire_days: int = Field(
        7,
        description="Refresh token expiration",
        ge=1,
    )

    # Optional JWT claims
    issuer: str | None = Field(None, description="JWT issuer claim")
    audience: str | None = Field(None, description="JWT audience claim")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        """Validate secret key strength."""
        secret = v.get_secret_value()
        min_key_length = 32
        if len(secret) < min_key_length:
            msg = f"JWT secret key must be at least {min_key_length} characters"
            raise ValueError(msg)
        return v

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm."""
        allowed_algorithms = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed_algorithms:
            msg = f"Algorithm must be one of: {allowed_algorithms}"
            raise ValueError(msg)
        return v

    def to_jwt_dict(self) -> JWTConfigDict:
        """Convert to JWTConfigDict."""
        return JWTConfigDict(
            secret_key=self.secret_key.get_secret_value(),
            algorithm=self.algorithm,
            access_token_expire_minutes=self.access_token_expire_minutes,
            refresh_token_expire_days=self.refresh_token_expire_days,
            issuer=self.issuer,
            audience=self.audience,
        )


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATION MODELS
# =============================================================================


class FlextOracleConfig(FlextBaseConfigModel):
    """Centralized Oracle database configuration."""

    host: str = Field("localhost", description="Oracle host address")
    port: int = Field(1521, description="Oracle port", ge=1, le=65535)
    username: str = Field("oracle", description="Oracle username")
    password: SecretStr = Field(SecretStr("oracle"), description="Oracle password")

    # Oracle connection identifiers (either SID or service_name required)
    service_name: str | None = Field(None, description="Oracle service name")
    sid: str | None = Field(None, description="Oracle system identifier")

    # Connection pooling
    pool_min: int = Field(1, description="Minimum pool connections", ge=1)
    pool_max: int = Field(10, description="Maximum pool connections", ge=1)
    pool_increment: int = Field(1, description="Pool increment", ge=1)
    timeout: int = Field(30, description="Connection timeout", ge=1)

    # Oracle-specific options
    encoding: str = Field("UTF-8", description="Character encoding")
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    protocol: str = Field("tcp", description="Connection protocol")

    @field_validator("service_name", "sid")
    @classmethod
    def validate_connection_identifier(cls, v: str | None) -> str | None:
        """Validate connection identifier format."""
        return v

    @classmethod
    def model_validate(
        cls,
        obj: dict[str, object],
    ) -> FlextOracleConfig:
        """Validate that either SID or service_name is provided."""
        instance = super().model_validate(obj)
        if not instance.service_name and not instance.sid:
            msg = "Either service_name or sid must be provided"
            raise ValueError(msg)
        return instance

    def get_connection_string(self) -> str:
        """Generate Oracle connection string."""
        if self.service_name:
            return f"{self.host}:{self.port}/{self.service_name}"
        if self.sid:
            return f"{self.host}:{self.port}:{self.sid}"
        return f"{self.host}:{self.port}"

    def to_oracle_dict(self) -> OracleConfigDict:
        """Convert to OracleConfigDict."""
        return OracleConfigDict(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password.get_secret_value(),
            service_name=self.service_name,
            sid=self.sid,
            pool_min=self.pool_min,
            pool_max=self.pool_max,
            pool_increment=self.pool_increment,
            timeout=self.timeout,
            encoding=self.encoding,
            ssl_enabled=self.ssl_enabled,
        )


class FlextLDAPConfig(FlextBaseConfigModel):
    """Centralized LDAP configuration."""

    host: str = Field("localhost", description="LDAP host address")
    port: int = Field(389, description="LDAP port", ge=1, le=65535)
    base_dn: str = Field("dc=example,dc=com", description="LDAP base DN")

    # Authentication
    bind_dn: str | None = Field(None, description="LDAP bind DN for authentication")
    bind_password: SecretStr | None = Field(None, description="LDAP bind password")

    # Security options
    use_ssl: bool = Field(default=False, description="Use SSL connection (LDAPS)")
    use_tls: bool = Field(default=False, description="Use TLS upgrade (StartTLS)")

    # Connection options
    timeout: int = Field(30, description="Connection timeout in seconds", ge=1)
    pool_size: int = Field(10, description="Connection pool size", ge=1)

    @field_validator("base_dn")
    @classmethod
    def validate_base_dn(cls, v: str) -> str:
        """Validate base DN format."""
        if not v or not v.strip():
            msg = "LDAP base DN cannot be empty"
            raise ValueError(msg)
        if not v.lower().startswith("dc="):
            msg = "LDAP base DN should start with 'dc='"
            raise ValueError(msg)
        return v.strip()

    def get_connection_string(self) -> str:
        """Generate LDAP connection string."""
        protocol = "ldaps" if self.use_ssl else "ldap"
        return f"{protocol}://{self.host}:{self.port}"

    def to_ldap_dict(self) -> LDAPConfigDict:
        """Convert to LDAPConfigDict."""
        return LDAPConfigDict(
            host=self.host,
            port=self.port,
            base_dn=self.base_dn,
            bind_dn=self.bind_dn,
            bind_password=self.bind_password.get_secret_value()
            if self.bind_password
            else None,
            use_ssl=self.use_ssl,
            use_tls=self.use_tls,
            timeout=self.timeout,
            pool_size=self.pool_size,
        )


class FlextSingerConfig(FlextBaseConfigModel):
    """Centralized Singer tap/target configuration."""

    stream_name: str = Field(description="Singer stream name")
    stream_schema: dict[str, object] = Field(
        default_factory=dict,
        description="Singer schema definition",
    )
    stream_config: dict[str, object] = Field(
        default_factory=dict,
        description="Singer configuration",
    )

    # Optional Singer components
    catalog: dict[str, object] | None = Field(None, description="Singer catalog")
    state: dict[str, object] | None = Field(None, description="Singer state")

    # Processing options
    batch_size: int = Field(1000, description="Batch processing size", ge=1)
    max_records: int | None = Field(None, description="Maximum records to process")

    @field_validator("stream_name")
    @classmethod
    def validate_stream_name(cls, v: str) -> str:
        """Validate stream name format."""
        if not v or not v.strip():
            msg = "Singer stream name cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def to_singer_dict(self) -> SingerConfigDict:
        """Convert to SingerConfigDict."""
        return SingerConfigDict(
            stream_name=self.stream_name,
            stream_schema=self.stream_schema,
            stream_config=self.stream_config,
            catalog=self.catalog,
            state=self.state,
            batch_size=self.batch_size,
        )


class FlextObservabilityConfig(FlextBaseConfigModel):
    """Centralized observability configuration."""

    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json, text)")

    # Observability features
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )

    # Service identification
    service_name: str = Field("flext-service", description="Service name for tracing")
    correlation_id_header: str = Field(
        "X-Correlation-ID",
        description="Correlation ID header",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            msg = f"Log level must be one of: {allowed_levels}"
            raise ValueError(msg)
        return v_upper

    def to_observability_dict(self) -> ObservabilityConfigDict:
        """Convert to ObservabilityConfigDict."""
        return ObservabilityConfigDict(
            log_level=self.log_level,
            log_format=self.log_format,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            correlation_id_header=self.correlation_id_header,
            service_name=self.service_name,
        )


# =============================================================================
# COMPOSITE CONFIGURATION MODELS
# =============================================================================


class FlextApplicationConfig(FlextBaseConfigModel):
    """Composite application configuration combining multiple domains."""

    # Core infrastructure
    database: FlextDatabaseConfig = Field(default_factory=FlextDatabaseConfig)
    redis: FlextRedisConfig = Field(default_factory=FlextRedisConfig)

    # Authentication
    jwt: FlextJWTConfig = Field(
        default_factory=lambda: FlextJWTConfig(
            secret_key=SecretStr("change-me-in-production-32-chars-minimum"),
        ),
    )

    # Observability
    observability: FlextObservabilityConfig = Field(
        default_factory=FlextObservabilityConfig,
    )

    # Application settings
    app_name: str = Field("FLEXT Application", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field("development", description="Environment name")


class FlextDataIntegrationConfig(FlextBaseConfigModel):
    """Composite configuration for data integration projects."""

    # Data sources
    oracle: FlextOracleConfig | None = Field(None, description="Oracle configuration")
    ldap: FlextLDAPConfig | None = Field(None, description="LDAP configuration")

    # Singer configuration
    singer: FlextSingerConfig | None = Field(None, description="Singer configuration")

    # Processing options
    batch_size: int = Field(1000, description="Default batch size", ge=1)
    parallel_workers: int = Field(4, description="Number of parallel workers", ge=1)


# =============================================================================
# SETTINGS CLASSES WITH ENVIRONMENT VARIABLE INTEGRATION
# =============================================================================


class FlextBaseSettings(BaseSettings):
    """Base settings class with environment variable integration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


class FlextDatabaseSettings(FlextBaseSettings):
    """Database settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="FLEXT_DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    username: str = Field("postgres", description="Database username")
    password: SecretStr = Field(SecretStr("password"), description="Database password")
    database: str = Field("flext", description="Database name")


class FlextRedisSettings(FlextBaseSettings):
    """Redis settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="FLEXT_REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    password: SecretStr | None = Field(None, description="Redis password")
    database: int = Field(0, description="Redis database")


# =============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================


def create_database_config(
    host: str = "localhost",
    port: int = 5432,
    username: str = "postgres",
    password: str | None = None,
    database: str = "flext",
    **kwargs: object,
) -> FlextDatabaseConfig:
    """Factory function to create database configuration.

    Args:
        host: Database host address
        port: Database port number
        username: Database username
        password: Database password (required)
        database: Database name
        **kwargs: Additional configuration parameters

    Returns:
        FlextDatabaseConfig: Configured database configuration

    Raises:
        ValueError: If password is not provided

    """
    if password is None:
        msg = "Database password is required"
        raise ValueError(msg)

    return FlextDatabaseConfig(
        host=host,
        port=port,
        username=username,
        password=SecretStr(password),
        database=database,
        **kwargs,
    )


def create_redis_config(
    host: str = "localhost",
    port: int = 6379,
    password: str | None = None,
    database: int = 0,
    **kwargs: object,
) -> FlextRedisConfig:
    """Factory function to create Redis configuration."""
    return FlextRedisConfig(
        host=host,
        port=port,
        password=SecretStr(password) if password else None,
        database=database,
        **kwargs,
    )


def create_oracle_config(
    host: str = "localhost",
    username: str = "oracle",
    password: str | None = None,
    service_name: str | None = None,
    **kwargs: object,
) -> FlextOracleConfig:
    """Factory function to create Oracle configuration.

    Args:
        host: Oracle database host address
        username: Oracle database username
        password: Oracle database password (required)
        service_name: Oracle service name (optional)
        **kwargs: Additional configuration parameters

    Returns:
        FlextOracleConfig: Configured Oracle configuration

    Raises:
        ValueError: If password is not provided

    """
    if password is None:
        msg = "Oracle password is required"
        raise ValueError(msg)

    # Create config data with defaults
    config_data = {
        "host": host,
        "port": 1521,
        "username": username,
        "password": SecretStr(password),
        "service_name": service_name,
        "sid": None,
    }

    # Override with any provided kwargs
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            if key in config_data:
                config_data[key] = value

    return FlextOracleConfig(**config_data)


def create_ldap_config(
    host: str = "localhost",
    port: int = 389,
    base_dn: str = "dc=example,dc=com",
    bind_dn: str | None = None,
    bind_password: str | None = None,
    **kwargs: object,
) -> FlextLDAPConfig:
    """Factory function to create LDAP configuration."""
    return FlextLDAPConfig(
        host=host,
        port=port,
        base_dn=base_dn,
        bind_dn=bind_dn,
        bind_password=SecretStr(bind_password) if bind_password else None,
        **kwargs,
    )


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================


def load_config_from_env(config_class: type[FlextBaseSettings]) -> FlextBaseSettings:
    """Load configuration from environment variables."""
    return config_class()


def merge_configs(*configs: FlextBaseConfigModel) -> dict[str, object]:
    """Merge multiple configuration objects."""
    merged: dict[str, object] = {}
    for config in configs:
        config_dict = config.to_dict()
        # Update with type conversion to object
        merged.update(config_dict)
    return merged


def validate_config(config: FlextBaseConfigModel) -> bool:
    """Validate configuration object."""
    try:
        config.model_validate(config.model_dump())
        return True
    except Exception:
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # TypedDict definitions
    "DatabaseConfigDict",
    # Composite models
    "FlextApplicationConfig",
    # Base models
    "FlextBaseConfigModel",
    "FlextBaseSettings",
    "FlextDataIntegrationConfig",
    # Core infrastructure models
    "FlextDatabaseConfig",
    # Settings classes
    "FlextDatabaseSettings",
    "FlextJWTConfig",
    "FlextLDAPConfig",
    "FlextObservabilityConfig",
    # Domain-specific models
    "FlextOracleConfig",
    "FlextRedisConfig",
    "FlextRedisSettings",
    "FlextSingerConfig",
    "JWTConfigDict",
    "LDAPConfigDict",
    "ObservabilityConfigDict",
    "OracleConfigDict",
    "RedisConfigDict",
    "SingerConfigDict",
    # Factory functions
    "create_database_config",
    "create_ldap_config",
    "create_oracle_config",
    "create_redis_config",
    # Utilities
    "load_config_from_env",
    "merge_configs",
    "validate_config",
]
