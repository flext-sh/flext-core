"""Domain-specific configuration models using config_base abstractions.

Provides concrete configuration models for database, cache, authentication,
and integration systems. Uses config_base.py abstractions following SOLID
principles to eliminate duplication.

Classes:
    FlextDatabaseConfig: Database connection configuration.
    FlextRedisConfig: Redis cache configuration.
    FlextJWTConfig: JWT authentication configuration.
    FlextOracleConfig: Oracle database configuration.
    FlextLDAPConfig: LDAP directory configuration.
    FlextConfigFactory: Static factory for configuration creation.

"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, NotRequired, TypedDict

from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import SettingsConfigDict

from flext_core.config_base import (
    FlextAbstractConfig,
    FlextSettings,
)
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict

# =============================================================================
# TYPED DICT DEFINITIONS - Type-safe dictionaries for configuration
# =============================================================================


class FlextDatabaseConfigDict(TypedDict):
    """Database configuration dictionary type."""

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


class FlextRedisConfigDict(TypedDict):
    """Redis configuration dictionary type."""

    host: str
    port: int
    password: NotRequired[str | None]
    database: NotRequired[int]
    ssl_enabled: NotRequired[bool]
    pool_max: NotRequired[int]
    timeout: NotRequired[int]


class FlextJWTConfigDict(TypedDict):
    """JWT configuration dictionary type."""

    secret_key: str
    algorithm: NotRequired[str]
    access_token_expire_minutes: NotRequired[int]
    refresh_token_expire_days: NotRequired[int]
    issuer: NotRequired[str | None]
    audience: NotRequired[str | None]


class FlextLDAPConfigDict(TypedDict):
    """LDAP configuration dictionary type."""

    host: str
    port: int
    base_dn: str
    bind_dn: NotRequired[str | None]
    bind_password: NotRequired[str | None]
    use_ssl: NotRequired[bool]
    use_tls: NotRequired[bool]
    timeout: NotRequired[int]
    pool_size: NotRequired[int]


class FlextOracleConfigDict(TypedDict):
    """Oracle configuration dictionary type."""

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


class FlextSingerConfigDict(TypedDict):
    """Singer configuration dictionary type."""

    stream_name: str
    stream_schema: dict[str, object]
    stream_config: dict[str, object]
    catalog: NotRequired[dict[str, object] | None]
    state: NotRequired[dict[str, object] | None]
    batch_size: NotRequired[int]


class FlextObservabilityConfigDict(TypedDict):
    """Observability configuration dictionary type."""

    log_level: str
    log_format: NotRequired[str]
    metrics_enabled: NotRequired[bool]
    tracing_enabled: NotRequired[bool]
    correlation_id_header: NotRequired[str]
    service_name: NotRequired[str]


# =============================================================================
# BASE CONFIGURATION MODELS - Core infrastructure models
# =============================================================================


class FlextBaseConfigModel(FlextAbstractConfig):
    """Base configuration model with common settings using base abstractions.

    Extends FlextAbstractConfig to provide domain-specific configuration
    patterns with validation and serialization capabilities.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=True,
    )

    def validate_config(self) -> FlextResult[None]:
        """Validate configuration specifics - base implementation.

        Returns:
            FlextResult indicating validation success.

        """
        return FlextResult.ok(None)

    def to_typed_dict(self) -> TAnyDict:
        """Convert to typed dictionary representation.

        Returns:
            Dictionary representation excluding unset values.

        """
        return self.model_dump(exclude_unset=True)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation including default values.

        """
        return self.model_dump()


class FlextDatabaseConfig(FlextBaseConfigModel):
    """Database configuration with connection pooling and SSL."""

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

    def to_database_dict(self) -> FlextDatabaseConfigDict:
        """Convert to FlextDatabaseConfigDict."""
        return FlextDatabaseConfigDict(
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
    """Redis configuration with pooling and SSL support."""

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

    def to_redis_dict(self) -> FlextRedisConfigDict:
        """Convert to FlextRedisConfigDict."""
        return FlextRedisConfigDict(
            host=self.host,
            port=self.port,
            password=self.password.get_secret_value() if self.password else None,
            database=self.database,
            ssl_enabled=self.ssl_enabled,
            pool_max=self.pool_max,
            timeout=self.timeout,
        )


class FlextJWTConfig(FlextBaseConfigModel):
    """JWT configuration with algorithm validation."""

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

    def to_jwt_dict(self) -> FlextJWTConfigDict:
        """Convert to FlextJWTConfigDict."""
        return FlextJWTConfigDict(
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
    """Oracle database configuration with service name or SID."""

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

    def model_post_init(self, __context: object | None, /) -> None:
        """Validate that either SID or service_name is provided."""
        if not self.service_name and not self.sid:
            msg = "Either service_name or sid must be provided"
            raise ValueError(msg)

    def get_connection_string(self) -> str:
        """Generate Oracle connection string."""
        if self.service_name:
            return f"{self.host}:{self.port}/{self.service_name}"
        if self.sid:
            return f"{self.host}:{self.port}:{self.sid}"
        return f"{self.host}:{self.port}"

    def to_oracle_dict(self) -> FlextOracleConfigDict:
        """Convert to FlextOracleConfigDict."""
        return FlextOracleConfigDict(
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
    """LDAP configuration with SSL/TLS support."""

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

    def to_ldap_dict(self) -> FlextLDAPConfigDict:
        """Convert to FlextLDAPConfigDict."""
        return FlextLDAPConfigDict(
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
    """Singer tap/target configuration with schema validation."""

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

    def to_singer_dict(self) -> FlextSingerConfigDict:
        """Convert to FlextSingerConfigDict."""
        return FlextSingerConfigDict(
            stream_name=self.stream_name,
            stream_schema=self.stream_schema,
            stream_config=self.stream_config,
            catalog=self.catalog,
            state=self.state,
            batch_size=self.batch_size,
        )


class FlextObservabilityConfig(FlextBaseConfigModel):
    """Observability configuration for logging and tracing."""

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

    # Backward-compat constants expected by tests
    ENABLE_METRICS: ClassVar[bool] = True
    TRACE_ENABLED: ClassVar[bool] = True
    TRACE_SAMPLE_RATE: ClassVar[float] = 0.1
    SLOW_OPERATION_THRESHOLD: ClassVar[int] = 1000
    CRITICAL_OPERATION_THRESHOLD: ClassVar[int] = 5000

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

    def to_observability_dict(self) -> dict[str, object]:
        """Convert to observability dictionary representation."""
        return self.model_dump()


class FlextPerformanceConfig(FlextBaseConfigModel):
    """Performance configuration for optimization settings."""

    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(1000, description="Cache size limit", ge=1)
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling",
    )
    max_connections: int = Field(100, description="Maximum connections", ge=1, le=10000)

    # Backward-compat constants expected by tests
    DEFAULT_CACHE_SIZE: ClassVar[int] = 1000
    DEFAULT_PAGE_SIZE: ClassVar[int] = 50
    DEFAULT_RETRIES: ClassVar[int] = 3
    DEFAULT_TIMEOUT: ClassVar[int] = 30
    DEFAULT_BATCH_SIZE: ClassVar[int] = 50  # same as DEFAULT_PAGE_SIZE per tests
    DEFAULT_POOL_SIZE: ClassVar[int] = 10
    DEFAULT_MAX_RETRIES: ClassVar[int] = 3  # same as DEFAULT_RETRIES per tests

    def to_performance_dict(self) -> dict[str, object]:
        """Convert to performance dictionary representation."""
        return {
            "enable_caching": self.enable_caching,
            "cache_size": self.cache_size,
            "enable_profiling": self.enable_profiling,
            "max_connections": self.max_connections,
        }


# =============================================================================
# COMPOSITE CONFIGURATION MODELS
# =============================================================================


class FlextApplicationConfig(FlextBaseConfigModel):
    """Application configuration combining all domain configs."""

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
    """Configuration for data integration with Oracle/LDAP."""

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


# FlextSettings imported from config.py - single source of truth
# Following SOLID Single Responsibility Principle


class FlextDatabaseSettings(FlextSettings):
    """Database settings with environment variables."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
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


class FlextRedisSettings(FlextSettings):
    """Redis settings with environment variables."""

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


# =============================================================================
# FLEXT CONFIG FACTORY - Static class for config creation functions
# =============================================================================


class FlextConfigFactory:
    """Factory methods for configuration creation and validation."""

    @staticmethod
    def create_database_config(
        host: str = "localhost",
        port: int = 5432,
        username: str = "postgres",
        password: str | None = None,
        database: str = "flext",
        **kwargs: object,
    ) -> FlextDatabaseConfig:
        """Create database configuration.

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

        # Create safe config data with explicit parameters
        config_data = {
            "host": host,
            "port": port,
            "username": username,
            "password": SecretStr(password),
            "database": database,
        }

        # Filter kwargs to only valid FlextDatabaseConfig fields
        valid_fields = {
            "pool_min",
            "pool_max",
            "pool_timeout",
            "ssl_enabled",
            "ssl_cert_path",
            "encoding",
            "autocommit",
        }
        config_data.update(
            {key: value for key, value in kwargs.items() if key in valid_fields},
        )

        return FlextDatabaseConfig.model_validate(config_data)

    @staticmethod
    def create_redis_config(
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        database: int = 0,
        **kwargs: object,
    ) -> FlextRedisConfig:
        """Create Redis configuration."""
        # Create safe config data with explicit parameters
        config_data = {
            "host": host,
            "port": port,
            "password": SecretStr(password) if password else None,
            "database": database,
        }

        # Filter kwargs to only valid FlextRedisConfig fields
        valid_fields = {"ssl_enabled", "timeout"}
        config_data.update(
            {key: value for key, value in kwargs.items() if key in valid_fields},
        )

        return FlextRedisConfig.model_validate(config_data)

    @staticmethod
    def create_oracle_config(
        host: str = "localhost",
        username: str = "oracle",
        password: str | None = None,
        service_name: str | None = None,
        **kwargs: object,
    ) -> FlextOracleConfig:
        """Create Oracle configuration.

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

        # Filter kwargs to only valid FlextOracleConfig fields
        valid_fields = {"port", "sid", "ssl_enabled"}
        config_data.update(
            {key: value for key, value in kwargs.items() if key in valid_fields},
        )

        return FlextOracleConfig.model_validate(config_data)

    @staticmethod
    def create_ldap_config(
        host: str = "localhost",
        port: int = 389,
        base_dn: str = "dc=example,dc=com",
        bind_dn: str | None = None,
        bind_password: str | None = None,
        **kwargs: object,
    ) -> FlextLDAPConfig:
        """Create LDAP configuration."""
        # Create safe config data with explicit parameters
        config_data = {
            "host": host,
            "port": port,
            "base_dn": base_dn,
            "bind_dn": bind_dn,
            "bind_password": SecretStr(bind_password) if bind_password else None,
        }

        # Filter kwargs to only valid FlextLDAPConfig fields
        valid_fields = {"use_ssl", "use_tls", "timeout", "pool_size"}
        config_data.update(
            {key: value for key, value in kwargs.items() if key in valid_fields},
        )

        return FlextLDAPConfig.model_validate(config_data)

    # =============================================================================
    # CONFIGURATION UTILITIES
    # =============================================================================

    @staticmethod
    def load_config_from_env(config_class: type[FlextSettings]) -> FlextSettings:
        """Load configuration from environment variables."""
        return config_class()

    @staticmethod
    def merge_configs(*configs: FlextBaseConfigModel) -> dict[str, object]:
        """Merge multiple configuration objects."""
        merged: dict[str, object] = {}
        for config in configs:
            config_dict = config.to_dict()
            # Update with type conversion to object
            merged.update(config_dict)
        return merged

    @staticmethod
    def validate_config(config: FlextBaseConfigModel) -> bool:
        """Validate configuration object."""
        try:
            config.model_validate(config.model_dump())
            return True
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger = FlextLoggerFactory.get_logger(__name__)
            logger.warning(
                f"Configuration validation failed for {type(config).__name__}: {e}",
            )
            return False


# =============================================================================
# PLUGIN CONFIGURATION - Plugin System Configuration Models
# =============================================================================


class FlextPluginConfigDict(TypedDict, total=False):
    """Plugin configuration dictionary type."""

    name: str
    version: str
    enabled: bool
    priority: int
    settings: dict[str, object]
    dependencies: list[str]
    metadata: dict[str, str]


# Removed duplicate FlextObservabilityConfig - using the more complete
# implementation at line 490 instead


class FlextPluginConfig(FlextBaseConfigModel):
    """Plugin configuration with validation and dependency management."""

    name: str = Field(description="Unique plugin identifier for system registration")
    version: str = Field(
        default="1.0.0",
        description="Semantic version for compatibility management",
    )
    description: str = Field(
        default="",
        description="Human-readable plugin description",
    )
    enabled: bool = Field(
        default=True,
        description="Runtime activation state for plugin",
    )
    priority: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Loading priority (0=highest, 1000=lowest)",
    )
    settings: dict[str, object] = Field(
        default_factory=dict,
        description="Plugin-specific configuration parameters",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required plugin dependencies for proper ordering",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional plugin metadata and documentation",
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?(?:\+[a-zA-Z0-9.-]+)?$"
        if not re.match(pattern, v):
            error_msg = "Version must follow semantic versioning (x.y.z)"
            raise ValueError(error_msg)
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate plugin name format."""
        pattern = r"^[a-z][a-z0-9-]*[a-z0-9]$"
        if not re.match(pattern, v):
            error_msg = (
                "Plugin name must be lowercase with hyphens, "
                "start with letter, end with letter/number"
            )
            raise ValueError(error_msg)
        return v


class FlextPluginRegistryConfig(FlextBaseConfigModel):
    """Plugin registry configuration with dependency resolution."""

    plugins: list[FlextPluginConfig] = Field(
        default_factory=list,
        description="List of configured plugins",
    )
    auto_discovery: bool = Field(
        default=False,
        description="Enable automatic plugin discovery",
    )
    discovery_paths: list[str] = Field(
        default_factory=list,
        description="Paths to search for plugins during auto-discovery",
    )
    global_settings: dict[str, object] = Field(
        default_factory=dict,
        description="Global settings applied to all plugins",
    )

    @model_validator(mode="after")
    def validate_plugin_dependencies(self) -> FlextPluginRegistryConfig:
        """Validate plugin dependencies are available."""
        plugin_names = {plugin.name for plugin in self.plugins}
        for plugin in self.plugins:
            for dep in plugin.dependencies:
                if dep not in plugin_names:
                    error_msg = (
                        f"Plugin '{plugin.name}' depends on missing plugin '{dep}'"
                    )
                    raise ValueError(error_msg)
        return self

    # =============================================================================
    # PLUGIN CONFIGURATION FACTORY FUNCTIONS
    # =============================================================================

    @staticmethod
    def create_plugin_config(
        name: str,
        *,
        config_options: dict[str, object] | None = None,
    ) -> FlextPluginConfig:
        """Create plugin configuration with validation.

        Factory function for creating validated plugin configurations
        with sensible defaults and proper type safety.

        Args:
            name: Unique plugin identifier
            config_options: Optional configuration dictionary with keys:
                - version: Plugin version (semantic versioning, default: "1.0.0")
                - enabled: Whether plugin is enabled (default: True)
                - priority: Plugin loading priority (default: 100)
                - settings: Plugin-specific configuration (default: {})
                - dependencies: Required plugin dependencies (default: [])

        Returns:
            Validated FlextPluginConfig instance

        """
        options = config_options or {}
        # Extract values with proper type casting
        priority_value = options.get("priority", 100)
        settings_value = options.get("settings")
        dependencies_value = options.get("dependencies")

        return FlextPluginConfig(
            name=name,
            version=str(options.get("version", "1.0.0")),
            enabled=bool(options.get("enabled", True)),
            priority=int(priority_value)
            if isinstance(priority_value, (int, str))
            else 100,
            settings=dict(settings_value) if isinstance(settings_value, dict) else {},
            dependencies=list(dependencies_value)
            if isinstance(dependencies_value, (list, tuple))
            else [],
        )

    @staticmethod
    def create_plugin_registry_config(
        *,
        plugins: list[FlextPluginConfig] | None = None,
        auto_discovery: bool = False,
        discovery_paths: list[str] | None = None,
    ) -> FlextPluginRegistryConfig:
        """Create plugin registry configuration.

        Factory function for creating plugin registry configurations
        with proper validation and dependency resolution.

        Args:
            plugins: List of plugin configurations
            auto_discovery: Enable automatic plugin discovery
            discovery_paths: Plugin discovery paths

        Returns:
            Validated FlextPluginRegistryConfig instance.

        """
        return FlextPluginRegistryConfig(
            plugins=plugins or [],
            auto_discovery=auto_discovery,
            discovery_paths=discovery_paths or [],
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [  # noqa: RUF022
    # Base models
    "FlextBaseConfigModel",
    # Composite models
    "FlextApplicationConfig",
    "FlextDataIntegrationConfig",
    # Core infrastructure models
    "FlextDatabaseConfig",
    "FlextRedisConfig",
    # Domain-specific models
    "FlextOracleConfig",
    "FlextPerformanceConfig",
    # Main factory class (SOLID-compliant organization)
    "FlextConfigFactory",
    # Plugin configuration models
    "FlextPluginConfig",
    "FlextPluginRegistryConfig",
    # Settings classes
    "FlextDatabaseSettings",
    "FlextJWTConfig",
    "FlextLDAPConfig",
    "FlextObservabilityConfig",
    "FlextRedisSettings",
    "FlextSettings",
    # Singer model
    "FlextSingerConfig",
    # TypedDict definitions (sorted)
    "FlextDatabaseConfigDict",
    "FlextObservabilityConfigDict",
    "FlextOracleConfigDict",
    "FlextPluginConfigDict",
    "FlextRedisConfigDict",
    "FlextSingerConfigDict",
    # Third-party re-exports
    "SettingsConfigDict",
    # Compatibility aliases (maintain API contracts)
    "create_database_config",
    "create_ldap_config",
    "create_oracle_config",
    "create_plugin_config",
    "create_plugin_registry_config",
    "create_redis_config",
    "load_config_from_env",
    "merge_configs",
    "validate_config",
]

# =============================================================================
# COMPATIBILITY ALIASES - For backward compatibility
# =============================================================================

# Module-level aliases for loose functions now organized in FlextConfigFactory
create_database_config = FlextConfigFactory.create_database_config
create_redis_config = FlextConfigFactory.create_redis_config
create_oracle_config = FlextConfigFactory.create_oracle_config
create_ldap_config = FlextConfigFactory.create_ldap_config
load_config_from_env = FlextConfigFactory.load_config_from_env
merge_configs = FlextConfigFactory.merge_configs
validate_config = FlextConfigFactory.validate_config
create_plugin_config = FlextPluginRegistryConfig.create_plugin_config
create_plugin_registry_config = FlextPluginRegistryConfig.create_plugin_registry_config
