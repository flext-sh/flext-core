"""Example: Extending FLEXT Core Configuration in a Subproject.

This example demonstrates how to extend FLEXT Core's configuration
system in a subproject (e.g., flext-api, flext-auth, flext-observability).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Optional
from pydantic import ValidationInfo
from datetime import datetime
import os
import yaml
from pathlib import Path
from pydantic import Field, field_validator, model_validator, ConfigDict
from flext_core import (
    FlextModels,
    FlextConfig,
    FlextResult,
    FlextConstants,
    FlextLogger,
    FlextExceptions,
)

logger = FlextLogger(__name__)


# =============================================================================
# STEP 1: Define Your Domain-Specific Configuration Model
# =============================================================================


class ApiEndpointConfig(FlextModels.Config):
    """Configuration for a single API endpoint."""

    path: str = Field(..., description="API endpoint path")
    method: str = Field(default="GET", pattern="^(GET|POST|PUT|DELETE|PATCH)$")
    timeout: int = Field(default=30, ge=1, le=300)
    rate_limit: int = Field(default=100, ge=1, le=10000)
    requires_auth: bool = Field(default=True)
    cache_ttl: int | None = Field(None, ge=0, le=86400)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path starts with /."""
        if not v.startswith("/"):
            return f"/{v}"
        return v


class ApiConfig(FlextModels.SystemConfigs.BaseSystemConfig):
    """Extended configuration for API service.

    This demonstrates:
    1. Extending BaseSystemConfig
    2. Adding domain-specific fields
    3. Custom validators
    4. Environment-specific adjustments
    5. Nested configuration models
    """

    # API-specific fields
    api_version: str = Field(default="v1", pattern=r"^v\d+$")
    base_url: str = Field(default="https://api.example.com")
    api_key: str | None = Field(None, min_length=32)
    jwt_secret: str | None = Field(None, min_length=32)

    # Service configuration
    service_name: str = Field(default="flext-api")
    service_port: int = Field(default=8080, ge=1024, le=65535)
    worker_count: int = Field(default=4, ge=1, le=32)

    # Feature flags
    enable_swagger: bool = Field(default=True)
    enable_cors: bool = Field(default=False)
    enable_rate_limiting: bool = Field(default=True)
    enable_request_logging: bool = Field(default=True)
    enable_response_caching: bool = Field(default=False)

    # Nested configurations
    endpoints: list[ApiEndpointConfig] = Field(default_factory=list)
    cors_config: dict[str, object] = Field(
        default_factory=lambda: {
            "allowed_origins": ["*"],
            "allowed_methods": ["GET", "POST"],
            "allowed_headers": ["Content-Type", "Authorization"],
            "max_age": 3600,
        }
    )

    # Rate limiting configuration
    rate_limit_config: FlextTypes.Core.CounterDict = Field(
        default_factory=lambda: {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
        }
    )

    @field_validator("api_key", "jwt_secret")
    @classmethod
    def validate_secrets(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate secret format and warn if exposed."""
        if (
            v
            and v.startswith("sk_live_")
            and info.data.get("environment") != "production"
        ):
            logger.warning("Production API key used in non-production environment")
        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            msg = "base_url must start with http:// or https://"
            raise ValueError(msg)
        if v.endswith("/"):
            return v[:-1]  # Remove trailing slash
        return v

    @model_validator(mode="after")
    def validate_environment_settings(self) -> ApiConfig:
        """Apply environment-specific validation and adjustments."""
        # Production requirements
        if self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value:
            # Enforce security in production
            if not self.api_key:
                msg = "api_key is required in production"
                raise ValueError(msg)
            if not self.jwt_secret:
                msg = "jwt_secret is required in production"
                raise ValueError(msg)
            if not self.base_url.startswith("https://"):
                msg = "HTTPS required in production"
                raise ValueError(msg)

            # Adjust settings for production
            self.debug = False
            self.log_level = FlextConstants.Config.LogLevel.WARNING.value
            self.enable_swagger = False  # Disable Swagger in production
            self.enable_response_caching = True
            self.worker_count = max(self.worker_count, 8)  # Minimum 8 workers

        # Development settings
        elif (
            self.environment
            == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
        ):
            self.debug = True
            self.log_level = FlextConstants.Config.LogLevel.DEBUG.value
            self.enable_swagger = True
            self.enable_cors = True  # Allow CORS in development
            self.enable_request_logging = True

        # Local settings
        elif self.environment == internal.invalid.value:
            self.debug = True
            self.enable_swagger = True
            self.enable_cors = True
            self.enable_rate_limiting = False  # No rate limiting locally
            self.base_url = "http://localhost:8080"  # Override for local

        return self

    def add_endpoint(self, endpoint: dict[str, object]) -> ApiConfig:
        """Add an endpoint configuration."""
        endpoint_config = ApiEndpointConfig.model_validate(endpoint)
        self.endpoints.append(endpoint_config)
        return self

    def get_endpoint(self, path: str) -> ApiEndpointConfig | None:
        """Get endpoint configuration by path."""
        for endpoint in self.endpoints:
            if endpoint.path == path:
                return endpoint
        return None

    def to_deployment_config(self) -> dict[str, object]:
        """Generate deployment configuration (e.g., for Kubernetes)."""
        return {
            "name": self.service_name,
            "replicas": self.worker_count,
            "port": self.service_port,
            "environment": self.environment,
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "1000m", "memory": "512Mi"},
            },
            "env": {
                "LOG_LEVEL": self.log_level,
                "API_VERSION": self.api_version,
                "BASE_URL": self.base_url,
            },
        }


# =============================================================================
# STEP 2: Create Extended Settings Class (Optional)
# =============================================================================


class ApiSettings(FlextConfig.Settings):
    """Extended settings for API service with environment variable support."""

    # Add API-specific settings fields
    api_version: str = "v1"
    base_url: str = "https://api.example.com"
    api_key: str | None = None
    jwt_secret: str | None = None
    service_port: int = 8080
    worker_count: int = 4

    model_config = ConfigDict(
        env_prefix="FLEXT_API_",  # Environment variable prefix
        env_nested_delimiter="__",  # For nested configs
        case_sensitive=False,
    )

    def to_config(self) -> ApiConfig:
        """Convert Settings to ApiConfig model."""
        return ApiConfig(
            # Base fields from parent
            environment=self.environment,
            log_level=self.log_level,
            debug=self.debug,
            # API-specific fields
            api_version=self.api_version,
            base_url=self.base_url,
            api_key=self.api_key,
            jwt_secret=self.jwt_secret,
            service_port=self.service_port,
            worker_count=self.worker_count,
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> ApiSettings:
        """Load settings from YAML file."""
        with Path(file_path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# =============================================================================
# STEP 3: Create Configuration Functions with FlextResult
# =============================================================================


def configure_api_system(config: dict[str, object]) -> FlextResult[dict[str, object]]:
    """Configure API system with validation.

    This follows the FLEXT pattern:
    1. Accept dict input (for compatibility)
    2. Validate with Pydantic model
    3. Apply business logic
    4. Return dict wrapped in FlextResult
    """
    try:
        # Validate configuration
        api_config = ApiConfig.model_validate(config)

        # Apply additional business logic
        if api_config.enable_cors and api_config.environment == "production":
            logger.warning("CORS enabled in production - ensure origins are restricted")

        # Register with global registry (optional)
        FlextConfig.SettingsRegistry.register(
            api_config.service_name, ApiSettings.model_validate(config)
        )

        # Return as dict for compatibility
        return FlextResult.ok(api_config.model_dump())

    except ValidationError as e:
        errors = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return FlextResult.fail(
            f"API configuration validation failed: {errors}",
            error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
        )
    except Exception as e:
        return FlextResult.fail(
            f"Failed to configure API system: {e!s}",
            error_code=FlextConstants.Errors.SYSTEM_ERROR,
        )


def create_api_config_for_environment(
    environment: str, **overrides: object
) -> FlextResult[FlextTypes.Core.Dict]:
    """Create environment-specific API configuration."""
    # Base configurations per environment
    env_configs = {
        "production": {
            "environment": "production",
            "base_url": "https://api.production.example.com",
            "enable_swagger": False,
            "enable_rate_limiting": True,
            "worker_count": 16,
        },
        "staging": {
            "environment": "staging",
            "base_url": "https://api.staging.example.com",
            "enable_swagger": True,
            "enable_rate_limiting": True,
            "worker_count": 8,
        },
        "development": {
            "environment": "development",
            "base_url": "https://api.dev.example.com",
            "enable_swagger": True,
            "enable_cors": True,
            "worker_count": 4,
        },
        "local": {
            "environment": "local",
            "base_url": "http://localhost:8080",
            "enable_swagger": True,
            "enable_cors": True,
            "enable_rate_limiting": False,
            "worker_count": 2,
        },
    }

    if environment not in env_configs:
        return FlextResult.fail(
            f"Unknown environment: {environment}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

    # Merge base config with overrides
    config = {**env_configs[environment], **overrides}

    return configure_api_system(config)


# =============================================================================
# STEP 4: Usage Examples
# =============================================================================


def example_basic_usage() -> None:
    """Basic usage example."""
    # Create configuration from dict
    config_dict = {
        "environment": "production",
        "api_key": "sk_live_" + "x" * 32,
        "jwt_secret": "secret_" + "x" * 32,
        "base_url": "https://api.mycompany.com",
        "service_port": 8443,
    }

    result = configure_api_system(config_dict)
    if result.success:
        config = result.unwrap()
        print(f"API configured: {config['base_url']}")
        print(f"Port: {config['service_port']}")
        print(f"Workers: {config['worker_count']}")
    else:
        print(f"Configuration failed: {result.error}")


def example_with_endpoints() -> None:
    """Example with endpoint configuration."""
    config = ApiConfig(
        environment="development",
        api_version="v2",
        base_url="https://api.dev.example.com",
    )

    # Add endpoints
    config.add_endpoint(
        {
            "path": "/users",
            "method": "GET",
            "rate_limit": 1000,
            "cache_ttl": 300,
        }
    )

    config.add_endpoint(
        {
            "path": "/users/{id}",
            "method": "GET",
            "rate_limit": 2000,
            "requires_auth": True,
        }
    )

    # Get endpoint config
    users_endpoint = config.get_endpoint("/users")
    if users_endpoint:
        print(f"Users endpoint: {users_endpoint.method} {users_endpoint.path}")
        print(f"Rate limit: {users_endpoint.rate_limit}")


def example_from_environment() -> None:
    """Example loading from environment variables."""
    # Set environment variables
    os.environ["FLEXT_API_ENVIRONMENT"] = "staging"
    os.environ["FLEXT_API_API_VERSION"] = "v3"
    os.environ["FLEXT_API_BASE_URL"] = "https://api.staging.myapp.com"
    os.environ["FLEXT_API_SERVICE_PORT"] = "9000"
    os.environ["FLEXT_API_WORKER_COUNT"] = "8"

    # Load from environment
    settings = ApiSettings.from_sources(
        env_prefix="FLEXT_API_",
        json_file="config.json",  # Also check JSON file
    )

    # Convert to config
    config = settings.to_config()
    print(f"Loaded config: {config.service_name} on port {config.service_port}")


def example_deployment_generation() -> None:
    """Example generating deployment configuration."""
    result = create_api_config_for_environment("production", worker_count=32)

    if result.success:
        config_dict = result.unwrap()
        api_config = ApiConfig.model_validate(config_dict)

        # Generate Kubernetes deployment config
        deployment = api_config.to_deployment_config()
        print(f"Deployment config: {deployment}")


def example_with_registry() -> None:
    """Example using the Settings Registry."""
    # Register multiple services
    services = {
        "api": ApiSettings(api_version="v1", service_port=8080),
        "auth": ApiSettings(api_version="v1", service_port=8081),
        "REDACTED_LDAP_BIND_PASSWORD": ApiSettings(api_version="v1", service_port=8082),
    }

    for name, settings in services.items():
        FlextConfig.SettingsRegistry.register(name, settings)

    # Update runtime configuration
    FlextConfig.SettingsRegistry.update_runtime(
        "api", {"log_level": "DEBUG", "worker_count": 16}
    )

    # Reload from sources
    result = FlextConfig.SettingsRegistry.reload_from_sources(
        "api", env_prefix="FLEXT_API_"
    )

    if result.success:
        print("Configuration reloaded successfully")


# =============================================================================
# STEP 5: Integration with Application
# =============================================================================


class ApiApplication:
    """Example API application using the configuration."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize application with configuration."""
        self.config = self._load_configuration(config)
        self.logger = FlextLogger(__name__)

    def _load_configuration(self, config_dict: dict[str, object] | None) -> ApiConfig:
        """Load and validate configuration."""
        if config_dict:
            # Use provided configuration
            result = configure_api_system(config_dict)
        else:
            # Load from environment
            result = self._load_from_environment()

        if result.is_failure:
            msg = f"Failed to load configuration: {result.error}"
            raise FlextExceptions.ConfigurationError(msg)

        return ApiConfig.model_validate(result.unwrap())

    def _load_from_environment(self) -> FlextResult[dict[str, object]]:
        """Load configuration from environment."""
        try:
            settings = ApiSettings.from_sources(
                env_prefix="FLEXT_API_", json_file="api_config.json"
            )
            config = settings.to_config()
            return FlextResult.ok(config.model_dump())
        except Exception as e:
            return FlextResult.fail(str(e))

    def start(self) -> None:
        """Start the API application."""
        self.logger.info(
            f"Starting {self.config.service_name}",
            port=self.config.service_port,
            workers=self.config.worker_count,
            environment=self.config.environment,
        )

        # Configure based on settings
        if self.config.enable_swagger:
            self._setup_swagger()
        if self.config.enable_cors:
            self._setup_cors()
        if self.config.enable_rate_limiting:
            self._setup_rate_limiting()

        # Start workers
        for i in range(self.config.worker_count):
            self._start_worker(i)

    def _setup_swagger(self) -> None:
        """Setup Swagger documentation."""
        self.logger.debug("Setting up Swagger UI")

    def _setup_cors(self) -> None:
        """Setup CORS configuration."""
        self.logger.debug("Setting up CORS", config=self.config.cors_config)

    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting."""
        self.logger.debug(
            "Setting up rate limiting", config=self.config.rate_limit_config
        )

    def _start_worker(self, worker_id: int) -> None:
        """Start a worker process."""
        self.logger.debug(f"Starting worker {worker_id}")


# =============================================================================
# Run examples if executed directly
# =============================================================================

if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()

    print("\n=== With Endpoints ===")
    example_with_endpoints()

    print("\n=== From Environment ===")
    example_from_environment()

    print("\n=== Deployment Generation ===")
    example_deployment_generation()

    print("\n=== With Registry ===")
    example_with_registry()

    print("\n=== Application Integration ===")
    app = ApiApplication(
        {
            "environment": "development",
            "api_version": "v2",
            "service_port": 9000,
        }
    )
    app.start()
