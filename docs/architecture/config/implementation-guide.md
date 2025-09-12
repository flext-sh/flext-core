# FlextConfig Implementation Guide

**Version**: 0.9.0
**Target**: FLEXT Library Developers
**Complexity**: Intermediate
**Estimated Time**: 1-3 hours per library

## 📋 Overview

This guide provides step-by-step instructions for implementing FlextConfig configuration management patterns in FLEXT ecosystem libraries. It covers configuration design, environment integration, validation strategies, and best practices.

## 🎯 Implementation Phases

### Phase 1: Configuration Analysis (30 minutes)
### Phase 2: Configuration Design (45 minutes)
### Phase 3: Implementation (1-2 hours)
### Phase 4: Integration & Testing (30 minutes)

---

## 🔍 Phase 1: Configuration Analysis

### 1.1 Identify Configuration Requirements

**Configuration Types to Consider**:
- **Service Settings**: Host, port, timeout, retry configuration
- **Database Settings**: Connection strings, pool settings, credentials
- **Security Settings**: API keys, certificates, encryption settings
- **Feature Flags**: Enable/disable functionality, experimental features
- **Performance Settings**: Batch sizes, worker counts, memory limits
- **Environment Settings**: Development, staging, production differences

### 1.2 Current Configuration Analysis Template

```python
# Analyze your current configuration approach
class CurrentConfigurationApproach:
    """Document what you currently have"""

    # ❌ Identify scattered configuration
    def __init__(self):
        self.host = "localhost"  # Hardcoded values
        self.port = 8080
        self.debug = True

    # ❌ Identify manual environment variable handling
    def load_env_vars(self):
        self.host = os.getenv("HOST", "localhost")
        # Manual, error-prone environment loading

    # ❌ Identify missing validation
    def validate(self):
        # No validation or inconsistent validation
        pass
```

### 1.3 Configuration Discovery Checklist

- [ ] **Service configuration**: Network, timeouts, retries identified
- [ ] **Database configuration**: Connection settings, credentials documented
- [ ] **Security requirements**: Authentication, encryption, certificates mapped
- [ ] **Environment differences**: Dev vs staging vs production requirements
- [ ] **Performance settings**: Concurrency, batch sizes, resource limits
- [ ] **Feature flags**: Optional functionality, experimental features
- [ ] **Validation rules**: Business constraints, security requirements

---

## 🏗️ Phase 2: Configuration Design

### 2.1 Basic Configuration Structure

```python
from flext_core import FlextConfig, FlextResult
from pydantic import Field, field_validator

class YourLibraryConfig(FlextConfig):
    """Configuration for YourLibrary with comprehensive settings."""

    # Service configuration
    service_host: str = Field(
        default="localhost",
        description="Service host address"
    )
    service_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Service port number"
    )

    # Performance settings
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of worker threads"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )

    # Feature flags
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business-specific configuration rules."""
        # Call parent validation first
        base_result = super().validate_business_rules()
        if base_result.is_failure:
            return base_result

        # Custom business validation
        if self.environment == "production" and self.enable_caching is False:
            return FlextResult[None].fail(
                "Caching must be enabled in production environment"
            )

        return FlextResult[None].ok(None)
```

### 2.2 Environment Variable Integration

```python
class YourLibraryConfig(FlextConfig):
    """Configuration with automatic environment variable loading."""

    # These fields will automatically load from environment variables:
    # FLEXT_DATABASE_HOST, FLEXT_DATABASE_PORT, etc.
    database_host: str = Field(default="localhost")
    database_port: int = Field(default=5432)
    database_name: str = Field(default="app_db")
    database_user: str = Field(default="user")
    database_password: str = Field(default="", min_length=8)

    class Settings(FlextConfig.Settings):
        """Environment-aware settings with custom prefix."""

        model_config = SettingsConfigDict(
            env_prefix="FLEXT_YOURLIB_",  # Custom prefix
            env_file=".env",
            case_sensitive=False
        )

    @field_validator("database_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate database password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Database password must be at least 8 characters")

        if v == "password" or v == "123456":
            raise ValueError("Database password is too weak")

        return v
```

### 2.3 Complex Nested Configuration

```python
class YourLibraryConfig(FlextConfig):
    """Configuration with nested structures for complex services."""

    class DatabaseConfig(FlextConfig.BaseModel):
        """Database connection configuration."""
        host: str = "localhost"
        port: int = 5432
        database: str = "app_db"
        username: str = Field(min_length=1)
        password: str = Field(min_length=8)
        pool_size: int = Field(default=10, ge=1, le=100)

        @field_validator("host")
        @classmethod
        def validate_host(cls, v: str) -> str:
            if not v or v.isspace():
                raise ValueError("Database host cannot be empty")
            return v.strip()

    class CacheConfig(FlextConfig.BaseModel):
        """Cache configuration."""
        enabled: bool = True
        host: str = "localhost"
        port: int = 6379
        timeout: int = Field(default=5, ge=1, le=60)
        max_connections: int = Field(default=10, ge=1, le=100)

    class SecurityConfig(FlextConfig.BaseModel):
        """Security configuration."""
        api_key: str = Field(min_length=32)
        jwt_secret: str = Field(min_length=32)
        token_expiry_hours: int = Field(default=24, ge=1, le=168)  # Max 1 week

    # Nested configuration instances
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate cross-component business rules."""
        base_result = super().validate_business_rules()
        if base_result.is_failure:
            return base_result

        # Cross-component validation
        if self.cache.enabled and not self.cache.host:
            return FlextResult[None].fail("Cache host must be specified when cache is enabled")

        # Environment-specific validation
        if self.environment == "production":
            if self.security.token_expiry_hours > 24:
                return FlextResult[None].fail(
                    "Token expiry cannot exceed 24 hours in production"
                )

        return FlextResult[None].ok(None)
```

---

## ⚙️ Phase 3: Implementation

### 3.1 Configuration Factory Pattern

```python
class YourLibraryConfigFactory:
    """Factory for creating validated configuration instances."""

    @staticmethod
    def create_development_config() -> FlextResult[YourLibraryConfig]:
        """Create development configuration with appropriate defaults."""
        try:
            config = YourLibraryConfig(
                environment="development",
                debug=True,
                database_host="localhost",
                database_port=5432,
                cache_enabled=False,  # Disable cache in dev
                log_level="DEBUG"
            )

            validation_result = config.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[YourLibraryConfig].fail(validation_result.error)

            return FlextResult[YourLibraryConfig].ok(config)

        except Exception as e:
            return FlextResult[YourLibraryConfig].fail(f"Failed to create dev config: {e}")

    @staticmethod
    def create_production_config() -> FlextResult[YourLibraryConfig]:
        """Create production configuration with security defaults."""
        try:
            # Load from environment variables for production
            config = YourLibraryConfig.Settings()

            # Override production-specific settings
            config.environment = "production"
            config.debug = False
            config.log_level = "INFO"

            validation_result = config.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[YourLibraryConfig].fail(validation_result.error)

            return FlextResult[YourLibraryConfig].ok(config)

        except Exception as e:
            return FlextResult[YourLibraryConfig].fail(f"Failed to create prod config: {e}")

    @staticmethod
    def create_from_file(config_path: str) -> FlextResult[YourLibraryConfig]:
        """Create configuration from JSON file."""
        try:
            # Use FlextConfig's built-in file loading
            config_result = YourLibraryConfig.load_and_validate_from_file(
                file_path=config_path,
                required_keys=["database_host", "database_user", "api_key"]
            )

            if config_result.is_failure:
                return FlextResult[YourLibraryConfig].fail(config_result.error)

            # Convert dict to config instance
            config_data = config_result.value
            config = YourLibraryConfig.model_validate(config_data)

            return FlextResult[YourLibraryConfig].ok(config)

        except Exception as e:
            return FlextResult[YourLibraryConfig].fail(f"Failed to load config from file: {e}")
```

### 3.2 Configuration Manager Pattern

```python
class YourLibraryConfigManager:
    """Central configuration manager for the library."""

    def __init__(self):
        self._config: YourLibraryConfig | None = None
        self._config_source: str = "none"

    def load_configuration(self,
                          config_source: str = "environment",
                          config_path: str | None = None) -> FlextResult[None]:
        """Load configuration from specified source."""
        try:
            if config_source == "development":
                config_result = YourLibraryConfigFactory.create_development_config()
            elif config_source == "production":
                config_result = YourLibraryConfigFactory.create_production_config()
            elif config_source == "file" and config_path:
                config_result = YourLibraryConfigFactory.create_from_file(config_path)
            elif config_source == "environment":
                # Load from environment variables
                config = YourLibraryConfig.Settings()
                config_result = FlextResult[YourLibraryConfig].ok(config)
            else:
                return FlextResult[None].fail(f"Unknown config source: {config_source}")

            if config_result.is_failure:
                return FlextResult[None].fail(config_result.error)

            self._config = config_result.value
            self._config_source = config_source

            # Log configuration loading
            logger.info(f"Configuration loaded from {config_source}",
                       environment=self._config.environment,
                       debug=self._config.debug)

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Configuration loading failed: {e}")

    @property
    def config(self) -> YourLibraryConfig:
        """Get current configuration, loading default if none set."""
        if self._config is None:
            # Load default configuration
            default_result = self.load_configuration("environment")
            if default_result.is_failure:
                raise RuntimeError(f"Failed to load default configuration: {default_result.error}")

        return self._config

    def reload_configuration(self) -> FlextResult[None]:
        """Reload configuration from the same source."""
        if self._config_source == "none":
            return FlextResult[None].fail("No configuration source set for reload")

        return self.load_configuration(self._config_source)

    def validate_current_config(self) -> FlextResult[None]:
        """Validate the current configuration."""
        if self._config is None:
            return FlextResult[None].fail("No configuration loaded")

        return self._config.validate_business_rules()
```

### 3.3 Library Integration Pattern

```python
# your_library/__init__.py
from .config import YourLibraryConfig, YourLibraryConfigManager

# Global configuration manager instance
config_manager = YourLibraryConfigManager()

def initialize_library(config_source: str = "environment",
                      config_path: str | None = None) -> FlextResult[None]:
    """Initialize the library with configuration."""
    return config_manager.load_configuration(config_source, config_path)

def get_config() -> YourLibraryConfig:
    """Get the current library configuration."""
    return config_manager.config

# Export configuration classes for external use
__all__ = [
    "YourLibraryConfig",
    "YourLibraryConfigManager",
    "initialize_library",
    "get_config"
]
```

---

## 🔗 Phase 4: Integration & Testing

### 4.1 Configuration Testing Strategy

```python
import pytest
import tempfile
import json
from pathlib import Path

class TestYourLibraryConfiguration:
    """Comprehensive configuration testing."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = YourLibraryConfig()

        assert config.service_host == "localhost"
        assert config.service_port == 8080
        assert config.max_workers == 4
        assert config.timeout_seconds == 30

        # Validate default configuration
        result = config.validate_business_rules()
        assert result.success

    def test_environment_variable_loading(self, monkeypatch):
        """Test automatic environment variable loading."""
        monkeypatch.setenv("FLEXT_YOURLIB_SERVICE_HOST", "api.example.com")
        monkeypatch.setenv("FLEXT_YOURLIB_SERVICE_PORT", "9000")
        monkeypatch.setenv("FLEXT_YOURLIB_MAX_WORKERS", "8")

        config = YourLibraryConfig.Settings()

        assert config.service_host == "api.example.com"
        assert config.service_port == 9000
        assert config.max_workers == 8

    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        config = YourLibraryConfig(
            environment="development",
            enable_caching=True,
            service_port=8080
        )

        result = config.validate_business_rules()
        assert result.success

    def test_configuration_validation_failure(self):
        """Test configuration validation failures."""
        config = YourLibraryConfig(
            environment="production",
            enable_caching=False  # Invalid in production
        )

        result = config.validate_business_rules()
        assert result.is_failure
        assert "production" in result.error
        assert "caching" in result.error.lower()

    def test_nested_configuration_validation(self):
        """Test nested configuration validation."""
        config = YourLibraryConfig()

        # Invalid database configuration
        config.database.host = ""  # Empty host
        config.database.password = "123"  # Too short

        with pytest.raises(ValidationError):
            YourLibraryConfig.model_validate(config.model_dump())

    def test_configuration_file_loading(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "service_host": "file.example.com",
            "service_port": 9090,
            "environment": "staging",
            "database": {
                "host": "db.example.com",
                "port": 5432,
                "username": "app_user",
                "password": "secure_password_123"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            result = YourLibraryConfigFactory.create_from_file(config_file)

            assert result.success
            config = result.value
            assert config.service_host == "file.example.com"
            assert config.service_port == 9090
            assert config.database.host == "db.example.com"
        finally:
            Path(config_file).unlink()

    def test_configuration_factory_patterns(self):
        """Test configuration factory methods."""
        # Development configuration
        dev_result = YourLibraryConfigFactory.create_development_config()
        assert dev_result.success
        dev_config = dev_result.value
        assert dev_config.environment == "development"
        assert dev_config.debug is True

        # Production configuration
        prod_result = YourLibraryConfigFactory.create_production_config()
        assert prod_result.success
        prod_config = prod_result.value
        assert prod_config.environment == "production"
        assert prod_config.debug is False

    def test_configuration_manager(self):
        """Test configuration manager functionality."""
        manager = YourLibraryConfigManager()

        # Load development configuration
        result = manager.load_configuration("development")
        assert result.success

        config = manager.config
        assert config.environment == "development"

        # Test configuration validation
        validation_result = manager.validate_current_config()
        assert validation_result.success
```

### 4.2 Integration Testing Patterns

```python
class TestConfigurationIntegration:
    """Test configuration integration with library functionality."""

    def test_library_initialization_with_config(self):
        """Test library initialization using configuration."""
        # Initialize with development configuration
        result = initialize_library("development")
        assert result.success

        # Verify configuration is accessible
        config = get_config()
        assert config.environment == "development"

    def test_configuration_hot_reload(self):
        """Test configuration reloading functionality."""
        manager = YourLibraryConfigManager()

        # Load initial configuration
        load_result = manager.load_configuration("development")
        assert load_result.success

        initial_config = manager.config
        assert initial_config.environment == "development"

        # Reload configuration
        reload_result = manager.reload_configuration()
        assert reload_result.success

        reloaded_config = manager.config
        assert reloaded_config.environment == "development"

    def test_configuration_error_handling(self):
        """Test configuration error handling in integration scenarios."""
        manager = YourLibraryConfigManager()

        # Test loading invalid configuration source
        result = manager.load_configuration("invalid_source")
        assert result.is_failure
        assert "Unknown config source" in result.error

        # Test reloading without initial load
        reload_result = manager.reload_configuration()
        assert reload_result.is_failure
        assert "No configuration source set" in reload_result.error
```

---

## ✅ Implementation Checklist

### Pre-Implementation
- [ ] **Configuration requirements identified**: All configuration needs documented
- [ ] **Environment variables planned**: FLEXT_* prefix naming convention
- [ ] **Validation rules designed**: Business constraints and security requirements
- [ ] **Integration points mapped**: How configuration connects to library functionality

### Core Implementation
- [ ] **Configuration class implemented**: Inherits from FlextConfig with proper fields
- [ ] **Environment integration added**: Settings class with proper env_prefix
- [ ] **Validation implemented**: Business rules validation method
- [ ] **Field constraints added**: Pydantic Field constraints and validators
- [ ] **Nested configurations**: Complex structures broken into nested classes

### Factory and Manager Implementation
- [ ] **Configuration factory implemented**: Methods for different environments
- [ ] **Configuration manager implemented**: Centralized configuration management
- [ ] **File loading support**: JSON configuration file support
- [ ] **Error handling comprehensive**: All configuration operations use FlextResult

### Integration & Testing
- [ ] **Library integration complete**: Configuration accessible throughout library
- [ ] **Unit tests implemented**: Configuration creation, validation, loading tested
- [ ] **Integration tests added**: Configuration usage in real scenarios tested
- [ ] **Error scenarios covered**: Invalid configurations and edge cases tested
- [ ] **Documentation updated**: Configuration usage documented with examples

### Quality Assurance
- [ ] **Type safety verified**: All configuration fields properly typed
- [ ] **Environment variable testing**: Automatic loading from environment verified
- [ ] **Business rule validation**: Custom validation logic thoroughly tested
- [ ] **Production readiness**: Security and performance considerations addressed

---

## 🚨 Common Pitfalls & Solutions

### 1. **Hardcoded Configuration Values**
```python
# ❌ Don't hardcode configuration values
class BadConfig(FlextConfig):
    database_host: str = "prod-db.company.com"  # Hardcoded!

# ✅ Use environment-aware defaults
class GoodConfig(FlextConfig):
    database_host: str = Field(default="localhost")  # Environment-aware default

    class Settings(FlextConfig.Settings):
        model_config = SettingsConfigDict(env_prefix="FLEXT_")
```

### 2. **Missing Business Validation**
```python
# ❌ Don't skip business rule validation
class BadConfig(FlextConfig):
    pass  # No validation!

# ✅ Implement business rule validation
class GoodConfig(FlextConfig):
    def validate_business_rules(self) -> FlextResult[None]:
        base_result = super().validate_business_rules()
        if base_result.is_failure:
            return base_result

        # Custom validation logic
        if self.environment == "production" and self.debug:
            return FlextResult[None].fail("Debug mode not allowed in production")

        return FlextResult[None].ok(None)
```

### 3. **Inconsistent Environment Variable Naming**
```python
# ❌ Don't use inconsistent environment variable naming
class BadConfig(FlextConfig):
    class Settings(FlextConfig.Settings):
        model_config = SettingsConfigDict(env_prefix="MYAPP_")  # Inconsistent!

# ✅ Use consistent FLEXT_* naming
class GoodConfig(FlextConfig):
    class Settings(FlextConfig.Settings):
        model_config = SettingsConfigDict(env_prefix="FLEXT_MYLIB_")  # Consistent!
```

### 4. **Manual Error Handling**
```python
# ❌ Don't handle errors manually
def bad_config_loading():
    try:
        config = YourConfig()
        return config
    except Exception:
        return None  # Lost error information!

# ✅ Use FlextResult for error handling
def good_config_loading() -> FlextResult[YourConfig]:
    try:
        config = YourConfig()
        validation_result = config.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[YourConfig].fail(validation_result.error)
        return FlextResult[YourConfig].ok(config)
    except Exception as e:
        return FlextResult[YourConfig].fail(f"Configuration loading failed: {e}")
```

---

## 📈 Success Metrics

Track these metrics to measure implementation success:

### Configuration Quality
- **Type Coverage**: 100% type annotations on configuration fields
- **Validation Coverage**: >95% of configuration constraints validated
- **Environment Integration**: 100% configuration values support environment loading

### Reliability
- **Error Handling**: All configuration operations return FlextResult
- **Validation Accuracy**: Business rules catch 100% of invalid configurations
- **Production Safety**: No hardcoded values, secure defaults

### Developer Experience
- **Configuration Clarity**: Clear field descriptions and validation messages
- **Loading Flexibility**: Multiple loading methods (environment, file, factory)
- **Testing Support**: Comprehensive test utilities and patterns

---

## 🔗 Next Steps

1. **Start Simple**: Begin with basic configuration fields and validation
2. **Add Environment Integration**: Implement Settings class with environment loading
3. **Enhance Validation**: Add business rule validation specific to your domain
4. **Test Thoroughly**: Implement comprehensive configuration testing
5. **Document Usage**: Provide clear examples and usage patterns

This implementation guide provides the foundation for successful FlextConfig adoption. Adapt the patterns to your specific library needs while maintaining consistency with FLEXT architectural principles.
