"""FLEXT Core Hierarchical Configuration - Configuration Layer Composition.

Hierarchical configuration management system that organizes configuration models
across architectural layers, providing structured composition patterns for complex
enterprise applications in the FLEXT ecosystem. Enables clean separation between
infrastructure, domain, and application-level configuration concerns.

Module Role in Architecture:
    Configuration Layer → Hierarchical Composition → Application Configuration

    This configuration module enables:
    - Layered configuration composition for enterprise applications
    - Clean separation between infrastructure and domain configuration
    - Type-safe configuration aggregation across multiple services
    - Environment-aware configuration with secure defaults
    - Standardized configuration patterns across all 32 ecosystem projects

Configuration Architecture Layers:
    Base Layer: Core configuration models and foundational patterns
    Domain Layer: Business domain configurations (Database, LDAP, Oracle, WMS)
    Integration Layer: External service integration (Singer, Meltano, observability)
    Application Layer: Complete application configuration compositions
    Settings Layer: Environment variable integration and deployment settings

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Base configuration patterns, domain model composition
    ✅ Implemented: Plugin configuration management with FlextPluginConfig and registry
    🔄 Enhancement: Advanced configuration validation and environment management

Core Configuration Patterns:
    FlextHierarchicalConfigManager: Central configuration orchestration
    create_application_project_config(): Application-level configuration factory
    create_infrastructure_project_config(): Infrastructure service configuration
    create_integration_project_config(): External service integration configuration
    import_complete_config_system(): Unified import for all configuration layers

Ecosystem Usage Patterns:
    # Application-level configuration in services
    app_config = create_application_project_config({
        "database": {"host": "localhost", "port": 5432},
        "ldap": {"host": "ldap.company.com", "port": 389},
        "observability": {"metrics_enabled": True}
    })

    # Infrastructure-specific configuration
    infra_config = create_infrastructure_project_config({
        "oracle": {"connection_pool_size": 20},
        "redis": {"max_connections": 100}
    })

    # Complete configuration system import
    config_system = import_complete_config_system()
    config = config_system.load_environment_config()

Hierarchical Configuration Philosophy:
    - Configuration should be composed from specialized layers
    - Each layer should have clear responsibilities and boundaries
    - Environment-specific configuration should override defaults safely
    - Configuration validation should occur at composition time
    - Secrets should be handled separately from general configuration

Configuration Composition Patterns:
    - Factory functions create pre-configured instances for common scenarios
    - Hierarchical managers orchestrate complex configuration assemblies
    - Type-safe composition prevents configuration errors at runtime
    - Environment variable integration provides deployment flexibility
    - Default value inheritance reduces configuration repetition

Quality Standards:
    - All configuration must be validated at composition time
    - Secrets must never be logged or exposed in error messages
    - Configuration changes must be backward compatible within major versions
    - Environment variables must have clear naming conventions
    - Configuration documentation must be kept up to date

Enterprise Configuration Requirements:
    - Security: Secure handling of sensitive configuration values
    - Flexibility: Support for multiple deployment environments
    - Validation: Comprehensive validation of configuration values
    - Documentation: Clear documentation of all configuration options
    - Monitoring: Configuration changes must be auditable

See Also:
    src/flext_core/config_models.py: Individual configuration model definitions
    src/flext_core/config.py: Base configuration patterns and settings
    docs/TODO.md: Configuration system enhancement roadmap

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

# =============================================================================
# LAYER 1: BASE CONFIGURATION MODELS - Foundation layer
# =============================================================================
# =============================================================================
# LAYER 2: DOMAIN CONFIGURATION MODELS - Domain-specific configs
# =============================================================================
# =============================================================================
# LAYER 3: INTEGRATION CONFIGURATION MODELS - External services
# =============================================================================
# =============================================================================
# LAYER 4: APPLICATION CONFIGURATION MODELS - Complete applications
# =============================================================================
# =============================================================================
# LAYER 5: SETTINGS CLASSES - Environment integration
# =============================================================================
# =============================================================================
# HIERARCHICAL CONFIGURATION FACTORY SYSTEM
# =============================================================================
from flext_core.config_models import (
    # Core TypedDict definitions for type safety
    DatabaseConfigDict,
    # Complete application configuration
    FlextApplicationConfig,
    # Base model foundation
    FlextBaseConfigModel,
    FlextBaseSettings,
    # Infrastructure domain configurations
    FlextDatabaseConfig,
    # Environment-aware settings
    FlextDatabaseSettings,
    # Composite integration configurations
    FlextDataIntegrationConfig,
    # Security domain configurations
    FlextJWTConfig,
    FlextLDAPConfig,
    # Observability domain configurations
    FlextObservabilityConfig,
    FlextOracleConfig,
    # Plugin configuration models
    FlextPluginConfig,
    FlextPluginRegistryConfig,
    FlextRedisConfig,
    FlextRedisSettings,
    # Data integration domain configurations
    FlextSingerConfig,
    JWTConfigDict,
    LDAPConfigDict,
    ObservabilityConfigDict,
    OracleConfigDict,
    PluginConfigDict,
    RedisConfigDict,
    SingerConfigDict,
    # Type-safe factory functions
    create_database_config,
    create_ldap_config,
    create_oracle_config,
    create_plugin_config,
    create_plugin_registry_config,
    create_redis_config,
    # Configuration utilities
    load_config_from_env,
    merge_configs,
    validate_config,
)

# =============================================================================
# HIERARCHICAL IMPORT HELPERS - Layer-specific import functions
# =============================================================================


def get_base_layer_imports() -> dict[str, object]:
    """Get base layer configuration imports."""
    return {
        # Base models
        "FlextBaseConfigModel": FlextBaseConfigModel,
        "FlextBaseSettings": FlextBaseSettings,
        # TypedDict definitions
        "DatabaseConfigDict": DatabaseConfigDict,
        "RedisConfigDict": RedisConfigDict,
        "JWTConfigDict": JWTConfigDict,
        "LDAPConfigDict": LDAPConfigDict,
        "OracleConfigDict": OracleConfigDict,
        "PluginConfigDict": PluginConfigDict,
        "SingerConfigDict": SingerConfigDict,
        "ObservabilityConfigDict": ObservabilityConfigDict,
    }


def get_domain_layer_imports() -> dict[str, object]:
    """Get domain layer configuration imports."""
    return {
        # Infrastructure domain
        "FlextDatabaseConfig": FlextDatabaseConfig,
        "FlextRedisConfig": FlextRedisConfig,
        "FlextOracleConfig": FlextOracleConfig,
        "FlextLDAPConfig": FlextLDAPConfig,
        # Security domain
        "FlextJWTConfig": FlextJWTConfig,
        # Plugin domain
        "FlextPluginConfig": FlextPluginConfig,
        "FlextPluginRegistryConfig": FlextPluginRegistryConfig,
        # Data integration domain
        "FlextSingerConfig": FlextSingerConfig,
        # Observability domain
        "FlextObservabilityConfig": FlextObservabilityConfig,
    }


def get_integration_layer_imports() -> dict[str, object]:
    """Get integration layer configuration imports."""
    return {
        "FlextDataIntegrationConfig": FlextDataIntegrationConfig,
    }


def get_application_layer_imports() -> dict[str, object]:
    """Get application layer configuration imports."""
    return {
        "FlextApplicationConfig": FlextApplicationConfig,
    }


def get_settings_layer_imports() -> dict[str, object]:
    """Get settings layer configuration imports."""
    return {
        "FlextDatabaseSettings": FlextDatabaseSettings,
        "FlextRedisSettings": FlextRedisSettings,
    }


def get_factory_layer_imports() -> dict[str, object]:
    """Get factory layer configuration imports."""
    return {
        # Factory functions
        "create_database_config": create_database_config,
        "create_redis_config": create_redis_config,
        "create_oracle_config": create_oracle_config,
        "create_ldap_config": create_ldap_config,
        "create_plugin_config": create_plugin_config,
        "create_plugin_registry_config": create_plugin_registry_config,
        # Utilities
        "load_config_from_env": load_config_from_env,
        "merge_configs": merge_configs,
        "validate_config": validate_config,
    }


# =============================================================================
# HIERARCHICAL CONFIGURATION MANAGER
# =============================================================================


class FlextHierarchicalConfigManager:
    """Manages hierarchical configuration imports and layer organization."""

    def __init__(self) -> None:
        """Initialize hierarchical configuration manager."""
        self._layers = {
            "base": get_base_layer_imports(),
            "domain": get_domain_layer_imports(),
            "integration": get_integration_layer_imports(),
            "application": get_application_layer_imports(),
            "settings": get_settings_layer_imports(),
            "factory": get_factory_layer_imports(),
        }

    def get_layer_imports(self, layer: str) -> dict[str, object]:
        """Get imports for a specific layer."""
        if layer not in self._layers:
            valid_layers = list(self._layers.keys())
            msg = f"Invalid layer '{layer}'. Valid layers: {valid_layers}"
            raise ValueError(msg)
        return self._layers[layer]

    def get_all_imports(self) -> dict[str, object]:
        """Get all configuration imports from all layers."""
        all_imports = {}
        for layer_imports in self._layers.values():
            all_imports.update(layer_imports)
        return all_imports

    def get_imports_by_layers(self, layers: list[str]) -> dict[str, object]:
        """Get imports for specific layers."""
        layer_imports = {}
        for layer in layers:
            layer_imports.update(self.get_layer_imports(layer))
        return layer_imports


# =============================================================================
# CONVENIENCE FUNCTIONS FOR LAYER-SPECIFIC IMPORTS
# =============================================================================


def import_base_config_layer() -> dict[str, object]:
    """Import base configuration layer for foundational usage."""
    return get_base_layer_imports()


def import_domain_config_layer() -> dict[str, object]:
    """Import domain configuration layer for domain-specific projects."""
    return get_domain_layer_imports()


def import_integration_config_layer() -> dict[str, object]:
    """Import integration configuration layer for external services."""
    return get_integration_layer_imports()


def import_application_config_layer() -> dict[str, object]:
    """Import application configuration layer for complete applications."""
    return get_application_layer_imports()


def import_settings_config_layer() -> dict[str, object]:
    """Import settings configuration layer for environment integration."""
    return get_settings_layer_imports()


def import_factory_config_layer() -> dict[str, object]:
    """Import factory configuration layer for configuration creation."""
    return get_factory_layer_imports()


def import_complete_config_system() -> dict[str, object]:
    """Import the complete hierarchical configuration system."""
    manager = FlextHierarchicalConfigManager()
    return manager.get_all_imports()


# =============================================================================
# HIERARCHICAL CONFIGURATION USE CASES
# =============================================================================


def create_infrastructure_project_config() -> dict[str, object]:
    """Create configuration imports for infrastructure projects."""
    manager = FlextHierarchicalConfigManager()
    return manager.get_imports_by_layers(["base", "domain", "settings", "factory"])


def create_application_project_config() -> dict[str, object]:
    """Create configuration imports for application projects (flext-auth, flext-api)."""
    manager = FlextHierarchicalConfigManager()
    return manager.get_imports_by_layers(
        ["base", "domain", "application", "settings", "factory"],
    )


def create_integration_project_config() -> dict[str, object]:
    """Create configuration imports for integration projects (Singer taps/targets)."""
    manager = FlextHierarchicalConfigManager()
    return manager.get_imports_by_layers(["base", "domain", "integration", "factory"])


# =============================================================================
# EXPORTS - Hierarchical configuration system
# =============================================================================

__all__ = [
    # Hierarchical manager
    "FlextHierarchicalConfigManager",
    "create_application_project_config",
    # Use case configurations
    "create_infrastructure_project_config",
    "create_integration_project_config",
    "get_application_layer_imports",
    # Layer import functions
    "get_base_layer_imports",
    "get_domain_layer_imports",
    "get_factory_layer_imports",
    "get_integration_layer_imports",
    "get_settings_layer_imports",
    "import_application_config_layer",
    # Convenience import functions
    "import_base_config_layer",
    "import_complete_config_system",
    "import_domain_config_layer",
    "import_factory_config_layer",
    "import_integration_config_layer",
    "import_settings_config_layer",
    # Direct imports are available through layer functions
]
