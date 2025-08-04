"""Comprehensive tests for flext_core.config_hierarchical module.

Tests for hierarchical configuration management system covering all functions,
classes, and layer organization patterns.
"""

import pytest

# Force import of the module to ensure coverage tracking
import flext_core.config_hierarchical  # noqa: F401
from flext_core.config_hierarchical import (
    FlextHierarchicalConfigManager,
    create_application_project_config,
    create_infrastructure_project_config,
    create_integration_project_config,
    get_application_layer_imports,
    get_base_layer_imports,
    get_domain_layer_imports,
    get_factory_layer_imports,
    get_integration_layer_imports,
    get_settings_layer_imports,
    import_application_config_layer,
    import_base_config_layer,
    import_complete_config_system,
    import_domain_config_layer,
    import_factory_config_layer,
    import_integration_config_layer,
    import_settings_config_layer,
)


@pytest.mark.unit
class TestLayerImportFunctions:
    """Test individual layer import functions."""

    def test_get_base_layer_imports(self) -> None:
        """Test get_base_layer_imports returns expected base layer imports."""
        imports = get_base_layer_imports()

        # Check return type
        assert isinstance(imports, dict)

        # Check key classes are present
        expected_keys = [
            "FlextBaseConfigModel",
            "FlextBaseSettings",
            "DatabaseConfigDict",
            "RedisConfigDict",
            "JWTConfigDict",
            "LDAPConfigDict",
            "OracleConfigDict",
            "SingerConfigDict",
            "ObservabilityConfigDict",
        ]

        for key in expected_keys:
            assert key in imports, (
                f"Expected key '{key}' not found in base layer imports"
            )
            assert imports[key] is not None, f"Import '{key}' is None"

    def test_get_domain_layer_imports(self) -> None:
        """Test get_domain_layer_imports returns expected domain layer imports."""
        imports = get_domain_layer_imports()

        assert isinstance(imports, dict)

        expected_keys = [
            "FlextDatabaseConfig",
            "FlextRedisConfig",
            "FlextOracleConfig",
            "FlextLDAPConfig",
            "FlextJWTConfig",
            "FlextSingerConfig",
            "FlextObservabilityConfig",
        ]

        for key in expected_keys:
            assert key in imports, (
                f"Expected key '{key}' not found in domain layer imports"
            )
            assert imports[key] is not None

    def test_get_integration_layer_imports(self) -> None:
        """Test get_integration_layer_imports returns expected integration imports."""
        imports = get_integration_layer_imports()

        assert isinstance(imports, dict)
        assert "FlextDataIntegrationConfig" in imports
        assert imports["FlextDataIntegrationConfig"] is not None

    def test_get_application_layer_imports(self) -> None:
        """Test get_application_layer_imports returns expected application imports."""
        imports = get_application_layer_imports()

        assert isinstance(imports, dict)
        assert "FlextApplicationConfig" in imports
        assert imports["FlextApplicationConfig"] is not None

    def test_get_settings_layer_imports(self) -> None:
        """Test get_settings_layer_imports returns expected settings imports."""
        imports = get_settings_layer_imports()

        assert isinstance(imports, dict)

        expected_keys = ["FlextDatabaseSettings", "FlextRedisSettings"]
        for key in expected_keys:
            assert key in imports, (
                f"Expected key '{key}' not found in settings layer imports"
            )
            assert imports[key] is not None

    def test_get_factory_layer_imports(self) -> None:
        """Test get_factory_layer_imports returns expected factory imports."""
        imports = get_factory_layer_imports()

        assert isinstance(imports, dict)

        expected_keys = [
            "create_database_config",
            "create_redis_config",
            "create_oracle_config",
            "create_ldap_config",
            "load_config_from_env",
            "merge_configs",
            "validate_config",
        ]

        for key in expected_keys:
            assert key in imports, (
                f"Expected key '{key}' not found in factory layer imports"
            )
            assert imports[key] is not None


@pytest.mark.unit
class TestFlextHierarchicalConfigManager:
    """Test FlextHierarchicalConfigManager class."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes with correct layers."""
        manager = FlextHierarchicalConfigManager()

        # Check internal layers structure
        assert hasattr(manager, "_layers")
        assert isinstance(manager._layers, dict)

        expected_layers = [
            "base",
            "domain",
            "integration",
            "application",
            "settings",
            "factory",
        ]
        for layer in expected_layers:
            assert layer in manager._layers, f"Expected layer '{layer}' not found"

    def test_get_layer_imports_valid_layer(self) -> None:
        """Test get_layer_imports with valid layer names."""
        manager = FlextHierarchicalConfigManager()

        # Test each valid layer
        valid_layers = [
            "base",
            "domain",
            "integration",
            "application",
            "settings",
            "factory",
        ]

        for layer in valid_layers:
            imports = manager.get_layer_imports(layer)
            assert isinstance(imports, dict)
            assert len(imports) > 0, f"Layer '{layer}' should have at least one import"

    def test_get_layer_imports_invalid_layer(self) -> None:
        """Test get_layer_imports with invalid layer name raises ValueError."""
        manager = FlextHierarchicalConfigManager()

        with pytest.raises(ValueError, match="Invalid layer 'invalid_layer'"):
            manager.get_layer_imports("invalid_layer")

    def test_get_all_imports(self) -> None:
        """Test get_all_imports combines all layer imports."""
        manager = FlextHierarchicalConfigManager()

        all_imports = manager.get_all_imports()

        assert isinstance(all_imports, dict)
        assert len(all_imports) > 0

        # Should contain imports from all layers
        # Check a few key imports from different layers
        expected_from_base = "FlextBaseConfigModel"
        expected_from_domain = "FlextDatabaseConfig"
        expected_from_factory = "create_database_config"

        assert expected_from_base in all_imports
        assert expected_from_domain in all_imports
        assert expected_from_factory in all_imports

    def test_get_imports_by_layers(self) -> None:
        """Test get_imports_by_layers with multiple layers."""
        manager = FlextHierarchicalConfigManager()

        # Test with multiple layers
        selected_layers = ["base", "domain"]
        imports = manager.get_imports_by_layers(selected_layers)

        assert isinstance(imports, dict)
        assert len(imports) > 0

        # Should contain imports from selected layers
        assert "FlextBaseConfigModel" in imports  # from base
        assert "FlextDatabaseConfig" in imports  # from domain

        # Should not contain imports from unselected layers
        # (though some overlap might exist)

    def test_get_imports_by_layers_single_layer(self) -> None:
        """Test get_imports_by_layers with single layer."""
        manager = FlextHierarchicalConfigManager()

        imports = manager.get_imports_by_layers(["base"])
        base_direct = manager.get_layer_imports("base")

        # Should be equivalent to direct layer import
        assert imports == base_direct

    def test_get_imports_by_layers_empty_list(self) -> None:
        """Test get_imports_by_layers with empty layer list."""
        manager = FlextHierarchicalConfigManager()

        imports = manager.get_imports_by_layers([])

        assert isinstance(imports, dict)
        assert len(imports) == 0

    def test_get_imports_by_layers_invalid_layer(self) -> None:
        """Test get_imports_by_layers with invalid layer raises ValueError."""
        manager = FlextHierarchicalConfigManager()

        with pytest.raises(ValueError, match="Invalid layer"):
            manager.get_imports_by_layers(["base", "invalid_layer"])


@pytest.mark.unit
class TestConvenienceImportFunctions:
    """Test convenience import functions."""

    def test_import_base_config_layer(self) -> None:
        """Test import_base_config_layer returns base layer imports."""
        imports = import_base_config_layer()
        direct_imports = get_base_layer_imports()

        assert imports == direct_imports

    def test_import_domain_config_layer(self) -> None:
        """Test import_domain_config_layer returns domain layer imports."""
        imports = import_domain_config_layer()
        direct_imports = get_domain_layer_imports()

        assert imports == direct_imports

    def test_import_integration_config_layer(self) -> None:
        """Test import_integration_config_layer returns integration layer imports."""
        imports = import_integration_config_layer()
        direct_imports = get_integration_layer_imports()

        assert imports == direct_imports

    def test_import_application_config_layer(self) -> None:
        """Test import_application_config_layer returns application layer imports."""
        imports = import_application_config_layer()
        direct_imports = get_application_layer_imports()

        assert imports == direct_imports

    def test_import_settings_config_layer(self) -> None:
        """Test import_settings_config_layer returns settings layer imports."""
        imports = import_settings_config_layer()
        direct_imports = get_settings_layer_imports()

        assert imports == direct_imports

    def test_import_factory_config_layer(self) -> None:
        """Test import_factory_config_layer returns factory layer imports."""
        imports = import_factory_config_layer()
        direct_imports = get_factory_layer_imports()

        assert imports == direct_imports

    def test_import_complete_config_system(self) -> None:
        """Test import_complete_config_system returns all configuration imports."""
        imports = import_complete_config_system()

        manager = FlextHierarchicalConfigManager()
        expected_imports = manager.get_all_imports()

        assert imports == expected_imports


@pytest.mark.unit
class TestProjectConfigFactories:
    """Test project configuration factory functions."""

    def test_create_infrastructure_project_config(self) -> None:
        """Test create_infrastructure_project_config returns appropriate imports."""
        config = create_infrastructure_project_config()

        assert isinstance(config, dict)
        assert len(config) > 0

        # Should include base, domain, settings, factory layers
        # Check for representative imports from each expected layer
        assert "FlextBaseConfigModel" in config  # base layer
        assert "FlextDatabaseConfig" in config  # domain layer
        assert "FlextDatabaseSettings" in config  # settings layer
        assert "create_database_config" in config  # factory layer

    def test_create_application_project_config(self) -> None:
        """Test create_application_project_config returns appropriate imports."""
        config = create_application_project_config()

        assert isinstance(config, dict)
        assert len(config) > 0

        # Should include base, domain, application, settings, factory layers
        assert "FlextBaseConfigModel" in config  # base layer
        assert "FlextDatabaseConfig" in config  # domain layer
        assert "FlextApplicationConfig" in config  # application layer
        assert "FlextDatabaseSettings" in config  # settings layer
        assert "create_database_config" in config  # factory layer

    def test_create_integration_project_config(self) -> None:
        """Test create_integration_project_config returns appropriate imports."""
        config = create_integration_project_config()

        assert isinstance(config, dict)
        assert len(config) > 0

        # Should include base, domain, integration, factory layers
        assert "FlextBaseConfigModel" in config  # base layer
        assert "FlextDatabaseConfig" in config  # domain layer
        assert "FlextDataIntegrationConfig" in config  # integration layer
        assert "create_database_config" in config  # factory layer


@pytest.mark.unit
class TestImportConsistency:
    """Test consistency between different import methods."""

    def test_layer_import_function_consistency(self) -> None:
        """Test that layer import functions are consistent."""
        # Test that convenience functions match get_*_layer_imports
        assert import_base_config_layer() == get_base_layer_imports()
        assert import_domain_config_layer() == get_domain_layer_imports()
        assert import_integration_config_layer() == get_integration_layer_imports()
        assert import_application_config_layer() == get_application_layer_imports()
        assert import_settings_config_layer() == get_settings_layer_imports()
        assert import_factory_config_layer() == get_factory_layer_imports()

    def test_manager_vs_function_consistency(self) -> None:
        """Test that manager methods match standalone functions."""
        manager = FlextHierarchicalConfigManager()

        assert manager.get_layer_imports("base") == get_base_layer_imports()
        assert manager.get_layer_imports("domain") == get_domain_layer_imports()
        assert (
            manager.get_layer_imports("integration") == get_integration_layer_imports()
        )
        assert (
            manager.get_layer_imports("application") == get_application_layer_imports()
        )
        assert manager.get_layer_imports("settings") == get_settings_layer_imports()
        assert manager.get_layer_imports("factory") == get_factory_layer_imports()

    def test_complete_system_vs_manager_consistency(self) -> None:
        """Test that import_complete_config_system matches manager.get_all_imports."""
        complete_system = import_complete_config_system()

        manager = FlextHierarchicalConfigManager()
        manager_all = manager.get_all_imports()

        assert complete_system == manager_all


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_imports_are_not_none(self) -> None:
        """Test that all imported objects are not None."""
        complete_system = import_complete_config_system()

        for name, obj in complete_system.items():
            assert obj is not None, f"Import '{name}' should not be None"

    def test_imports_are_callable_or_classes(self) -> None:
        """Test that all imports are callable functions or classes."""
        complete_system = import_complete_config_system()

        for name, obj in complete_system.items():
            # Should be either a class (type) or callable function
            assert callable(obj) or isinstance(obj, type), (
                f"Import '{name}' should be callable or a class"
            )

    def test_no_duplicate_imports_in_layers(self) -> None:
        """Test that each layer has unique imports (no key conflicts within layer)."""
        manager = FlextHierarchicalConfigManager()

        for layer_name in [
            "base",
            "domain",
            "integration",
            "application",
            "settings",
            "factory",
        ]:
            layer_imports = manager.get_layer_imports(layer_name)

            # Check no duplicate keys (this would be a dict implementation issue)
            import_names = list(layer_imports.keys())
            unique_names = set(import_names)

            assert len(import_names) == len(unique_names), (
                f"Layer '{layer_name}' has duplicate import names"
            )

    def test_manager_layers_immutability(self) -> None:
        """Test that manager internal layers cannot be accidentally modified."""
        manager = FlextHierarchicalConfigManager()

        # Get reference to internal layers
        original_layer_count = len(manager._layers)

        # Get layer imports (should not affect internal state)
        manager.get_layer_imports("base")
        manager.get_all_imports()

        # Internal layers should remain unchanged
        assert len(manager._layers) == original_layer_count


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_infrastructure_project_has_required_configs(self) -> None:
        """Test that infrastructure project config has all required components."""
        config = create_infrastructure_project_config()

        # Should have database configuration capabilities
        assert "FlextDatabaseConfig" in config
        assert "create_database_config" in config
        assert "FlextDatabaseSettings" in config

        # Should have Redis configuration
        assert "FlextRedisConfig" in config
        assert "create_redis_config" in config

        # Should have Oracle configuration
        assert "FlextOracleConfig" in config
        assert "create_oracle_config" in config

    def test_application_project_has_complete_config_stack(self) -> None:
        """Test that application project config has complete configuration stack."""
        config = create_application_project_config()

        # Should have all infrastructure configs
        assert "FlextDatabaseConfig" in config
        assert "FlextRedisConfig" in config
        assert "FlextOracleConfig" in config

        # Should have application-level config
        assert "FlextApplicationConfig" in config

        # Should have settings for environment integration
        assert "FlextDatabaseSettings" in config
        assert "FlextRedisSettings" in config

        # Should have factory functions
        assert "create_database_config" in config
        assert "merge_configs" in config
        assert "validate_config" in config

    def test_integration_project_has_data_integration_focus(self) -> None:
        """Test that integration project config focuses on data integration."""
        config = create_integration_project_config()

        # Should have data integration specific config
        assert "FlextDataIntegrationConfig" in config

        # Should have Singer config for data pipeline integration
        assert "FlextSingerConfig" in config

        # Should have observability for monitoring integration
        assert "FlextObservabilityConfig" in config

        # Should have factory functions for creating configs
        assert "create_database_config" in config
        assert "validate_config" in config
