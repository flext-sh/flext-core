"""Comprehensive tests for FlextConfig Namespace Registration System.

Tests the namespace architecture that enables unified configuration management
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from flext_core import FlextConfig, FlextResult

# ============================================================================
# Test Fixtures - Mock Namespace Configs
# ============================================================================


class MockLdapConfig(BaseModel):
    """Mock LDAP configuration for namespace testing."""

    host: str = Field(default="localhost")
    port: int = Field(default=389, ge=1, le=65535)
    bind_dn: str = Field(default="cn=REDACTED_LDAP_BIND_PASSWORD")

    @classmethod
    def register_as_namespace(cls) -> FlextResult[bool]:
        """Register as 'ldap' namespace."""
        try:
            FlextConfig.register_namespace("ldap", cls)
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))


class MockLdifConfig(BaseModel):
    """Mock LDIF configuration for namespace testing."""

    ldif_encoding: str = Field(default="utf-8")
    ldif_max_entries: int = Field(default=1000, ge=1)

    @classmethod
    def register_as_namespace(cls) -> FlextResult[bool]:
        """Register as 'ldif' namespace."""
        try:
            FlextConfig.register_namespace("ldif", cls)
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))


class MockCliConfig(BaseModel):
    """Mock CLI configuration for namespace testing."""

    cli_verbose: bool = Field(default=False)
    cli_color: bool = Field(default=True)

    _instances: ClassVar[dict[type, MockCliConfig]] = {}

    @classmethod
    def get_instance(cls) -> MockCliConfig:
        """Singleton pattern for testing."""
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]

    @classmethod
    def register_as_namespace(cls) -> FlextResult[bool]:
        """Register as 'cli' namespace with factory."""
        try:
            FlextConfig.register_namespace("cli", cls, factory=cls.get_instance)
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))


# ============================================================================
# Test Suite
# ============================================================================


class TestFlextConfigNamespaces:
    """Test suite for FlextConfig namespace registration system."""

    @pytest.fixture(autouse=True)
    def reset_namespaces(self) -> None:
        """Reset namespaces before each test."""
        FlextConfig.reset_namespaces()

    def test_register_namespace_basic(self) -> None:
        """Test basic namespace registration."""
        # Register namespace
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        # Verify registration
        assert FlextConfig.has_namespace("ldap")
        assert "ldap" in FlextConfig.list_namespaces()

    def test_register_multiple_namespaces(self) -> None:
        """Test registering multiple namespaces."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)
        FlextConfig.register_namespace("ldif", MockLdifConfig)
        FlextConfig.register_namespace("cli", MockCliConfig)

        # Verify all registered
        assert FlextConfig.has_namespace("ldap")
        assert FlextConfig.has_namespace("ldif")
        assert FlextConfig.has_namespace("cli")

        namespaces = FlextConfig.list_namespaces()
        assert "ldap" in namespaces
        assert "ldif" in namespaces
        assert "cli" in namespaces
        assert len(namespaces) == 3

    def test_access_namespace_via_attribute(self) -> None:
        """Test accessing namespace via attribute (config.ldap)."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        config = FlextConfig()
        ldap_config = config.ldap

        # Verify type and values
        assert isinstance(ldap_config, MockLdapConfig)
        assert ldap_config.host == "localhost"
        assert ldap_config.port == 389

    def test_namespace_lazy_loading(self) -> None:
        """Test that namespaces are lazy-loaded (created on first access)."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        # Namespace registered but not yet instantiated
        assert "ldap" in FlextConfig._namespaces
        assert "ldap" not in FlextConfig._namespace_instances

        # Access triggers instantiation
        config = FlextConfig()
        _ = config.ldap

        # Now instantiated
        assert "ldap" in FlextConfig._namespace_instances

    def test_namespace_singleton_per_access(self) -> None:
        """Test that multiple accesses return same namespace instance."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        config = FlextConfig()
        ldap1 = config.ldap
        ldap2 = config.ldap

        # Same instance
        assert ldap1 is ldap2

    def test_namespace_with_factory_function(self) -> None:
        """Test namespace registration with custom factory function."""
        # MockCliConfig uses get_instance() factory
        FlextConfig.register_namespace(
            "cli",
            MockCliConfig,
            factory=MockCliConfig.get_instance,
        )

        config = FlextConfig()
        cli1 = config.cli
        cli2 = config.cli

        # Factory ensures singleton
        assert cli1 is cli2
        assert isinstance(cli1, MockCliConfig)

    def test_protocol_compliance(self) -> None:
        """Test that mock configs satisfy NamespaceConfigProtocol."""
        # MockLdapConfig should satisfy protocol
        # Note: isinstance doesn't work with Protocol classmethod, but we can
        # verify the method exists
        assert hasattr(MockLdapConfig, "register_as_namespace")
        assert callable(MockLdapConfig.register_as_namespace)

        # Test registration works
        result = MockLdapConfig.register_as_namespace()
        assert result.is_success

    def test_register_via_protocol_method(self) -> None:
        """Test registering namespace via protocol's register_as_namespace()."""
        # Use protocol method
        result = MockLdapConfig.register_as_namespace()
        assert result.is_success

        # Verify registration
        assert FlextConfig.has_namespace("ldap")

        # Access namespace
        config = FlextConfig()
        ldap_config = config.ldap
        assert isinstance(ldap_config, MockLdapConfig)

    def test_list_namespaces_empty(self) -> None:
        """Test list_namespaces returns empty list initially."""
        namespaces = FlextConfig.list_namespaces()
        assert isinstance(namespaces, list)
        assert len(namespaces) == 0

    def test_has_namespace_false_for_unregistered(self) -> None:
        """Test has_namespace returns False for unregistered namespace."""
        assert FlextConfig.has_namespace("nonexistent") is False

    def test_access_unregistered_namespace_raises_error(self) -> None:
        """Test accessing unregistered namespace raises AttributeError."""
        config = FlextConfig()

        with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
            _ = config.nonexistent

    def test_register_non_basemodel_raises_error(self) -> None:
        """Test registering non-BaseModel class raises TypeError."""

        class NotABaseModel:
            pass

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel"):
            FlextConfig.register_namespace(
                "invalid", cast("type[BaseModel]", NotABaseModel)
            )

    def test_register_basesettings_raises_error(self) -> None:
        """Test registering BaseSettings (instead of BaseModel) raises TypeError."""

        class SettingsConfig(BaseSettings):
            field: str = "value"

        with pytest.raises(TypeError, match="inherits from BaseSettings"):
            FlextConfig.register_namespace("settings", SettingsConfig)

    def test_namespace_thread_safety_registration(self) -> None:
        """Test thread-safe namespace registration."""
        registration_results: list[bool] = []

        def register_namespace() -> None:
            try:
                FlextConfig.register_namespace("ldap", MockLdapConfig)
                registration_results.append(True)
            except Exception:
                registration_results.append(False)

        # Try registering same namespace from multiple threads
        threads = [threading.Thread(target=register_namespace) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Only one should succeed, others should be safe
        assert FlextConfig.has_namespace("ldap")
        # All threads completed safely
        assert len(registration_results) == 10

    def test_namespace_thread_safety_access(self) -> None:
        """Test thread-safe namespace access."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        config = FlextConfig()
        accessed_configs: list[MockLdapConfig] = []

        def access_namespace() -> None:
            ldap_config = cast("MockLdapConfig", config.ldap)
            accessed_configs.append(ldap_config)

        # Access namespace from multiple threads
        threads = [threading.Thread(target=access_namespace) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads got the same instance
        assert len(accessed_configs) == 10
        assert all(cfg is accessed_configs[0] for cfg in accessed_configs)

    def test_reset_namespaces(self) -> None:
        """Test reset_namespaces clears all registrations."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)
        FlextConfig.register_namespace("ldif", MockLdifConfig)

        assert len(FlextConfig.list_namespaces()) == 2

        # Reset
        FlextConfig.reset_namespaces()

        # All cleared
        assert len(FlextConfig.list_namespaces()) == 0
        assert not FlextConfig.has_namespace("ldap")
        assert not FlextConfig.has_namespace("ldif")

    def test_namespace_independence(self) -> None:
        """Test that namespaces are independent of each other."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)
        FlextConfig.register_namespace("ldif", MockLdifConfig)

        config = FlextConfig()

        # Access both
        ldap_config = config.ldap
        ldif_config = config.ldif

        # Independent instances
        assert isinstance(ldap_config, MockLdapConfig)
        assert isinstance(ldif_config, MockLdifConfig)
        assert ldap_config is not ldif_config

    def test_namespace_default_values(self) -> None:
        """Test namespace configs use their default values."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        config = FlextConfig()
        ldap_config = cast("MockLdapConfig", config.ldap)

        # Check default values
        assert ldap_config.host == "localhost"
        assert ldap_config.port == 389
        assert ldap_config.bind_dn == "cn=REDACTED_LDAP_BIND_PASSWORD"

    def test_namespace_custom_values(self) -> None:
        """Test namespace configs can be instantiated with custom values."""

        def custom_factory() -> MockLdapConfig:
            return MockLdapConfig(host="custom.server.com", port=636)

        FlextConfig.register_namespace("ldap", MockLdapConfig, factory=custom_factory)

        config = FlextConfig()
        ldap_config = cast("MockLdapConfig", config.ldap)

        # Check custom values from factory
        assert ldap_config.host == "custom.server.com"
        assert ldap_config.port == 636

    def test_multiple_configs_same_namespace_forbidden(self) -> None:
        """Test that registering same namespace twice does nothing (idempotent)."""
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        # Try registering again - should be idempotent (no error, but no duplicate)
        FlextConfig.register_namespace("ldap", MockLdapConfig)

        # Still only one namespace
        namespaces = FlextConfig.list_namespaces()
        assert namespaces.count("ldap") == 1

    def test_namespace_integration_with_flextconfig(self) -> None:
        """Test namespace integration with FlextConfig singleton."""
        # Register namespaces
        MockLdapConfig.register_as_namespace()
        MockLdifConfig.register_as_namespace()

        # Get FlextConfig singleton
        config = FlextConfig.get_global_instance()

        # Access namespaces
        ldap = config.ldap
        ldif = config.ldif

        # Verify types
        assert isinstance(ldap, MockLdapConfig)
        assert isinstance(ldif, MockLdifConfig)

        # Verify independence
        assert ldap is not ldif


class TestNamespaceConfigProtocol:
    """Test suite for NamespaceConfigProtocol compliance."""

    @pytest.fixture(autouse=True)
    def reset_namespaces(self) -> None:
        """Reset namespaces before each test."""
        FlextConfig.reset_namespaces()

    def test_protocol_method_signature(self) -> None:
        """Test that protocol method has correct signature."""
        # Verify method exists
        assert hasattr(MockLdapConfig, "register_as_namespace")

        # Verify it's a classmethod
        method = MockLdapConfig.register_as_namespace
        assert callable(method)

    def test_protocol_method_returns_result(self) -> None:
        """Test that protocol method returns FlextResult[bool]."""
        result = MockLdapConfig.register_as_namespace()

        # Verify return type
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert isinstance(result.value, bool)

    def test_protocol_method_handles_errors(self) -> None:
        """Test that protocol method handles errors gracefully."""
        # Register once (success)
        result1 = MockLdapConfig.register_as_namespace()
        assert result1.is_success

        # Subsequent calls should be idempotent (no error)
        result2 = MockLdapConfig.register_as_namespace()
        # Should still be successful (idempotent behavior)
        assert result2.is_success

    def test_auto_registration_pattern(self) -> None:
        """Test auto-registration pattern (called on module import)."""
        # Simulate auto-registration in __init__.py
        result = MockLdapConfig.register_as_namespace()
        assert result.is_success

        # Namespace should be immediately available
        assert FlextConfig.has_namespace("ldap")

        # And usable
        config = FlextConfig()
        ldap = config.ldap
        assert isinstance(ldap, MockLdapConfig)


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================


class TestNamespaceEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture(autouse=True)
    def reset_namespaces(self) -> None:
        """Reset namespaces before each test."""
        FlextConfig.reset_namespaces()

    def test_empty_namespace_name(self) -> None:
        """Test registering namespace with empty name."""
        # Empty string is technically allowed (implementation detail)
        # But should work correctly
        FlextConfig.register_namespace("", MockLdapConfig)
        assert FlextConfig.has_namespace("")

    def test_namespace_name_with_special_characters(self) -> None:
        """Test namespace names with special characters."""
        # Test various valid Python identifiers
        FlextConfig.register_namespace("ldap_v2", MockLdapConfig)
        assert FlextConfig.has_namespace("ldap_v2")

        FlextConfig.register_namespace("ldap-legacy", MockLdapConfig)
        assert FlextConfig.has_namespace("ldap-legacy")

    def test_access_reserved_attribute(self) -> None:
        """Test that accessing reserved FlextConfig attributes still works."""
        config = FlextConfig()

        # Reserved attributes should work normally
        assert hasattr(config, "debug")
        assert hasattr(config, "log_level")
        assert hasattr(config, "model_dump")

        # Can access them
        _ = config.debug
        _ = config.log_level
        _ = config.model_dump()

    def test_namespace_does_not_override_config_attributes(self) -> None:
        """Test that namespace names don't override FlextConfig attributes."""
        # Try registering with name that conflicts with FlextConfig attribute
        # This should work because __getattr__ is only called for missing attributes
        FlextConfig.register_namespace("debug", MockLdapConfig)

        config = FlextConfig()

        # Original attribute should still work
        assert isinstance(config.debug, bool)

        # Namespace won't be accessible due to name conflict
        # (This is expected behavior - don't register namespaces with conflicting names)
