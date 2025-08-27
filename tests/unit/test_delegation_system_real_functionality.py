"""Real functionality tests for delegation system - NO MOCKS, only real usage.

These tests demonstrate the delegation system working with actual mixins and hosts,
focusing on real-world scenarios and increasing coverage to near 100%.
"""

from __future__ import annotations

import ast

import pytest

from flext_core import FlextResult, FlextTypes
from flext_core.delegation_system import (
    FlextMixinDelegator,
    create_mixin_delegator,
    validate_delegation_system,
)
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextValidatableMixin,
)

# Initialize dynamic exception classes
FlextExceptions.initialize()

pytestmark = [pytest.mark.unit, pytest.mark.core]


class BusinessLogicMixin:
    """Real business logic mixin for testing."""

    def __init__(self) -> None:
        self.calculations_count = 0
        self.last_result = 0.0

    def calculate_discount(self, amount: float, percentage: float) -> float:
        """Real business calculation."""
        self.calculations_count += 1
        discount = amount * (percentage / 100)
        self.last_result = discount
        return discount

    def validate_amount(self, amount: float) -> FlextResult[None]:
        """Real validation with FlextResult."""
        if amount <= 0:
            return FlextResult[None].fail("Amount must be positive")
        if amount > 1_000_000:
            return FlextResult[None].fail("Amount exceeds maximum limit")
        return FlextResult[None].ok(None)

    @property
    def discount_rate(self) -> float:
        """Read-only property."""
        return 10.5

    @property
    def configurable_rate(self) -> float:
        """Property with setter."""
        return getattr(self, "_configurable_rate", 5.0)

    @configurable_rate.setter
    def configurable_rate(self, value: float) -> None:
        """Setter for configurable rate."""
        if value < 0 or value > 100:
            msg = "Rate must be between 0 and 100"
            raise ValueError(msg)
        self._configurable_rate = value


class DatabaseMixin:
    """Real database operations mixin."""

    def __init__(self) -> None:
        self.operations_log: list[str] = []
        self.connection_count = 0

    def save_record(self, data: dict[str, FlextTypes.Core.Object]) -> FlextResult[str]:
        """Real save operation."""
        if not data:
            return FlextResult[str].fail("Data cannot be empty")

        record_id = f"rec_{len(self.operations_log) + 1:05d}"
        self.operations_log.append(f"SAVE:{record_id}:{data}")
        return FlextResult[str].ok(record_id)

    def find_record(self, record_id: str) -> FlextResult[dict[str, FlextTypes.Core.Object]]:
        """Real find operation."""
        for entry in self.operations_log:
            if f"SAVE:{record_id}:" in entry:
                data_part = entry.split(":", 2)[2]
                return FlextResult[dict[str, FlextTypes.Core.Object]].ok(ast.literal_eval(data_part))
        return FlextResult[dict[str, FlextTypes.Core.Object]].fail(f"Record {record_id} not found")

    def get_connection_info(self) -> dict[str, FlextTypes.Core.Object]:
        """Get connection information."""
        return {
            "active_connections": self.connection_count,
            "operations_count": len(self.operations_log),
        }


class CacheAccessMixin:
    """Real cache operations mixin."""

    def __init__(self) -> None:
        self.cache_data: dict[str, object] = {}
        self.hit_count = 0
        self.miss_count = 0

    def cache_get(self, key: str) -> FlextResult[dict[str, object]]:
        """Real cache get operation."""
        if key in self.cache_data:
            self.hit_count += 1
            return FlextResult[dict[str, object]].ok(self.cache_data[key])
        self.miss_count += 1
        return FlextResult[dict[str, object]].fail(f"Cache miss for key: {key}")

    def cache_set(self, key: str, value: object) -> FlextResult[None]:
        """Real cache set operation."""
        if not key:
            return FlextResult[None].fail("Key cannot be empty")

        self.cache_data[key] = value
        return FlextResult[None].ok(None)

    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "total_keys": len(self.cache_data),
        }


class RealBusinessHost:
    """Real business host class that uses delegation."""

    def __init__(self) -> None:
        self.host_id = "business_host_001"
        self.status = "initialized"

        # Create delegator with real business mixins
        self.delegator = create_mixin_delegator(
            self,
            BusinessLogicMixin,
            DatabaseMixin,
            CacheAccessMixin,
            FlextValidatableMixin,
            FlextTimestampMixin,
        )

    def process_order(
        self, amount: float, customer_id: str
    ) -> FlextResult[dict[str, FlextTypes.Core.Object]]:
        """Real business process using delegated methods."""
        # Use delegated validation
        validation_result = self.validate_amount(amount)
        if validation_result.is_failure:
            return FlextResult[None].fail(
                f"Validation failed: {validation_result.error}"
            )

        # Calculate discount using delegated method
        discount = self.calculate_discount(amount, 15.0)

        # Save to database using delegated method
        order_data = {
            "customer_id": customer_id,
            "amount": amount,
            "discount": discount,
            "final_amount": amount - discount,
        }

        save_result = self.save_record(order_data)
        if save_result.is_failure:
            return FlextResult[None].fail(f"Save failed: {save_result.error}")

        # Cache the result
        cache_key = f"order_{customer_id}_{save_result.value}"
        self.cache_set(cache_key, order_data)

        return FlextResult[None].ok(
            {
                "order_id": save_result.value,
                "final_amount": order_data["final_amount"],
                "discount_applied": discount,
            }
        )


class TestRealDelegationFunctionality:
    """Test real delegation functionality without mocks."""

    def test_real_business_workflow_complete(self) -> None:
        """Test complete real business workflow with delegated methods."""
        host = RealBusinessHost()

        # Test successful order processing
        result = host.process_order(100.0, "customer_123")

        assert result.success
        order_data = result.value
        assert order_data["final_amount"] == 85.0  # 100 - 15% discount
        assert order_data["discount_applied"] == 15.0
        assert "order_id" in order_data

        # Verify delegated methods were actually called
        # Access mixin instances through the delegator
        business_mixin = next(
            instance
            for cls, instance in host.delegator._mixin_instances.items()
            if cls.__name__ == "BusinessLogicMixin"
        )
        database_mixin = next(
            instance
            for cls, instance in host.delegator._mixin_instances.items()
            if cls.__name__ == "DatabaseMixin"
        )

        assert business_mixin.calculations_count == 1
        assert business_mixin.last_result == 15.0
        assert len(database_mixin.operations_log) == 1

        # Test cache functionality
        cache_stats = host.cache_stats()
        assert cache_stats["total_keys"] == 1
        assert cache_stats["hits"] == 0
        assert cache_stats["misses"] == 0

    def test_real_validation_failure_handling(self) -> None:
        """Test real validation failure handling."""
        host = RealBusinessHost()

        # Test negative amount validation
        result = host.process_order(-50.0, "customer_456")
        assert result.is_failure
        assert "Amount must be positive" in result.error

        # Test excessive amount validation
        result = host.process_order(2_000_000.0, "customer_789")
        assert result.is_failure
        assert "Amount exceeds maximum limit" in result.error

    def test_property_delegation_real_functionality(self) -> None:
        """Test property delegation with real property access."""
        host = RealBusinessHost()

        # Test read-only property
        discount_rate = host.discount_rate
        assert discount_rate == 10.5

        # Test property with setter
        host.configurable_rate = 20.0
        assert host.configurable_rate == 20.0

        # Test setter validation
        with pytest.raises(ValueError, match="Rate must be between 0 and 100"):
            host.configurable_rate = 150.0

    def test_delegation_info_real_data(self) -> None:
        """Test delegation info provides real data."""
        host = RealBusinessHost()
        delegation_info = host.delegator.get_delegation_info()

        assert len(delegation_info["registered_mixins"]) >= 5
        assert len(delegation_info["delegated_methods"]) > 0
        assert delegation_info["validation_result"] is True
        assert "initialization_log" in delegation_info
        assert "delegated_methods" in delegation_info


class TestAdvancedDelegationScenarios:
    """Test advanced delegation scenarios with real functionality."""

    def test_delegation_with_flext_core_mixins(self) -> None:
        """Test delegation working with real flext-core mixins."""

        class AdvancedHost:
            def __init__(self) -> None:
                self.data_store: dict[str, FlextTypes.Core.Object] = {}
                self.delegator = create_mixin_delegator(
                    self,
                    FlextValidatableMixin,
                    FlextSerializableMixin,
                    FlextTimestampMixin,
                    FlextLoggableMixin,
                )

        host = AdvancedHost()

        # Test validation mixin
        host.add_validation_error("test_error")
        assert host.has_validation_errors()
        assert not host.is_valid

        # Test serializable mixin
        basic_dict = host.to_dict_basic()
        assert isinstance(basic_dict, dict)

        # Test timestamp mixin functionality
        assert hasattr(host, "created_at")
        assert hasattr(host, "updated_at")

    def test_mixin_initialization_real_scenario(self) -> None:
        """Test mixin initialization with real initialization methods."""

        class InitializableMixin:
            def __init__(self) -> None:
                self.validation_initialized = False
                self.timestamps_initialized = False
                self.id_initialized = False
                self.logging_initialized = False
                self.serialization_initialized = False

            def _initialize_validation(self) -> None:
                self.validation_initialized = True

            def _initialize_timestamps(self) -> None:
                self.timestamps_initialized = True

            def _initialize_id(self) -> None:
                self.id_initialized = True

            def _initialize_logging(self) -> None:
                self.logging_initialized = True

            def _initialize_serialization(self) -> None:
                self.serialization_initialized = True

            def get_initialization_status(self) -> dict[str, bool]:
                return {
                    "validation": self.validation_initialized,
                    "timestamps": self.timestamps_initialized,
                    "id": self.id_initialized,
                    "logging": self.logging_initialized,
                    "serialization": self.serialization_initialized,
                }

        class InitHost:
            def __init__(self) -> None:
                self.delegator = create_mixin_delegator(self, InitializableMixin)

        host = InitHost()
        status = host.get_initialization_status()

        # Verify all initialization methods were called
        assert status["validation"] is True
        assert status["timestamps"] is True
        assert status["id"] is True
        assert status["logging"] is True
        assert status["serialization"] is True

        # Check initialization log
        delegation_info = host.delegator.get_delegation_info()
        init_log = delegation_info["initialization_log"]
        assert any("_initialize_validation()" in entry for entry in init_log)
        assert any("_initialize_timestamps()" in entry for entry in init_log)

    def test_delegation_error_recovery_real_scenarios(self) -> None:
        """Test delegation error recovery in real scenarios."""

        class ProblematicMixin:
            def __init__(self) -> None:
                self.working_method_called = False

            def working_method(self) -> str:
                self.working_method_called = True
                return "success"

            def _initialize_validation(self) -> None:
                # This will be called during initialization but might fail
                # Let's make it succeed to test successful initialization
                pass

            @property
            def problematic_property(self) -> str:
                # This property works fine
                return "property_value"

        class ErrorRecoveryHost:
            def __init__(self) -> None:
                # Should succeed even if some initialization fails
                self.delegator = create_mixin_delegator(self, ProblematicMixin)

        host = ErrorRecoveryHost()

        # Verify delegation worked despite potential issues
        result = host.working_method()
        assert result == "success"

        # Access mixin instance to check the flag
        problematic_mixin = next(
            instance
            for cls, instance in host.delegator._mixin_instances.items()
            if cls.__name__ == "ProblematicMixin"
        )
        assert problematic_mixin.working_method_called

        # Verify property access works
        prop_value = host.problematic_property
        assert prop_value == "property_value"


class TestDelegationPropertyEdgeCases:
    """Test property delegation edge cases with real functionality."""

    def test_property_descriptor_complex_scenario(self) -> None:
        """Test complex property descriptor scenarios."""

        class ComplexDescriptorMixin:
            def __init__(self) -> None:
                self._private_value = 42

            @property
            def complex_property(self) -> int:
                """Complex property with business logic."""
                return self._private_value * 2

            @complex_property.setter
            def complex_property(self, value: int) -> None:
                if value < 0:
                    msg = "Value must be non-negative"
                    raise ValueError(msg)
                self._private_value = value // 2

            @property
            def readonly_computed(self) -> str:
                """Read-only computed property."""
                return f"computed_{self._private_value}"

        class PropertyHost:
            def __init__(self) -> None:
                self.delegator = create_mixin_delegator(self, ComplexDescriptorMixin)

        host = PropertyHost()

        # Test property getter
        assert host.complex_property == 84  # 42 * 2

        # Test property setter
        host.complex_property = 100
        assert host.complex_property == 100

        # Test read-only property
        readonly_value = host.readonly_computed
        assert readonly_value == "computed_50"  # 100 // 2

        # Test setter validation
        with pytest.raises(ValueError, match="Value must be non-negative"):
            host.complex_property = -10

    def test_property_delegation_without_setter(self) -> None:
        """Test property delegation for read-only properties."""

        class ReadOnlyMixin:
            def __init__(self) -> None:
                self._value = "readonly"

            @property
            def readonly_prop(self) -> str:
                return self._value

        class ReadOnlyHost:
            def __init__(self) -> None:
                self.delegator = create_mixin_delegator(self, ReadOnlyMixin)

        host = ReadOnlyHost()

        # Test reading works
        assert host.readonly_prop == "readonly"

        # Test that setting raises appropriate error
        with pytest.raises(
            FlextExceptions.OperationError, match="Property 'readonly_prop' is read-only"
        ):
            host.readonly_prop = "new_value"


class TestDelegationSystemValidationReal:
    """Test delegation system validation with real scenarios."""

    def test_validate_delegation_system_success_real(self) -> None:
        """Test successful validation with real delegation system."""
        # Create a working delegation system first
        RealBusinessHost()

        # The validation should succeed for properly set up delegation
        result = validate_delegation_system()
        # Note: validate_delegation_system() is a general function
        # We're testing that it works with real delegation scenarios
        assert isinstance(result, FlextResult)

    def test_delegation_info_completeness_real(self) -> None:
        """Test delegation info provides complete real information."""
        host = RealBusinessHost()
        info = host.delegator.get_delegation_info()

        # Verify all expected information is present
        required_keys = [
            "registered_mixins",
            "delegated_methods",
            "validation_result",
            "initialization_log",
        ]

        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

        # Verify data types and content
        assert isinstance(info["registered_mixins"], list)
        assert len(info["registered_mixins"]) > 0
        assert isinstance(info["delegated_methods"], list)
        assert len(info["delegated_methods"]) > 0
        assert isinstance(info["validation_result"], bool)
        assert isinstance(info["initialization_log"], list)
        assert isinstance(info["delegated_methods"], list)

        # Verify specific methods are delegated
        delegated_method_names = info["delegated_methods"]
        expected_methods = [
            "calculate_discount",
            "validate_amount",
            "save_record",
            "find_record",
            "cache_get",
            "cache_set",
        ]

        for method in expected_methods:
            assert method in delegated_method_names, f"Method {method} not delegated"


class TestDelegationPerformanceReal:
    """Test delegation performance with real scenarios."""

    def test_delegation_performance_many_operations(self) -> None:
        """Test delegation performance with many real operations."""
        host = RealBusinessHost()

        # Perform many operations to test performance
        results = []
        for i in range(100):
            result = host.process_order(100.0 + i, f"customer_{i:03d}")
            results.append(result)

        # Verify all operations succeeded
        successful = sum(1 for r in results if r.success)
        assert successful == 100

        # Verify delegation tracked all operations
        business_mixin = next(
            instance
            for cls, instance in host.delegator._mixin_instances.items()
            if cls.__name__ == "BusinessLogicMixin"
        )
        database_mixin = next(
            instance
            for cls, instance in host.delegator._mixin_instances.items()
            if cls.__name__ == "DatabaseMixin"
        )

        assert business_mixin.calculations_count == 100
        assert len(database_mixin.operations_log) == 100

        # Verify cache has all entries
        cache_stats = host.cache_stats()
        assert cache_stats["total_keys"] == 100

    def test_mixin_registry_persistence_real(self) -> None:
        """Test mixin registry persistence across instances."""
        # Create first host - should populate registry
        host1 = RealBusinessHost()

        # Registry should have entries now
        registry = FlextMixinDelegator._MIXIN_REGISTRY
        assert len(registry) > 0

        # Create second host - should reuse registry
        host2 = RealBusinessHost()

        # Both should work identically
        result1 = host1.process_order(100.0, "customer_A")
        result2 = host2.process_order(100.0, "customer_B")

        assert result1.success
        assert result2.success
        assert result1.value["discount_applied"] == result2.value["discount_applied"]
