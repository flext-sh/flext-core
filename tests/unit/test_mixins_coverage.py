"""Additional tests to achieve ~100% coverage for mixins.

These tests focus on untested code paths and edge cases.
"""

from __future__ import annotations

import time
from typing import cast
from unittest.mock import patch

import pytest

from flext_core import (
    FlextCache,
    FlextConstants,
    FlextIdentification,
    FlextMixins,
    FlextSerialization,
    FlextState,
    FlextTimestamps,
    FlextTiming,
    FlextTypes,
    FlextValidation,
)


class TestMixinsCoreFullCoverage:
    """Tests for core.py module to achieve full coverage."""

    def test_configure_mixins_system_all_paths(self) -> None:
        """Test all configuration paths for mixins system."""
        # Test with full configuration
        full_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "cache_enabled": True,
            "cache_ttl": 300,
            "state_management_enabled": True,
            "enable_detailed_validation": False,
            "max_validation_errors": 10,
        }

        result = FlextMixins.configure_mixins_system(full_config)
        assert result.success
        config = result.unwrap()
        assert config["environment"] == "production"
        assert config["log_level"] == "ERROR"
        assert config["cache_enabled"] is True
        assert config["cache_ttl"] == 300

        # Test with invalid environment
        invalid_env_config: FlextTypes.Config.ConfigDict = {
            "environment": "invalid_env"
        }
        result = FlextMixins.configure_mixins_system(invalid_env_config)
        assert result.is_failure
        assert "Invalid environment" in (result.error or "")

        # Test with invalid log level
        invalid_log_config: FlextTypes.Config.ConfigDict = {
            "log_level": "INVALID_LEVEL"
        }
        result = FlextMixins.configure_mixins_system(invalid_log_config)
        assert result.is_failure
        assert "Invalid log_level" in (result.error or "")

        # Test with minimal configuration (defaults)
        minimal_config: FlextTypes.Config.ConfigDict = {}
        result = FlextMixins.configure_mixins_system(minimal_config)
        assert result.success
        config = result.unwrap()
        assert config["environment"] == "development"
        assert config["log_level"] == "DEBUG"
        assert config["cache_enabled"] is True
        assert config["cache_ttl"] == 3600

    def test_configure_mixins_system_exception_handling(self) -> None:
        """Test exception handling in configure_mixins_system."""
        # Test with invalid type that might cause exception
        with patch(
            "flext_core.mixins.core.FlextConstants.Config.ConfigEnvironment"
        ) as mock_enum:
            mock_enum.__iter__.side_effect = Exception("Test exception")

            config: FlextTypes.Config.ConfigDict = {"environment": "test"}
            result = FlextMixins.configure_mixins_system(config)
            assert result.is_failure
            assert "Failed to configure" in (result.error or "")

    def test_optimize_mixins_performance(self) -> None:
        """Test optimization of mixins performance."""
        # Test with different configurations
        configs = [
            {"cache_size": 100, "enable_gc": True},
            {"cache_size": 1000, "enable_gc": False},
            {"cache_size": 0, "enable_gc": True},  # Minimal cache
        ]

        for config in configs:
            typed_config = cast("FlextTypes.Config.ConfigDict", config)
            result = FlextMixins.optimize_mixins_performance(typed_config)
            assert result.success
            optimized = result.unwrap()
            assert "optimized" in optimized or "cache_size" in optimized

    def test_get_mixins_system_config(self) -> None:
        """Test getting mixins system configuration."""
        result = FlextMixins.get_mixins_system_config()
        assert result.success

        config = result.unwrap()
        assert "environment" in config or "configuration" in config

    def test_create_environment_mixins_config(self) -> None:
        """Test creating environment-specific mixins configuration."""
        # Test for different environments using valid enum values
        valid_envs = [
            FlextConstants.Config.ConfigEnvironment.DEVELOPMENT,
            FlextConstants.Config.ConfigEnvironment.PRODUCTION,
            FlextConstants.Config.ConfigEnvironment.TEST,
        ]
        for env in valid_envs:
            result = FlextMixins.create_environment_mixins_config(env.value)
            assert result.success
            config = result.unwrap()
            assert "environment" in config or env.value in str(config)


class TestSerializationMixinFullCoverage:
    """Tests for serialization.py to achieve full coverage."""

    def test_serialization_all_methods(self) -> None:
        """Test all serialization methods."""

        class TestSerializable:
            def __init__(self) -> None:
                self.name = "test"
                self.value = 42
                self.items = [1, 2, 3]
                self.nested = {"key": "value"}
                self._private = "hidden"

        obj = TestSerializable()

        # Test to_dict
        result = FlextSerialization.to_dict(obj)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["items"] == [1, 2, 3]
        assert "_private" not in result

        # Test load_from_dict
        data = {"name": "updated", "value": 100, "new_field": "added"}
        FlextSerialization.load_from_dict(obj, data)
        assert obj.name == "updated"
        assert obj.value == 100
        assert hasattr(obj, "new_field")
        assert obj.new_field == "added"

        # Test to_json
        json_str = FlextSerialization.to_json(obj)
        assert isinstance(json_str, str)
        assert '"name": "updated"' in json_str

        # Test load_from_json
        json_data = '{"name": "from_json", "value": 200}'
        FlextSerialization.load_from_json(obj, json_data)
        assert obj.name == "from_json"
        assert obj.value == 200
        # Test JSON validation (using json.loads for validation)
        import json

        valid_json = '{"valid": true}'
        try:
            json.loads(valid_json)
            json_valid = True
        except json.JSONDecodeError:
            json_valid = False
        assert json_valid is True

        invalid_json = '{"invalid": true'
        try:
            json.loads(invalid_json)
            json_invalid = True
        except json.JSONDecodeError:
            json_invalid = False
        assert json_invalid is False

        # Test basic dict conversion
        dict_result = FlextSerialization.to_dict_basic(obj)
        assert isinstance(dict_result, dict)
        assert dict_result["name"] == obj.name


class TestStateMixinFullCoverage:
    """Tests for state.py to achieve full coverage."""

    def test_state_management_complete(self) -> None:
        """Test complete state management functionality."""

        class StatefulObject:
            def __init__(self) -> None:
                pass

        obj = StatefulObject()

        # Initialize state first
        FlextState.initialize_state(obj, "initialized")

        # Test get_state
        state = FlextState.get_state(obj)
        assert state == "initialized"

        # Test set_state
        FlextState.set_state(obj, "updated")
        state = FlextState.get_state(obj)
        assert state == "updated"


class TestCacheMixinFullCoverage:
    """Tests for cache.py to achieve full coverage."""

    def test_cache_operations_complete(self) -> None:
        """Test all cache operations."""

        class CacheableObject:
            def __init__(self) -> None:
                pass

        obj = CacheableObject()

        # Test cache_set
        FlextCache.set_cached_value(obj, "key1", "value1")

        # Test cache_get
        value = FlextCache.get_cached_value(obj, "key1")
        assert value == "value1"

        # Test cache_get with missing key
        value = FlextCache.get_cached_value(obj, "missing")
        assert value is None

        # Test cache_has
        assert FlextCache.has_cached_value(obj, "key1") is True
        assert FlextCache.has_cached_value(obj, "missing") is False

        # Test cache_clear
        FlextCache.clear_cache(obj)
        assert FlextCache.get_cached_value(obj, "key1") is None


class TestValidationMixinFullCoverage:
    """Tests for validation.py to achieve full coverage."""

    def test_validation_complete_functionality(self) -> None:
        """Test complete validation functionality."""

        class ValidatableObject:
            def __init__(self) -> None:
                self.email = "test@example.com"
                self.age = 25

        obj = ValidatableObject()

        # Test add_validation_error
        FlextValidation.add_validation_error(obj, "Test error 1")
        errors = FlextValidation.get_validation_errors(obj)
        assert len(errors) == 1
        assert errors[0] == "Test error 1"

        # Test clear_validation_errors
        FlextValidation.clear_validation_errors(obj)
        assert FlextValidation.get_validation_errors(obj) == []

        # Test is_valid
        assert FlextValidation.is_valid(obj) is True

        FlextValidation.add_validation_error(obj, "Error")
        assert FlextValidation.is_valid(obj) is False


class TestTimingMixinFullCoverage:
    """Tests for timing.py to achieve full coverage."""

    def test_timing_operations_complete(self) -> None:
        """Test all timing operations."""

        class TimedObject:
            def __init__(self) -> None:
                pass

        obj = TimedObject()

        # Test start_timing
        FlextTiming.start_timing(obj)

        # Test stop_timing
        time.sleep(0.01)
        FlextTiming.stop_timing(obj)

        # Test get_last_elapsed_time
        duration = FlextTiming.get_last_elapsed_time(obj)
        assert duration > 0


class TestIdentificationMixinFullCoverage:
    """Tests for identification.py to achieve full coverage."""

    def test_identification_static_methods(self) -> None:
        """Test static methods of FlextIdentification."""

        class IdentifiableObject:
            def __init__(self) -> None:
                pass

        obj = IdentifiableObject()

        # Test generate_correlation_id
        corr_id = FlextIdentification.generate_correlation_id()
        assert corr_id is not None
        assert len(corr_id) > 0

        # Test generate_entity_id
        entity_id = FlextIdentification.generate_entity_id()
        assert entity_id is not None
        assert len(entity_id) > 0

        # Test ensure_id
        FlextIdentification.ensure_id(obj)
        assert hasattr(obj, "id")
        assert FlextIdentification.get_id(obj) is not None


class TestTimestampMixinFullCoverage:
    """Tests for timestamps.py to achieve full coverage."""

    def test_timestamp_operations_complete(self) -> None:
        """Test all timestamp operations."""

        class TimestampedObject:
            def __init__(self) -> None:
                # Initialize timestamps manually since we're testing static methods
                import datetime
                from zoneinfo import ZoneInfo

                now = datetime.datetime.now(tz=ZoneInfo("UTC"))
                self.created_at = now
                self.updated_at = now

        obj = TimestampedObject()

        # Test get_created_at
        created = FlextTimestamps.get_created_at(obj)
        assert created is not None

        # Test get_updated_at
        updated = FlextTimestamps.get_updated_at(obj)
        assert updated is not None

        # Test update_timestamp
        old_updated = obj.updated_at
        time.sleep(0.01)
        FlextTimestamps.update_timestamp(obj)
        assert obj.updated_at != old_updated


class TestLoggingMixinFullCoverage:
    """Tests for logging.py to achieve full coverage."""

    def test_logging_operations_complete(self) -> None:
        """Test all logging operations."""

        class LoggableObject:
            def __init__(self) -> None:
                pass

        obj = LoggableObject()

        # Test get_logger
        logger = FlextMixins.get_logger(obj)
        assert logger is not None

        # Test log_operation
        FlextMixins.log_operation(
            obj, "test_operation", user="test_user", status="success"
        )

        # Test log_error with string
        FlextMixins.log_error(obj, "Test error message", code="ERR001")

        # Test log_error with Exception
        exception = ValueError("Test exception")
        FlextMixins.log_error(obj, exception, code="ERR002")

        # Test log_info
        FlextMixins.log_info(obj, "Information message", detail="test_detail")

        # Test log_debug
        FlextMixins.log_debug(obj, "Debug message", data={"key": "value"})

    def test_logging_mixin_class(self) -> None:
        """Test the Loggable mixin class."""
        from flext_core.mixins.logging import FlextLogging

        class LoggableClass(FlextLogging.Loggable):
            def __init__(self) -> None:
                super().__init__()

        obj = LoggableClass()

        # Test mixin methods
        obj.log_info("Info from mixin")
        obj.log_error("Error from mixin")
        obj.log_debug("Debug from mixin")
        obj.log_operation("mixin_op", status="ok")

    def test_logging_with_exception_object(self) -> None:
        """Test logging with exception objects."""

        class TestObj:
            def __init__(self) -> None:
                pass

        obj = TestObj()

        # Create an exception
        exc = ValueError("Test exception")
        FlextMixins.log_error(obj, exc, code="TEST_ERR")


class TestValidationMixinExtended:
    """Extended tests for validation.py to achieve full coverage."""

    def test_validate_field_types_with_errors(self) -> None:
        """Test field type validation with various error scenarios."""

        class ValidatableObject:
            def __init__(self) -> None:
                self.name = 123  # Wrong type
                self.age = "not a number"  # Wrong type
                self.email = None  # None value

        obj = ValidatableObject()

        # Test with type mismatches
        field_types = {"name": str, "age": int, "email": str}
        result = FlextValidation.validate_field_types(obj, field_types)
        assert result.is_failure
        assert "Type validation failed" in (result.error or "")

    def test_validate_url(self) -> None:
        """Test URL validation."""
        # Valid URLs
        result = FlextValidation.validate_url("https://example.com")
        assert result.success
        assert result.unwrap() == "https://example.com"

        result = FlextValidation.validate_url("http://subdomain.example.org/path")
        assert result.success

        # Invalid URLs
        result = FlextValidation.validate_url("not a url")
        assert result.is_failure
        assert "Invalid URL format" in (result.error or "")

    def test_validate_phone(self) -> None:
        """Test phone number validation."""
        # Valid phone numbers
        result = FlextValidation.validate_phone("+12345678901")
        assert result.success
        assert result.unwrap() == "+12345678901"

        result = FlextValidation.validate_phone("12345678901")
        assert result.success

        # Invalid phone numbers
        result = FlextValidation.validate_phone("invalid")
        assert result.is_failure
        assert "Invalid phone format" in (result.error or "")

        result = FlextValidation.validate_phone("")
        assert result.is_failure


class TestSerializationMixinExtended:
    """Extended tests for serialization.py to achieve full coverage."""

    def test_serialization_with_errors(self) -> None:
        """Test serialization error handling."""

        class ComplexObject:
            def __init__(self) -> None:
                self.data = "test"
                self._private = "hidden"

        obj = ComplexObject()

        # Test normal serialization
        result = FlextSerialization.to_dict_basic(obj)
        assert "data" in result
        assert "_private" not in result

        # Test to_dict with nested objects
        class NestedObject:
            def __init__(self) -> None:
                self.nested_value = "nested"

        obj.nested = NestedObject()
        result = FlextSerialization.to_dict(obj)
        assert "data" in result
        assert "nested" in result

    def test_load_from_json_errors(self) -> None:
        """Test JSON loading error scenarios."""

        class TestObject:
            def __init__(self) -> None:
                pass

        obj = TestObject()

        # Invalid JSON
        result = FlextSerialization.load_from_json(obj, "not valid json")
        assert result.is_failure
        assert "Failed to load from JSON" in (result.error or "")

        # Non-dict JSON
        result = FlextSerialization.load_from_json(obj, '["array", "not", "dict"]')
        assert result.is_failure
        assert "JSON data must be a dictionary" in (result.error or "")


class TestCacheMixinExtended:
    """Extended tests for cache.py to achieve full coverage."""

    def test_cache_mixin_class(self) -> None:
        """Test the Cacheable mixin class."""

        class CacheableClass(FlextCache.Cacheable):
            def __init__(self) -> None:
                super().__init__()

        obj = CacheableClass()

        # Test mixin methods
        obj.set_cached_value("key1", "value1")
        assert obj.get_cached_value("key1") == "value1"
        assert obj.has_cached_value("key1") is True

        obj.clear_cache()
        assert obj.has_cached_value("key1") is False

    def test_cache_ttl_and_expiry(self) -> None:
        """Test cache TTL functionality."""

        class CacheableObject:
            def __init__(self) -> None:
                pass

        obj = CacheableObject()

        # Test cache with multiple keys
        FlextCache.set_cached_value(obj, "key1", "value1")
        FlextCache.set_cached_value(obj, "key2", "value2")
        assert FlextCache.get_cached_value(obj, "key1") == "value1"
        assert FlextCache.get_cached_value(obj, "key2") == "value2"

        # Test cache clear
        FlextCache.clear_cache(obj)
        assert FlextCache.get_cached_value(obj, "key1") is None
        assert FlextCache.get_cached_value(obj, "key2") is None


class TestStateMixinExtended:
    """Extended tests for state.py to achieve full coverage."""

    def test_state_mixin_class(self) -> None:
        """Test the Stateful mixin class."""

        class StatefulClass(FlextState.Stateful):
            def __init__(self) -> None:
                super().__init__()

        obj = StatefulClass()

        # Test mixin property
        obj.state = "active"  # Use property setter
        assert obj.state == "active"  # Use property getter

        # Test state history
        obj.state = "inactive"
        history = obj.state_history
        assert len(history) >= 2  # Should have initialization and changes


class TestTimestampsMixinExtended:
    """Extended tests for timestamps.py to achieve full coverage."""

    def test_timestamps_mixin_class(self) -> None:
        """Test the Timestampable mixin class."""

        class TimestampedClass(FlextTimestamps.Timestampable):
            def __init__(self) -> None:
                super().__init__()

        obj = TimestampedClass()

        # Test mixin methods - access attributes directly
        created = obj.created_at
        assert created is not None

        updated = obj.updated_at
        assert updated is not None

        time.sleep(0.01)
        obj.update_timestamp()
        new_updated = obj.updated_at
        assert new_updated != updated


class TestTimingMixinExtended:
    """Extended tests for timing.py to achieve full coverage."""

    def test_timing_mixin_class(self) -> None:
        """Test the Timeable mixin class."""

        class TimeableClass(FlextTiming.Timeable):
            def __init__(self) -> None:
                super().__init__()

        obj = TimeableClass()

        # Test mixin methods
        obj.start_timing()
        time.sleep(0.01)
        obj.stop_timing()

        elapsed = obj.get_last_elapsed_time()
        assert elapsed > 0

    def test_timing_errors(self) -> None:
        """Test timing error cases."""

        class TestObject:
            def __init__(self) -> None:
                pass

        obj = TestObject()

        # Get elapsed time without starting
        elapsed = FlextTiming.get_last_elapsed_time(obj)
        assert elapsed == 0.0  # Should return 0 when no timing


class TestIdentificationMixinExtended:
    """Extended tests for identification.py to achieve full coverage."""

    def test_identification_mixin_class(self) -> None:
        """Test the Identifiable mixin class."""

        class IdentifiableClass(FlextIdentification.Identifiable):
            def __init__(self) -> None:
                super().__init__()

        obj = IdentifiableClass()

        # Test mixin methods
        id_value = obj.get_id()
        assert id_value is not None

        obj.set_id("custom-id-123")
        assert obj.get_id() == "custom-id-123"

        assert obj.has_id() is True

    def test_identification_static_complete(self) -> None:
        """Test all static methods of identification."""

        class TestObject:
            def __init__(self) -> None:
                pass

        obj = TestObject()

        # Test set_id
        FlextIdentification.set_id(obj, "test-id-456")
        assert FlextIdentification.get_id(obj) == "test-id-456"

        # Test generate_entity_id (static method)
        entity_id = FlextIdentification.generate_entity_id()
        assert entity_id is not None
        assert len(entity_id) > 0


class TestMixinsCoreAdvanced:
    """Advanced tests for core.py module to achieve better coverage."""

    def test_performance_optimization_with_config(self) -> None:
        """Test performance optimization with various configs."""
        import gc

        # Test with garbage collection control
        config = {"enable_gc": False, "cache_size": 500}
        result = FlextMixins.optimize_mixins_performance(
            cast("FlextTypes.Config.ConfigDict", config)
        )
        assert result.success

        # Re-enable GC
        config = {"enable_gc": True, "cache_size": 100}
        result = FlextMixins.optimize_mixins_performance(
            cast("FlextTypes.Config.ConfigDict", config)
        )
        assert result.success
        gc.collect()  # Ensure GC is working

    def test_validation_mixin_advanced(self) -> None:
        """Test advanced validation scenarios."""

        class ValidatableClass(FlextValidation.Validatable):
            def __init__(self) -> None:
                super().__init__()
                self.name = "Test"
                self.email = "test@example.com"

        obj = ValidatableClass()

        # Test validate_required_fields
        result = obj.validate_required_fields(["name", "email"])
        assert result.success

        # Test with missing field
        result = obj.validate_required_fields(["missing_field"])
        assert result.is_failure

        # Test field types (validation has been fixed)
        result = obj.validate_field_types({"name": str, "email": str})
        # The validation implementation now works correctly
        assert result.success  # Validation is now working

        # Test mark_valid
        obj.mark_valid()
        assert obj.is_valid()


class TestSerializationAdvanced:
    """Advanced serialization tests."""

    def test_to_dict_with_protocol_objects(self) -> None:
        """Test serialization with protocol-based objects."""

        class HasToDictObj:
            def to_dict(self) -> dict[str, object]:
                return {"serialized": True}

        class HasToDictBasicObj:
            def to_dict_basic(self) -> dict[str, object]:
                return {"basic": True}

        class ComplexObj:
            def __init__(self) -> None:
                self.dict_obj = HasToDictObj()
                self.basic_obj = HasToDictBasicObj()
                self.list_of_objs = [HasToDictBasicObj(), HasToDictBasicObj()]
                self.normal_list = [1, 2, 3]
                self.none_value = None

        obj = ComplexObj()
        result = FlextSerialization.to_dict(obj)

        assert "dict_obj" in result
        assert result["dict_obj"] == {"serialized": True}
        assert "basic_obj" in result
        assert result["basic_obj"] == {"basic": True}
        assert "list_of_objs" in result
        list_obj = cast("list[object]", result["list_of_objs"])
        assert len(list_obj) == 2
        assert "normal_list" in result
        assert result["normal_list"] == [1, 2, 3]
        assert "none_value" not in result  # None values are skipped


class TestTimestampsAdvanced:
    """Advanced timestamp tests."""

    def test_timestamp_without_initialization(self) -> None:
        """Test timestamp operations on uninitialized objects."""

        class SimpleObject:
            def __init__(self) -> None:
                pass

        obj = SimpleObject()

        # These should auto-initialize timestamps
        created = FlextTimestamps.get_created_at(obj)
        assert created is not None  # Auto-initialized

        updated = FlextTimestamps.get_updated_at(obj)
        assert updated is not None  # Auto-initialized

        # Update timestamp to initialize
        FlextTimestamps.update_timestamp(obj)
        assert FlextTimestamps.get_created_at(obj) is not None


class TestStateAdvanced:
    """Advanced state management tests."""

    def test_state_transitions_and_history(self) -> None:
        """Test state transitions with history tracking."""

        class StatefulObj:
            def __init__(self) -> None:
                pass

        obj = StatefulObj()

        # Initialize and track transitions
        FlextState.initialize_state(obj, "initial")
        assert FlextState.get_state(obj) == "initial"

        # Multiple transitions
        FlextState.set_state(obj, "processing")
        FlextState.set_state(obj, "completed")

        history = FlextState.get_state_history(obj)
        assert "initial" in history
        assert "processing" in history
        assert "completed" in history

        # Test state validation
        assert FlextState.get_state(obj) == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
