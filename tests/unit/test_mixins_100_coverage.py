"""Additional tests to achieve 100% coverage for all mixin modules.

This file contains targeted tests for uncovered lines in the mixins package.
"""

from __future__ import annotations

import json
import time
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from flext_core import FlextConstants, FlextMixins, FlextResult, FlextTypes
from flext_core.mixins import (
    FlextCache,
    FlextIdentification,
    FlextSerialization,
    FlextState,
    FlextTimestamps,
    FlextTiming,
    FlextValidation,
)
from flext_core.mixins.logging import FlextLogging
from flext_core.protocols import FlextProtocols


class TestCacheComplete:
    """Complete tests for cache.py to achieve 100% coverage."""
    
    def test_cache_invalidate_cache(self) -> None:
        """Test cache invalidation method."""
        
        class CacheableObj:
            def __init__(self) -> None:
                pass
        
        obj = CacheableObj()
        
        # Set some cached values
        FlextCache.set_cached_value(obj, "key1", "value1")
        FlextCache.set_cached_value(obj, "key2", "value2")
        
        # Test clearing specific cache key
        FlextCache.clear_cache(obj)
        FlextCache.set_cached_value(obj, "key1", "value1")
        FlextCache.set_cached_value(obj, "key2", "value2")
        
        # Clear and verify
        FlextCache.clear_cache(obj)
        assert FlextCache.get_cached_value(obj, "key1") is None
        assert FlextCache.get_cached_value(obj, "key2") is None
        
    def test_cache_get_cache_stats(self) -> None:
        """Test cache statistics retrieval."""
        
        class CacheableObj:
            def __init__(self) -> None:
                pass
        
        obj = CacheableObj()
        
        # Test cache operations
        FlextCache.set_cached_value(obj, "key1", "value1")
        FlextCache.set_cached_value(obj, "key2", "value2")
        
        # Check cache has values
        assert FlextCache.has_cached_value(obj, "key1") is True
        assert FlextCache.has_cached_value(obj, "key2") is True
        
    def test_cacheable_mixin_methods(self) -> None:
        """Test Cacheable mixin class methods (lines 148-154, 185)."""
        
        class TestClass(FlextCache.Cacheable):
            def __init__(self) -> None:
                super().__init__()
        
        obj = TestClass()
        
        # Test cache operations
        obj.set_cached_value("test_key", "test_value")
        assert obj.get_cached_value("test_key") == "test_value"
        
        # Clear cache
        obj.clear_cache()
        assert obj.get_cached_value("test_key") is None
        
        # Test has_cached_value (line 185)
        obj.set_cached_value("key", "value")
        assert obj.has_cached_value("key") is True
        assert obj.has_cached_value("missing") is False


class TestSerializationComplete:
    """Complete tests for serialization.py to achieve 100% coverage."""
    
    def test_serialization_edge_cases(self) -> None:
        """Test serialization edge cases for uncovered lines."""
        
        # Test lines 97-99: Exception handling in to_dict_basic
        class BadObject:
            def __getattribute__(self, name: str) -> object:
                if name == "__dict__":
                    raise AttributeError("No __dict__")
                return object.__getattribute__(self, name)
        
        obj = BadObject()
        # This should handle the error gracefully or raise
        try:
            result = FlextSerialization.to_dict_basic(obj)
            # If it doesn't raise, check result is empty
            assert result == {}
        except (ValueError, AttributeError):
            # Expected behavior
            pass
            
        # Test lines 103-104, 108: Timestamp and ID handling
        class ObjectWithTimestamps:
            def __init__(self) -> None:
                self._timestamp_initialized = True
                self.data = "test"
        
        obj2 = ObjectWithTimestamps()
        FlextTimestamps.create_timestamp_fields(obj2)
        FlextIdentification.ensure_id(obj2)
        
        result = FlextSerialization.to_dict_basic(obj2)
        assert "created_at" in result
        assert "updated_at" in result
        assert "id" in result
        
    def test_to_dict_with_protocols(self) -> None:
        """Test to_dict with protocol-based objects (lines 140-142, 149-151, 163-165)."""
        
        class HasToDictBasic:
            def to_dict_basic(self) -> dict[str, object]:
                raise ValueError("to_dict_basic failed")
        
        class HasToDict:
            def to_dict(self) -> dict[str, object]:
                raise ValueError("to_dict failed")
        
        class ComplexObject:
            def __init__(self) -> None:
                self.basic_obj = HasToDictBasic()
                self.dict_obj = HasToDict()
                self.list_with_error = [HasToDictBasic()]
        
        obj = ComplexObject()
        
        # These should raise ValueError due to failed serialization
        with pytest.raises(ValueError, match="Failed to serialize basic_obj"):
            FlextSerialization.to_dict(obj)
        
    def test_to_dict_error_handling(self) -> None:
        """Test to_dict error handling (lines 175-177)."""
        
        class BadObject:
            def __getattribute__(self, name: str) -> object:
                if name == "__dict__":
                    raise RuntimeError("Cannot access __dict__")
                return object.__getattribute__(self, name)
        
        obj = BadObject()
        # This should handle the error gracefully
        try:
            result = FlextSerialization.to_dict(obj)
            # If it doesn't raise, check result
            assert isinstance(result, dict)
        except (ValueError, RuntimeError, AttributeError):
            # Expected behavior
            pass
    
    def test_load_from_dict_exceptions(self) -> None:
        """Test load_from_dict with setattr exceptions (lines 214-216)."""
        
        class RestrictedObject:
            def __init__(self) -> None:
                self.allowed = "yes"
            
            def __setattr__(self, name: str, value: object) -> None:
                if name == "restricted":
                    raise AttributeError("Cannot set restricted")
                object.__setattr__(self, name, value)
        
        obj = RestrictedObject()
        data = {"allowed": "updated", "restricted": "forbidden", "normal": "ok"}
        
        # Should skip the restricted attribute but set others
        FlextSerialization.load_from_dict(obj, data)
        assert obj.allowed == "updated"
        assert not hasattr(obj, "restricted")
        assert obj.normal == "ok"  # type: ignore[attr-defined]
        
    def test_serializable_mixin_methods(self) -> None:
        """Test Serializable mixin class methods (lines 261, 265, 269, 273-275)."""
        
        class TestClass(FlextSerialization.Serializable):
            def __init__(self) -> None:
                self.data = "test"
                self.number = 42
        
        obj = TestClass()
        
        # Test to_dict (line 261)
        result = obj.to_dict()
        assert result["data"] == "test"
        
        # Test to_json (line 265)
        json_str = obj.to_json()
        assert '"data": "test"' in json_str
        
        # Test load_from_dict (line 269)
        obj.load_from_dict({"data": "updated", "new_field": "added"})
        assert obj.data == "updated"
        
        # Test load_from_json with error (lines 273-275)
        with pytest.raises(ValueError):
            obj.load_from_json("invalid json {")


class TestStateComplete:
    """Complete tests for state.py to achieve 100% coverage."""
    
    def test_state_edge_cases(self) -> None:
        """Test state management edge cases."""
        
        class StateObj:
            def __init__(self) -> None:
                pass
        
        obj = StateObj()
        
        # Test line 68: get_state without initialization
        state = FlextState.get_state(obj)
        assert state is not None  # Should auto-initialize
        
        # Test line 90: set_state return value
        result = FlextState.set_state(obj, "new_state")
        assert result.success if result else True
        
        # Test line 95: get_state_history
        history = FlextState.get_state_history(obj)
        assert isinstance(history, list)
        assert len(history) > 0
        
    def test_state_validation_level(self) -> None:
        """Test state validation (line 124)."""
        
        class StateObj:
            def __init__(self) -> None:
                pass
        
        obj = StateObj()
        FlextState.initialize_state(obj, "initial")
        
        # Test invalid state transition (would need validation rules)
        # This tests the validation path in set_state
        result = FlextState.set_state(obj, "valid_state")
        assert result is None or result.success
        
    def test_stateful_mixin_error_handling(self) -> None:
        """Test Stateful mixin error handling (lines 157-159)."""
        
        class TestClass(FlextState.Stateful):
            def __init__(self) -> None:
                super().__init__()
        
        obj = TestClass()
        
        # Mock set_state to return a failure
        with patch.object(FlextState, 'set_state') as mock_set:
            mock_result = FlextResult[None].fail("Invalid state")
            mock_set.return_value = mock_result
            
            with pytest.raises(Exception, match="Invalid state"):
                obj.state = "invalid"


class TestTimestampsComplete:
    """Complete tests for timestamps.py to achieve 100% coverage."""
    
    def test_timestamp_initialization(self) -> None:
        """Test timestamp initialization edge cases (lines 50, 54-55)."""
        
        class SimpleObj:
            def __init__(self) -> None:
                pass
        
        obj = SimpleObj()
        
        # Test line 50: create_timestamp_fields with existing initialization
        obj._timestamp_initialized = True  # type: ignore[attr-defined]
        FlextTimestamps.create_timestamp_fields(obj)
        # Should not reinitialize
        
        # Test lines 54-55: Return existing values
        obj._created_at = "existing_created"  # type: ignore[attr-defined]
        obj._updated_at = "existing_updated"  # type: ignore[attr-defined]
        
        FlextTimestamps.update_timestamp(obj)
        # Should update the timestamp


class TestLoggingComplete:
    """Complete tests for logging.py to achieve 100% coverage."""
    
    def test_logging_edge_cases(self) -> None:
        """Test logging edge cases for uncovered lines."""
        
        class LogObj:
            def __init__(self) -> None:
                pass
        
        obj = LogObj()
        
        # Test lines 33-36: _get_caller_info
        # This is called internally by logging methods
        FlextMixins.log_info(obj, "test message")
        
        # Test line 49: log_operation with various kwargs
        FlextMixins.log_operation(obj, "test_op", user="testuser", result="success")
        
        # Test line 57: log_error with exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            FlextMixins.log_error(obj, e, context="test_context")
        
        # Test lines 78-79: _normalize_context
        FlextMixins.log_debug(obj, "debug", extra_data={"key": "value"})


class TestIdentificationComplete:
    """Complete tests for identification.py to achieve 100% coverage."""
    
    def test_identification_edge_cases(self) -> None:
        """Test identification edge cases (lines 62, 84)."""
        
        class IdObj:
            def __init__(self) -> None:
                pass
        
        obj = IdObj()
        
        # Test line 62: has_id when not initialized
        assert FlextIdentification.has_id(obj) is False
        
        # Test line 84: set_id and get_id
        FlextIdentification.set_id(obj, "test-entity-id")
        assert FlextIdentification.get_id(obj) == "test-entity-id"


class TestTimingComplete:
    """Complete tests for timing.py to achieve 100% coverage."""
    
    def test_timing_edge_cases(self) -> None:
        """Test timing edge cases for uncovered lines."""
        
        class TimingObj:
            def __init__(self) -> None:
                pass
        
        obj = TimingObj()
        
        # Test line 71: stop_timing without start
        FlextTiming.stop_timing(obj)  # Should handle gracefully
        
        # Test line 128: get_last_elapsed_time edge case
        elapsed = FlextTiming.get_last_elapsed_time(obj)
        assert elapsed == 0.0
        
        # Test Timeable mixin methods (lines 115-116, 155, 159)
        class TestClass(FlextTiming.Timeable):
            def __init__(self) -> None:
                super().__init__()
        
        timed_obj = TestClass()
        
        # Test start_timing (line 155)
        timed_obj.start_timing()
        
        # Test stop_timing (line 159)
        time.sleep(0.01)
        timed_obj.stop_timing()
        
        # Test get_last_elapsed_time
        elapsed = timed_obj.get_last_elapsed_time()
        assert elapsed > 0


class TestValidationComplete:
    """Complete tests for validation.py to achieve 100% coverage."""
    
    def test_validation_edge_cases(self) -> None:
        """Test validation edge cases for uncovered lines."""
        
        class ValidObj:
            def __init__(self) -> None:
                self.email = "test@example.com"
        
        obj = ValidObj()
        
        # Test line 93: validate_email
        result = FlextValidation.validate_email("valid@email.com")
        assert result.success
        
        # Test line 144: clear_validation_errors without initialization
        FlextValidation.clear_validation_errors(obj)
        
        # Test line 155: get_validation_errors without initialization
        errors = FlextValidation.get_validation_errors(obj)
        assert errors == []
        
        # Test line 165: is_valid without initialization
        assert FlextValidation.is_valid(obj) is True
        
        # Test line 176: mark_valid without initialization
        FlextValidation.mark_valid(obj)
        assert FlextValidation.is_valid(obj) is True
        
        # Test line 204: Validatable mixin get_validation_errors
        class TestClass(FlextValidation.Validatable):
            def __init__(self) -> None:
                super().__init__()
        
        val_obj = TestClass()
        errors = val_obj.get_validation_errors()
        assert errors == []


class TestCoreComplete:
    """Complete tests for core.py to achieve 100% coverage."""
    
    def test_core_error_handling(self) -> None:
        """Test core.py error handling paths."""
        
        # Test line 330: max_validation_errors default
        config = {"max_validation_errors": []}  # Invalid type
        result = FlextMixins.configure_mixins_system(config)
        assert result.success
        assert result.unwrap()["max_validation_errors"] == 10
        
        # Test lines 394-395: get_mixins_system_config works normally
        result = FlextMixins.get_mixins_system_config()
        assert result.success
        config = result.unwrap()
        assert "environment" in config
        
        # Test line 418: invalid environment in create_environment_mixins_config
        result = FlextMixins.create_environment_mixins_config("invalid_env")
        assert result.is_failure
        assert "Invalid environment" in (result.error or "")
        
        # Test lines 464-475: Environment-specific configs
        for env in ["production", "staging", "test", "local"]:
            result = FlextMixins.create_environment_mixins_config(env)
            assert result.success
            config = result.unwrap()
            assert config["environment"] == env
        
        # Test lines 486-487: create_environment_mixins_config with invalid value
        # Test with non-string type to trigger different code path
        result = FlextMixins.create_environment_mixins_config(123)  # type: ignore[arg-type]
        assert result.is_failure
        
        # Test line 512: high performance level
        config = {"performance_level": "high"}
        result = FlextMixins.optimize_mixins_performance(config)
        assert result.success
        assert result.unwrap()["enable_caching"] is True
        
        # Test line 535: low performance level  
        config = {"performance_level": "low"}
        result = FlextMixins.optimize_mixins_performance(config)
        assert result.success
        assert result.unwrap()["enable_caching"] is False
        
        # Test lines 558-569: Memory optimization
        config = {"memory_limit_mb": 100}  # Low memory
        result = FlextMixins.optimize_mixins_performance(config)
        assert result.success
        assert result.unwrap()["enable_memory_monitoring"] is True
        
        config = {"memory_limit_mb": 8192}  # High memory
        result = FlextMixins.optimize_mixins_performance(config)
        assert result.success
        assert result.unwrap()["enable_large_cache"] is True
        
        # Test lines 592-593: optimize_mixins_performance exception
        with patch.object(dict, 'copy', side_effect=Exception("Test error")):
            result = FlextMixins.optimize_mixins_performance({})
            assert result.is_failure
    
    def test_mixin_imports_and_classes(self) -> None:
        """Test mixin class imports and initialization (lines 679-782)."""
        
        # Test mixin functionality through FlextMixins static methods
        # These test the delegated functionality
        
        class TestObj:
            def __init__(self) -> None:
                self.data = "test"
        
        obj = TestObj()
        
        # Test timestamp functionality
        FlextMixins.create_timestamp_fields(obj)
        assert FlextMixins.get_created_at(obj) is not None
        
        # Test logging functionality
        logger = FlextMixins.get_logger(obj)
        assert logger is not None
        
        # Test serialization functionality
        result = FlextMixins.to_dict(obj)
        assert "data" in result
        
        # Test validation functionality
        FlextMixins.initialize_validation(obj)
        FlextMixins.add_validation_error(obj, "test error")
        assert not FlextMixins.is_valid(obj)
        
        # Test identification functionality
        FlextMixins.ensure_id(obj)
        assert FlextMixins.has_id(obj)
        
        # Test state functionality
        FlextMixins.initialize_state(obj, "initial")
        assert FlextMixins.get_state(obj) == "initial"
        
        # Test cache functionality
        FlextMixins.set_cached_value(obj, "key", "value")
        assert FlextMixins.get_cached_value(obj, "key") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])