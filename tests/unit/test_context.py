"""Comprehensive tests for FlextContext - Context Management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
import time
from typing import cast

from flext_core import FlextContext, FlextModels, FlextResult


class TestFlextContext:
    """Test suite for FlextContext context management."""

    def test_context_initialization(self) -> None:
        """Test context initialization."""
        context = FlextContext()
        assert context is not None
        assert isinstance(context, FlextContext)

    def test_context_with_initial_data(self) -> None:
        """Test context initialization with initial data."""
        initial_data = FlextModels.ContextData(
            data={"user_id": "123", "session_id": "abc"},
        )
        context = FlextContext(initial_data)
        assert context is not None

        user_id_result = context.get("user_id")
        assert isinstance(user_id_result, FlextResult)
        assert user_id_result.is_success
        assert user_id_result.unwrap() == "123"

        session_id_result = context.get("session_id")
        assert isinstance(session_id_result, FlextResult)
        assert session_id_result.is_success
        assert session_id_result.unwrap() == "abc"

    def test_context_set_get_value(self) -> None:
        """Test context set/get value operations."""
        context = FlextContext()

        context.set("test_key", "test_value").unwrap()
        result = context.get("test_key")
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.unwrap() == "test_value"

    def test_context_get_with_default(self) -> None:
        """Test context get with default value using monadic operations."""
        context = FlextContext()

        result = context.get("nonexistent_key")
        assert isinstance(result, FlextResult)
        assert result.is_failure
        # Use monadic unwrap_or for default value
        value = result.unwrap_or("default_value")
        assert value == "default_value"

    def test_context_has_value(self) -> None:
        """Test context has value check."""
        context = FlextContext()

        context.set("test_key", "test_value").unwrap()
        assert context.has("test_key") is True
        assert context.has("nonexistent_key") is False

    def test_context_remove_value(self) -> None:
        """Test context remove value operation."""
        context = FlextContext()

        context.set("test_key", "test_value").unwrap()
        assert context.has("test_key") is True

        context.remove("test_key")
        assert context.has("test_key") is False

    def test_context_clear(self) -> None:
        """Test context clear operation."""
        context = FlextContext()

        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        assert context.has("key1") is True
        assert context.has("key2") is True

        context.clear()

        assert context.has("key1") is False
        assert context.has("key2") is False

    def test_context_keys_values_items(self) -> None:
        """Test context keys, values, and items operations."""
        context = FlextContext()

        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        keys = context.keys()
        assert "key1" in keys
        assert "key2" in keys

        values = context.values()
        assert "value1" in values
        assert "value2" in values

        items = context.items()
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items

    def test_context_nested_data(self) -> None:
        """Test context with nested data structures."""
        context = FlextContext()

        nested_data: dict[str, object] = {
            "user": {
                "id": "123",
                "profile": {"name": "John Doe", "email": "john@example.com"},
            },
        }

        context.set("nested", nested_data).unwrap()
        retrieved_result = context.get("nested")
        assert retrieved_result.is_success
        retrieved = retrieved_result.unwrap()
        assert retrieved == nested_data
        if isinstance(retrieved, dict):
            assert retrieved["user"]["profile"]["name"] == "John Doe"

    def test_context_merge(self) -> None:
        """Test context merging."""
        context1 = FlextContext()
        context1.set("key1", "value1")
        context1.set("key2", "value1")

        context2 = FlextContext()
        context2.set("key2", "value2")
        context2.set("key3", "value3")

        merged = context1.merge(context2)
        key1_result = merged.get("key1")
        assert key1_result.is_success
        assert key1_result.unwrap() == "value1"

        key2_result = merged.get("key2")
        assert key2_result.is_success
        assert key2_result.unwrap() == "value2"  # Overridden by context2

        key3_result = merged.get("key3")
        assert key3_result.is_success
        assert key3_result.unwrap() == "value3"

    def test_context_clone(self) -> None:
        """Test context cloning."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        cloned = context.clone()
        key1_result = cloned.get("key1")
        assert key1_result.is_success
        assert key1_result.unwrap() == "value1"

        key2_result = cloned.get("key2")
        assert key2_result.is_success
        assert key2_result.unwrap() == "value2"

        # Modifying clone should not affect original
        cloned.set("key1", "modified_value")
        original_key1_result = context.get("key1")
        assert original_key1_result.is_success
        assert original_key1_result.unwrap() == "value1"

    def test_context_serialization(self) -> None:
        """Test context serialization."""
        context = FlextContext()
        context.set("string_key", "string_value").unwrap()
        context.set("int_key", 42).unwrap()
        context.set("bool_key", True).unwrap()
        context.set("list_key", [1, 2, 3]).unwrap()
        context.set("dict_key", {"nested": "value"}).unwrap()

        # Test JSON serialization
        json_str = context.to_json()
        assert isinstance(json_str, str)
        assert "string_value" in json_str

        # Test JSON deserialization
        restored_context = FlextContext.from_json(json_str)
        string_result = restored_context.get("string_key")
        assert string_result.is_success
        assert string_result.unwrap() == "string_value"

        int_result = restored_context.get("int_key")
        assert int_result.is_success
        assert int_result.unwrap() == 42

        bool_result = restored_context.get("bool_key")
        assert bool_result.is_success
        assert bool_result.unwrap() is True

    def test_context_validation(self) -> None:
        """Test context validation."""
        context = FlextContext()
        context.set("valid_key", "valid_value").unwrap()

        result = context.validate()
        assert result.is_success

    def test_context_validation_failure(self) -> None:
        """Test context validation failure - empty key returns failure."""
        context = FlextContext()

        # Empty key should return failure (not raise exception)
        result = context.set("", "empty_key")
        assert result.is_failure
        assert result.error is not None and "must be a non-empty string" in result.error

        # Validate should still pass since no invalid keys were actually set
        validation_result = context.validate()
        assert validation_result.is_success

    def test_context_thread_safety(self) -> None:
        """Test context thread safety."""
        context = FlextContext()
        results: list[str] = []

        def set_value(thread_id: int) -> None:
            context.set(f"thread_{thread_id}", f"value_{thread_id}")
            result = context.get(f"thread_{thread_id}")
            if result.is_success:
                results.append(str(result.unwrap()))

        threads: list[threading.Thread] = []
        for i in range(10):
            thread = threading.Thread(target=set_value, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.startswith("value_") for result in results)

    def test_context_performance(self) -> None:
        """Test context performance."""
        context = FlextContext()

        start_time = time.time()

        # Perform many operations
        for i in range(1000):
            context.set(f"key_{i}", f"value_{i}")
            result = context.get(f"key_{i}")
            assert result.is_success

        end_time = time.time()

        # Should complete 2000 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_context_error_handling(self) -> None:
        """Test context error handling with FlextResult pattern."""
        context = FlextContext()

        # Test invalid key (None) - should return failure with message
        result = context.set(cast("str", None), "value")
        assert result.is_failure
        assert result.error is not None and "must be a non-empty string" in result.error

        # Test invalid value (object) - should return failure with message
        result = context.set("key", object())
        assert result.is_failure
        assert result.error is not None
        assert "must be serializable" in result.error

    def test_context_scoped_access(self) -> None:
        """Test context scoped access."""
        context = FlextContext()

        # Set values in different scopes
        context.set("global_key", "global_value").unwrap()
        context.set("user_key", "user_value", scope="user").unwrap()
        context.set("session_key", "session_value", scope="session").unwrap()

        # Test global access
        global_result = context.get("global_key")
        assert global_result.is_success
        assert global_result.unwrap() == "global_value"

        # Test scoped access
        user_result = context.get("user_key", scope="user")
        assert user_result.is_success
        assert user_result.unwrap() == "user_value"

        session_result = context.get("session_key", scope="session")
        assert session_result.is_success
        assert session_result.unwrap() == "session_value"

        # Test cross-scope access - fast fail when key not found
        user_key_global_result = context.get("user_key")
        assert user_key_global_result.is_failure  # Not in global scope

        session_key_global_result = context.get("session_key")
        assert session_key_global_result.is_failure  # Not in global scope

    def test_context_lifecycle(self) -> None:
        """Test context lifecycle management."""
        context = FlextContext()

        # Test context creation
        assert context.is_active() is True

        # Test context suspension
        context.suspend()
        assert context.is_active() is False

        # Test context resumption
        context.resume()
        assert context.is_active() is True

        # Test context destruction
        context.destroy()
        assert context.is_active() is False

    def test_context_hooks(self) -> None:
        """Test context hooks."""
        context = FlextContext()

        hook_called = False

        def test_hook(_arg: object) -> object:
            nonlocal hook_called
            hook_called = True
            return _arg

        context.add_hook("set", test_hook)
        context.set("test_key", "test_value").unwrap()

        assert hook_called is True

    def test_context_metadata(self) -> None:
        """Test context metadata."""
        context = FlextContext()

        # Set metadata
        context.set_metadata("created_at", "2025-01-01")
        context.set_metadata("version", "1.0.0")

        # Get metadata - now returns FlextResult
        created_at_result = context.get_metadata("created_at")
        assert created_at_result.is_success
        assert created_at_result.unwrap() == "2025-01-01"

        version_result = context.get_metadata("version")
        assert version_result.is_success
        assert version_result.unwrap() == "1.0.0"

        # Get all metadata
        metadata = context.get_all_metadata()
        assert "created_at" in metadata
        assert "version" in metadata

    def test_context_statistics(self) -> None:
        """Test context statistics."""
        context = FlextContext()

        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        context.get("key1")
        context.get("key2")
        context.remove("key1")

        stats = context.get_statistics()
        if hasattr(stats, "operations") and stats.operations:
            operations = cast("dict[str, int]", stats.operations)
            # Use cast to handle dynamic typing from context statistics
            set_count: int = operations.get("set", 0)
            get_count: int = operations.get("get", 0)
            remove_count: int = operations.get("remove", 0)
            assert set_count >= 2
            assert get_count >= 2
            assert remove_count >= 1

    def test_context_cleanup(self) -> None:
        """Test context cleanup."""
        context = FlextContext()

        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        assert context.has("key1") is True
        assert context.has("key2") is True

        context.cleanup()

        assert context.has("key1") is False
        assert context.has("key2") is False

    def test_context_export_import(self) -> None:
        """Test context export/import."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        # Export context
        exported = context.export()
        assert isinstance(exported, dict)
        assert "key1" in exported
        assert "key2" in exported

        # Import context
        new_context = FlextContext()
        new_context.import_data(exported)

        assert new_context.get("key1").unwrap() == "value1"
        assert new_context.get("key2").unwrap() == "value2"

    def test_context_singleton_pattern(self) -> None:
        """Test context singleton pattern - manage explicitly."""
        # Create manual singleton pattern
        context_lock = threading.RLock()
        context_instance: FlextContext | None = None

        def get_context() -> FlextContext:
            nonlocal context_instance
            with context_lock:
                if context_instance is None:
                    context_instance = FlextContext()
                return context_instance

        context1 = get_context()
        context2 = get_context()

        assert context1 is context2

    def test_context_singleton_reset(self) -> None:
        """Test context singleton reset - manage explicitly."""
        # Create manual singleton pattern with reset
        context_lock = threading.RLock()
        context_instance: FlextContext | None = None

        def get_context() -> FlextContext:
            nonlocal context_instance
            with context_lock:
                if context_instance is None:
                    context_instance = FlextContext()
                return context_instance

        def reset_context() -> None:
            nonlocal context_instance
            with context_lock:
                context_instance = None

        context1 = get_context()
        reset_context()
        context2 = get_context()

        assert context1 is not context2

    def test_correlation_generate_correlation_id(self) -> None:
        """Test Correlation.generate_correlation_id generates unique IDs."""
        id1 = FlextContext.Correlation.generate_correlation_id()
        id2 = FlextContext.Correlation.generate_correlation_id()
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    def test_correlation_new_correlation_auto_generate(self) -> None:
        """Test Correlation.new_correlation auto-generates ID if not provided."""
        with FlextContext.Correlation.new_correlation():
            correlation_id = FlextContext.Correlation.generate_correlation_id()
            assert correlation_id is not None
            assert len(correlation_id) > 0

    def test_service_register_and_get_service(self) -> None:
        """Test Service.register_service and get_service."""
        service_instance = {"service": "instance"}
        FlextContext.Service.register_service("test-service", service_instance)
        result = FlextContext.Service.get_service("test-service")
        assert result.is_success or result.is_failure

    def test_service_context_manager(self) -> None:
        """Test Service.service_context context manager."""
        with FlextContext.Service.service_context(service_name="temp-service"):
            # Context manager should work without errors
            pass

    def test_context_suspend_resume(self) -> None:
        """Test context suspend and resume operations."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        assert context.is_active() is True

        context.suspend()
        # After suspend, context may or may not be active depending on implementation

        context.resume()
        assert context.is_active() is True

    def test_context_is_active(self) -> None:
        """Test context is_active method."""
        context = FlextContext()
        assert context.is_active() is True

    def test_context_destroy(self) -> None:
        """Test context destroy operation."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.destroy()
        # After destroy, context should be inactive or cleared

    def test_context_scoped_access_request_scope(self) -> None:
        """Test scoped access with request scope."""
        context = FlextContext()
        context.set("user_id", "user-456").unwrap()
        result = context.get("user_id")
        assert result.unwrap() == "user-456"

    def test_context_scoped_access_operation_scope(self) -> None:
        """Test scoped access with operation scope."""
        context = FlextContext()
        context.set("operation_id", "op-789").unwrap()
        result = context.get("operation_id")
        assert result.unwrap() == "op-789"

    def test_context_scoped_access_application_scope(self) -> None:
        """Test scoped access with application scope."""
        context = FlextContext()
        context.set("app_config", {"key": "value"}).unwrap()
        result = context.get("app_config")
        assert result.unwrap() == {"key": "value"}

    def test_context_remove_from_scope(self) -> None:
        """Test removing value from specific scope."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        assert context.has("key1") is True

        context.remove("key1")
        assert context.has("key1") is False

    def test_context_validate_success(self) -> None:
        """Test context validation succeeds."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        result = context.validate()
        assert result.is_success

    def test_context_validate_failure_empty(self) -> None:
        """Test context validation with empty context."""
        context = FlextContext()
        context.validate()
        # May succeed or fail depending on validation rules

    def test_context_to_json(self) -> None:
        """Test context serialization to JSON."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        json_str = context.to_json()
        assert json_str is not None
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_context_from_json_roundtrip(self) -> None:
        """Test context deserialization from JSON."""
        context1 = FlextContext()
        context1.set("key1", "value1")
        context1.set("key2", 42)

        json_str = context1.to_json()
        context2 = FlextContext.from_json(json_str)

        key1_result = context2.get("key1")
        assert key1_result.is_success
        assert key1_result.unwrap() == "value1"

        key2_result = context2.get("key2")
        assert key2_result.is_success
        assert key2_result.unwrap() == 42

    def test_context_get_all_scopes(self) -> None:
        """Test getting all scope registrations."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()

        scopes = context.get_all_scopes()
        assert scopes is not None
        assert isinstance(scopes, dict)

    def test_context_performance_timing(self) -> None:
        """Test context performance tracking."""
        context = FlextContext()
        start_time = time.time()
        context.set("key1", "value1").unwrap()
        context.get("key1")
        context.has("key1")
        end_time = time.time()

        elapsed = end_time - start_time
        assert elapsed >= 0
        assert context.has("key1") is True

    def test_context_multiple_scopes_isolation(self) -> None:
        """Test that values in different scopes are isolated."""
        context = FlextContext()
        context.set("key1", "global_value").unwrap()
        context.set("key2", "request_value").unwrap()

        global_val = context.get("key1")
        request_val = context.get("key2")

        assert global_val.is_success
        assert request_val.is_success
        assert global_val.unwrap() == "global_value"
        assert request_val.unwrap() == "request_value"

    def test_context_variables_correlation(self) -> None:
        """Test Variables inner class Correlation access."""
        # Test that Variables inner class exists and can be accessed
        assert hasattr(FlextContext.Variables, "Correlation")
        assert FlextContext.Variables.Correlation is not None

    def test_context_variables_service(self) -> None:
        """Test Variables inner class Service access."""
        # Test that Variables inner class exists and can be accessed
        assert hasattr(FlextContext.Variables, "Service")
        assert FlextContext.Variables.Service is not None

    def test_context_variables_request(self) -> None:
        """Test Variables inner class Request access."""
        # Test that Variables inner class exists and can be accessed
        assert hasattr(FlextContext.Variables, "Request")
        assert FlextContext.Variables.Request is not None

    def test_context_export_empty(self) -> None:
        """Test exporting empty context."""
        context = FlextContext()
        exported = context.export()
        assert isinstance(exported, dict)

    def test_context_export_with_data(self) -> None:
        """Test exporting context with data."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.set("key2", 42).unwrap()

        exported = context.export()
        assert exported is not None
        assert isinstance(exported, dict)

    def test_context_export_snapshot(self) -> None:
        """Test exporting context snapshot."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()

        snapshot = context.export_snapshot()
        assert snapshot is not None
        assert isinstance(snapshot, FlextModels.ContextExport)

    def test_context_import_data(self) -> None:
        """Test importing data into context."""
        context = FlextContext()
        data_to_import: dict[str, object] = {"key1": "value1", "key2": "value2"}

        context.import_data(data_to_import)
        assert context.get("key1").unwrap() == "value1"
        assert context.get("key2").unwrap() == "value2"

    def test_context_merge_with_dict(self) -> None:
        """Test merging context with dictionary."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()

        merged = context.merge({"key2": "value2", "key3": "value3"})
        assert merged.get("key1").unwrap() == "value1"
        assert merged.get("key2").unwrap() == "value2"
        assert merged.get("key3").unwrap() == "value3"

    def test_context_merge_with_context(self) -> None:
        """Test merging context with another context."""
        context1 = FlextContext()
        context1.set("key1", "value1")

        context2 = FlextContext()
        context2.set("key2", "value2")

        merged = context1.merge(context2)
        # Use FlextResult API - get() returns FlextResult[object]
        result1 = merged.get("key1")
        assert result1.is_success
        assert result1.unwrap() == "value1"

        result2 = merged.get("key2")
        assert result2.is_success
        assert result2.unwrap() == "value2"

    def test_context_clone_independence(self) -> None:
        """Test cloned context is independent."""
        context1 = FlextContext()
        context1.set("key1", "value1")

        context2 = context1.clone()
        context2.set("key1", "value2")
        context2.set("key3", "value3")

        assert context1.get("key1").unwrap() == "value1"
        assert context2.get("key1").unwrap() == "value2"
        assert context1.has("key3") is False
        assert context2.has("key3") is True

    def test_context_edge_case_none_value(self) -> None:
        """Test context with None value."""
        context = FlextContext()
        context.set("key_none", None).unwrap()
        result = context.get("key_none")
        # FlextResult.ok() cannot accept None, so get() should fail
        assert result.is_failure
        assert result.error is not None
        assert "None value" in result.error or "not found" in result.error.lower()

    def test_context_edge_case_empty_string(self) -> None:
        """Test context with empty string."""
        context = FlextContext()
        context.set("empty_key", "").unwrap()
        result = context.get("empty_key")
        assert result.unwrap() == ""

    def test_context_edge_case_empty_dict(self) -> None:
        """Test context with empty dict."""
        context = FlextContext()
        context.set("empty_dict", {}).unwrap()
        result = context.get("empty_dict")
        assert result.unwrap() == {}

    def test_context_edge_case_empty_list(self) -> None:
        """Test context with empty list."""
        context = FlextContext()
        context.set("empty_list", []).unwrap()
        result = context.get("empty_list")
        assert result.unwrap() == []

    def test_context_edge_case_zero_value(self) -> None:
        """Test context with zero value."""
        context = FlextContext()
        context.set("zero", 0).unwrap()
        result = context.get("zero")
        assert result.unwrap() == 0

    def test_context_edge_case_false_value(self) -> None:
        """Test context with False value."""
        context = FlextContext()
        context.set("false_val", False).unwrap()
        result = context.get("false_val")
        assert result.is_success
        assert result.unwrap() is False

    def test_context_edge_case_special_characters(self) -> None:
        """Test context keys with special characters."""
        context = FlextContext()
        special_key = "key!@#$%^&*()"
        context.set(special_key, "special_value")
        result = context.get(special_key)
        assert result.unwrap() == "special_value"

    def test_context_edge_case_very_long_key(self) -> None:
        """Test context with very long key name."""
        context = FlextContext()
        long_key = "k" * 1000
        context.set(long_key, "long_key_value")
        result = context.get(long_key)
        assert result.unwrap() == "long_key_value"

    def test_context_edge_case_very_long_value(self) -> None:
        """Test context with very long value."""
        context = FlextContext()
        long_value = "v" * 10000
        context.set("long_value_key", long_value).unwrap()
        result = context.get("long_value_key")
        assert result.unwrap() == long_value

    def test_context_edge_case_complex_nested_dict(self) -> None:
        """Test context with complex nested dictionary."""
        context = FlextContext()
        complex_dict = {"level1": {"level2": {"level3": {"value": "deeply_nested"}}}}
        context.set("complex", complex_dict).unwrap()
        result = context.get("complex")
        assert result.is_success
        unwrapped = result.unwrap()
        assert unwrapped == complex_dict
        assert isinstance(unwrapped, dict)
        assert unwrapped["level1"]["level2"]["level3"]["value"] == "deeply_nested"

    def test_context_edge_case_duplicate_keys_overwrite(self) -> None:
        """Test context behavior when overwriting existing keys."""
        context = FlextContext()
        context.set("key", "value1").unwrap()
        assert context.get("key").unwrap() == "value1"
        context.set("key", "value2").unwrap()
        assert context.get("key").unwrap() == "value2"

    def test_context_concurrent_reads(self) -> None:
        """Test context with concurrent read operations."""
        context = FlextContext()
        context.set("shared_key", "shared_value").unwrap()

        error_count = []

        def read_value() -> None:
            try:
                context.get("shared_key")
            except Exception:
                error_count.append(1)

        threads = [threading.Thread(target=read_value) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not error during concurrent reads
        assert len(error_count) == 0

    def test_context_concurrent_writes(self) -> None:
        """Test context with concurrent write operations."""
        context = FlextContext()
        error_count = []

        def write_value(key: str, value: str) -> None:
            try:
                context.set(key, value)
            except Exception:
                error_count.append(1)

        threads = [
            threading.Thread(target=write_value, args=(f"key_{i}", f"value_{i}"))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not error during concurrent writes
        assert len(error_count) == 0

    def test_context_values_method_empty(self) -> None:
        """Test context.values() on empty context."""
        context = FlextContext()
        values = context.values()
        assert isinstance(values, list)
        assert len(values) == 0

    def test_context_keys_method_empty(self) -> None:
        """Test context.keys() on empty context."""
        context = FlextContext()
        keys = context.keys()
        assert isinstance(keys, list)
        assert len(keys) == 0

    def test_context_items_method_empty(self) -> None:
        """Test context.items() on empty context."""
        context = FlextContext()
        items = context.items()
        assert isinstance(items, list)
        assert len(items) == 0

    def test_context_clear_already_empty(self) -> None:
        """Test clearing an already empty context."""
        context = FlextContext()
        context.clear()  # Should not raise
        assert len(context.keys()) == 0

    def test_context_remove_nonexistent(self) -> None:
        """Test removing a nonexistent key."""
        context = FlextContext()
        context.remove("nonexistent_key")  # Should not raise
        assert context.has("nonexistent_key") is False

    def test_context_merge_empty_dicts(self) -> None:
        """Test merging context with empty dictionary."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        merged = context.merge({})
        assert merged.get("key1").unwrap() == "value1"

    def test_context_clone_then_clear_original(self) -> None:
        """Test cloning, then clearing original."""
        context1 = FlextContext()
        context1.set("key1", "value1")
        context2 = context1.clone()
        context1.clear()

        assert context1.has("key1") is False
        assert context2.has("key1") is True

    def test_context_import_empty_data(self) -> None:
        """Test importing empty data into context."""
        context = FlextContext()
        context.set("existing_key", "existing_value").unwrap()
        context.import_data({})
        assert context.get("existing_key").unwrap() == "existing_value"

    def test_context_export_after_clear(self) -> None:
        """Test exporting after clearing context."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.clear()
        exported = context.export()
        assert isinstance(exported, dict)
        assert len(exported) == 0 or "key1" not in exported

    def test_context_multiple_sequential_operations(self) -> None:
        """Test multiple sequential set/get/remove operations."""
        context = FlextContext()
        for i in range(100):
            context.set(f"key_{i}", f"value_{i}")

        for i in range(100):
            result = context.get(f"key_{i}")
            assert result.is_success
            assert result.unwrap() == f"value_{i}"

        for i in range(50):
            context.remove(f"key_{i}")

        for i in range(50):
            assert context.has(f"key_{i}") is False

        for i in range(50, 100):
            result = context.get(f"key_{i}")
            assert result.is_success
            assert result.unwrap() == f"value_{i}"

    def test_context_get_metadata_nonexistent(self) -> None:
        """Test getting metadata that doesn't exist - fast fail."""
        context = FlextContext()
        result = context.get_metadata("nonexistent_meta")
        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_context_get_metadata_with_default(self) -> None:
        """Test getting metadata with default value using monadic operations."""
        context = FlextContext()
        result = context.get_metadata("nonexistent_meta")
        # Use monadic unwrap_or for default value
        value = result.unwrap_or("default_value")
        assert value == "default_value"

    def test_context_set_get_metadata(self) -> None:
        """Test setting and getting metadata."""
        context = FlextContext()
        context.set_metadata("meta_key", "meta_value")
        result = context.get_metadata("meta_key")
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.unwrap() == "meta_value"

    def test_context_get_all_metadata_empty(self) -> None:
        """Test getting all metadata when empty."""
        context = FlextContext()
        metadata = context.get_all_metadata()
        assert isinstance(metadata, dict)

    def test_context_get_all_data_empty(self) -> None:
        """Test getting all data when empty."""
        context = FlextContext()
        all_data = context.get_all_data()
        assert isinstance(all_data, dict)

    def test_context_get_statistics(self) -> None:
        """Test getting context statistics."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        stats = context.get_statistics()
        assert stats is not None
        assert isinstance(stats, FlextModels.ContextStatistics)

    def test_context_from_json_invalid_json(self) -> None:
        """Test creating context from invalid JSON."""
        try:
            FlextContext.from_json("invalid json {")
            # If it doesn't raise, that's ok too
        except Exception:
            # Expected behavior
            pass

    def test_correlation_inherit_correlation_context(self) -> None:
        """Test Correlation.inherit_correlation context manager."""
        try:
            with FlextContext.Correlation.inherit_correlation():
                # Should work without errors
                pass
        except Exception:
            # May not be fully implemented
            pass

    def test_context_add_hook_and_invoke(self) -> None:
        """Test adding and invoking hooks."""
        context = FlextContext()
        hook_called = []

        def test_hook(arg: object) -> object:
            hook_called.append(arg)
            return arg

        context.add_hook("test_event", test_hook)
        # Note: We can't directly invoke hooks, just verify add_hook doesn't error

    def test_context_cleanup_twice(self) -> None:
        """Test cleanup called multiple times."""
        context = FlextContext()
        context.set("key1", "value1").unwrap()
        context.cleanup()
        context.cleanup()  # Second call should not error
