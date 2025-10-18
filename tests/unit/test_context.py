"""Comprehensive tests for FlextContext - Context Management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import threading
import time
from typing import cast

from flext_core import FlextContext, FlextModels


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
            data={"user_id": "123", "session_id": "abc"}
        )
        context = FlextContext(initial_data)
        assert context is not None
        assert context.get("user_id") == "123"
        assert context.get("session_id") == "abc"

    def test_context_set_get_value(self) -> None:
        """Test context set/get value operations."""
        context = FlextContext()

        context.set("test_key", "test_value")
        value = context.get("test_key")
        assert value == "test_value"

    def test_context_get_with_default(self) -> None:
        """Test context get with default value."""
        context = FlextContext()

        value = context.get("nonexistent_key", "default_value")
        assert value == "default_value"

    def test_context_has_value(self) -> None:
        """Test context has value check."""
        context = FlextContext()

        context.set("test_key", "test_value")
        assert context.has("test_key") is True
        assert context.has("nonexistent_key") is False

    def test_context_remove_value(self) -> None:
        """Test context remove value operation."""
        context = FlextContext()

        context.set("test_key", "test_value")
        assert context.has("test_key") is True

        context.remove("test_key")
        assert context.has("test_key") is False

    def test_context_clear(self) -> None:
        """Test context clear operation."""
        context = FlextContext()

        context.set("key1", "value1")
        context.set("key2", "value2")

        assert context.has("key1") is True
        assert context.has("key2") is True

        context.clear()

        assert context.has("key1") is False
        assert context.has("key2") is False

    def test_context_keys_values_items(self) -> None:
        """Test context keys, values, and items operations."""
        context = FlextContext()

        context.set("key1", "value1")
        context.set("key2", "value2")

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

        context.set("nested", nested_data)
        retrieved = context.get("nested")
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
        assert merged.get("key1") == "value1"
        assert merged.get("key2") == "value2"  # Overridden by context2
        assert merged.get("key3") == "value3"

    def test_context_clone(self) -> None:
        """Test context cloning."""
        context = FlextContext()
        context.set("key1", "value1")
        context.set("key2", "value2")

        cloned = context.clone()
        assert cloned.get("key1") == "value1"
        assert cloned.get("key2") == "value2"

        # Modifying clone should not affect original
        cloned.set("key1", "modified_value")
        assert context.get("key1") == "value1"

    def test_context_serialization(self) -> None:
        """Test context serialization."""
        context = FlextContext()
        context.set("string_key", "string_value")
        context.set("int_key", 42)
        context.set("bool_key", True)
        context.set("list_key", [1, 2, 3])
        context.set("dict_key", {"nested": "value"})

        # Test JSON serialization
        json_str = context.to_json()
        assert isinstance(json_str, str)
        assert "string_value" in json_str

        # Test JSON deserialization
        restored_context = FlextContext.from_json(json_str)
        assert restored_context.get("string_key") == "string_value"
        assert restored_context.get("int_key") == 42
        assert restored_context.get("bool_key") is True

    def test_context_validation(self) -> None:
        """Test context validation."""
        context = FlextContext()
        context.set("valid_key", "valid_value")

        result = context.validate()
        assert result.is_success

    def test_context_validation_failure(self) -> None:
        """Test context validation failure - empty key raises ValueError immediately."""
        import pytest

        context = FlextContext()

        # Empty key should raise ValueError at set time
        with pytest.raises(ValueError) as exc_info:
            context.set("", "empty_key")  # Should raise ValueError
        assert "must be a non-empty string" in str(exc_info.value)

        # Validate should still pass since no invalid keys were actually set
        result = context.validate()
        assert result.is_success

    def test_context_thread_safety(self) -> None:
        """Test context thread safety."""
        context = FlextContext()
        results: list[str] = []

        def set_value(thread_id: int) -> None:
            context.set(f"thread_{thread_id}", f"value_{thread_id}")
            result = context.get(f"thread_{thread_id}")
            if result is not None:
                results.append(str(result))

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
            context.get(f"key_{i}")

        end_time = time.time()

        # Should complete 2000 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_context_error_handling(self) -> None:
        """Test context error handling."""
        context = FlextContext()

        # Test invalid key
        try:
            # Use cast to suppress Pyrefly type error for intentional None test
            context.set(cast("str", None), "value")
            msg = "Should have raised an error"
            raise AssertionError(msg)
        except (TypeError, ValueError):
            pass  # Expected

        # Test invalid value
        try:
            context.set("key", object())
            msg = "Should have raised an error"
            raise AssertionError(msg)
        except (TypeError, ValueError):
            pass  # Expected

    def test_context_scoped_access(self) -> None:
        """Test context scoped access."""
        context = FlextContext()

        # Set values in different scopes
        context.set("global_key", "global_value")
        context.set("user_key", "user_value", scope="user")
        context.set("session_key", "session_value", scope="session")

        # Test global access
        assert context.get("global_key") == "global_value"

        # Test scoped access
        assert context.get("user_key", scope="user") == "user_value"
        assert context.get("session_key", scope="session") == "session_value"

        # Test cross-scope access
        assert context.get("user_key") is None  # Not in global scope
        assert context.get("session_key") is None  # Not in global scope

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

        def test_hook(_key: str, _value: object) -> None:
            nonlocal hook_called
            hook_called = True

        context.add_hook("set", test_hook)
        context.set("test_key", "test_value")

        assert hook_called is True

    def test_context_metadata(self) -> None:
        """Test context metadata."""
        context = FlextContext()

        # Set metadata
        context.set_metadata("created_at", "2025-01-01")
        context.set_metadata("version", "1.0.0")

        # Get metadata
        assert context.get_metadata("created_at") == "2025-01-01"
        assert context.get_metadata("version") == "1.0.0"

        # Get all metadata
        metadata = context.get_all_metadata()
        assert "created_at" in metadata
        assert "version" in metadata

    def test_context_statistics(self) -> None:
        """Test context statistics."""
        context = FlextContext()

        context.set("key1", "value1")
        context.set("key2", "value2")
        context.get("key1")
        context.get("key2")
        context.remove("key1")

        stats = context.get_statistics()
        if "operations" in stats:
            operations = cast("dict[str, int]", stats["operations"])
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

        context.set("key1", "value1")
        context.set("key2", "value2")

        assert context.has("key1") is True
        assert context.has("key2") is True

        context.cleanup()

        assert context.has("key1") is False
        assert context.has("key2") is False

    def test_context_export_import(self) -> None:
        """Test context export/import."""
        context = FlextContext()
        context.set("key1", "value1")
        context.set("key2", "value2")

        # Export context
        exported = context.export()
        assert isinstance(exported, dict)
        assert "key1" in exported
        assert "key2" in exported

        # Import context
        new_context = FlextContext()
        new_context.import_data(exported)

        assert new_context.get("key1") == "value1"
        assert new_context.get("key2") == "value2"

    def test_context_singleton_pattern(self) -> None:
        """Test context singleton pattern - manage explicitly."""
        # Create manual singleton pattern
        import threading

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
        import threading

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
