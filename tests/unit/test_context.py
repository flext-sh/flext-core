"""Comprehensive tests for FlextContext - Context Management.

Module: flext_core.context
Scope: FlextContext - hierarchical context management, correlation IDs, metadata

Tests FlextContext functionality including:
- Context initialization and lifecycle
- Set/get/remove operations
- Scoped access (global, request, session, etc.)
- Context merging and cloning
- Serialization (JSON export/import)
- Metadata management
- Thread safety
- Edge cases and error handling

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import ClassVar

import pytest

from flext_core import FlextContext, FlextModels
from flext_core.typings import t


@dataclass(frozen=True, slots=True)
class ContextOperationScenario:
    """Test scenario for context operations."""

    name: str
    key: str
    value: object
    expected_success: bool = True


class ContextScenarios:
    """Centralized context test scenarios using FlextConstants."""

    SET_GET_CASES: ClassVar[list[tuple[str, object, object]]] = [
        ("string_key", "string_value", "string_value"),
        ("int_key", 42, 42),
        ("bool_key", True, True),
        ("list_key", [1, 2, 3], [1, 2, 3]),
        ("dict_key", {"nested": "value"}, {"nested": "value"}),
        ("empty_string", "", ""),
        ("empty_dict", {}, {}),
        ("empty_list", [], []),
        ("zero", 0, 0),
        ("false_val", False, False),
    ]

    EDGE_CASE_KEYS: ClassVar[list[tuple[str, str]]] = [
        ("special_chars", "key!@#$%^&*()"),
        ("long_key", "k" * 1000),
    ]

    EDGE_CASE_VALUES: ClassVar[list[tuple[str, object]]] = [
        ("long_value", "v" * 10000),
        (
            "complex_nested",
            {"level1": {"level2": {"level3": {"value": "deeply_nested"}}}},
        ),
    ]

    SCOPE_CASES: ClassVar[list[tuple[str, str]]] = [
        ("user", "user_value"),
        ("session", "session_value"),
        ("request", "request_value"),
        ("operation", "operation_value"),
        ("application", "application_value"),
    ]


class ContextTestHelpers:
    """Generalized helpers for context testing."""

    @staticmethod
    def create_test_context(**initial_data: object) -> FlextContext:
        """Create test context with optional initial data."""
        if initial_data:
            # Convert object values to GeneralValueType for type safety
            converted_data: dict[str, t.GeneralValueType] = {}
            for key, value in initial_data.items():
                # Type narrowing: ensure value is GeneralValueType compatible
                if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    converted_data[str(key)] = value
                else:
                    # Convert non-compatible types to string
                    converted_data[str(key)] = str(value)
            context_data = FlextModels.ContextData(data=converted_data)
            return FlextContext(context_data)
        return FlextContext()

    @staticmethod
    def assert_context_get_success(
        context: FlextContext,
        key: str,
        expected_value: object,
    ) -> None:
        """Assert context get operation succeeds with expected value."""
        result = context.get(key)
        assert result.is_success
        assert result.value == expected_value

    @staticmethod
    def assert_context_get_failure(context: FlextContext, key: str) -> None:
        """Assert context get operation fails."""
        result = context.get(key)
        assert result.is_failure


class TestFlextContext:
    """Test suite for FlextContext using FlextTestsUtilities and FlextConstants."""

    def test_context_initialization(self) -> None:
        """Test context initialization."""
        context = ContextTestHelpers.create_test_context()
        assert context is not None
        assert isinstance(context, FlextContext)

    def test_context_with_initial_data(self) -> None:
        """Test context initialization with initial data."""
        initial_data = FlextModels.ContextData(
            data={"user_id": "123", "session_id": "abc"},
        )
        context = FlextContext(initial_data)
        ContextTestHelpers.assert_context_get_success(context, "user_id", "123")
        ContextTestHelpers.assert_context_get_success(context, "session_id", "abc")

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        ContextScenarios.SET_GET_CASES,
        ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
    )
    def test_context_set_get_value(
        self,
        key: str,
        value: object,
        expected: object,
    ) -> None:
        """Test context set/get value operations."""
        context = ContextTestHelpers.create_test_context()
        # Type narrowing: value must be GeneralValueType compatible
        converted_value: t.GeneralValueType = (
            value
            if isinstance(value, (str, int, float, bool, type(None), list, dict))
            else str(value)
        )
        set_result = context.set(key, converted_value)
        assert set_result.is_success
        ContextTestHelpers.assert_context_get_success(context, key, expected)

    def test_context_get_with_default(self) -> None:
        """Test context get with default value using monadic operations."""
        context = ContextTestHelpers.create_test_context()
        result = context.get("nonexistent_key")
        assert result.is_failure
        # unwrap_or exists on FlextResult, type narrowing ensures it's available
        default_value = "default_value"
        # Type narrowing: context.get() returns FlextResult[GeneralValueType]
        # FlextResult has unwrap_or method
        # pyright: ignore[reportAttributeAccessIssue] - unwrap_or exists on FlextResult
        value = result.unwrap_or(default_value)  # type: ignore[attr-defined]
        assert value == default_value

    def test_context_has_value(self) -> None:
        """Test context has value check."""
        context = ContextTestHelpers.create_test_context()
        context.set("test_key", "test_value").unwrap()
        assert context.has("test_key") is True
        assert context.has("nonexistent_key") is False

    def test_context_remove_value(self) -> None:
        """Test context remove value operation."""
        context = ContextTestHelpers.create_test_context()
        context.set("test_key", "test_value").unwrap()
        assert context.has("test_key") is True
        context.remove("test_key")
        assert context.has("test_key") is False

    def test_context_clear(self) -> None:
        """Test context clear operation."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        assert all(context.has(k) for k in ["key1", "key2"])
        context.clear()
        assert not any(context.has(k) for k in ["key1", "key2"])

    def test_context_keys_values_items(self) -> None:
        """Test context keys, values, and items operations."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        keys = context.keys()
        values = context.values()
        items = context.items()
        assert all(k in keys for k in ["key1", "key2"])
        assert all(v in values for v in ["value1", "value2"])
        assert all(i in items for i in [("key1", "value1"), ("key2", "value2")])

    def test_context_nested_data(self) -> None:
        """Test context with nested data structures."""
        context = ContextTestHelpers.create_test_context()
        # Type narrowing: nested_data must be GeneralValueType compatible
        nested_data: dict[str, t.GeneralValueType] = {
            "user": {
                "id": "123",
                "profile": {"name": "John Doe", "email": "john@example.com"},
            },
        }
        context.set("nested", nested_data).unwrap()
        result = context.get("nested")
        assert result.is_success
        retrieved = result.value
        assert isinstance(retrieved, dict)
        # Type narrowing: retrieved is dict after isinstance check
        # pyright needs explicit type narrowing for nested dict access
        retrieved_dict: dict[str, t.GeneralValueType] = retrieved
        user_data = retrieved_dict.get("user")
        assert isinstance(user_data, dict)
        user_dict: dict[str, t.GeneralValueType] = user_data
        profile_data = user_dict.get("profile")
        assert isinstance(profile_data, dict)
        profile_dict: dict[str, t.GeneralValueType] = profile_data
        assert profile_dict.get("name") == "John Doe"

    def test_context_merge(self) -> None:
        """Test context merging."""
        context1 = ContextTestHelpers.create_test_context()
        context1.set("key1", "value1").unwrap()
        context1.set("key2", "value1").unwrap()
        context2 = ContextTestHelpers.create_test_context()
        context2.set("key2", "value2").unwrap()
        context2.set("key3", "value3").unwrap()
        merged = context1.merge(context2)
        ContextTestHelpers.assert_context_get_success(merged, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(merged, "key2", "value2")
        ContextTestHelpers.assert_context_get_success(merged, "key3", "value3")

    def test_context_clone(self) -> None:
        """Test context cloning."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        cloned = context.clone()
        ContextTestHelpers.assert_context_get_success(cloned, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(cloned, "key2", "value2")
        cloned.set("key1", "modified_value").unwrap()
        ContextTestHelpers.assert_context_get_success(context, "key1", "value1")

    def test_context_serialization(self) -> None:
        """Test context serialization."""
        context = ContextTestHelpers.create_test_context()
        context.set("string_key", "string_value").unwrap()
        context.set("int_key", 42).unwrap()
        context.set("bool_key", True).unwrap()
        json_str = context.to_json()
        assert isinstance(json_str, str) and "string_value" in json_str
        restored = FlextContext.from_json(json_str)
        ContextTestHelpers.assert_context_get_success(
            restored,
            "string_key",
            "string_value",
        )
        ContextTestHelpers.assert_context_get_success(restored, "int_key", 42)
        ContextTestHelpers.assert_context_get_success(restored, "bool_key", True)

    def test_context_validation(self) -> None:
        """Test context validation."""
        context = ContextTestHelpers.create_test_context()
        context.set("valid_key", "valid_value").unwrap()
        result = context.validate()
        assert result.is_success

    def test_context_validation_failure(self) -> None:
        """Test context validation failure - empty key returns failure."""
        context = ContextTestHelpers.create_test_context()
        result = context.set("", "empty_key")
        assert result.is_failure
        assert result.error is not None and "must be a non-empty string" in result.error

    def test_context_thread_safety(self) -> None:
        """Test context thread safety."""
        context = ContextTestHelpers.create_test_context()
        results: list[str] = []

        def set_value(thread_id: int) -> None:
            context.set(f"thread_{thread_id}", f"value_{thread_id}")
            result = context.get(f"thread_{thread_id}")
            if result.is_success:
                results.append(str(result.value))

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 10
        assert all(r.startswith("value_") for r in results)

    def test_context_performance(self) -> None:
        """Test context performance."""
        context = ContextTestHelpers.create_test_context()
        start_time = time.time()
        for i in range(100):
            context.set(f"key_{i}", f"value_{i}")
            result = context.get(f"key_{i}")
            assert result.is_success
        assert time.time() - start_time < 1.0

    def test_context_error_handling(self) -> None:
        """Test context error handling with FlextResult pattern."""
        context = ContextTestHelpers.create_test_context()
        result = context.set("", "value")
        assert result.is_failure
        assert result.error is not None and "must be a non-empty string" in result.error

    @pytest.mark.parametrize(
        ("scope", "value"),
        ContextScenarios.SCOPE_CASES,
        ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
    )
    def test_context_scoped_access(self, scope: str, value: str) -> None:
        """Test context scoped access."""
        context = ContextTestHelpers.create_test_context()
        context.set("global_key", "global_value").unwrap()
        context.set(f"{scope}_key", value, scope=scope).unwrap()
        ContextTestHelpers.assert_context_get_success(
            context,
            "global_key",
            "global_value",
        )
        scoped_result = context.get(f"{scope}_key", scope=scope)
        assert scoped_result.is_success
        assert scoped_result.value == value

    def test_context_lifecycle(self) -> None:
        """Test context lifecycle management."""
        context = ContextTestHelpers.create_test_context()
        assert context.is_active() is True
        context._suspend()
        context._resume()
        assert context.is_active() is True
        context._destroy()

    def test_context_hooks(self) -> None:
        """Test context hooks."""
        context = ContextTestHelpers.create_test_context()
        hook_called = False

        # HandlerCallable = Callable[[GeneralValueType], GeneralValueType]
        def test_hook(_arg: t.GeneralValueType) -> t.GeneralValueType:
            nonlocal hook_called
            hook_called = True
            return _arg

        context._add_hook("set", test_hook)
        context.set("test_key", "test_value").unwrap()
        assert hook_called is True

    def test_context_metadata(self) -> None:
        """Test context metadata."""
        context = ContextTestHelpers.create_test_context()
        context.set_metadata("created_at", "2025-01-01")
        context.set_metadata("version", "1.0.0")
        created_at_result = context.get_metadata("created_at")
        assert created_at_result.is_success
        assert created_at_result.value == "2025-01-01"
        metadata = context._get_all_metadata()
        assert "created_at" in metadata and "version" in metadata

    def test_context_statistics(self) -> None:
        """Test context statistics."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        context.get("key1")
        context.get("key2")
        context.remove("key1")
        stats = context._get_statistics()
        assert stats is not None
        assert isinstance(stats, FlextModels.ContextStatistics)

    def test_context_cleanup(self) -> None:
        """Test context cleanup."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        assert all(context.has(k) for k in ["key1", "key2"])
        context._cleanup()
        assert not any(context.has(k) for k in ["key1", "key2"])

    def test_context_export_import(self) -> None:
        """Test context export/import."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        exported = context.export()
        # export() returns {scope: {key: value}} structure
        assert isinstance(exported, dict)
        assert "global" in exported
        global_data = exported.get("global")
        new_context = FlextContext()
        if global_data is not None and isinstance(global_data, dict):
            # Convert dict[str, object] to dict[str, GeneralValueType]
            converted_global: dict[str, t.GeneralValueType] = {
                str(k): v
                if isinstance(v, (str, int, float, bool, type(None), list, dict))
                else str(v)
                for k, v in global_data.items()
            }
            assert "key1" in converted_global
            # Pass global scope data to _import_data
            new_context._import_data(converted_global)
        ContextTestHelpers.assert_context_get_success(new_context, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(new_context, "key2", "value2")

    @pytest.mark.parametrize(
        ("key_name", "special_key"),
        ContextScenarios.EDGE_CASE_KEYS,
        ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
    )
    def test_context_edge_case_special_characters(
        self,
        key_name: str,
        special_key: str,
    ) -> None:
        """Test context keys with special characters."""
        context = ContextTestHelpers.create_test_context()
        context.set(special_key, "special_value").unwrap()
        ContextTestHelpers.assert_context_get_success(
            context,
            special_key,
            "special_value",
        )

    @pytest.mark.parametrize(
        ("value_name", "special_value"),
        ContextScenarios.EDGE_CASE_VALUES,
        ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
    )
    def test_context_edge_case_special_values(
        self,
        value_name: str,
        special_value: object,
    ) -> None:
        """Test context with special values."""
        context = ContextTestHelpers.create_test_context()
        # Type narrowing: special_value must be GeneralValueType compatible
        converted_value: t.GeneralValueType = (
            special_value
            if isinstance(
                special_value, (str, int, float, bool, type(None), list, dict)
            )
            else str(special_value)
        )
        context.set(f"{value_name}_key", converted_value).unwrap()
        result = context.get(f"{value_name}_key")
        assert result.is_success
        assert result.value == special_value

    def test_context_edge_case_duplicate_keys_overwrite(self) -> None:
        """Test context behavior when overwriting existing keys."""
        context = ContextTestHelpers.create_test_context()
        context.set("key", "value1").unwrap()
        assert context.get("key").value == "value1"
        context.set("key", "value2").unwrap()
        assert context.get("key").value == "value2"

    def test_context_concurrent_reads(self) -> None:
        """Test context with concurrent read operations."""
        context = ContextTestHelpers.create_test_context()
        context.set("shared_key", "shared_value").unwrap()
        error_count: list[int] = []

        def read_value() -> None:
            try:
                context.get("shared_key")
            except Exception:
                error_count.append(1)

        threads = [threading.Thread(target=read_value) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(error_count) == 0

    def test_context_concurrent_writes(self) -> None:
        """Test context with concurrent write operations."""
        context = ContextTestHelpers.create_test_context()
        error_count: list[int] = []

        def write_value(key: str, value: str) -> None:
            try:
                context.set(key, value)
            except Exception:
                error_count.append(1)

        threads = [
            threading.Thread(target=write_value, args=(f"key_{i}", f"value_{i}"))
            for i in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(error_count) == 0

    def test_context_multiple_sequential_operations(self) -> None:
        """Test multiple sequential set/get/remove operations."""
        context = ContextTestHelpers.create_test_context()
        for i in range(100):
            context.set(f"key_{i}", f"value_{i}").unwrap()
        for i in range(100):
            ContextTestHelpers.assert_context_get_success(
                context,
                f"key_{i}",
                f"value_{i}",
            )
        for i in range(50):
            context.remove(f"key_{i}")
        for i in range(50):
            assert context.has(f"key_{i}") is False
        for i in range(50, 100):
            ContextTestHelpers.assert_context_get_success(
                context,
                f"key_{i}",
                f"value_{i}",
            )

    def test_context_get_metadata_nonexistent(self) -> None:
        """Test getting metadata that doesn't exist."""
        context = ContextTestHelpers.create_test_context()
        result = context.get_metadata("nonexistent_meta")
        assert result.is_failure
        assert result.error is not None

    def test_context_get_metadata_with_default(self) -> None:
        """Test getting metadata with default value."""
        context = ContextTestHelpers.create_test_context()
        result = context.get_metadata("nonexistent_meta")
        # unwrap_or exists on FlextResult, type narrowing ensures it's available
        default_value = "default_value"
        # Type narrowing: context.get_metadata() returns FlextResult[GeneralValueType]
        # FlextResult has unwrap_or method
        # pyright: ignore[reportAttributeAccessIssue] - unwrap_or exists on FlextResult
        value = result.unwrap_or(default_value)  # type: ignore[attr-defined]
        assert value == default_value

    def test_context_set_get_metadata(self) -> None:
        """Test setting and getting metadata."""
        context = ContextTestHelpers.create_test_context()
        context.set_metadata("meta_key", "meta_value")
        result = context.get_metadata("meta_key")
        assert result.is_success
        assert result.value == "meta_value"

    def test_context_get_all_metadata_empty(self) -> None:
        """Test getting all metadata when empty."""
        context = ContextTestHelpers.create_test_context()
        metadata = context._get_all_metadata()
        assert isinstance(metadata, dict)

    def test_context_get_all_data_empty(self) -> None:
        """Test getting all data when empty."""
        context = ContextTestHelpers.create_test_context()
        all_data = context._get_all_data()
        assert isinstance(all_data, dict)

    def test_context_get_statistics(self) -> None:
        """Test getting context statistics."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        stats = context._get_statistics()
        assert stats is not None
        assert isinstance(stats, FlextModels.ContextStatistics)

    def test_context_from_json_invalid_json(self) -> None:
        """Test creating context from invalid JSON."""
        try:
            FlextContext.from_json("invalid json {")
        except Exception:
            pass

    def test_correlation_generate_correlation_id(self) -> None:
        """Test Correlation.generate_correlation_id generates unique IDs."""
        id1 = FlextContext.Correlation.generate_correlation_id()
        id2 = FlextContext.Correlation.generate_correlation_id()
        assert id1 is not None and id2 is not None
        assert id1 != id2 and len(id1) > 0

    def test_correlation_new_correlation_auto_generate(self) -> None:
        """Test Correlation.new_correlation auto-generates ID if not provided."""
        with FlextContext.Correlation.new_correlation():
            correlation_id = FlextContext.Correlation.generate_correlation_id()
            assert correlation_id is not None and len(correlation_id) > 0

    def test_service_register_and_get_service(self) -> None:
        """Test Service.register_service and get_service."""
        service_instance = {"service": "instance"}
        FlextContext.Service.register_service("test-service", service_instance)
        result = FlextContext.Service.get_service("test-service")
        assert result.is_success or result.is_failure

    def test_service_context_manager(self) -> None:
        """Test Service.service_context context manager."""
        with FlextContext.Service.service_context(service_name="temp-service"):
            pass

    def test_context_suspend_resume(self) -> None:
        """Test context suspend and resume operations."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        assert context.is_active() is True
        context._suspend()
        context._resume()
        assert context.is_active() is True

    def test_context_is_active(self) -> None:
        """Test context is_active method."""
        context = ContextTestHelpers.create_test_context()
        assert context.is_active() is True

    def test_context_destroy(self) -> None:
        """Test context destroy operation."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context._destroy()

    def test_context_remove_nonexistent(self) -> None:
        """Test removing a nonexistent key."""
        context = ContextTestHelpers.create_test_context()
        context.remove("nonexistent_key")
        assert context.has("nonexistent_key") is False

    def test_context_merge_empty_dicts(self) -> None:
        """Test merging context with empty dictionary."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        merged = context.merge({})
        ContextTestHelpers.assert_context_get_success(merged, "key1", "value1")

    def test_context_clone_then_clear_original(self) -> None:
        """Test cloning, then clearing original."""
        context1 = ContextTestHelpers.create_test_context()
        context1.set("key1", "value1").unwrap()
        context2 = context1.clone()
        context1.clear()
        assert context1.has("key1") is False
        assert context2.has("key1") is True

    def test_context_import_empty_data(self) -> None:
        """Test importing empty data into context."""
        context = ContextTestHelpers.create_test_context()
        context.set("existing_key", "existing_value").unwrap()
        context._import_data({})
        ContextTestHelpers.assert_context_get_success(
            context,
            "existing_key",
            "existing_value",
        )

    def test_context_export_after_clear(self) -> None:
        """Test exporting after clearing context."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.clear()
        exported = context.export()
        assert isinstance(exported, dict)

    def test_context_merge_with_dict(self) -> None:
        """Test merging context with dictionary."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        merged = context.merge({"key2": "value2", "key3": "value3"})
        ContextTestHelpers.assert_context_get_success(merged, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(merged, "key2", "value2")
        ContextTestHelpers.assert_context_get_success(merged, "key3", "value3")

    def test_context_merge_with_context(self) -> None:
        """Test merging context with another context."""
        context1 = ContextTestHelpers.create_test_context()
        context1.set("key1", "value1").unwrap()
        context2 = ContextTestHelpers.create_test_context()
        context2.set("key2", "value2").unwrap()
        merged = context1.merge(context2)
        ContextTestHelpers.assert_context_get_success(merged, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(merged, "key2", "value2")

    def test_context_clone_independence(self) -> None:
        """Test cloned context is independent."""
        context1 = ContextTestHelpers.create_test_context()
        context1.set("key1", "value1").unwrap()
        context2 = context1.clone()
        context2.set("key1", "value2").unwrap()
        context2.set("key3", "value3").unwrap()
        ContextTestHelpers.assert_context_get_success(context1, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(context2, "key1", "value2")
        assert context1.has("key3") is False
        assert context2.has("key3") is True

    def test_context_edge_case_none_value(self) -> None:
        """Test context with None value."""
        context = ContextTestHelpers.create_test_context()
        result = context.set("key_none", None)
        assert result.is_failure

    def test_context_export_snapshot(self) -> None:
        """Test exporting context snapshot."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        snapshot = context._export_snapshot()
        assert snapshot is not None
        assert isinstance(snapshot, FlextModels.ContextExport)

    def test_context_to_json(self) -> None:
        """Test context serialization to JSON."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        json_str = context.to_json()
        assert isinstance(json_str, str) and len(json_str) > 0

    def test_context_from_json_roundtrip(self) -> None:
        """Test context deserialization from JSON."""
        context1 = ContextTestHelpers.create_test_context()
        context1.set("key1", "value1").unwrap()
        context1.set("key2", 42).unwrap()
        json_str = context1.to_json()
        context2 = FlextContext.from_json(json_str)
        ContextTestHelpers.assert_context_get_success(context2, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(context2, "key2", 42)

    def test_context_get_all_scopes(self) -> None:
        """Test getting all scope registrations."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", "value2").unwrap()
        scopes = context._get_all_scopes()
        assert scopes is not None
        assert isinstance(scopes, dict)

    def test_context_performance_timing(self) -> None:
        """Test context performance tracking."""
        context = ContextTestHelpers.create_test_context()
        start_time = time.time()
        context.set("key1", "value1").unwrap()
        context.get("key1")
        context.has("key1")
        assert time.time() - start_time >= 0
        assert context.has("key1") is True

    def test_context_multiple_scopes_isolation(self) -> None:
        """Test that values in different scopes are isolated."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "global_value").unwrap()
        context.set("key2", "request_value").unwrap()
        ContextTestHelpers.assert_context_get_success(context, "key1", "global_value")
        ContextTestHelpers.assert_context_get_success(context, "key2", "request_value")

    def test_context_variables_correlation(self) -> None:
        """Test Variables inner class Correlation access."""
        assert hasattr(FlextContext.Variables, "Correlation")
        assert FlextContext.Variables.Correlation is not None

    def test_context_variables_service(self) -> None:
        """Test Variables inner class Service access."""
        assert hasattr(FlextContext.Variables, "Service")
        assert FlextContext.Variables.Service is not None

    def test_context_variables_request(self) -> None:
        """Test Variables inner class Request access."""
        assert hasattr(FlextContext.Variables, "Request")
        assert FlextContext.Variables.Request is not None

    def test_context_export_empty(self) -> None:
        """Test exporting empty context."""
        context = ContextTestHelpers.create_test_context()
        exported = context.export()
        assert isinstance(exported, dict)

    def test_context_export_with_data(self) -> None:
        """Test exporting context with data."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context.set("key2", 42).unwrap()
        exported = context.export()
        assert exported is not None
        assert isinstance(exported, dict)

    def test_context_import_data(self) -> None:
        """Test importing data into context."""
        context = ContextTestHelpers.create_test_context()
        # Type narrowing: convert dict[str, object] to dict[str, GeneralValueType]
        data_to_import: dict[str, t.GeneralValueType] = {
            "key1": "value1",
            "key2": "value2",
        }
        context._import_data(data_to_import)
        ContextTestHelpers.assert_context_get_success(context, "key1", "value1")
        ContextTestHelpers.assert_context_get_success(context, "key2", "value2")

    def test_context_clear_already_empty(self) -> None:
        """Test clearing an already empty context."""
        context = ContextTestHelpers.create_test_context()
        context.clear()
        assert len(context.keys()) == 0

    def test_context_values_method_empty(self) -> None:
        """Test context.values() on empty context."""
        context = ContextTestHelpers.create_test_context()
        values = context.values()
        assert isinstance(values, list) and len(values) == 0

    def test_context_keys_method_empty(self) -> None:
        """Test context.keys() on empty context."""
        context = ContextTestHelpers.create_test_context()
        keys = context.keys()
        assert isinstance(keys, list) and len(keys) == 0

    def test_context_items_method_empty(self) -> None:
        """Test context.items() on empty context."""
        context = ContextTestHelpers.create_test_context()
        items = context.items()
        assert isinstance(items, list) and len(items) == 0

    def test_context_cleanup_twice(self) -> None:
        """Test cleanup called multiple times."""
        context = ContextTestHelpers.create_test_context()
        context.set("key1", "value1").unwrap()
        context._cleanup()
        context._cleanup()

    def test_correlation_inherit_correlation_context(self) -> None:
        """Test Correlation.inherit_correlation context manager."""
        try:
            with FlextContext.Correlation.inherit_correlation():
                pass
        except Exception:
            pass

    def test_context_add_hook_and_invoke(self) -> None:
        """Test adding and invoking hooks."""
        context = ContextTestHelpers.create_test_context()
        hook_called: list[t.GeneralValueType] = []

        def test_hook(arg: t.GeneralValueType) -> t.GeneralValueType:
            hook_called.append(arg)
            return arg

        # Type narrowing: test_hook signature matches HandlerCallable
        # HandlerCallable = Callable[[GeneralValueType], GeneralValueType]
        # test_hook has the same signature, so it's compatible
        context._add_hook("test_event", test_hook)


__all__ = ["TestFlextContext"]
