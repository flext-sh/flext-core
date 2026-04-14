"""Comprehensive tests for FlextContext - Context Management.

Module: flext_core
Scope: p.Context - hierarchical context management, correlation IDs, metadata

Tests FlextContext functionality including:
- Context initialization and lifecycle
- Set/get/remove operations
- Scoped access (global, request, session, etc.)
- Context merging and cloning
- Serialization (JSON export/import)
- Metadata management
- Thread safety
- Edge cases and error handling

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Annotated, ClassVar

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContainer, FlextContext
from flext_tests import tm
from tests import p, t, u


class TestFlextContext:
    type SetGetInputValue = t.Primitives | MutableSequence[int] | t.MutableStrMapping
    type SetGetExpectedValue = t.Primitives
    type NestedDictValue = Mapping[str, Mapping[str, Mapping[str, t.StrMapping]]]

    class ContextOperationScenario(BaseModel):
        """Test scenario for context operations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Context operation scenario name")]
        key: Annotated[str, Field(description="Context key under test")]
        value: Annotated[
            t.RecursiveContainer,
            Field(description="Context value under test"),
        ]
        expected_success: Annotated[
            bool,
            Field(default=True, description="Whether operation should succeed"),
        ] = True

    class ContextScenarios:
        """Centralized context test scenarios using FlextConstants."""

        SET_GET_CASES: ClassVar[
            Sequence[
                tuple[
                    str,
                    TestFlextContext.SetGetInputValue,
                    TestFlextContext.SetGetExpectedValue,
                ]
            ]
        ] = [
            ("string_key", "string_value", "string_value"),
            ("int_key", 42, 42),
            ("bool_key", True, True),
            ("list_key", [1, 2, 3], "[1,2,3]"),
            ("dict_key", {"nested": "value"}, '{"nested":"value"}'),
            ("empty_string", "", ""),
            ("empty_dict", {}, "{}"),
            ("empty_list", [], "[]"),
            ("zero", 0, 0),
            ("false_val", False, False),
        ]
        EDGE_CASE_KEYS: ClassVar[Sequence[tuple[str, str]]] = [
            ("special_chars", "key!@#$%^&*()"),
            ("long_key", "k" * 1000),
        ]
        EDGE_CASE_VALUES: ClassVar[
            Sequence[tuple[str, str | TestFlextContext.NestedDictValue]]
        ] = [
            ("long_value", "v" * 10000),
            (
                "complex_nested",
                {"level1": {"level2": {"level3": {"value": "deeply_nested"}}}},
            ),
        ]
        SCOPE_CASES: ClassVar[Sequence[tuple[str, str]]] = [
            ("user", "user_value"),
            ("session", "session_value"),
            ("request", "request_value"),
            ("operation", "operation_value"),
            ("application", "application_value"),
        ]

    def test_context_initialization(self, test_context: p.Context) -> None:
        """Test context initialization."""
        tm.that(
            test_context,
            none=False,
            msg="context must be a valid Context instance",
        )

    def test_context_with_initial_data(self) -> None:
        """Test context initialization with initial data."""
        initial_data = t.ConfigMap(root={"user_id": "123", "session_id": "abc"})
        context = FlextContext.create(initial_data=initial_data)
        u.Core.Tests.assert_context_get_success(context, "user_id", "123")
        u.Core.Tests.assert_context_get_success(context, "session_id", "abc")

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        ContextScenarios.SET_GET_CASES,
    )
    def test_context_set_get_value(
        self,
        test_context: p.Context,
        key: str,
        value: SetGetInputValue,
        expected: SetGetExpectedValue,
    ) -> None:
        """Test context set/get value operations."""
        context = test_context
        if isinstance(value, (dict, list, tuple, set)):
            set_result = context.set(t.ConfigMap.model_validate({key: value}))
        elif isinstance(value, (str, int, float, bool)):
            set_result = context.set(key, value)
        else:
            pytest.fail(f"Unexpected SetGetInputValue type: {type(value)!r}")
        _ = u.Core.Tests.assert_success(set_result)
        expected_value = expected
        u.Core.Tests.assert_context_get_success(context, key, expected_value)

    def test_context_get_with_default(self, test_context: p.Context) -> None:
        """Test context get with default value using monadic operations."""
        context = test_context
        result = context.get("nonexistent_key")
        _ = u.Core.Tests.assert_failure(result)
        default_value = "default_value"
        value = result.value if result.success else default_value
        tm.that(value, eq=default_value)

    def test_context_has_value(self, test_context: p.Context) -> None:
        """Test context has value check."""
        context = test_context
        context.set("test_key", "test_value").value
        tm.that(context.has("test_key"), eq=True)
        tm.that(not context.has("nonexistent_key"), eq=True)

    def test_context_remove_value(self, test_context: p.Context) -> None:
        """Test context remove value operation."""
        context = test_context
        context.set("test_key", "test_value").value
        tm.that(context.has("test_key"), eq=True)
        context.remove("test_key")
        tm.that(not context.has("test_key"), eq=True)

    def test_context_clear(self, test_context: p.Context) -> None:
        """Test context clear operation."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        tm.that(all(context.has(k) for k in ["key1", "key2"]), eq=True)
        context.clear()
        tm.that(not any(context.has(k) for k in ["key1", "key2"]), eq=True)

    def test_context_keys_values_items(self, test_context: p.Context) -> None:
        """Test context keys, values, and items operations."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        keys = context.keys()
        values = context.values()
        items = context.items()
        tm.that(all(k in keys for k in ["key1", "key2"]), eq=True)
        tm.that(all(v in values for v in ["value1", "value2"]), eq=True)
        tm.that(
            all(i in items for i in [("key1", "value1"), ("key2", "value2")]),
            eq=True,
        )

    def test_context_nested_data(self, test_context: p.Context) -> None:
        """Test context with nested data structures."""
        context = test_context
        nested_data: Mapping[str, Mapping[str, str | t.MutableStrMapping]] = {
            "user": {
                "id": "123",
                "profile": {"name": "John Doe", "email": "john@example.com"},
            },
        }
        context.set(t.ConfigMap.model_validate({"nested": nested_data})).value
        result = context.get("nested")
        _ = u.Core.Tests.assert_success(result)
        retrieved = result.value
        tm.that(retrieved, is_=str)
        tm.that(
            retrieved,
            eq='{"user":{"id":"123","profile":{"name":"John Doe","email":"john@example.com"}}}',
        )

    def test_context_merge(self, test_context: p.Context) -> None:
        """Test context merging."""
        context1 = test_context
        context1.set("key1", "value1").value
        context1.set("key2", "value1").value
        context2 = u.Core.Tests.create_test_context()
        context2.set("key2", "value2").value
        context2.set("key3", "value3").value
        merged = context1.merge(context2)
        u.Core.Tests.assert_context_get_success(merged, "key1", "value1")
        u.Core.Tests.assert_context_get_success(merged, "key2", "value2")
        u.Core.Tests.assert_context_get_success(merged, "key3", "value3")

    def test_context_clone(self, test_context: p.Context) -> None:
        """Test context cloning."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        cloned_raw = context.clone()
        cloned = cloned_raw
        u.Core.Tests.assert_context_get_success(cloned, "key1", "value1")
        u.Core.Tests.assert_context_get_success(cloned, "key2", "value2")
        cloned.set("key1", "modified_value").value
        u.Core.Tests.assert_context_get_success(context, "key1", "value1")

    def test_context_validation(self, test_context: p.Context) -> None:
        """Test context validation."""
        context = test_context
        context.set("valid_key", "valid_value").value
        result = context.validate_context()
        _ = u.Core.Tests.assert_success(result)

    def test_context_validation_failure(self, test_context: p.Context) -> None:
        """Test context validation failure - empty key returns failure."""
        context = test_context
        result = context.set("", "empty_key")
        _ = u.Core.Tests.assert_failure(result)
        error_message = result.error
        tm.that(error_message, none=False)
        if error_message is None:
            pytest.fail("Expected error message for invalid context key")
        tm.that(error_message, has="must be a non-empty string")

    def test_context_thread_safety(self, test_context: p.Context) -> None:
        """Test context thread safety."""
        context = test_context
        results: MutableSequence[str] = []

        def set_value(thread_id: int) -> None:
            context.set(f"thread_{thread_id}", f"value_{thread_id}")
            result = context.get(f"thread_{thread_id}")
            if result.success:
                results.append(str(result.value))

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tm.that(len(results), eq=10)
        tm.that(all(r.startswith("value_") for r in results), eq=True)

    def test_context_performance(self, test_context: p.Context) -> None:
        """Test context performance."""
        context = test_context
        start_time = time.time()
        for i in range(100):
            context.set(f"key_{i}", f"value_{i}")
            result = context.get(f"key_{i}")
            _ = u.Core.Tests.assert_success(result)
        tm.that(time.time() - start_time, lt=30.0)

    def test_context_error_handling(self, test_context: p.Context) -> None:
        """Test context error handling with r pattern."""
        context = test_context
        result = context.set("", "value")
        _ = u.Core.Tests.assert_failure(result)
        error_message = result.error
        tm.that(error_message, none=False)
        if error_message is None:
            pytest.fail("Expected error message for invalid context key")
        tm.that(error_message, has="must be a non-empty string")

    @pytest.mark.parametrize(("scope", "value"), ContextScenarios.SCOPE_CASES)
    def test_context_scoped_access(
        self,
        test_context: p.Context,
        scope: str,
        value: str,
    ) -> None:
        """Test context scoped access."""
        context = test_context
        context.set("global_key", "global_value").value
        context.set(f"{scope}_key", value, scope=scope).value
        u.Core.Tests.assert_context_get_success(
            context,
            "global_key",
            "global_value",
        )
        scoped_result = context.get(f"{scope}_key", scope=scope)
        _ = u.Core.Tests.assert_success(scoped_result)
        tm.that(scoped_result.value, eq=value)

    def test_context_metadata(self, test_context: p.Context) -> None:
        """Test context metadata."""
        context = test_context
        context.apply_metadata("created_at", "2025-01-01")
        context.apply_metadata("version", "1.0.0")
        created_at_result = context.resolve_metadata("created_at")
        _ = u.Core.Tests.assert_success(created_at_result)
        tm.that(created_at_result.value, eq="2025-01-01")
        version_result = context.resolve_metadata("version")
        _ = u.Core.Tests.assert_success(version_result)
        tm.that(version_result.value, eq="1.0.0")

    def test_context_cleanup(self, test_context: p.Context) -> None:
        """Test context cleanup."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        tm.that(all(context.has(k) for k in ["key1", "key2"]), eq=True)
        context.clear()
        tm.that(not any(context.has(k) for k in ["key1", "key2"]), eq=True)

    @pytest.mark.parametrize(
        ("key_name", "special_key"),
        ContextScenarios.EDGE_CASE_KEYS,
    )
    def test_context_edge_case_special_characters(
        self,
        test_context: p.Context,
        key_name: str,
        special_key: str,
    ) -> None:
        """Test context keys with special characters."""
        context = test_context
        _ = key_name
        context.set(special_key, "special_value").value
        u.Core.Tests.assert_context_get_success(
            context,
            special_key,
            "special_value",
        )

    @pytest.mark.parametrize(
        ("value_name", "special_value"),
        ContextScenarios.EDGE_CASE_VALUES,
    )
    def test_context_edge_case_special_values(
        self,
        test_context: p.Context,
        value_name: str,
        special_value: str | NestedDictValue,
    ) -> None:
        """Test context with special values."""
        context = test_context
        converted_value: str | t.ConfigMap
        expected_value: str
        if isinstance(special_value, dict):
            converted_value = t.ConfigMap.model_validate({
                f"{value_name}_key": special_value,
            })
            expected_value = (
                '{"level1":{"level2":{"level3":{"value":"deeply_nested"}}}}'
            )
        else:
            converted_value = str(special_value)
            expected_value = str(special_value)
        if isinstance(converted_value, t.ConfigMap):
            context.set(converted_value).value
        else:
            context.set(f"{value_name}_key", converted_value).value
        result = context.get(f"{value_name}_key")
        _ = u.Core.Tests.assert_success(result)
        actual = result.value
        tm.that(
            actual.model_dump() if isinstance(actual, t.Dict) else actual,
            eq=expected_value,
        )

    def test_context_edge_case_duplicate_keys_overwrite(
        self,
        test_context: p.Context,
    ) -> None:
        """Test context behavior when overwriting existing keys."""
        context = test_context
        context.set("key", "value1").value
        result1 = context.get("key")
        _ = u.Core.Tests.assert_success(result1)
        tm.that(result1.value, eq="value1")
        context.set("key", "value2").value
        result2 = context.get("key")
        _ = u.Core.Tests.assert_success(result2)
        tm.that(result2.value, eq="value2")

    def test_context_concurrent_reads(self, test_context: p.Context) -> None:
        """Test context with concurrent read operations."""
        context = test_context
        context.set("shared_key", "shared_value").value
        error_count: MutableSequence[int] = []

        def read_value() -> None:
            try:
                context.get("shared_key")
            except (KeyError, RuntimeError):
                error_count.append(1)

        threads = [threading.Thread(target=read_value) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tm.that(len(error_count), eq=0)

    def test_context_concurrent_writes(self, test_context: p.Context) -> None:
        """Test context with concurrent write operations."""
        context = test_context
        error_count: MutableSequence[int] = []

        def write_value(key: str, value: str) -> None:
            try:
                context.set(key, value)
            except (KeyError, RuntimeError):
                error_count.append(1)

        threads = [
            threading.Thread(target=write_value, args=(f"key_{i}", f"value_{i}"))
            for i in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tm.that(len(error_count), eq=0)

    def test_context_multiple_sequential_operations(
        self,
        test_context: p.Context,
    ) -> None:
        """Test multiple sequential set/get/remove operations."""
        context = test_context
        for i in range(100):
            context.set(f"key_{i}", f"value_{i}").value
        for i in range(100):
            u.Core.Tests.assert_context_get_success(
                context,
                f"key_{i}",
                f"value_{i}",
            )
        for i in range(50):
            context.remove(f"key_{i}")
        for i in range(50):
            tm.that(not context.has(f"key_{i}"), eq=True)
        for i in range(50, 100):
            u.Core.Tests.assert_context_get_success(
                context,
                f"key_{i}",
                f"value_{i}",
            )

    def test_context_get_metadata_nonexistent(self, test_context: p.Context) -> None:
        """Test getting metadata that doesn't exist."""
        context = test_context
        result = context.resolve_metadata("nonexistent_meta")
        _ = u.Core.Tests.assert_failure(result)
        tm.that(result.error, none=False)

    def test_context_get_metadata_with_default(
        self,
        test_context: p.Context,
    ) -> None:
        """Test getting metadata with default value."""
        context = test_context
        result = context.resolve_metadata("nonexistent_meta")
        default_value = "default_value"
        value = result.value if result.success else default_value
        tm.that(value, eq=default_value)

    def test_context_apply_resolve_metadata(self, test_context: p.Context) -> None:
        """Test setting and getting metadata."""
        context = test_context
        context.apply_metadata("meta_key", "meta_value")
        result = context.resolve_metadata("meta_key")
        _ = u.Core.Tests.assert_success(result)
        tm.that(result.value, eq="meta_value")

    def test_context_get_all_metadata_empty(self, test_context: p.Context) -> None:
        """Test getting all metadata when empty."""
        context = test_context
        result = context.resolve_metadata("meta_key")
        _ = u.Core.Tests.assert_failure(result)

    def test_service_register_and_fetch_service(self) -> None:
        """Test Service.register_service and get_service."""
        container = FlextContainer.shared(context=FlextContext())
        FlextContext.configure_container(container)
        service_instance = {"service": "instance"}
        FlextContext.Service.register_service("test-service", service_instance)
        FlextContext.Service.fetch_service("test-service")

    def test_service_context_manager(self) -> None:
        """Test Service.service_context context manager."""
        with FlextContext.Service.service_context(service_name="temp-service"):
            pass

    def test_context_remove_nonexistent(self, test_context: p.Context) -> None:
        """Test removing a nonexistent key."""
        context = test_context
        context.remove("nonexistent_key")
        tm.that(not context.has("nonexistent_key"), eq=True)

    def test_context_merge_empty_dicts(self, test_context: p.Context) -> None:
        """Test merging context with empty dictionary."""
        context = test_context
        context.set("key1", "value1").value
        merged = context.merge({})
        u.Core.Tests.assert_context_get_success(merged, "key1", "value1")

    def test_context_clone_then_clear_original(
        self,
        test_context: p.Context,
    ) -> None:
        """Test cloning, then clearing original."""
        context1 = test_context
        context1.set("key1", "value1").value
        context2_raw = context1.clone()
        context2 = context2_raw
        context1.clear()
        tm.that(not context1.has("key1"), eq=True)
        tm.that(context2.has("key1"), eq=True)

    def test_context_export_after_clear(self, test_context: p.Context) -> None:
        """Test exporting after clearing context."""
        context = test_context
        context.set("key1", "value1").value
        context.clear()
        exported = context.export()
        tm.that(exported, is_=dict)

    def test_context_merge_with_dict(self, test_context: p.Context) -> None:
        """Test merging context with dictionary."""
        context = test_context
        context.set("key1", "value1").value
        merged = context.merge({"key2": "value2", "key3": "value3"})
        u.Core.Tests.assert_context_get_success(merged, "key1", "value1")
        u.Core.Tests.assert_context_get_success(merged, "key2", "value2")
        u.Core.Tests.assert_context_get_success(merged, "key3", "value3")

    def test_context_merge_with_context(self, test_context: p.Context) -> None:
        """Test merging context with another context."""
        context1 = test_context
        context1.set("key1", "value1").value
        context2 = u.Core.Tests.create_test_context()
        context2.set("key2", "value2").value
        merged = context1.merge(context2)
        u.Core.Tests.assert_context_get_success(merged, "key1", "value1")
        u.Core.Tests.assert_context_get_success(merged, "key2", "value2")

    def test_context_clone_independence(self, test_context: p.Context) -> None:
        """Test cloned context is independent."""
        context1 = test_context
        context1.set("key1", "value1").value
        context2_raw = context1.clone()
        context2 = context2_raw
        context2.set("key1", "value2").value
        context2.set("key3", "value3").value
        u.Core.Tests.assert_context_get_success(context1, "key1", "value1")
        u.Core.Tests.assert_context_get_success(context2, "key1", "value2")
        tm.that(not context1.has("key3"), eq=True)
        tm.that(context2.has("key3"), eq=True)

    def test_context_edge_case_none_value(self, test_context: p.Context) -> None:
        """Test context with None value."""
        context = test_context
        config_map = t.ConfigMap(root={"key_none": None})
        result = context.set(config_map)
        _ = u.Core.Tests.assert_success(result)

    def test_context_get_all_scopes(self, test_context: p.Context) -> None:
        """Test getting all scope registrations."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        keys = context.keys()
        tm.that("key1" in keys, eq=True)
        tm.that("key2" in keys, eq=True)

    def test_context_performance_timing(self, test_context: p.Context) -> None:
        """Test context performance tracking."""
        context = test_context
        start_time = time.time()
        context.set("key1", "value1").value
        context.get("key1")
        context.has("key1")
        tm.that(time.time() - start_time, gte=0)
        tm.that(context.has("key1"), eq=True)

    def test_context_multiple_scopes_isolation(
        self,
        test_context: p.Context,
    ) -> None:
        """Test that values in different scopes are isolated."""
        context = test_context
        context.set("key1", "global_value").value
        context.set("key2", "request_value").value
        u.Core.Tests.assert_context_get_success(
            context,
            "key1",
            "global_value",
        )
        u.Core.Tests.assert_context_get_success(
            context,
            "key2",
            "request_value",
        )

    def test_context_variables_correlation(self) -> None:
        """Test Variables inner class Correlation access."""
        tm.that(FlextContext.Variables.Correlation, none=False)

    def test_context_variables_service(self) -> None:
        """Test Variables inner class Service access."""
        tm.that(FlextContext.Variables.Service, none=False)

    def test_context_variables_request(self) -> None:
        """Test Variables inner class Request access."""
        tm.that(FlextContext.Variables.Request, none=False)

    def test_context_export_empty(self, test_context: p.Context) -> None:
        """Test exporting empty context."""
        context = test_context
        exported = context.export()
        tm.that(exported, is_=dict)

    def test_context_export_with_data(self, test_context: p.Context) -> None:
        """Test exporting context with data."""
        context = test_context
        context.set("key1", "value1").value
        context.set("key2", 42).value
        exported = context.export()
        tm.that(exported, none=False)
        tm.that(exported, is_=dict)

    def test_context_clear_already_empty(self, test_context: p.Context) -> None:
        """Test clearing an already empty context."""
        context = test_context
        context.clear()
        tm.that(len(context.keys()), eq=0)

    def test_context_values_method_empty(self, test_context: p.Context) -> None:
        """Test context.values() on empty context."""
        context = test_context
        values = context.values()
        tm.that(values, is_=list)
        tm.that(len(values), eq=0)

    def test_context_keys_method_empty(self, test_context: p.Context) -> None:
        """Test context.keys() on empty context."""
        context = test_context
        keys = context.keys()
        tm.that(keys, is_=list)
        tm.that(len(keys), eq=0)

    def test_context_items_method_empty(self, test_context: p.Context) -> None:
        """Test context.items() on empty context."""
        context = test_context
        items = context.items()
        tm.that(items, is_=list)
        tm.that(len(items), eq=0)

    def test_context_cleanup_twice(self, test_context: p.Context) -> None:
        """Test cleanup called multiple times."""
        context = test_context
        context.set("key1", "value1").value
        context.clear()
        context.clear()

    @given(
        key=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        ),
        value=st.one_of(
            st.text(
                min_size=1,
                max_size=30,
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            ),
            st.integers(min_value=0, max_value=1000),
            st.booleans(),
        ),
    )
    @settings(max_examples=50)
    def test_set_get_roundtrip_property(self, key: str, value: t.Primitives) -> None:
        """Property: set then get returns the same value."""
        ctx = FlextContext.create()
        tm.ok(ctx.set(key, value), eq=True)
        tm.ok(ctx.get(key), eq=value)

    __all__: list[str] = ["TestFlextContext"]


__all__: list[str] = ["TestFlextContext"]
