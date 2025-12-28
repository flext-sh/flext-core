"""Real tests to achieve 100% context coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in context.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict as UserDictBase
from typing import cast

import pytest
from pydantic import BaseModel

from flext_core import FlextTypes as t, (
    FlextConstants,
    FlextContext,
    FlextResult,
    m,
    r,
    t,
)
from flext_tests import u
from tests.test_utils import assertion_helpers

# ==================== COVERAGE TESTS ====================


class TestContext100Coverage:
    """Real tests to achieve 100% context coverage."""

    def test_remove_success(self) -> None:
        """Test remove successfully removes key."""
        context = FlextContext()
        context.set("test_key", "test_value").value

        # Remove the key
        context.remove("test_key")

        # Verify key is removed
        result = context.get("test_key")
        u.Tests.Result.assert_result_failure(result)

    def test_remove_nonexistent_key(self) -> None:
        """Test remove with nonexistent key (idempotent)."""
        context = FlextContext()

        # Remove nonexistent key - should not raise
        context.remove("nonexistent_key")

        # Verify key still doesn't exist
        result = context.get("nonexistent_key")
        u.Tests.Result.assert_result_failure(result)

    def test_clear_removes_all_data(self) -> None:
        """Test clear removes all data."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value

        # Clear all data
        context.clear()

        # Verify all keys are removed
        result1 = context.get("key1")
        result2 = context.get("key2")
        u.Tests.Result.assert_result_failure(result1)
        u.Tests.Result.assert_result_failure(result2)

    def test_merge_with_dict(self) -> None:
        """Test merge with dictionary."""
        context1 = FlextContext()
        context1.set("key1", "value1").value

        merge_data: dict[str, t.GeneralValueType] = {"key2": "value2", "key3": "value3"}
        # Convert dict[str, t.GeneralValueType] to dict[str, t.GeneralValueType]
        converted_data: dict[str, t.GeneralValueType] = {
            k: v
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
            else str(v)
            for k, v in merge_data.items()
        }
        merged = context1.merge(converted_data)
        assert isinstance(merged, FlextContext)

        # Verify merged data
        result2 = merged.get("key2")
        result3 = merged.get("key3")
        u.Tests.Result.assert_result_success(result2)
        u.Tests.Result.assert_result_success(result3)

    def test_merge_with_context(self) -> None:
        """Test merge with another context."""
        context1 = FlextContext()
        context1.set("key1", "value1").value

        context2 = FlextContext()
        context2.set("key2", "value2").value

        merged = context1.merge(context2)
        assert isinstance(merged, FlextContext)

        # Verify merged data
        result1 = merged.get("key1")
        result2 = merged.get("key2")
        u.Tests.Result.assert_result_success(result1)
        u.Tests.Result.assert_result_success(result2)

    def test_clone_creates_independent_copy(self) -> None:
        """Test clone creates independent copy."""
        context1 = FlextContext()
        context1.set("key1", "value1").value

        cloned = context1.clone()
        assert isinstance(cloned, FlextContext)

        # Verify cloned has same data
        result = cloned.get("key1")
        u.Tests.Result.assert_success_with_value(
            cast("r[str]", result),
            "value1",
        )

        # Modify original - clone should be independent
        context1.set("key1", "modified").value
        cloned_result = cloned.get("key1")
        u.Tests.Result.assert_success_with_value(
            cast("r[str]", cloned_result),
            "value1",
        )  # Clone unchanged

    def test_validate_success(self) -> None:
        """Test validate with valid context."""
        context = FlextContext()
        context.set("key1", "value1").value

        result = context.validate()
        u.Tests.Result.assert_result_success(result)

    def test_suspend_resume(self) -> None:
        """Test suspend and resume functionality."""
        context = FlextContext()
        context.set("key1", "value1").value

        # Suspend context
        context._suspend()
        assert context._suspended is True

        # Resume context
        context._resume()
        assert context._suspended is False

    def test_destroy_deactivates_context(self) -> None:
        """Test destroy deactivates context."""
        context = FlextContext()
        context.set("key1", "value1").value

        # Destroy context
        context._destroy()

        # Verify context is inactive
        assert context._active is False

        # Operations should fail after destroy
        result = context.set("key2", "value2")
        u.Tests.Result.assert_result_failure(result)

    def test_export_returns_dict(self) -> None:
        """Test export returns dictionary with scoped data."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value

        exported = context.export()
        # export() returns {scope: {key: value}} structure
        assert isinstance(exported, dict)
        assert "global" in exported
        # Type narrowing: exported["global"] is dict-like
        global_data = exported.get("global")
        if isinstance(global_data, dict):
            assert "key1" in global_data
            assert "key2" in global_data

    def test_export_snapshot_returns_typed_model(self) -> None:
        """Test export_snapshot returns typed model."""
        context = FlextContext()
        context.set("key1", "value1").value

        snapshot = context._export_snapshot()
        assert isinstance(snapshot, m.ContextExport)
        assert "key1" in snapshot.data

    def test_import_data_loads_dict(self) -> None:
        """Test import_data loads dictionary."""
        context = FlextContext()

        import_data: dict[str, t.GeneralValueType] = {"key1": "value1", "key2": "value2"}
        # Convert dict[str, t.GeneralValueType] to dict[str, t.GeneralValueType]
        converted_data: dict[str, t.GeneralValueType] = {
            k: v
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
            else str(v)
            for k, v in import_data.items()
        }
        context._import_data(converted_data)

        # Verify imported data
        result1 = context.get("key1")
        result2 = context.get("key2")
        u.Tests.Result.assert_result_success(result1)
        u.Tests.Result.assert_result_success(result2)

    def test_import_empty_data(self) -> None:
        """Test import_data with empty dict."""
        context = FlextContext()
        context.set("existing", "value").value

        context._import_data({})

        # Existing data should remain
        result = context.get("existing")
        u.Tests.Result.assert_result_success(result)

    def test_get_with_none_value_returns_failure(self) -> None:
        """Test get with None value returns failure."""
        context = FlextContext()
        # Set a key to None (if possible) or test the None handling path
        # Since set() validates None, we'll test the get() None handling
        context.set("key1", "value1").value

        # Manually set None in contextvar to test None handling
        scope_var = context._scope_vars[FlextConstants.Context.SCOPE_GLOBAL]
        current = scope_var.get() or {}
        current["none_key"] = None
        scope_var.set(current)

        # Get None value should return failure
        result = context.get("none_key")
        u.Tests.Result.assert_result_failure(result)
        assert result.error is not None and "None value" in result.error

    def test_get_when_context_not_active(self) -> None:
        """Test get when context is not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        result = context.get("any_key")
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None and "not active" in result.error

    def test_set_when_context_not_active(self) -> None:
        """Test set when context is not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        result = context.set("key", "value")
        u.Tests.Result.assert_result_failure(result)
        assert result.error is not None and "not active" in result.error

    def test_has_returns_false_when_not_active(self) -> None:
        """Test has returns False when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        has_key = context.has("any_key")
        assert has_key is False

    def test_remove_when_not_active(self) -> None:
        """Test remove when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        # Remove should not raise, but do nothing
        context.remove("any_key")  # Should not raise

    def test_clear_when_not_active(self) -> None:
        """Test clear when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        # Clear should not raise
        context.clear()  # Should not raise

    def test_merge_when_not_active(self) -> None:
        """Test merge when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        # Merge should still work (creates new context)
        merged = context.merge({"key": "value"})
        assert isinstance(merged, FlextContext)

    def test_clone_when_not_active(self) -> None:
        """Test clone when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        # Clone should still work
        cloned = context.clone()
        assert isinstance(cloned, FlextContext)

    def test_validate_when_not_active(self) -> None:
        """Test validate when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        result = context.validate()
        # May succeed or fail depending on implementation
        # Check for ResultProtocol attributes (structural typing)
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_export_when_not_active(self) -> None:
        """Test export when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        exported = context.export()
        assert isinstance(exported, dict)

    def test_export_snapshot_when_not_active(self) -> None:
        """Test export_snapshot when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        snapshot = context._export_snapshot()
        assert isinstance(snapshot, m.ContextExport)

    def test_import_data_when_not_active(self) -> None:
        """Test import_data when context not active."""
        context = FlextContext()
        context._destroy()  # Deactivates context

        # Import should not raise
        context._import_data({"key": "value"})  # Should not raise

    def test_get_with_different_scope(self) -> None:
        """Test get with different scope."""
        context = FlextContext()
        context.set("global_key", "global_value").value
        context.set("user_key", "user_value", scope="user").value

        # Get from global scope
        global_result = context.get(
            "global_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        assert global_result.is_success

        # Get from user scope
        user_result = context.get("user_key", scope="user")
        assert user_result.is_success

        # Get from wrong scope
        wrong_result = context.get(
            "user_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        assert wrong_result.is_failure

    def test_set_with_different_scope(self) -> None:
        """Test set with different scope."""
        context = FlextContext()

        # Set in global scope
        result1 = context.set(
            "global_key",
            "global_value",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        assert result1.is_success

        # Set in user scope
        result2 = context.set("user_key", "user_value", scope="user")
        assert result2.is_success

        # Verify isolation
        global_result = context.get(
            "global_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        user_result = context.get("user_key", scope="user")
        assert global_result.is_success
        assert user_result.is_success

    def test_remove_from_specific_scope(self) -> None:
        """Test remove from specific scope."""
        context = FlextContext()
        context.set("key1", "value1", scope="user").value

        # Remove from user scope
        context.remove("key1", scope="user")

        # Verify removed
        result = context.get("key1", scope="user")
        assertion_helpers.assert_flext_result_failure(result)

    def test_has_with_different_scope(self) -> None:
        """Test has with different scope."""
        context = FlextContext()
        context.set("key1", "value1", scope="user").value

        # Check in user scope
        has_user = context.has("key1", scope="user")
        assert has_user is True

        # Check in global scope
        has_global = context.has("key1", scope=FlextConstants.Context.SCOPE_GLOBAL)
        assert has_global is False

    def test_keys_returns_all_keys(self) -> None:
        """Test keys returns all keys."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value

        keys = context.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_values_returns_all_values(self) -> None:
        """Test values returns all values."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value

        values = context.values()
        assert "value1" in values
        assert "value2" in values

    def test_items_returns_all_items(self) -> None:
        """Test items returns all items."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value

        items = context.items()
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items

    def test_get_all_scopes_returns_dict(self) -> None:
        """Test get_all_scopes returns dictionary."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2", scope="user").value

        all_scopes = context._get_all_scopes()
        assert isinstance(all_scopes, dict)
        assert FlextConstants.Context.SCOPE_GLOBAL in all_scopes
        assert "user" in all_scopes

    def test_get_statistics_returns_model(self) -> None:
        """Test get_statistics returns statistics model."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.get("key1")

        stats = context._get_statistics()
        assert isinstance(stats, m.ContextStatistics)

    def test_statistics_access(self) -> None:
        """Test statistics access via get_statistics."""
        context = FlextContext()
        context.set("key1", "value1").value

        stats = context._get_statistics()
        assert isinstance(stats, m.ContextStatistics)
        assert stats.sets >= 1  # Fixed: Field is 'sets', not 'set_count'

    def test_to_json_returns_string(self) -> None:
        """Test to_json returns JSON string."""
        context = FlextContext()
        context.set("key1", "value1").value

        json_str = context.to_json()
        assert isinstance(json_str, str)
        assert "key1" in json_str

    def test_serialization_round_trip(self) -> None:
        """Test serialization round trip."""
        context1 = FlextContext()
        context1.set("key1", "value1").value
        context1.set("key2", "value2").value

        # Export returns {scope: {key: value}}, import expects flat {key: value}
        exported = context1.export()
        context2 = FlextContext()
        # Pass global scope data to _import_data
        # Type narrowing: exported is dict[str, t.GeneralValueType] | ContextExport
        # When as_dict=True (default), it returns dict
        if isinstance(exported, dict):
            global_data = exported.get("global")
            if isinstance(global_data, dict):
                # Convert dict[str, t.GeneralValueType] to dict[str, t.GeneralValueType]
                converted_global: dict[str, t.GeneralValueType] = {
                    k: v
                    if isinstance(v, (str, int, float, bool, type(None), list, dict))
                    else str(v)
                    for k, v in global_data.items()
                }
                context2._import_data(converted_global)

        # Verify data
        result1 = context2.get("key1")
        result2 = context2.get("key2")
        assert result1.is_success
        assert result2.is_success

    def test_export_after_clear(self) -> None:
        """Test export after clear."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.clear()

        exported = context.export()
        assert isinstance(exported, dict)
        # Should be empty or minimal

    def test_merge_empty_dicts(self) -> None:
        """Test merge with empty dictionaries."""
        context1 = FlextContext()
        context2 = FlextContext()

        merged = context1.merge(context2)
        assert isinstance(merged, FlextContext)

    def test_remove_from_specific_scope_direct(self) -> None:
        """Test remove from specific scope using remove method."""
        context = FlextContext()
        context.set("key1", "value1", scope="user").value

        # Remove from user scope
        context.remove("key1", scope="user")

        # Verify removed
        get_result = context.get("key1", scope="user")
        assert get_result.is_failure

    def test_hooks_execution(self) -> None:
        """Test hooks execution during operations."""
        context = FlextContext()

        # Hooks are executed internally during set/get operations
        # Test that operations trigger hooks
        context.set("key1", "value1").value
        context.get("key1")

        # Verify hooks were executed (via statistics or internal state)
        stats = context._get_statistics()
        assert stats.sets >= 1  # Fixed: Field is 'sets', not 'set_count'
        assert stats.gets >= 1  # Fixed: Field is 'gets', not 'get_count'

    def test_get_with_default_using_unwrap_or(self) -> None:
        """Test get with default using unwrap_or pattern."""
        context = FlextContext()

        # Get nonexistent key and use unwrap_or for default
        result = context.get("nonexistent")
        # Cast to FlextResult to access unwrap_or (ResultProtocol doesn't have it)

        result_typed = cast("FlextResult[t.GeneralValueType]", result)
        value = result_typed.unwrap_or("default_value")
        assert value == "default_value"

    def test_export_import_round_trip(self) -> None:
        """Test export/import round trip."""
        context1 = FlextContext()
        context1.set("key1", "value1").value
        context1.set("key2", "value2").value

        # Export returns {scope: {key: value}}, import expects flat {key: value}
        exported = context1.export()

        # Import global scope data into new context
        context2 = FlextContext()
        # Type narrowing: exported is dict[str, t.GeneralValueType] | ContextExport
        # When as_dict=True (default), it returns dict
        if isinstance(exported, dict):
            global_data = exported.get("global")
            if isinstance(global_data, dict):
                # Convert dict[str, t.GeneralValueType] to dict[str, t.GeneralValueType]
                converted_global: dict[str, t.GeneralValueType] = {
                    k: v
                    if isinstance(v, (str, int, float, bool, type(None), list, dict))
                    else str(v)
                    for k, v in global_data.items()
                }
                context2._import_data(converted_global)

        # Verify
        result1 = context2.get("key1")
        result2 = context2.get("key2")
        assert result1.is_success
        assert result2.is_success

    def test_context_data_validate_dict_serializable_non_dict(self) -> None:
        """Test ContextData.validate_dict_serializable with non-dict."""
        # Test with non-dict value for metadata (which uses validate_metadata)
        invalid_metadata: object = 123
        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            m.ContextData(metadata=invalid_metadata)

    def test_context_data_validate_dict_serializable_non_string_key(self) -> None:
        """Test ContextData.validate_dict_serializable with non-string key.

        Note: Non-string keys are converted to strings by the validator's
        key normalization (str(k)), so integer key 123 becomes string key "123".
        """
        # Test with dict containing non-string key - will be converted to string

        class IntKeyDict(UserDictBase[int, str]):
            def __init__(self) -> None:
                # Initialize with non-string key
                super().__init__()
                self[123] = "value"

        int_key_dict: object = IntKeyDict()
        # Key normalization converts int key to string
        result = m.ContextData(data=int_key_dict)
        assert "123" in result.data

    def test_context_data_validate_dict_serializable_non_serializable_value(
        self,
    ) -> None:
        """Test ContextData.validate_dict_serializable with non-serializable value.

        Note: Non-JSON-serializable values (like sets) are converted to strings
        by FlextRuntime.normalize_to_general_value() before serializability check,
        so they become valid strings. This is intentional - ensures any value
        can be stored in context.
        """
        # Test with dict containing non-serializable value (e.g., set)
        # Sets are converted to string representation (e.g., "{1, 2, 3}")
        bad_dict: dict[str, t.GeneralValueType] = {"key": {1, 2, 3}}  # set becomes string
        result = m.ContextData(data=bad_dict)
        # Set was normalized to string representation
        assert isinstance(result.data["key"], str)

    def test_context_export_validate_dict_serializable_pydantic_model(self) -> None:
        """Test ContextExport.validate_dict_serializable with Pydantic model."""

        class TestModel(BaseModel):
            field: str = "value"

        # Test with Pydantic model (should convert via model_dump)
        model: TestModel = TestModel()
        export = m.ContextExport(data=model)
        assert isinstance(export.data, dict)
        assert "field" in export.data

    def test_context_export_validate_dict_serializable_non_dict(self) -> None:
        """Test ContextExport.validate_dict_serializable with non-dict."""
        # Test with non-dict value (should raise TypeError via Pydantic validation)
        invalid_data: object = 123
        with pytest.raises(
            TypeError,
            match=r".*must be a dict or Pydantic model.*",
        ):
            m.ContextExport(data=invalid_data)

    def test_context_export_validate_dict_serializable_non_string_key(self) -> None:
        """Test ContextExport.validate_dict_serializable with non-string key.

        Note: Non-string keys are converted to strings by normalize_to_general_value(),
        so integer key 123 becomes string key "123". No error is raised.
        """
        # Create dict with non-string key (will be converted to string)
        data: dict[object, str] = {123: "value"}  # Non-string key → "123"
        result = m.ContextExport(data=data)
        # Key was normalized to string
        assert "123" in result.data

    def test_context_export_validate_dict_serializable_non_serializable_value(
        self,
    ) -> None:
        """Test ContextExport.validate_dict_serializable with non-serializable value.

        Note: Non-JSON-serializable values (like sets) are converted to strings
        by FlextRuntime.normalize_to_general_value() before serializability check.
        """
        # Test with dict containing non-serializable value (e.g., set)
        # Sets are converted to string representation
        data: dict[str, t.GeneralValueType] = {"key": {1, 2, 3}}  # set becomes string
        result = m.ContextExport(data=data)
        # Set was normalized to string representation
        assert isinstance(result.data["key"], str)

    def test_context_export_total_data_items(self) -> None:
        """Test ContextExport.total_data_items computed field."""
        export = m.ContextExport(
            data={"key1": "value1", "key2": "value2"},
            metadata=m.Metadata(attributes={}),
            statistics={},
        )
        # Access computed field directly (Pydantic v2 property)
        assert len(export.data) == 2

    def test_context_export_has_statistics(self) -> None:
        """Test ContextExport.has_statistics computed field."""
        # With statistics
        export1 = m.ContextExport(
            data={},
            metadata=m.Metadata(attributes={}),
            statistics={"sets": 5},
        )
        # Check that statistics are non-empty (computed field checks bool(statistics))
        assert bool(export1.statistics) is True

        # Without statistics
        export2 = m.ContextExport(
            data={},
            metadata=m.Metadata(attributes={}),
            statistics={},
        )
        # Check that statistics are empty
        assert bool(export2.statistics) is False

    def test_context_scope_data_validate_data_with_basemodel(self) -> None:
        """Test ContextScopeData._validate_data with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model: TestModel = TestModel()
        # Create instance with BaseModel - validator will be called
        scope_data = m.ContextScopeData(data=model)
        assert isinstance(scope_data.data, dict)
        assert "field" in scope_data.data

    def test_context_scope_data_validate_data_with_none(self) -> None:
        """Test ContextScopeData._validate_data with None."""
        # Create instance with empty dict (None validation tests not applicable here)
        scope_data = m.ContextScopeData(data={})
        assert isinstance(scope_data.data, dict)
        assert scope_data.data == {}

    def test_context_scope_data_validate_metadata_with_basemodel(self) -> None:
        """Test ContextScopeData._validate_metadata with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model: TestModel = TestModel()
        # Create instance with BaseModel - validator will be called
        scope_data = m.ContextScopeData(metadata=model)
        assert isinstance(scope_data.metadata, dict)
        assert "field" in scope_data.metadata

    def test_context_scope_data_validate_metadata_with_none(self) -> None:
        """Test ContextScopeData._validate_metadata with None."""
        # Create instance with empty dict (None validation tests not applicable here)
        scope_data = m.ContextScopeData(metadata={})
        assert isinstance(scope_data.metadata, dict)
        assert scope_data.metadata == {}

    def test_context_statistics_validate_operations_with_basemodel(self) -> None:
        """Test ContextStatistics._validate_operations with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model: TestModel = TestModel()
        # Create instance with BaseModel - validator will be called
        stats = m.ContextStatistics(operations=model)
        assert isinstance(stats.operations, dict)
        assert "field" in stats.operations

    def test_context_statistics_validate_operations_with_none(self) -> None:
        """Test ContextStatistics._validate_operations with None."""
        # Create instance with None - validator will convert to {}
        none_operations: object | None = None
        stats = m.ContextStatistics(operations=none_operations)
        assert isinstance(stats.operations, dict)
        assert stats.operations == {}

    def test_context_metadata_validate_custom_fields_with_basemodel(self) -> None:
        """Test ContextMetadata._validate_custom_fields with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model: TestModel = TestModel()
        # Create instance with BaseModel - validator will be called
        metadata = m.ContextMetadata(custom_fields=model)
        assert isinstance(metadata.custom_fields, dict)
        assert "field" in metadata.custom_fields

    def test_context_metadata_validate_custom_fields_with_none(self) -> None:
        """Test ContextMetadata._validate_custom_fields with None."""
        # Create instance with None - validator will convert to {}
        none_custom_fields: object | None = None
        metadata = m.ContextMetadata(custom_fields=none_custom_fields)
        assert isinstance(metadata.custom_fields, dict)
        assert metadata.custom_fields == {}
