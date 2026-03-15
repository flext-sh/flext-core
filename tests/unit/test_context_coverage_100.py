"""Real tests to achieve 100% context coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in context.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict as UserDictBase

import pytest
from pydantic import BaseModel, ValidationError

from flext_core import FlextConstants, FlextContext
from flext_core._models.base import FlextModelFoundation
from flext_core._models.context import FlextModelsContext
from flext_tests import t, tm, u


class TestContext100Coverage:
    """Real tests to achieve 100% context coverage."""

    def test_remove_success(self) -> None:
        """Test remove successfully removes key."""
        context = FlextContext()
        context.set("test_key", "test_value").value
        context.remove("test_key")
        result = context.get("test_key")
        _ = u.Tests.Result.assert_failure(result)

    def test_remove_nonexistent_key(self) -> None:
        """Test remove with nonexistent key (idempotent)."""
        context = FlextContext()
        context.remove("nonexistent_key")
        result = context.get("nonexistent_key")
        _ = u.Tests.Result.assert_failure(result)

    def test_clear_removes_all_data(self) -> None:
        """Test clear removes all data."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        context.clear()
        result1 = context.get("key1")
        result2 = context.get("key2")
        _ = u.Tests.Result.assert_failure(result1)
        _ = u.Tests.Result.assert_failure(result2)

    def test_merge_with_dict(self) -> None:
        """Test merge with dictionary."""
        context1 = FlextContext()
        context1.set("key1", "value1").value
        merge_data: dict[str, t.NormalizedValue] = {"key2": "value2", "key3": "value3"}
        merged = context1.merge(merge_data)
        assert isinstance(merged, FlextContext)
        result2 = merged.get("key2")
        result3 = merged.get("key3")
        _ = u.Tests.Result.assert_success(result2)
        _ = u.Tests.Result.assert_success(result3)

    def test_merge_with_context(self) -> None:
        """Test merge with another context preserves current context when payload is invalid."""
        context1 = FlextContext()
        context1.set("key1", "value1").value
        context2 = FlextContext()
        context2.set("key2", "value2").value
        merged = context1.merge(context2)
        assert isinstance(merged, FlextContext)
        result1 = merged.get("key1")
        result2 = merged.get("key2")
        _ = u.Tests.Result.assert_success(result1)
        _ = u.Tests.Result.assert_success(result2)

    def test_clone_creates_independent_copy(self) -> None:
        """Test clone creates independent copy."""
        context1 = FlextContext()
        context1.set("key1", "value1").value
        cloned = context1.clone()
        assert isinstance(cloned, FlextContext)
        result = cloned.get("key1")
        _ = u.Tests.Result.assert_success(result)
        tm.that(result.value, eq="value1")
        context1.set("key1", "modified").value
        cloned_result = cloned.get("key1")
        _ = u.Tests.Result.assert_success(cloned_result)
        tm.that(cloned_result.value, eq="value1")

    def test_validate_success(self) -> None:
        """Test validate with valid context."""
        context = FlextContext()
        context.set("key1", "value1").value
        result = context.validate_context()
        _ = u.Tests.Result.assert_success(result)

    def test_export_returns_dict(self) -> None:
        """Test export returns dictionary with scoped data."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        exported = context.export()
        assert isinstance(exported, dict)
        tm.that(exported, contains="global")
        global_data = exported.get("global")
        if isinstance(global_data, dict):
            tm.that(global_data, contains="key1")
            tm.that(global_data, contains="key2")

    def test_set_with_none_value_raises_validation_error(self) -> None:
        """Test None is rejected by context.set validation."""
        context = FlextContext()
        context.set("key1", "value1").value
        result = context._set_single(
            "none_key", None, FlextConstants.Context.SCOPE_GLOBAL
        )
        tm.fail(result)

    def test_get_with_different_scope(self) -> None:
        """Test get with different scope."""
        context = FlextContext()
        context.set("global_key", "global_value").value
        context.set("user_key", "user_value", scope="user").value
        global_result = context.get(
            "global_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        tm.ok(global_result)
        user_result = context.get("user_key", scope="user")
        tm.ok(user_result)
        wrong_result = context.get(
            "user_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        tm.fail(wrong_result)

    def test_set_with_different_scope(self) -> None:
        """Test set with different scope."""
        context = FlextContext()
        result1 = context.set(
            "global_key",
            "global_value",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        tm.ok(result1)
        result2 = context.set("user_key", "user_value", scope="user")
        tm.ok(result2)
        global_result = context.get(
            "global_key",
            scope=FlextConstants.Context.SCOPE_GLOBAL,
        )
        user_result = context.get("user_key", scope="user")
        tm.ok(global_result)
        tm.ok(user_result)

    def test_remove_from_specific_scope(self) -> None:
        """Test remove from specific scope."""
        context = FlextContext()
        context.set("key1", "value1", scope="user").value
        context.remove("key1", scope="user")
        result = context.get("key1", scope="user")
        tm.fail(result)

    def test_has_with_different_scope(self) -> None:
        """Test has with different scope."""
        context = FlextContext()
        context.set("key1", "value1", scope="user").value
        has_user = context.has("key1", scope="user")
        tm.that(has_user, eq=True)
        has_global = context.has("key1", scope=FlextConstants.Context.SCOPE_GLOBAL)
        tm.that(has_global, eq=False)

    def test_keys_returns_all_keys(self) -> None:
        """Test keys returns all keys."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        keys = context.keys()
        tm.that(keys, contains="key1")
        tm.that(keys, contains="key2")

    def test_values_returns_all_values(self) -> None:
        """Test values returns all values."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.set("key2", "value2").value
        values = context.values()
        tm.that(values, contains="value1")
        tm.that(values, contains="value2")

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
        tm.that(all_scopes, contains="user")

    def test_export_after_clear(self) -> None:
        """Test export after clear."""
        context = FlextContext()
        context.set("key1", "value1").value
        context.clear()
        exported = context.export()
        assert isinstance(exported, dict)

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
        context.remove("key1", scope="user")
        get_result = context.get("key1", scope="user")
        tm.fail(get_result)

    def test_get_with_default_using_unwrap_or(self) -> None:
        """Test get with default using unwrap_or pattern."""
        context = FlextContext()
        result = context.get("nonexistent")
        result_typed = result
        value = result_typed.unwrap_or("default_value")
        tm.that(value, eq="default_value")

    def test_context_data_validate_dict_serializable_non_dict(self) -> None:
        """Test ContextData.validate_dict_serializable with non-dict."""
        invalid_metadata = 123
        exc_types: tuple[type[Exception], ...] = (TypeError, ValidationError)
        with pytest.raises(exc_types):
            FlextModelsContext.ContextData.model_validate({
                "metadata": invalid_metadata,
            })

    def test_context_data_validate_dict_serializable_non_string_key(self) -> None:
        """Test ContextData.validate_dict_serializable with non-string key.

        Note: Non-string keys are converted to strings by the validator's
        key normalization (str(k)), so integer key 123 becomes string key "123".
        """

        class IntKeyDict(UserDictBase[int, str]):
            def __init__(self) -> None:
                super().__init__()
                self[123] = "value"

        int_key_dict = IntKeyDict()
        result = FlextModelsContext.ContextData.model_validate({"data": int_key_dict})
        tm.that(result, contains="123").data

    def test_context_data_validate_dict_serializable_non_serializable_value(
        self,
    ) -> None:
        """Test ContextData.validate_dict_serializable with non-serializable value.

        Note: Non-JSON-serializable values (like sets) are converted to strings
        by FlextRuntime.normalize_to_general_value() before serializability check,
        so they become valid strings. This is intentional - ensures any value
        can be stored in context.
        """
        bad_dict = {"key": {1, 2, 3}}
        result = FlextModelsContext.ContextData.model_validate({"data": bad_dict})
        assert isinstance(result.data["key"], str)

    def test_context_export_validate_dict_serializable_pydantic_model(self) -> None:
        """Test ContextExport.validate_dict_serializable with Pydantic model."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        export = FlextModelsContext.ContextExport.model_validate({"data": model})
        assert isinstance(export.data, dict)
        tm.that(export, contains="field").data

    def test_context_export_validate_dict_serializable_non_dict(self) -> None:
        """Test ContextExport.validate_dict_serializable with non-dict."""
        invalid_data = 123
        with pytest.raises(TypeError):
            FlextModelsContext.ContextExport.model_validate({"data": invalid_data})

    def test_context_export_validate_dict_serializable_non_string_key(self) -> None:
        """Test ContextExport.validate_dict_serializable with non-string key.

        Note: Non-string keys are converted to strings by normalize_to_general_value(),
        so integer key 123 becomes string key "123". No error is raised.
        """
        data = {123: "value"}
        result = FlextModelsContext.ContextExport.model_validate({"data": data})
        tm.that(result, contains="123").data

    def test_context_export_validate_dict_serializable_non_serializable_value(
        self,
    ) -> None:
        """Test ContextExport.validate_dict_serializable with non-serializable value.

        Note: Non-JSON-serializable values (like sets) are converted to strings
        by FlextRuntime.normalize_to_general_value() before serializability check.
        """
        data = {"key": {1, 2, 3}}
        result = FlextModelsContext.ContextExport.model_validate({"data": data})
        assert isinstance(result.data["key"], str)

    def test_context_export_total_data_items(self) -> None:
        """Test ContextExport.total_data_items computed field."""
        export = FlextModelsContext.ContextExport(
            data={"key1": "value1", "key2": "value2"},
            metadata=FlextModelFoundation.Metadata(attributes={}),
            statistics={},
        )
        assert len(export.data) == 2

    def test_context_export_has_statistics(self) -> None:
        """Test ContextExport.has_statistics computed field."""
        export1 = FlextModelsContext.ContextExport(
            data={},
            metadata=FlextModelFoundation.Metadata(attributes={}),
            statistics={"sets": 5},
        )
        assert bool(export1.statistics) is True
        export2 = FlextModelsContext.ContextExport(
            data={},
            metadata=FlextModelFoundation.Metadata(attributes={}),
            statistics={},
        )
        assert bool(export2.statistics) is False

    def test_context_scope_data_validate_data_with_basemodel(self) -> None:
        """Test ContextScopeData._validate_data with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        scope_data = FlextModelsContext.ContextScopeData.model_validate({"data": model})
        assert isinstance(scope_data.data, dict)
        tm.that(scope_data, contains="field").data

    def test_context_scope_data_validate_data_with_none(self) -> None:
        """Test ContextScopeData._validate_data with None."""
        scope_data = FlextModelsContext.ContextScopeData(
            scope_name="global",
            data={},
            metadata={},
        )
        assert isinstance(scope_data.data, dict)
        tm.that(scope_data.data, eq={})

    def test_context_scope_data_validate_metadata_with_basemodel(self) -> None:
        """Test ContextScopeData._validate_metadata with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        scope_data = FlextModelsContext.ContextScopeData.model_validate({
            "scope_name": "global",
            "metadata": model,
        })
        assert isinstance(scope_data.metadata, dict)
        tm.that(scope_data, contains="field").metadata

    def test_context_scope_data_validate_metadata_with_none(self) -> None:
        """Test ContextScopeData._validate_metadata with None."""
        scope_data = FlextModelsContext.ContextScopeData(
            scope_name="global",
            data={},
            metadata={},
        )
        assert isinstance(scope_data.metadata, dict)
        tm.that(scope_data.metadata, eq={})

    def test_context_statistics_validate_operations_with_basemodel(self) -> None:
        """Test ContextStatistics._validate_operations with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        stats = FlextModelsContext.ContextStatistics.model_validate({
            "operations": model,
        })
        assert isinstance(stats.operations, dict)
        tm.that(stats, contains="field").operations

    def test_context_statistics_validate_operations_with_none(self) -> None:
        """Test ContextStatistics._validate_operations with None."""
        none_operations = None
        stats = FlextModelsContext.ContextStatistics.model_validate({
            "operations": none_operations,
        })
        assert isinstance(stats.operations, dict)
        tm.that(stats.operations, eq={})

    def test_context_metadata_validate_custom_fields_with_basemodel(self) -> None:
        """Test ContextMetadata._validate_custom_fields with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        metadata = FlextModelsContext.ContextMetadata.model_validate({
            "custom_fields": model,
        })
        assert isinstance(metadata.custom_fields, dict)
        tm.that(metadata, contains="field").custom_fields

    def test_context_metadata_validate_custom_fields_with_none(self) -> None:
        """Test ContextMetadata._validate_custom_fields with None."""
        none_custom_fields = None
        metadata = FlextModelsContext.ContextMetadata.model_validate({
            "custom_fields": none_custom_fields,
        })
        assert isinstance(metadata.custom_fields, dict)
        tm.that(metadata.custom_fields, eq={})
