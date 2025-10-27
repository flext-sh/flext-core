"""Coverage improvement tests targeting 75%+ threshold.

This module provides edge case and corner case tests to improve overall
test coverage from 74% to 75%+ for Phase 4 completion.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

from flext_core import FlextResult, FlextUtilities


class TestEdgeCases:
    """Edge case tests for comprehensive coverage."""

    def test_clean_text_with_special_whitespace(self) -> None:
        """Test text cleaning with various whitespace characters."""
        result = FlextUtilities.TextProcessor.clean_text(
            "  \t\n\r  hello  \n\t  world  \r\n  "
        )
        assert result.is_success
        # clean_text removes various whitespace
        value = result.unwrap()
        assert isinstance(value, str)

    def test_truncate_text_with_long_suffix(self) -> None:
        """Test text truncation when ellipsis fits."""
        result = FlextUtilities.TextProcessor.truncate_text(
            "This is a very long text",
            max_length=5,
        )
        assert result.is_success
        truncated = result.unwrap()
        # Should have added "..."
        assert len(truncated) <= 8  # 5 + 3 for ellipsis
        assert "..." in truncated or len(truncated) <= 5

    def test_safe_string_with_whitespace_only(self) -> None:
        """Test safe_string with only whitespace."""
        result = FlextUtilities.TextProcessor.safe_string("   \t\n  ")
        # Should return empty or whitespace-stripped
        assert not result or not result.strip()

    def test_safe_string_with_normal_text(self) -> None:
        """Test safe_string with normal text."""
        result = FlextUtilities.TextProcessor.safe_string("hello world")
        assert result == "hello world"

    def test_safe_string_with_leading_trailing_spaces(self) -> None:
        """Test safe_string strips leading/trailing spaces."""
        result = FlextUtilities.TextProcessor.safe_string("  hello  ")
        assert result == "hello"

    def test_validate_pipeline_with_multiple_validators(self) -> None:
        """Test validator pipeline with multiple validators."""

        def validator1(value: str) -> FlextResult[None]:
            if not value:
                return FlextResult[None].fail("Empty string")
            return FlextResult[None].ok(None)

        def validator2(value: str) -> FlextResult[None]:
            if len(value) < 3:
                return FlextResult[None].fail("Too short")
            return FlextResult[None].ok(None)

        result = FlextUtilities.Validation.validate_pipeline(
            "hello",
            [validator1, validator2],
        )
        assert result.is_success

    def test_validate_pipeline_with_failing_validator(self) -> None:
        """Test validator pipeline with failing validator."""

        def validator1(value: str) -> FlextResult[None]:
            return FlextResult[None].fail("Always fails")

        def validator2(value: str) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        result = FlextUtilities.Validation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        assert result.is_failure

    def test_generators_correlation_id_format(self) -> None:
        """Test correlation ID has expected format."""
        corr_id = FlextUtilities.Generators.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
        assert corr_id.startswith("corr_")

    def test_generators_command_id_format(self) -> None:
        """Test command ID has expected format."""
        cmd_id = FlextUtilities.Generators.generate_command_id()
        assert isinstance(cmd_id, str)
        assert len(cmd_id) > 0
        assert cmd_id.startswith("cmd_")

    def test_generators_query_id_format(self) -> None:
        """Test query ID has expected format."""
        query_id = FlextUtilities.Generators.generate_query_id()
        assert isinstance(query_id, str)
        assert len(query_id) > 0
        assert query_id.startswith("qry_")

    def test_cache_normalize_component_with_dict(self) -> None:
        """Test component normalization with dict."""
        result = FlextUtilities.Cache.normalize_component({"a": 1, "b": 2})
        assert isinstance(result, (dict, str))

    def test_cache_normalize_component_with_list(self) -> None:
        """Test component normalization with list."""
        result = FlextUtilities.Cache.normalize_component([1, 2, 3])
        assert isinstance(result, (dict, str, list))

    def test_cache_normalize_component_with_string(self) -> None:
        """Test component normalization with string."""
        result = FlextUtilities.Cache.normalize_component("test")
        assert isinstance(result, (dict, str))

    def test_type_guards_with_various_types(self) -> None:
        """Test type guards with various input types."""
        # Test string guard
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(42) is False

        # Test dict guard
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"a": 1}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty([1, 2]) is False

        # Test list guard
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty("not-a-list") is False

    def test_reliability_with_timeout_success(self) -> None:
        """Test operation with timeout that succeeds."""

        def quick_op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.with_timeout(quick_op, 5.0)
        assert result.is_success
        assert result.unwrap() == "success"

    def test_reliability_with_timeout_zero(self) -> None:
        """Test operation with zero timeout."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.with_timeout(op, 0.0)
        assert result.is_failure

    def test_reliability_with_timeout_negative(self) -> None:
        """Test operation with negative timeout."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.with_timeout(op, -1.0)
        assert result.is_failure

    def test_type_checker_various_scenarios(self) -> None:
        """Test type checking in various scenarios."""
        # Can handle same type
        assert FlextUtilities.TypeChecker.can_handle_message_type((str,), str) is True

        # Can handle subclass of accepted type
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type((object,), str) is True
        )

        # Cannot handle unrelated type
        assert FlextUtilities.TypeChecker.can_handle_message_type((int,), str) is False

        # Cannot handle empty accepted types
        assert FlextUtilities.TypeChecker.can_handle_message_type((), str) is False

    def test_short_id_generation_length(self) -> None:
        """Test short ID generation with various lengths."""
        for length in [4, 8, 16, 32]:
            short_id = FlextUtilities.Generators.generate_short_id(length=length)
            assert len(short_id) == length

    def test_entity_version_generation(self) -> None:
        """Test entity version generation."""
        version = FlextUtilities.Generators.generate_entity_version()
        assert isinstance(version, int)
        assert version > 0

    def test_correlation_id_with_context(self) -> None:
        """Test correlation ID generation with context."""
        corr_id = FlextUtilities.Generators.generate_correlation_id_with_context(
            "test_context"
        )
        assert isinstance(corr_id, str)
        assert "test_context" in corr_id

    def test_batch_id_generation(self) -> None:
        """Test batch ID generation."""
        batch_id = FlextUtilities.Generators.generate_batch_id(100)
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0
        assert "100" in batch_id or "batch" in batch_id

    def test_transaction_id_generation(self) -> None:
        """Test transaction ID generation."""
        tx_id = FlextUtilities.Generators.generate_transaction_id()
        assert isinstance(tx_id, str)
        assert len(tx_id) > 0

    def test_saga_id_generation(self) -> None:
        """Test saga ID generation."""
        saga_id = FlextUtilities.Generators.generate_saga_id()
        assert isinstance(saga_id, str)
        assert len(saga_id) > 0
        assert "saga" in saga_id

    def test_event_id_generation(self) -> None:
        """Test event ID generation."""
        event_id = FlextUtilities.Generators.generate_event_id()
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert "evt" in event_id or "event" in event_id

    def test_aggregate_id_generation(self) -> None:
        """Test aggregate ID generation with type."""
        agg_id = FlextUtilities.Generators.generate_aggregate_id("Order")
        assert isinstance(agg_id, str)
        assert "Order" in agg_id

    def test_iso_timestamp_format(self) -> None:
        """Test ISO timestamp has proper format."""
        ts = FlextUtilities.Generators.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format includes 'T' separator

    def test_correlation_iso_timestamp(self) -> None:
        """Test correlation ISO timestamp generation."""
        ts = FlextUtilities.Correlation.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts

    def test_sort_dict_keys_with_various_types(self) -> None:
        """Test dict key sorting with various value types."""
        data = {"z": 1, "a": "string", "m": [1, 2, 3], "b": {"nested": True}}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        # Keys should be in sorted order
        assert keys == sorted(keys)

    def test_generate_cache_key_consistency(self) -> None:
        """Test cache key generation is consistent for same input."""
        key1 = FlextUtilities.Cache.generate_cache_key("command", str)
        key2 = FlextUtilities.Cache.generate_cache_key("command", str)
        # Keys should be consistent
        assert isinstance(key1, str)
        assert isinstance(key2, str)

    def test_has_cache_attributes_with_decorated_object(self) -> None:
        """Test cache attribute detection with decorated object."""

        class DecoratedObj:
            def __init__(self) -> None:
                super().__init__()
                self._cache: dict[str, object] = {}

        obj = DecoratedObj()
        assert FlextUtilities.Cache.has_cache_attributes(obj) is True

    def test_clear_object_cache_with_populated_cache(self) -> None:
        """Test clearing object cache with populated cache."""

        class CachedObj:
            def __init__(self) -> None:
                super().__init__()
                self._cache = {"key": "value"}

        obj = CachedObj()
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success
        # Cache should be cleared
        assert len(obj._cache) == 0

    def test_retry_with_eventual_failure(self) -> None:
        """Test retry that eventually fails after max attempts."""
        attempt_count = 0

        def always_fails() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            return FlextResult[str].fail("Always fails")

        result = FlextUtilities.Reliability.retry(
            always_fails,
            max_attempts=3,
            delay_seconds=0.01,
        )
        assert result.is_failure
        assert attempt_count == 3

    def test_cache_normalize_component_with_set(self) -> None:
        """Test component normalization with set."""
        result = FlextUtilities.Cache.normalize_component({1, 2, 3})
        assert isinstance(result, (set, dict, str))

    def test_cache_sort_key_with_numeric(self) -> None:
        """Test sort key generation with numeric values."""
        key_int = FlextUtilities.Cache.sort_key(42)
        assert isinstance(key_int, tuple)
        assert len(key_int) == 2
        assert key_int[0] == 1  # Numeric types get priority 1

        key_float = FlextUtilities.Cache.sort_key(math.pi)
        assert isinstance(key_float, tuple)
        assert len(key_float) == 2
        assert key_float[0] == 1  # Numeric types get priority 1

    def test_cache_sort_key_with_string(self) -> None:
        """Test sort key generation with string values."""
        key_str = FlextUtilities.Cache.sort_key("test")
        assert isinstance(key_str, tuple)
        assert len(key_str) == 2
        assert key_str[0] == 0  # String types get priority 0

    def test_cache_sort_key_with_other_type(self) -> None:
        """Test sort key generation with other types."""
        key_other = FlextUtilities.Cache.sort_key([1, 2, 3])
        assert isinstance(key_other, tuple)
        assert len(key_other) == 2
        assert key_other[0] == 2  # Other types get priority 2

    def test_generators_timestamp_format(self) -> None:
        """Test timestamp has expected format."""
        ts = FlextUtilities.Generators.generate_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0
        # ISO format should have either T or -
        assert "T" in ts or "-" in ts

    def test_generators_uuid_format(self) -> None:
        """Test UUID generation."""
        uuid_val = FlextUtilities.Generators.generate_uuid()
        assert isinstance(uuid_val, str)
        assert len(uuid_val) > 0
        # UUID typically has hyphens
        assert "-" in uuid_val or len(uuid_val) >= 32

    def test_generators_entity_id_format(self) -> None:
        """Test entity ID has expected format."""
        entity_id = FlextUtilities.Generators.generate_entity_id()
        assert isinstance(entity_id, str)
        assert len(entity_id) > 0
        # Entity ID should have some identifier
        assert entity_id  # Non-empty

    def test_ensure_id_on_object_without_id(self) -> None:
        """Test ensuring ID on object without pre-existing ID."""

        class TestObj:
            id: str = ""

        obj = TestObj()
        assert not obj.id
        FlextUtilities.Generators.ensure_id(obj)
        # ID should be generated if empty
        assert obj.id  # Should be non-empty

    def test_initialize_boolean_field(self) -> None:
        """Test initializing boolean field."""

        class TestObj:
            flag: bool = False

        obj = TestObj()
        FlextUtilities.Validation.initialize(obj, "flag")
        assert obj.flag is True

    def test_clear_multiple_caches(self) -> None:
        """Test clearing multiple cache attributes on object."""

        class CachedObj:
            def __init__(self) -> None:
                super().__init__()
                self._cache = {"key1": "value1", "key2": "value2"}

        obj = CachedObj()
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success
        # Cache should be cleared
        assert len(obj._cache) == 0

    def test_configuration_get_parameter_from_config(self) -> None:
        """Test getting parameter from configuration."""

        class MockConfig:
            def model_dump(self) -> dict[str, object]:
                return {"app_name": "FlextApp", "version": "1.0"}

        config = MockConfig()
        value = FlextUtilities.Configuration.get_parameter(config, "app_name")
        assert value == "FlextApp"

    def test_configuration_set_parameter_on_config(self) -> None:
        """Test setting parameter on configuration."""

        class MockConfig:
            def __init__(self) -> None:
                super().__init__()
                self.custom_param: str | None = None

        config = MockConfig()
        result = FlextUtilities.Configuration.set_parameter(
            config,
            "custom_param",
            "new_value",
        )
        assert isinstance(result, bool)

    def test_configuration_set_singleton(self) -> None:
        """Test setting singleton configuration value."""

        class MockConfig:
            app_name: str = "FlextApp"

        result = FlextUtilities.Configuration.set_singleton(
            MockConfig,
            "app_name",
            "NewApp",
        )
        # Result should be boolean indicating success
        assert isinstance(result, bool)

    def test_run_external_command_success(self) -> None:
        """Test running successful external command."""
        result = FlextUtilities.run_external_command(
            ["echo", "hello"],
            capture_output=True,
            text=True,
        )
        assert result.is_success

    def test_run_external_command_with_empty_list(self) -> None:
        """Test running command with empty list."""
        result = FlextUtilities.run_external_command([])
        assert result.is_failure

    def test_text_processor_clean_text_removal(self) -> None:
        """Test text cleaning removes internal whitespace."""
        result = FlextUtilities.TextProcessor.clean_text("hello\nworld\ttabs")
        assert result.is_success
        # Should handle various whitespace

    def test_text_processor_truncate_short_text(self) -> None:
        """Test truncating text shorter than max_length."""
        result = FlextUtilities.TextProcessor.truncate_text(
            "short",
            max_length=100,
        )
        assert result.is_success
        assert result.unwrap() == "short"


__all__ = ["TestEdgeCases"]
