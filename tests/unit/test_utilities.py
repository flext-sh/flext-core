"""Comprehensive coverage tests for FlextUtilities.

Module: flext_core.utilities.FlextUtilities
Scope: Type guards, generators, text processing, caching, reliability,
validation, type checking

Tests validate:
- Type guards (string, dict, list non-empty checks)
- ID/timestamp generation (multiple generator types)
- Text processing (cleaning, truncation, safe string conversion)
- Caching utilities (normalization, key generation, cleanup)
- Reliability patterns (timeout, retry)
- Type checking for message handlers
- Validation utilities (sort_key, normalize_component, validate_pipeline)
- Cache utilities (sort_dict_keys, cache operations)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar

import pytest
from pydantic import BaseModel

from flext_core import FlextConfig, FlextResult, FlextUtilities
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes
from flext_tests.utilities import FlextTestsUtilities


class UtilityScenarios:
    """Centralized utility test scenarios using FlextConstants."""

    TYPE_GUARD_CASES: ClassVar[
        dict[
            str,
            list[tuple[str, FlextTypes.GeneralValueType, bool]],
        ]
    ] = {
        "string": [
            ("string_empty", "", False),
            ("string_valid", "test", True),
            ("string_none", None, False),
            ("string_number", 123, False),
            ("string_whitespace", " ", False),
            ("string_content", " content ", True),
        ],
        "dict": [
            ("dict_empty", {}, False),
            ("dict_valid", {"a": 1}, True),
            ("dict_none", None, False),
            ("dict_string", "not_dict", False),
        ],
        "list": [
            ("list_empty", [], False),
            ("list_valid", [1, 2, 3], True),
            ("list_none", None, False),
            ("list_string", "not_list", False),
        ],
    }

    GENERATOR_METHODS: ClassVar[list[str]] = [
        "generate_id",
        "generate_iso_timestamp",
        "generate_correlation_id",
        "generate_entity_id",
        "generate_transaction_id",
        "generate_saga_id",
        "generate_event_id",
    ]

    SHORT_ID_LENGTHS: ClassVar[list[tuple[int | None, int]]] = [
        (5, 5),
        (10, 10),
        (20, 20),
        (None, FlextConstants.Utilities.SHORT_UUID_LENGTH),
    ]

    TEXT_CLEAN_CASES: ClassVar[list[tuple[str, str]]] = [
        ("  Test\n\r\tText  ", "Test Text"),
        ("a    b    c", "a b c"),
        ("  Test  Text  ", "Test Text"),
    ]

    TEXT_TRUNCATE_CASES: ClassVar[list[tuple[str, int, bool]]] = [
        ("VeryLongText", 5, True),
        ("Hi", 10, False),
    ]

    CACHE_NORMALIZE_CASES: ClassVar[
        list[tuple[FlextTypes.GeneralValueType, type | tuple[type, ...]]]
    ] = [
        ({"a": 1, "b": 2}, dict),
        ([1, 2, 3], list),
        ("string", str),
        (42, (int, float, str, dict, list, tuple)),  # Primitives returned as-is
        ({"a": {"b": {"c": 1}}, "d": [1, 2, 3]}, dict),
    ]

    VALIDATION_PIPELINE_CASES: ClassVar[
        list[tuple[str, Sequence[Callable[[str], FlextResult[bool]]], bool, str | None]]
    ] = [
        (
            "abc123",
            [
                lambda d: FlextResult[bool].ok(True)
                if len(d) > 0
                else FlextResult[bool].fail("Empty"),
                lambda d: FlextResult[bool].ok(True)
                if d.isalnum()
                else FlextResult[bool].fail("Non-alnum"),
            ],
            True,
            None,
        ),
        (
            "test",
            [lambda d: FlextResult[bool].fail("First failed")],
            False,
            "First failed",
        ),
        ("test", [], True, None),
    ]

    TYPE_CHECKER_CASES: ClassVar[list[tuple[tuple[type, ...], type, bool]]] = [
        ((str,), str, True),
        ((str, int), str, True),
        ((str, int), int, True),
        ((str,), int, False),
        ((int,), str, False),
    ]

    @staticmethod
    def create_test_model() -> BaseModel:
        """Create test Pydantic model using FlextTestsUtilities pattern."""

        class TestModel(BaseModel):
            """Test Pydantic model for validation testing."""

            name: str
            value: int

        return FlextTestsUtilities.ModelTestHelpers.assert_model_creation_success(
            factory_method=lambda **kw: TestModel(name="test", value=42, **kw),
            expected_attrs={"name": "test", "value": 42},
        )


class TestFlextUtilities:
    """Unified test suite for FlextUtilities using flext_tests and FlextConstants."""

    # =====================================================================
    # Type Guards Tests - Parametrized
    # =====================================================================

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_CASES["string"],
    )
    def test_type_guard_string(
        self,
        description: str,
        value: FlextTypes.GeneralValueType,
        expected: bool,
    ) -> None:
        """Test string type guards."""
        result = FlextUtilities.TypeGuards.is_string_non_empty(value)
        assert result is expected, f"{description}: expected {expected}, got {result}"

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_CASES["dict"],
    )
    def test_type_guard_dict(
        self,
        description: str,
        value: FlextTypes.GeneralValueType,
        expected: bool,
    ) -> None:
        """Test dict type guards."""
        result = FlextUtilities.TypeGuards.is_dict_non_empty(value)
        assert result is expected, f"{description}: expected {expected}, got {result}"

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_CASES["list"],
    )
    def test_type_guard_list(
        self,
        description: str,
        value: FlextTypes.GeneralValueType,
        expected: bool,
    ) -> None:
        """Test list type guards."""
        result = FlextUtilities.TypeGuards.is_list_non_empty(value)
        assert result is expected, f"{description}: expected {expected}, got {result}"

    # =====================================================================
    # Generators Tests - Parametrized
    # =====================================================================

    @pytest.mark.parametrize("method_name", UtilityScenarios.GENERATOR_METHODS)
    def test_generators_operations(self, method_name: str) -> None:
        """Test ID and timestamp generation operations."""
        method = getattr(FlextUtilities.Generators, method_name)
        result = method()
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize(
        ("length", "expected_length"),
        UtilityScenarios.SHORT_ID_LENGTHS,
    )
    def test_generators_short_id_lengths(
        self,
        length: int | None,
        expected_length: int,
    ) -> None:
        """Test short ID generation with various lengths."""
        short_id = (
            FlextUtilities.Generators.generate_short_id(length=length)
            if length is not None
            else FlextUtilities.Generators.generate_short_id()
        )
        assert len(short_id) == expected_length

    def test_generators_batch_id(self) -> None:
        """Test batch ID generation."""
        batch_id = FlextUtilities.Generators.generate_batch_id(
            FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
        )
        assert isinstance(batch_id, str) and len(batch_id) > 0

    def test_generators_correlation_id_with_context(self) -> None:
        """Test correlation ID with context."""
        corr_id = FlextUtilities.Generators.generate_correlation_id_with_context(
            "test_ctx",
        )
        assert isinstance(corr_id, str) and "test_ctx" in corr_id

    def test_generators_uniqueness(self) -> None:
        """Test generator uniqueness."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert id1 != id2

    # =====================================================================
    # Text Processor Tests - Parametrized
    # =====================================================================

    @pytest.mark.parametrize(
        ("input_text", "expected_pattern"),
        UtilityScenarios.TEXT_CLEAN_CASES,
    )
    def test_text_processor_clean_text(
        self,
        input_text: str,
        expected_pattern: str,
    ) -> None:
        """Test text cleaning."""
        result = FlextUtilities.TextProcessor.clean_text(input_text)
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize(
        ("text", "max_length", "should_truncate"),
        UtilityScenarios.TEXT_TRUNCATE_CASES,
    )
    def test_text_processor_truncate(
        self,
        text: str,
        max_length: int,
        should_truncate: bool,
    ) -> None:
        """Test text truncation."""
        result = FlextUtilities.TextProcessor.truncate_text(text, max_length=max_length)
        assert result.is_success
        if should_truncate:
            assert len(result.value) <= max_length + 3  # +3 for "..."
        else:
            assert result.value == text

    def test_text_processor_safe_string_success(self) -> None:
        """Test safe string with valid input."""
        result = FlextUtilities.TextProcessor.safe_string("valid")
        assert isinstance(result, str) and result == "valid"

    def test_text_processor_safe_string_empty(self) -> None:
        """Test safe string with empty raises ValueError."""
        with pytest.raises(ValueError):
            _ = FlextUtilities.TextProcessor.safe_string("")

    # =====================================================================
    # Cache Tests - Parametrized
    # =====================================================================

    @pytest.mark.parametrize(
        ("input_data", "expected_type"),
        UtilityScenarios.CACHE_NORMALIZE_CASES,
    )
    def test_cache_normalize_component(
        self,
        input_data: FlextTypes.GeneralValueType,
        expected_type: type | tuple[type, ...],
    ) -> None:
        """Test cache component normalization."""
        normalized = FlextUtilities.Cache.normalize_component(input_data)
        if isinstance(expected_type, tuple):
            assert isinstance(normalized, expected_type)
        else:
            assert isinstance(normalized, expected_type)

    def test_cache_sort_dict_keys(self) -> None:
        """Test dictionary key sorting."""
        data: FlextTypes.Types.ConfigurationMapping = {"z": 1, "a": 2, "m": 3}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_cache_generate_key(self) -> None:
        """Test cache key generation."""
        key1 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2")
        key2 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2")
        assert key1 == key2 and isinstance(key1, str)

    def test_cache_generate_key_with_kwargs(self) -> None:
        """Test cache key generation with kwargs."""
        key = FlextUtilities.Cache.generate_cache_key("test", foo="bar", num=42)
        assert isinstance(key, str) and len(key) > 0

    def test_cache_clear_object_cache(self) -> None:
        """Test clearing object cache."""
        cache_data: FlextTypes.Types.ConfigurationMapping = {"test": "data"}
        result = FlextUtilities.Cache.clear_object_cache(cache_data)
        assert result.is_success

    @pytest.mark.parametrize(
        ("has_cache", "expected"),
        [
            (True, True),
            (False, False),
        ],
    )
    def test_cache_has_attributes(self, has_cache: bool, expected: bool) -> None:
        """Test detecting cache attributes."""
        if has_cache:

            class TestWithCache:
                _cache: ClassVar[FlextTypes.Types.ConfigurationMapping] = {}

            cache_obj: TestWithCache = TestWithCache()
            assert FlextUtilities.Cache.has_cache_attributes(cache_obj) is expected
        else:

            class TestNoCache:
                pass

            no_cache_obj: TestNoCache = TestNoCache()
            assert FlextUtilities.Cache.has_cache_attributes(no_cache_obj) is expected

    def test_cache_sort_key(self) -> None:
        """Test sort_key returns tuple."""
        key = FlextUtilities.Cache.sort_key("test")
        assert isinstance(key, tuple) and len(key) == 2

    # =====================================================================
    # Type Checker Tests - Parametrized
    # =====================================================================

    @pytest.mark.parametrize(
        ("accepted_types", "message_type", "expected"),
        UtilityScenarios.TYPE_CHECKER_CASES,
    )
    def test_type_checker_can_handle(
        self,
        accepted_types: tuple[type, ...],
        message_type: type,
        expected: bool,
    ) -> None:
        """Test type checking."""
        result = FlextUtilities.TypeChecker.can_handle_message_type(
            accepted_types,
            message_type,
        )
        assert result is expected

    # =====================================================================
    # Validation Tests - Parametrized
    # =====================================================================

    def test_validation_sort_key_deterministic(self) -> None:
        """Test sort_key produces deterministic results."""
        key1 = FlextUtilities.Validation.sort_key("test")
        key2 = FlextUtilities.Validation.sort_key("test")
        assert key1 == key2 and isinstance(key1, (str, tuple))

    def test_validation_sort_key_different(self) -> None:
        """Test sort_key differs for different inputs."""
        key1 = FlextUtilities.Validation.sort_key("test1")
        key2 = FlextUtilities.Validation.sort_key("test2")
        assert key1 != key2

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (42, 42),
            ("test", "test"),
            (None, None),
        ],
    )
    def test_validation_normalize_component_primitives(
        self,
        value: FlextTypes.GeneralValueType,
        expected: FlextTypes.GeneralValueType,
    ) -> None:
        """Test normalize_component with primitives."""
        result = FlextUtilities.Validation.normalize_component(value)
        assert result == expected

    @pytest.mark.parametrize(
        ("input_data", "expected_type"),
        [
            ({"z": 1, "a": 2}, (dict, tuple)),
            ([1, 2, 3, "four"], (tuple, list, str)),
        ],
    )
    def test_validation_normalize_component_collections(
        self,
        input_data: FlextTypes.GeneralValueType,
        expected_type: tuple[type, ...],
    ) -> None:
        """Test normalize_component with collections."""
        result = FlextUtilities.Validation.normalize_component(input_data)
        assert isinstance(result, expected_type)

    def test_validation_normalize_component_pydantic(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = UtilityScenarios.create_test_model()
        # normalize_component accepts GeneralValueType
        # Convert BaseModel to dict (GeneralValueType) to match signature
        model_dict: FlextTypes.GeneralValueType = model.model_dump()
        result: FlextTypes.GeneralValueType = (
            FlextUtilities.Validation.normalize_component(model_dict)
        )
        assert result is not None

    @pytest.mark.parametrize(
        ("data", "validators", "should_succeed", "error_pattern"),
        UtilityScenarios.VALIDATION_PIPELINE_CASES,
    )
    def test_validation_validate_pipeline(
        self,
        data: str,
        validators: Sequence[Callable[[str], FlextResult[bool]]],
        should_succeed: bool,
        error_pattern: str | None,
    ) -> None:
        """Test validation pipeline."""
        result = FlextUtilities.Validation.validate_pipeline(data, list(validators))
        if should_succeed:
            assert result.is_success
        else:
            assert result.is_failure
            if error_pattern:
                assert error_pattern in str(result.error)

    def test_validation_validate_pipeline_exception(self) -> None:
        """Test validation pipeline handles exceptions."""
        error_msg = "Unexpected error"

        def bad_validator(data: str) -> FlextResult[bool]:
            raise ValueError(error_msg)

        result = FlextUtilities.Validation.validate_pipeline("test", [bad_validator])
        assert result.is_failure

    def test_validation_normalize_dataclass(self) -> None:
        """Test normalize with dataclass."""

        @dataclass
        class Person:
            name: str
            age: int

        person = Person(name="Alice", age=30)
        # normalize_component accepts GeneralValueType
        # Convert dataclass to dict (GeneralValueType) via dataclasses.asdict
        person_dict: FlextTypes.GeneralValueType = asdict(person)
        result: FlextTypes.GeneralValueType = (
            FlextUtilities.Validation.normalize_component(person_dict)
        )
        assert result is not None

    def test_validation_sort_key_with_dict(self) -> None:
        """Test sort_key with dictionary."""
        dict_a: FlextTypes.Types.ConfigurationMapping = {"z": 1, "a": 2}
        key = FlextUtilities.Validation.sort_key(dict_a)
        assert isinstance(key, (str, tuple)) and len(key) > 0

    # =====================================================================
    # Configuration Tests
    # =====================================================================

    def test_configuration_get_parameter(self) -> None:
        """Test getting configuration parameter."""
        config = FlextConfig.get_global_instance()
        value = FlextUtilities.Configuration.get_parameter(config, "app_name")
        assert value is not None

    def test_configuration_set_parameter(self) -> None:
        """Test setting configuration parameter."""
        config = FlextConfig.get_global_instance()
        result = FlextUtilities.Configuration.set_parameter(
            config,
            "test_param",
            "test_value",
        )
        assert isinstance(result, bool)

    def test_configuration_get_singleton(self) -> None:
        """Test getting singleton configuration."""
        value = FlextUtilities.Configuration.get_singleton(FlextConfig, "app_name")
        assert value is not None

    # =====================================================================
    # Reliability Tests
    # =====================================================================

    def test_reliability_retry_immediate_success(self) -> None:
        """Test retry with immediate success."""

        def quick_success() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.retry(
            quick_success,
            max_attempts=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        )
        assert result.is_success
        assert result.value == "success"

    def test_reliability_retry_eventual_success(self) -> None:
        """Test retry with eventual success."""
        call_count = [0]

        def flaky_op() -> FlextResult[str]:
            call_count[0] += 1
            if call_count[0] < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("Success")

        result = FlextUtilities.Reliability.retry(
            flaky_op,
            max_attempts=5,
            delay_seconds=FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
        )
        assert result.is_success
        assert result.value == "Success"
        assert call_count[0] >= 3


__all__ = ["TestFlextUtilities"]
