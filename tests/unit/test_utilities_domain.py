"""Tests for u.Domain - Domain entity and value object operations.

Module: flext_core._utilities.domain
Scope: u.Domain - entity/value object comparison, hashing, validation

Tests u.Domain functionality including:
- Entity comparison by ID (compare_entities_by_id)
- Entity hashing by ID (hash_entity_by_id)
- Value object comparison by value (compare_value_objects_by_value)
- Value object hashing by value (hash_value_object_by_value)
- Entity ID validation (validate_entity_has_id)
- Value object immutability validation (validate_value_object_immutable)

Uses Python 3.13 patterns, FlextTestsUtilities, constants (c), types (t),
utilities (u), protocols (p), models (m) extensively for maximum code reuse.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from flext_tests import t, u
from pydantic import BaseModel

from tests.constants import TestsFlextConstants
from tests.models import TestsFlextModels

ComplexValue = TestsFlextModels.Core.ComplexValue
CustomEntity = TestsFlextModels.Core.CustomEntity
DomainTestEntity = TestsFlextModels.Core.DomainTestEntity
DomainTestValue = TestsFlextModels.Core.DomainTestValue
ImmutableObj = TestsFlextModels.Core.ImmutableObj
MutableObj = TestsFlextModels.Core.MutableObj
NoConfigNoSetattr = TestsFlextModels.Core.NoConfigNoSetattr
NoDict = TestsFlextModels.Core.NoDict
NoSetattr = TestsFlextModels.Core.NoSetattr
SimpleValue = TestsFlextModels.Core.SimpleValue


# Module-level helper functions (must be defined before class for @pytest.mark.parametrize)
type TestPayload = t.Tests.PayloadValue
type TestCaseMap = Mapping[str, TestPayload]
type InputPayloadMap = dict[str, object]


def _build_domain_test_entity(
    *,
    name: str,
    value: TestPayload,
) -> TestsFlextModels.Core.DomainTestEntity:
    return DomainTestEntity(name=name, value=cast("int", value))


def _convert_to_general_value(obj: object) -> TestPayload:
    """Convert object to t.GeneralValueType (handles Pydantic models).

    Args:
        obj: Object to convert (Pydantic model, dict, list, or primitive)

    Returns:
        t.GeneralValueType-compatible value

    """
    if isinstance(obj, BaseModel):
        # Convert Pydantic model to dict
        return obj.model_dump()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        # Recursively convert dict values
        return cast(
            "TestPayload",
            {str(k): _convert_to_general_value(v) for k, v in obj.items()},
        )
    if isinstance(obj, (list, tuple)):
        # Recursively convert list items
        return cast(
            "TestPayload",
            [_convert_to_general_value(item) for item in obj],
        )
    # For other objects, convert to string
    return str(obj)


def _convert_expected_result(
    expected: object,
) -> TestPayload:
    """Convert expected result to t.GeneralValueType (handles type objects).

    Args:
        expected: Expected result (type, value, or other)

    Returns:
        t.GeneralValueType-compatible value

    """
    if isinstance(expected, type):
        # Type objects are not t.GeneralValueType, convert to string
        return expected.__name__
    if isinstance(expected, (str, int, float, bool, type(None))):
        return expected
    # For other objects, convert to string
    return str(expected)


def _require_payload_str(value: TestPayload) -> str:
    if isinstance(value, str):
        return value
    msg = f"Expected str payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _require_payload_mapping(value: TestPayload) -> Mapping[str, TestPayload]:
    if isinstance(value, Mapping):
        return value
    msg = f"Expected mapping payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _as_test_payload(value: object) -> TestPayload:
    return cast("TestPayload", value)


def _as_payload_map(value: InputPayloadMap) -> Mapping[str, TestPayload]:
    return cast("Mapping[str, TestPayload]", value)


def create_compare_entities_cases() -> list[TestCaseMap]:
    """Create test cases for entity comparison using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create entities using u.Tests.DomainHelpers with batch creation
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
            TestsFlextConstants.TestDomain.ENTITY_NAME_BOB,
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
            TestsFlextConstants.TestDomain.ENTITY_VALUE_20,
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, False, True],  # alice_no_id without ID
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    alice_entity, bob_entity, alice_no_id = entities
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
        count=TestsFlextConstants.TestDomain.VALUE_COUNT_5,
        value_class=DomainTestValue,
    )
    custom1 = CustomEntity(TestsFlextConstants.TestDomain.CUSTOM_ID_1)
    custom2 = CustomEntity(TestsFlextConstants.TestDomain.CUSTOM_ID_1)

    # Pass objects directly (domain methods expect real objects, not dicts)
    input_data_same_id: InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": alice_entity,
    }
    input_data_different_id: InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": bob_entity,
    }
    input_data_different_type: InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": value_obj,
    }
    input_data_no_id: InputPayloadMap = {
        "entity_a": alice_no_id,
        "entity_b": bob_entity,
    }
    input_data_custom: InputPayloadMap = {
        "entity_a": custom1,
        "entity_b": custom2,
    }

    return cast(
        "list[TestCaseMap]",
        [
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="compare_entities_by_id",
                description="same_id",
                input_data=_as_payload_map(input_data_same_id),
                expected_result=True,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="compare_entities_by_id",
                description="different_id",
                input_data=_as_payload_map(input_data_different_id),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="compare_entities_by_id",
                description="different_type",
                input_data=_as_payload_map(input_data_different_type),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="compare_entities_by_id",
                description="no_id",
                input_data=_as_payload_map(input_data_no_id),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="compare_entities_by_id",
                description="custom_id_attr",
                input_data=_as_payload_map(input_data_custom),
                expected_result=True,
                id_attr="custom_id",
            ),
        ],
    )


def create_hash_entity_cases() -> list[TestCaseMap]:
    """Create test cases for entity hashing using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create entities using batch helper
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, True],  # alice_no_id without ID
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    alice_entity, alice_no_id = entities
    custom = CustomEntity(TestsFlextConstants.TestDomain.CUSTOM_ID_1)

    # Pass objects directly (domain methods expect real objects, not dicts)
    input_data_with_id: InputPayloadMap = {
        "entity": alice_entity,
    }
    input_data_no_id: InputPayloadMap = {
        "entity": alice_no_id,
    }
    input_data_custom: InputPayloadMap = {
        "entity": custom,
    }

    return cast(
        "list[TestCaseMap]",
        [
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="hash_entity_by_id",
                description="with_id",
                input_data=_as_payload_map(input_data_with_id),
                expected_result=_as_test_payload(int),
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="hash_entity_by_id",
                description="no_id",
                input_data=_as_payload_map(input_data_no_id),
                expected_result=_as_test_payload(int),
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="hash_entity_by_id",
                description="custom_id_attr",
                input_data=_as_payload_map(input_data_custom),
                expected_result=_as_test_payload(int),
                id_attr="custom_id",
            ),
        ],
    )


def create_compare_value_objects_cases() -> list[TestCaseMap]:
    """Create test cases for value object comparison using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create value objects using batch helper
    value_objs = u.Tests.DomainHelpers.create_test_value_objects_batch(
        data_list=[
            TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
            TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
        ],
        count_list=[
            TestsFlextConstants.TestDomain.VALUE_COUNT_5,
            TestsFlextConstants.TestDomain.VALUE_COUNT_10,
        ],
        value_class=DomainTestValue,
    )
    value1, value2 = value_objs
    alice_entity = u.Tests.DomainHelpers.create_test_entity_instance(
        name=TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
        value=TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
        entity_class=_build_domain_test_entity,
    )
    simple1 = SimpleValue(TestsFlextConstants.TestDomain.VALUE_DATA_TEST)
    simple2 = SimpleValue(TestsFlextConstants.TestDomain.VALUE_DATA_TEST)
    bad1 = u.Tests.BadObjects.BadModelDump()
    bad2 = u.Tests.BadObjects.BadModelDump()
    no_dict1 = NoDict(TestsFlextConstants.TestDomain.VALUE_COUNT_5)
    no_dict2 = NoDict(TestsFlextConstants.TestDomain.VALUE_COUNT_5)

    # Pass objects directly (domain methods expect real objects, not dicts)
    input_data_list: list[InputPayloadMap] = [
        {
            "obj_a": value1,
            "obj_b": value1,
        },
        {
            "obj_a": value1,
            "obj_b": value2,
        },
        {
            "obj_a": value1,
            "obj_b": alice_entity,
        },
        {
            "obj_a": simple1,
            "obj_b": simple2,
        },
        {
            "obj_a": bad1,
            "obj_b": bad2,
        },
        {
            "obj_a": no_dict1,
            "obj_b": no_dict2,
        },
    ]

    return cast(
        "list[TestCaseMap]",
        u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
            operation="compare_value_objects_by_value",
            descriptions=[
                "same_values",
                "different_values",
                "different_type",
                "no_model_dump",
                "model_dump_exception",
                "no_dict",
            ],
            input_data_list=cast("list[Mapping[str, TestPayload]]", input_data_list),
            expected_results=[
                True,
                False,
                False,
                True,
                _as_test_payload(bool),
                True,
            ],
        ),
    )


def create_hash_value_object_cases() -> list[TestCaseMap]:
    """Create test cases for value object hashing using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create value object using helper
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
        count=TestsFlextConstants.TestDomain.VALUE_COUNT_5,
        value_class=DomainTestValue,
    )
    simple_obj = SimpleValue(TestsFlextConstants.TestDomain.VALUE_DATA_TEST)
    bad_obj = u.Tests.BadObjects.BadModelDump()
    # Convert tuple to list for ComplexValue (expects list[str])
    complex_items_list: list[str] = list(
        TestsFlextConstants.TestDomain.COMPLEX_ITEMS,
    )
    complex_obj = ComplexValue(
        TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
        complex_items_list,
    )
    no_dict_obj = NoDict(TestsFlextConstants.TestDomain.VALUE_COUNT_5)

    # Pass objects directly (domain methods expect real objects, not dicts)
    input_data_list_hash: list[InputPayloadMap] = [
        {"obj": value_obj},
        {"obj": simple_obj},
        {"obj": bad_obj},
        {"obj": complex_obj},
        {"obj": no_dict_obj},
    ]

    return cast(
        "list[TestCaseMap]",
        u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
            operation="hash_value_object_by_value",
            descriptions=[
                "with_model_dump",
                "no_model_dump",
                "model_dump_exception",
                "non_hashable_values",
                "no_dict",
            ],
            input_data_list=cast(
                "list[Mapping[str, TestPayload]]", input_data_list_hash,
            ),
            expected_results=[
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
            ],
        ),
    )


def create_validate_entity_has_id_cases() -> list[TestCaseMap]:
    """Create test cases for entity ID validation using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create entities using batch helper
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
            TestsFlextConstants.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
            TestsFlextConstants.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, True],  # alice_no_id without ID
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    alice_entity, alice_no_id = entities
    custom = CustomEntity(TestsFlextConstants.TestDomain.CUSTOM_ID_1)

    # Pass objects directly (domain methods expect real objects, not dicts)
    input_data_has_id: InputPayloadMap = {
        "entity": alice_entity,
    }
    input_data_no_id_validate: InputPayloadMap = {
        "entity": alice_no_id,
    }
    input_data_custom_validate: InputPayloadMap = {
        "entity": custom,
    }

    return cast(
        "list[TestCaseMap]",
        [
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="validate_entity_has_id",
                description="has_id",
                input_data=_as_payload_map(input_data_has_id),
                expected_result=True,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="validate_entity_has_id",
                description="no_id",
                input_data=_as_payload_map(input_data_no_id_validate),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Tests.TestCaseHelpers.create_operation_test_case(
                operation="validate_entity_has_id",
                description="custom_attr",
                input_data=_as_payload_map(input_data_custom_validate),
                expected_result=True,
                id_attr="custom_id",
            ),
        ],
    )


def create_validate_value_object_immutable_cases() -> list[TestCaseMap]:
    """Create test cases for immutability validation using constants."""
    # Use TestsFlextConstants.TestDomain directly (c is imported from tests.helpers)
    # Create value object using helper
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=TestsFlextConstants.TestDomain.VALUE_DATA_TEST,
        count=TestsFlextConstants.TestDomain.VALUE_COUNT_5,
        value_class=DomainTestValue,
    )
    mutable_obj = MutableObj(TestsFlextConstants.TestDomain.VALUE_COUNT_5)
    immutable_obj = ImmutableObj(TestsFlextConstants.TestDomain.VALUE_COUNT_5)
    bad_config_obj = u.Tests.BadObjects.BadConfig()
    no_config_obj = NoConfigNoSetattr()
    no_setattr_obj = NoSetattr()

    return cast(
        "list[TestCaseMap]",
        u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
            operation="validate_value_object_immutable",
            descriptions=[
                "frozen",
                "mutable",
                "custom_setattr",
                "config_exception",
                "no_config_no_setattr",
                "no_setattr",
            ],
            input_data_list=cast(
                "list[Mapping[str, TestPayload]]",
                [
                    {"obj": value_obj},
                    {"obj": mutable_obj},
                    {"obj": immutable_obj},
                    {"obj": bad_config_obj},
                    {"obj": no_config_obj},
                    {"obj": no_setattr_obj},
                ],
            ),
            expected_results=[
                True,
                False,
                True,
                False,
                False,
                False,
            ],
        ),
    )


class TestuDomain:
    """Comprehensive tests for u.Domain using FlextTestsUtilities and constants extensively."""

    # ============================================================================
    # Parametrized Tests - Using FlextTestsUtilities Extensively
    # ============================================================================

    @pytest.mark.parametrize(
        "test_case",
        create_compare_entities_cases(),
        ids=lambda case: f"compare_entities_{case['description']}",
    )
    def test_compare_entities_by_id(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test compare_entities_by_id using FlextTestsUtilities."""
        # Type narrowing: execute_domain_operation returns object, but we know it's t.GeneralValueType
        # Cast lambda return type to t.GeneralValueType for type checker
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(
                test_case.get("id_attr", "unique_id"),
            ),
        )
        u.Tests.TestCaseHelpers.execute_and_assert_operation_result(
            lambda: operation_result,
            test_case,
        )

    @pytest.mark.parametrize(
        "test_case",
        create_hash_entity_cases(),
        ids=lambda case: f"hash_entity_{case['description']}",
    )
    def test_hash_entity_by_id(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test hash_entity_by_id using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(
                test_case.get("id_attr", "unique_id"),
            ),
        )
        expected: object = test_case.get("expected_result")
        if isinstance(expected, type):
            assert isinstance(operation_result, expected), (
                f"Expected type {expected}, got {type(operation_result)}"
            )
        else:
            assert operation_result == expected

    @pytest.mark.parametrize(
        "test_case",
        create_compare_value_objects_cases(),
        ids=lambda case: f"compare_value_objects_{case['description']}",
    )
    def test_compare_value_objects_by_value(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test compare_value_objects_by_value using FlextTestsUtilities."""
        # Type narrowing: execute_domain_operation returns object, but we know it's t.GeneralValueType
        # Cast lambda return type to t.GeneralValueType for type checker
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        expected: object = test_case.get("expected_result")
        if isinstance(expected, type):
            assert isinstance(operation_result, expected), (
                f"Expected {expected}, got {type(operation_result)}: {operation_result}"
            )
        else:
            assert operation_result == expected, (
                f"Expected {expected}, got {operation_result}"
            )

    @pytest.mark.parametrize(
        "test_case",
        create_hash_value_object_cases(),
        ids=lambda case: f"hash_value_object_{case['description']}",
    )
    def test_hash_value_object_by_value(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test hash_value_object_by_value using FlextTestsUtilities."""
        # Type narrowing: execute_domain_operation returns object, but we know it's t.GeneralValueType
        # Cast lambda return type to t.GeneralValueType for type checker
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        # hash_value_object_by_value returns an int; expected_result is the type `int`
        assert isinstance(operation_result, int), (
            f"Expected int, got {type(operation_result)}: {operation_result}"
        )

    @pytest.mark.parametrize(
        "test_case",
        create_validate_entity_has_id_cases(),
        ids=lambda case: f"validate_entity_has_id_{case['description']}",
    )
    def test_validate_entity_has_id(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test validate_entity_has_id using FlextTestsUtilities."""
        # Type narrowing: execute_domain_operation returns object, but we know it's t.GeneralValueType
        # Cast lambda return type to t.GeneralValueType for type checker
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(
                test_case.get("id_attr", "unique_id"),
            ),
        )
        u.Tests.TestCaseHelpers.execute_and_assert_operation_result(
            lambda: operation_result,
            test_case,
        )

    @pytest.mark.parametrize(
        "test_case",
        create_validate_value_object_immutable_cases(),
        ids=lambda case: f"validate_value_object_immutable_{case['description']}",
    )
    def test_validate_value_object_immutable(
        self,
        test_case: TestCaseMap,
    ) -> None:
        """Test validate_value_object_immutable using FlextTestsUtilities."""
        # Type narrowing: execute_domain_operation returns object, but we know it's t.GeneralValueType
        # Cast lambda return type to t.GeneralValueType for type checker
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        u.Tests.TestCaseHelpers.execute_and_assert_operation_result(
            lambda: operation_result,
            test_case,
        )

    # ============================================================================
    # Special Cases Not Covered by Parametrized Tests
    # ============================================================================

    def test_validate_immutable_config_type_error(self) -> None:
        """Test validation with config that raises TypeError using u.Domain directly."""
        obj = u.Tests.BadObjects.BadConfigTypeError()
        try:
            obj_value = cast("t.ConfigMapValue", cast("object", obj))
            result = u.Domain.validate_value_object_immutable(obj_value)
            assert isinstance(result, bool)
        except TypeError:
            # Exception propagation is also acceptable for coverage
            pass
