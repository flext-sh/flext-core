"""Tests for u - Domain entity and value object operations.

Module: flext_core._utilities.domain
Scope: u - entity/value object comparison, hashing, validation

Tests u functionality including:
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
from pydantic import BaseModel

from flext_tests import t, u
from tests.constants import c
from tests.models import m

from ._models import InputPayloadMap, TestCaseMap


def _build_domain_test_entity(
    *,
    name: str,
    value: t.Tests.object,
    **_kwargs: t.Tests.object,
) -> m.DomainTestEntity:
    return m.DomainTestEntity(name=name, value=cast("int", value), domain_events=[])


def _convert_to_general_value(obj: object) -> t.Tests.object:
    """Convert object to object (handles Pydantic models).

    Args:
        obj: Object to convert (Pydantic model, dict, list, or primitive)

    Returns:
        object-compatible value

    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return cast(
            "t.Tests.object",
            {
                str(key): _convert_to_general_value(val)
                for key, val in cast("dict[str, t.Tests.object]", obj).items()
            },
        )
    if isinstance(obj, (list, tuple)):
        return cast(
            "t.Tests.object",
            [
                _convert_to_general_value(elem)
                for elem in cast("list[t.Tests.object]", obj)
            ],
        )
    return str(obj)


def _require_payload_str(value: t.Tests.object) -> str:
    if isinstance(value, str):
        return value
    msg = f"Expected str payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _require_payload_mapping(
    value: t.Tests.object,
) -> Mapping[str, t.Tests.object]:
    if isinstance(value, Mapping):
        return value
    msg = f"Expected mapping payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _as_test_payload(
    value: type[t.Primitives],
) -> t.Tests.object:
    return cast("t.Tests.object", value)


def _as_payload_map(value: InputPayloadMap) -> Mapping[str, t.Tests.object]:
    return cast("Mapping[str, t.Tests.object]", value)


def create_compare_entities_cases() -> list[TestCaseMap]:
    """Create test cases for entity comparison using constants."""
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
            c.Tests.TestDomain.ENTITY_NAME_BOB,
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.Tests.TestDomain.ENTITY_VALUE_10,
            c.Tests.TestDomain.ENTITY_VALUE_20,
            c.Tests.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, False, True],
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    assert isinstance(entities, list)
    alice_entity, bob_entity, alice_no_id = entities
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=c.Tests.TestDomain.VALUE_DATA_TEST,
        count=c.Tests.TestDomain.VALUE_COUNT_5,
        value_class=m.DomainTestValue,
    )
    custom1 = m.CustomEntity(c.Tests.TestDomain.CUSTOM_ID_1)
    custom2 = m.CustomEntity(c.Tests.TestDomain.CUSTOM_ID_1)
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
    input_data_custom: InputPayloadMap = cast(
        "InputPayloadMap",
        {"entity_a": custom1, "entity_b": custom2},
    )
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
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.Tests.TestDomain.ENTITY_VALUE_10,
            c.Tests.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, True],
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    assert isinstance(entities, list)
    alice_entity, alice_no_id = entities
    custom = m.CustomEntity(c.Tests.TestDomain.CUSTOM_ID_1)
    input_data_with_id: InputPayloadMap = {"entity": alice_entity}
    input_data_no_id: InputPayloadMap = {"entity": alice_no_id}
    input_data_custom: InputPayloadMap = cast("InputPayloadMap", {"entity": custom})
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
    value_objs = u.Tests.DomainHelpers.create_test_value_objects_batch(
        data_list=[
            c.Tests.TestDomain.VALUE_DATA_TEST,
            c.Tests.TestDomain.VALUE_DATA_TEST,
        ],
        count_list=[
            c.Tests.TestDomain.VALUE_COUNT_5,
            c.Tests.TestDomain.VALUE_COUNT_10,
        ],
        value_class=m.DomainTestValue,
    )
    value1, value2 = value_objs
    alice_entity = u.Tests.DomainHelpers.create_test_entity_instance(
        name=c.Tests.TestDomain.ENTITY_NAME_ALICE,
        value=c.Tests.TestDomain.ENTITY_VALUE_10,
        entity_class=_build_domain_test_entity,
    )
    simple1 = m.SimpleValue(c.Tests.TestDomain.VALUE_DATA_TEST)
    simple2 = m.SimpleValue(c.Tests.TestDomain.VALUE_DATA_TEST)
    bad1 = u.Tests.BadObjects.BadModelDump()
    bad2 = u.Tests.BadObjects.BadModelDump()
    no_dict1 = m.NoDict(c.Tests.TestDomain.VALUE_COUNT_5)
    no_dict2 = m.NoDict(c.Tests.TestDomain.VALUE_COUNT_5)
    input_data_list: list[InputPayloadMap] = cast(
        "list[InputPayloadMap]",
        [
            {"obj_a": value1, "obj_b": value1},
            {"obj_a": value1, "obj_b": value2},
            {"obj_a": value1, "obj_b": alice_entity},
            {"obj_a": simple1, "obj_b": simple2},
            {"obj_a": bad1, "obj_b": bad2},
            {"obj_a": no_dict1, "obj_b": no_dict2},
        ],
    )
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
            input_data_list=cast(
                "list[Mapping[str, t.Tests.object]]",
                input_data_list,
            ),
            expected_results=[True, False, False, True, _as_test_payload(bool), True],
        ),
    )


def create_hash_value_object_cases() -> list[TestCaseMap]:
    """Create test cases for value object hashing using constants."""
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=c.Tests.TestDomain.VALUE_DATA_TEST,
        count=c.Tests.TestDomain.VALUE_COUNT_5,
        value_class=m.DomainTestValue,
    )
    simple_obj = m.SimpleValue(c.Tests.TestDomain.VALUE_DATA_TEST)
    bad_obj = u.Tests.BadObjects.BadModelDump()
    complex_items_list: list[str] = list(c.Tests.TestDomain.COMPLEX_ITEMS)
    complex_obj = m.ComplexValue(
        c.Tests.TestDomain.VALUE_DATA_TEST,
        complex_items_list,
    )
    no_dict_obj = m.NoDict(c.Tests.TestDomain.VALUE_COUNT_5)
    input_data_list_hash: list[InputPayloadMap] = cast(
        "list[InputPayloadMap]",
        [
            {"obj": value_obj},
            {"obj": simple_obj},
            {"obj": bad_obj},
            {"obj": complex_obj},
            {"obj": no_dict_obj},
        ],
    )
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
                "list[Mapping[str, t.Tests.object]]",
                input_data_list_hash,
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
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
            c.Tests.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.Tests.TestDomain.ENTITY_VALUE_10,
            c.Tests.TestDomain.ENTITY_VALUE_10,
        ],
        entity_class=_build_domain_test_entity,
        remove_ids=[False, True],
    )
    assert entities_result.is_success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities = entities_result.value
    assert isinstance(entities, list)
    alice_entity, alice_no_id = entities
    custom = m.CustomEntity(c.Tests.TestDomain.CUSTOM_ID_1)
    input_data_has_id: InputPayloadMap = {"entity": alice_entity}
    input_data_no_id_validate: InputPayloadMap = {"entity": alice_no_id}
    input_data_custom_validate: InputPayloadMap = cast(
        "InputPayloadMap",
        {"entity": custom},
    )
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
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=c.Tests.TestDomain.VALUE_DATA_TEST,
        count=c.Tests.TestDomain.VALUE_COUNT_5,
        value_class=m.DomainTestValue,
    )
    mutable_obj = m.MutableObj(c.Tests.TestDomain.VALUE_COUNT_5)
    immutable_obj = m.ImmutableObj(c.Tests.TestDomain.VALUE_COUNT_5)
    bad_config_obj = u.Tests.BadObjects.BadConfig()
    no_config_obj = m.NoConfigNoSetattr()
    no_setattr_obj = m.NoSetattr()
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
                "list[Mapping[str, t.Tests.object]]",
                [
                    {"obj": value_obj},
                    {"obj": mutable_obj},
                    {"obj": immutable_obj},
                    {"obj": bad_config_obj},
                    {"obj": no_config_obj},
                    {"obj": no_setattr_obj},
                ],
            ),
            expected_results=[True, False, True, False, False, False],
        ),
    )


class TestuDomain:
    """Comprehensive tests for u using FlextTestsUtilities and constants extensively."""

    @pytest.mark.parametrize(
        "test_case",
        create_compare_entities_cases(),
        ids=lambda case: f"compare_entities_{case['description']}",
    )
    def test_compare_entities_by_id(self, test_case: TestCaseMap) -> None:
        """Test compare_entities_by_id using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(test_case.get("id_attr", "unique_id")),
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
    def test_hash_entity_by_id(self, test_case: TestCaseMap) -> None:
        """Test hash_entity_by_id using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(test_case.get("id_attr", "unique_id")),
        )
        expected = test_case.get("expected_result")
        if isinstance(expected, type):
            expected_type = cast("type[t.Tests.object]", expected)
            assert isinstance(operation_result, expected_type), (
                f"Expected type {expected}, got {type(operation_result)}"
            )
        else:
            assert operation_result == expected

    @pytest.mark.parametrize(
        "test_case",
        create_compare_value_objects_cases(),
        ids=lambda case: f"compare_value_objects_{case['description']}",
    )
    def test_compare_value_objects_by_value(self, test_case: TestCaseMap) -> None:
        """Test compare_value_objects_by_value using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        expected = test_case.get("expected_result")
        if isinstance(expected, type):
            expected_type = cast("type[t.Tests.object]", expected)
            assert isinstance(operation_result, expected_type), (
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
    def test_hash_value_object_by_value(self, test_case: TestCaseMap) -> None:
        """Test hash_value_object_by_value using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        assert isinstance(operation_result, int), (
            f"Expected int, got {type(operation_result)}: {operation_result}"
        )

    @pytest.mark.parametrize(
        "test_case",
        create_validate_entity_has_id_cases(),
        ids=lambda case: f"validate_entity_has_id_{case['description']}",
    )
    def test_validate_entity_has_id(self, test_case: TestCaseMap) -> None:
        """Test validate_entity_has_id using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(test_case.get("id_attr", "unique_id")),
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
    def test_validate_value_object_immutable(self, test_case: TestCaseMap) -> None:
        """Test validate_value_object_immutable using FlextTestsUtilities."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        u.Tests.TestCaseHelpers.execute_and_assert_operation_result(
            lambda: operation_result,
            test_case,
        )

    def test_validate_immutable_config_type_error(self) -> None:
        """Test validation with config that raises TypeError using u directly."""
        obj = u.Tests.BadObjects.BadConfigTypeError()
        try:
            obj_value = cast("t.NormalizedValue", obj)
            result = u.validate_value_object_immutable(obj_value)
            assert isinstance(result, bool)
        except TypeError:
            pass
