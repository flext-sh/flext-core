"""Tests for u - Domain entity and value t.NormalizedValue operations.

Module: flext_core._utilities.domain
Scope: u - entity/value t.NormalizedValue comparison, hashing, validation

Tests u functionality including:
- Entity comparison by ID (compare_entities_by_id)
- Entity hashing by ID (hash_entity_by_id)
- Value t.NormalizedValue comparison by value (compare_value_objects_by_value)
- Value t.NormalizedValue hashing by value (hash_value_object_by_value)
- Entity ID validation (validate_entity_has_id)
- Value t.NormalizedValue immutability validation (validate_value_object_immutable)

Uses Python 3.13 patterns, u, constants (c), types (t),
utilities (u), protocols (p), models (m) extensively for maximum code reuse.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest
from flext_tests import t, u
from pydantic import BaseModel

from tests import c, m

from ._models import TestUnitModels


def _build_domain_test_entity(
    *,
    name: str,
    value: t.NormalizedValue,
    **_kwargs: t.NormalizedValue,
) -> m.Core.DomainTestEntity:
    return m.Core.DomainTestEntity(
        name=name, value=cast("int", value), domain_events=[]
    )


def _convert_to_general_value(obj: t.NormalizedValue) -> t.NormalizedValue:
    """Convert t.NormalizedValue to t.NormalizedValue (handles Pydantic models).

    Args:
        obj: Object to convert (Pydantic model, dict, list, or primitive)

    Returns:
        t.NormalizedValue-compatible value

    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return cast(
            "t.Tests.NormalizedValue",
            {
                str(key): _convert_to_general_value(val)
                for key, val in cast(
                    "Mapping[str, t.Tests.NormalizedValue]", obj
                ).items()
            },
        )
    if isinstance(obj, (list, tuple)):
        return cast(
            "t.Tests.NormalizedValue",
            [
                _convert_to_general_value(elem)
                for elem in cast("Sequence[t.Tests.NormalizedValue]", obj)
            ],
        )
    return str(obj)


def _require_payload_str(value: t.NormalizedValue) -> str:
    if isinstance(value, str):
        return value
    msg = f"Expected str payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _require_payload_mapping(
    value: t.NormalizedValue,
) -> Mapping[str, t.NormalizedValue]:
    if isinstance(value, Mapping):
        return value
    msg = f"Expected mapping payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _as_test_payload(
    value: type[t.Primitives],
) -> t.NormalizedValue:
    return cast("t.Tests.NormalizedValue", value)


def _as_payload_map(
    value: TestUnitModels.InputPayloadMap,
) -> Mapping[str, t.NormalizedValue]:
    return cast("Mapping[str, t.Tests.NormalizedValue]", value)


def create_compare_entities_cases() -> Sequence[TestUnitModels.TestCaseMap]:
    """Create test cases for entity comparison using constants."""
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.TestDomain.ENTITY_NAME_ALICE,
            c.TestDomain.ENTITY_NAME_BOB,
            c.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.TestDomain.ENTITY_VALUE_10,
            c.TestDomain.ENTITY_VALUE_20,
            c.TestDomain.ENTITY_VALUE_10,
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
        data=c.TestDomain.VALUE_DATA_TEST,
        count=c.TestDomain.VALUE_COUNT_5,
        value_class=m.Core.DomainTestValue,
    )
    custom1 = m.Core.CustomEntity(c.TestDomain.CUSTOM_ID_1)
    custom2 = m.Core.CustomEntity(c.TestDomain.CUSTOM_ID_1)
    input_data_same_id: TestUnitModels.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": alice_entity,
    }
    input_data_different_id: TestUnitModels.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": bob_entity,
    }
    input_data_different_type: TestUnitModels.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": value_obj,
    }
    input_data_no_id: TestUnitModels.InputPayloadMap = {
        "entity_a": alice_no_id,
        "entity_b": bob_entity,
    }
    input_data_custom: TestUnitModels.InputPayloadMap = cast(
        "TestUnitModels.InputPayloadMap",
        {"entity_a": custom1, "entity_b": custom2},
    )
    return cast(
        "Sequence[TestUnitModels.TestCaseMap]",
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


def create_hash_entity_cases() -> Sequence[TestUnitModels.TestCaseMap]:
    """Create test cases for entity hashing using constants."""
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.TestDomain.ENTITY_NAME_ALICE,
            c.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.TestDomain.ENTITY_VALUE_10,
            c.TestDomain.ENTITY_VALUE_10,
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
    custom = m.Core.CustomEntity(c.TestDomain.CUSTOM_ID_1)
    input_data_with_id: TestUnitModels.InputPayloadMap = {"entity": alice_entity}
    input_data_no_id: TestUnitModels.InputPayloadMap = {"entity": alice_no_id}
    input_data_custom: TestUnitModels.InputPayloadMap = cast(
        "TestUnitModels.InputPayloadMap",
        {"entity": custom},
    )
    return cast(
        "Sequence[TestUnitModels.TestCaseMap]",
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


def create_compare_value_objects_cases() -> Sequence[TestUnitModels.TestCaseMap]:
    """Create test cases for value t.NormalizedValue comparison using constants."""
    value_objs = u.Tests.DomainHelpers.create_test_value_objects_batch(
        data_list=[
            c.TestDomain.VALUE_DATA_TEST,
            c.TestDomain.VALUE_DATA_TEST,
        ],
        count_list=[
            c.TestDomain.VALUE_COUNT_5,
            c.TestDomain.VALUE_COUNT_10,
        ],
        value_class=m.Core.DomainTestValue,
    )
    value1, value2 = value_objs
    alice_entity = u.Tests.DomainHelpers.create_test_entity_instance(
        name=c.TestDomain.ENTITY_NAME_ALICE,
        value=c.TestDomain.ENTITY_VALUE_10,
        entity_class=_build_domain_test_entity,
    )
    simple1 = m.Core.SimpleValue(c.TestDomain.VALUE_DATA_TEST)
    simple2 = m.Core.SimpleValue(c.TestDomain.VALUE_DATA_TEST)
    bad1 = u.Tests.BadObjects.BadModelDump()
    bad2 = u.Tests.BadObjects.BadModelDump()
    no_dict1 = m.Core.NoDict(c.TestDomain.VALUE_COUNT_5)
    no_dict2 = m.Core.NoDict(c.TestDomain.VALUE_COUNT_5)
    input_data_list: Sequence[TestUnitModels.InputPayloadMap] = cast(
        "Sequence[TestUnitModels.InputPayloadMap]",
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
        "Sequence[TestUnitModels.TestCaseMap]",
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
                "Sequence[Mapping[str, t.Tests.NormalizedValue]]",
                input_data_list,
            ),
            expected_results=[True, False, False, True, _as_test_payload(bool), True],
        ),
    )


def create_hash_value_object_cases() -> Sequence[TestUnitModels.TestCaseMap]:
    """Create test cases for value t.NormalizedValue hashing using constants."""
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=c.TestDomain.VALUE_DATA_TEST,
        count=c.TestDomain.VALUE_COUNT_5,
        value_class=m.Core.DomainTestValue,
    )
    simple_obj = m.Core.SimpleValue(c.TestDomain.VALUE_DATA_TEST)
    bad_obj = u.Tests.BadObjects.BadModelDump()
    complex_items_list: Sequence[str] = list(c.TestDomain.COMPLEX_ITEMS)
    complex_obj = m.Core.ComplexValue(
        c.TestDomain.VALUE_DATA_TEST,
        complex_items_list,
    )
    no_dict_obj = m.Core.NoDict(c.TestDomain.VALUE_COUNT_5)
    input_data_list_hash: Sequence[TestUnitModels.InputPayloadMap] = cast(
        "Sequence[TestUnitModels.InputPayloadMap]",
        [
            {"obj": value_obj},
            {"obj": simple_obj},
            {"obj": bad_obj},
            {"obj": complex_obj},
            {"obj": no_dict_obj},
        ],
    )
    return cast(
        "Sequence[TestUnitModels.TestCaseMap]",
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
                "Sequence[Mapping[str, t.Tests.NormalizedValue]]",
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


def create_validate_entity_has_id_cases() -> Sequence[TestUnitModels.TestCaseMap]:
    """Create test cases for entity ID validation using constants."""
    entities_result = u.Tests.DomainHelpers.create_test_entities_batch(
        names=[
            c.TestDomain.ENTITY_NAME_ALICE,
            c.TestDomain.ENTITY_NAME_ALICE,
        ],
        values=[
            c.TestDomain.ENTITY_VALUE_10,
            c.TestDomain.ENTITY_VALUE_10,
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
    custom = m.Core.CustomEntity(c.TestDomain.CUSTOM_ID_1)
    input_data_has_id: TestUnitModels.InputPayloadMap = {"entity": alice_entity}
    input_data_no_id_validate: TestUnitModels.InputPayloadMap = {"entity": alice_no_id}
    input_data_custom_validate: TestUnitModels.InputPayloadMap = cast(
        "TestUnitModels.InputPayloadMap",
        {"entity": custom},
    )
    return cast(
        "Sequence[TestUnitModels.TestCaseMap]",
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


def create_validate_value_object_immutable_cases() -> Sequence[
    TestUnitModels.TestCaseMap
]:
    """Create test cases for immutability validation using constants."""
    value_obj = u.Tests.DomainHelpers.create_test_value_object_instance(
        data=c.TestDomain.VALUE_DATA_TEST,
        count=c.TestDomain.VALUE_COUNT_5,
        value_class=m.Core.DomainTestValue,
    )
    mutable_obj = m.Core.MutableObj(c.TestDomain.VALUE_COUNT_5)
    immutable_obj = m.Core.ImmutableObj(c.TestDomain.VALUE_COUNT_5)
    bad_config_obj = u.Tests.BadObjects.BadConfig()
    no_config_obj = m.Core.NoConfigNoSetattr()
    no_setattr_obj = m.Core.NoSetattr()
    return cast(
        "Sequence[TestUnitModels.TestCaseMap]",
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
                "Sequence[Mapping[str, t.Tests.NormalizedValue]]",
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
    """Comprehensive tests for u using u and constants extensively."""

    @pytest.mark.parametrize(
        "test_case",
        create_compare_entities_cases(),
        ids=lambda case: f"compare_entities_{case['description']}",
    )
    def test_compare_entities_by_id(
        self, test_case: TestUnitModels.TestCaseMap
    ) -> None:
        """Test compare_entities_by_id using u."""
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
    def test_hash_entity_by_id(self, test_case: TestUnitModels.TestCaseMap) -> None:
        """Test hash_entity_by_id using u."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(test_case.get("id_attr", "unique_id")),
        )
        expected = test_case.get("expected_result")
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
        self, test_case: TestUnitModels.TestCaseMap
    ) -> None:
        """Test compare_value_objects_by_value using u."""
        operation_result = u.Tests.DomainHelpers.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        expected = test_case.get("expected_result")
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
        self, test_case: TestUnitModels.TestCaseMap
    ) -> None:
        """Test hash_value_object_by_value using u."""
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
    def test_validate_entity_has_id(
        self, test_case: TestUnitModels.TestCaseMap
    ) -> None:
        """Test validate_entity_has_id using u."""
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
    def test_validate_value_object_immutable(
        self, test_case: TestUnitModels.TestCaseMap
    ) -> None:
        """Test validate_value_object_immutable using u."""
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
