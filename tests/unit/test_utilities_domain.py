"""Tests for u - Domain entity and value t.RecursiveContainer operations.

Module: flext_core
Scope: u - entity/value t.RecursiveContainer comparison, hashing, validation

Tests u functionality including:
- Entity comparison by ID (compare_entities_by_id)
- Entity hashing by ID (hash_entity_by_id)
- Value t.RecursiveContainer comparison by value (compare_value_objects_by_value)
- Value t.RecursiveContainer hashing by value (hash_value_object_by_value)
- Entity ID validation (validate_entity_has_id)
- Value t.RecursiveContainer immutability validation (validate_value_object_immutable)

Uses Python 3.13 patterns, u, constants (c), types (t),
utilities (u), protocols (p), models (m) extensively for maximum code reuse.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import pytest
from pydantic import BaseModel

from tests import c, m, p, r, t, u


def _create_entities_batch(
    names: t.StrSequence,
    values: Sequence[t.Core.Tests.TestobjectSerializable],
    entity_class: Callable[..., m.Core.Tests.DomainTestEntity],
    remove_ids: Sequence[bool] | None = None,
) -> r[Sequence[m.Core.Tests.DomainTestEntity]]:
    return u.Core.Tests.create_test_entities_batch(
        names=names,
        values=values,
        entity_class=entity_class,
        remove_ids=remove_ids,
    )


def _create_entity(
    name: str,
    value: t.Core.Tests.TestobjectSerializable,
    entity_class: Callable[..., m.Core.Tests.DomainTestEntity],
) -> m.Core.Tests.DomainTestEntity:
    return u.Core.Tests.create_test_entity_instance(
        name=name,
        value=value,
        entity_class=entity_class,
    )


def _create_value_object(
    data: str,
    count: int,
    value_class: p.Core.Tests.ValueFactory[m.Core.Tests.DomainTestValue],
) -> m.Core.Tests.DomainTestValue:
    return u.Core.Tests.create_test_value_object_instance(
        data=data,
        count=count,
        value_class=value_class,
    )


def _create_value_objects_batch(
    data_list: t.StrSequence,
    count_list: Sequence[int],
    value_class: p.Core.Tests.ValueFactory[m.Core.Tests.DomainTestValue],
) -> Sequence[m.Core.Tests.DomainTestValue]:
    return u.Core.Tests.create_test_value_objects_batch(
        data_list=data_list,
        count_list=count_list,
        value_class=value_class,
    )


def _build_domain_test_entity(
    *,
    name: str,
    value: t.Core.Tests.TestobjectSerializable,
    **_kwargs: t.Core.Tests.TestobjectSerializable,
) -> m.Core.Tests.DomainTestEntity:
    return m.Core.Tests.DomainTestEntity(
        name=name,
        value=cast("int", value),
        domain_events=[],
    )


def _convert_to_general_value(
    obj: t.Core.Tests.TestobjectSerializable,
) -> t.RecursiveContainer:
    """Convert test object to NormalizedValue (handles Pydantic models).

    Args:
        obj: Object to convert (Pydantic model, dict, list, or primitive)

    Returns:
        t.RecursiveContainer-compatible value

    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(key): _convert_to_general_value(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_general_value(elem) for elem in obj]
    return str(obj)


def _require_payload_str(value: t.Core.Tests.TestobjectSerializable) -> str:
    if isinstance(value, str):
        return value
    msg = f"Expected str payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _require_payload_mapping(
    value: t.Core.Tests.TestobjectSerializable,
) -> Mapping[str, t.Core.Tests.TestobjectSerializable]:
    if isinstance(value, Mapping):
        return value
    msg = f"Expected mapping payload, got {type(value).__name__}"
    raise AssertionError(msg)


def _as_test_payload(
    value: type[t.Primitives],
) -> t.Core.Tests.TestobjectSerializable:
    return value


def _as_payload_map(
    value: m.Core.Tests.InputPayloadMap,
) -> Mapping[str, t.Core.Tests.TestobjectSerializable]:
    return value


def create_compare_entities_cases() -> Sequence[m.Core.Tests.TestCaseMap]:
    """Create test cases for entity comparison using constants."""
    entities_result: r[Sequence[m.Core.Tests.DomainTestEntity]] = (
        _create_entities_batch(
            names=[
                c.Core.Tests.TestDomain.ENTITY_NAME_ALICE,
                c.Core.Tests.TestDomain.ENTITY_NAME_BOB,
                c.Core.Tests.TestDomain.ENTITY_NAME_ALICE,
            ],
            values=[
                c.Core.Tests.TestDomain.ENTITY_VALUE_10,
                c.Core.Tests.TestDomain.ENTITY_VALUE_20,
                c.Core.Tests.TestDomain.ENTITY_VALUE_10,
            ],
            entity_class=_build_domain_test_entity,
            remove_ids=[False, False, True],
        )
    )
    assert entities_result.success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities: Sequence[m.Core.Tests.DomainTestEntity] = entities_result.value
    assert isinstance(entities, list)
    alice_entity: m.Core.Tests.DomainTestEntity
    bob_entity: m.Core.Tests.DomainTestEntity
    alice_no_id: m.Core.Tests.DomainTestEntity
    alice_entity, bob_entity, alice_no_id = entities
    value_obj: m.Core.Tests.DomainTestValue = _create_value_object(
        data=c.Core.Tests.TestDomain.VALUE_DATA_TEST,
        count=c.Core.Tests.TestDomain.VALUE_COUNT_5,
        value_class=m.Core.Tests.DomainTestValue,
    )
    custom1 = m.Core.Tests.CustomEntity(c.Core.Tests.TestDomain.CUSTOM_ID_1)
    custom2 = m.Core.Tests.CustomEntity(c.Core.Tests.TestDomain.CUSTOM_ID_1)
    input_data_same_id: m.Core.Tests.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": alice_entity,
    }
    input_data_different_id: m.Core.Tests.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": bob_entity,
    }
    input_data_different_type: m.Core.Tests.InputPayloadMap = {
        "entity_a": alice_entity,
        "entity_b": value_obj,
    }
    input_data_no_id: m.Core.Tests.InputPayloadMap = {
        "entity_a": alice_no_id,
        "entity_b": bob_entity,
    }
    input_data_custom: m.Core.Tests.InputPayloadMap = cast(
        "m.Core.Tests.InputPayloadMap",
        {"entity_a": custom1, "entity_b": custom2},
    )
    return cast(
        "Sequence[m.Core.Tests.TestCaseMap]",
        [
            u.Core.Tests.create_operation_test_case(
                operation="compare_entities_by_id",
                description="same_id",
                input_data=_as_payload_map(input_data_same_id),
                expected_result=True,
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="compare_entities_by_id",
                description="different_id",
                input_data=_as_payload_map(input_data_different_id),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="compare_entities_by_id",
                description="different_type",
                input_data=_as_payload_map(input_data_different_type),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="compare_entities_by_id",
                description="no_id",
                input_data=_as_payload_map(input_data_no_id),
                expected_result=False,
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="compare_entities_by_id",
                description="custom_id_attr",
                input_data=_as_payload_map(input_data_custom),
                expected_result=True,
                id_attr="custom_id",
            ),
        ],
    )


def create_hash_entity_cases() -> Sequence[m.Core.Tests.TestCaseMap]:
    """Create test cases for entity hashing using constants."""
    entities_result: r[Sequence[m.Core.Tests.DomainTestEntity]] = (
        _create_entities_batch(
            names=[
                c.Core.Tests.TestDomain.ENTITY_NAME_ALICE,
                c.Core.Tests.TestDomain.ENTITY_NAME_ALICE,
            ],
            values=[
                c.Core.Tests.TestDomain.ENTITY_VALUE_10,
                c.Core.Tests.TestDomain.ENTITY_VALUE_10,
            ],
            entity_class=_build_domain_test_entity,
            remove_ids=[False, True],
        )
    )
    assert entities_result.success, (
        f"Failed to create entities: {entities_result.error}"
    )
    entities: Sequence[m.Core.Tests.DomainTestEntity] = entities_result.value
    assert isinstance(entities, list)
    alice_entity: m.Core.Tests.DomainTestEntity
    alice_no_id: m.Core.Tests.DomainTestEntity
    alice_entity, alice_no_id = entities
    custom = m.Core.Tests.CustomEntity(c.Core.Tests.TestDomain.CUSTOM_ID_1)
    input_data_with_id: m.Core.Tests.InputPayloadMap = {"entity": alice_entity}
    input_data_no_id: m.Core.Tests.InputPayloadMap = {"entity": alice_no_id}
    input_data_custom: m.Core.Tests.InputPayloadMap = cast(
        "m.Core.Tests.InputPayloadMap",
        {"entity": custom},
    )
    return cast(
        "Sequence[m.Core.Tests.TestCaseMap]",
        [
            u.Core.Tests.create_operation_test_case(
                operation="hash_entity_by_id",
                description="with_id",
                input_data=_as_payload_map(input_data_with_id),
                expected_result=_as_test_payload(int),
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="hash_entity_by_id",
                description="no_id",
                input_data=_as_payload_map(input_data_no_id),
                expected_result=_as_test_payload(int),
                id_attr="unique_id",
            ),
            u.Core.Tests.create_operation_test_case(
                operation="hash_entity_by_id",
                description="custom_id_attr",
                input_data=_as_payload_map(input_data_custom),
                expected_result=_as_test_payload(int),
                id_attr="custom_id",
            ),
        ],
    )


def create_compare_value_objects_cases() -> Sequence[m.Core.Tests.TestCaseMap]:
    """Create test cases for value t.RecursiveContainer comparison using constants."""
    value_objs: Sequence[m.Core.Tests.DomainTestValue] = _create_value_objects_batch(
        data_list=[
            c.Core.Tests.TestDomain.VALUE_DATA_TEST,
            c.Core.Tests.TestDomain.VALUE_DATA_TEST,
        ],
        count_list=[
            c.Core.Tests.TestDomain.VALUE_COUNT_5,
            c.Core.Tests.TestDomain.VALUE_COUNT_10,
        ],
        value_class=m.Core.Tests.DomainTestValue,
    )
    value1: m.Core.Tests.DomainTestValue
    value2: m.Core.Tests.DomainTestValue
    value1, value2 = value_objs
    alice_entity: m.Core.Tests.DomainTestEntity = _create_entity(
        name=c.Core.Tests.TestDomain.ENTITY_NAME_ALICE,
        value=c.Core.Tests.TestDomain.ENTITY_VALUE_10,
        entity_class=_build_domain_test_entity,
    )
    simple1 = m.Core.Tests.SimpleValue(c.Core.Tests.TestDomain.VALUE_DATA_TEST)
    simple2 = m.Core.Tests.SimpleValue(c.Core.Tests.TestDomain.VALUE_DATA_TEST)
    bad1 = u.Core.Tests.BadModelDump()
    bad2 = u.Core.Tests.BadModelDump()
    no_dict1 = m.Core.Tests.NoDict(c.Core.Tests.TestDomain.VALUE_COUNT_5)
    no_dict2 = m.Core.Tests.NoDict(c.Core.Tests.TestDomain.VALUE_COUNT_5)
    input_data_list: Sequence[m.Core.Tests.InputPayloadMap] = cast(
        "Sequence[m.Core.Tests.InputPayloadMap]",
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
        "Sequence[m.Core.Tests.TestCaseMap]",
        u.Core.Tests.create_batch_operation_test_cases(
            operation="compare_value_objects_by_value",
            descriptions=[
                "same_values",
                "different_values",
                "different_type",
                "no_model_dump",
                "model_dump_exception",
                "no_dict",
            ],
            input_data_list=input_data_list,
            expected_results=[True, False, False, True, _as_test_payload(bool), True],
        ),
    )


def create_hash_value_object_cases() -> Sequence[m.Core.Tests.TestCaseMap]:
    """Create test cases for value t.RecursiveContainer hashing using constants."""
    value_obj: m.Core.Tests.DomainTestValue = _create_value_object(
        data=c.Core.Tests.TestDomain.VALUE_DATA_TEST,
        count=c.Core.Tests.TestDomain.VALUE_COUNT_5,
        value_class=m.Core.Tests.DomainTestValue,
    )
    simple_obj = m.Core.Tests.SimpleValue(c.Core.Tests.TestDomain.VALUE_DATA_TEST)
    bad_obj = u.Core.Tests.BadModelDump()
    complex_items_list: t.StrSequence = list(c.Core.Tests.TestDomain.COMPLEX_ITEMS)
    complex_obj = m.Core.Tests.ComplexValue(
        c.Core.Tests.TestDomain.VALUE_DATA_TEST,
        complex_items_list,
    )
    no_dict_obj = m.Core.Tests.NoDict(c.Core.Tests.TestDomain.VALUE_COUNT_5)
    input_data_list_hash: Sequence[m.Core.Tests.InputPayloadMap] = cast(
        "Sequence[m.Core.Tests.InputPayloadMap]",
        [
            {"obj": value_obj},
            {"obj": simple_obj},
            {"obj": bad_obj},
            {"obj": complex_obj},
            {"obj": no_dict_obj},
        ],
    )
    return cast(
        "Sequence[m.Core.Tests.TestCaseMap]",
        u.Core.Tests.create_batch_operation_test_cases(
            operation="hash_value_object_by_value",
            descriptions=[
                "with_model_dump",
                "no_model_dump",
                "model_dump_exception",
                "non_hashable_values",
                "no_dict",
            ],
            input_data_list=input_data_list_hash,
            expected_results=[
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
                _as_test_payload(int),
            ],
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
        self,
        test_case: m.Core.Tests.TestCaseMap,
    ) -> None:
        """Test compare_entities_by_id using u."""
        operation_result = u.Core.Tests.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
            id_attr=_require_payload_str(test_case.get("id_attr", "unique_id")),
        )
        u.Core.Tests.execute_and_assert_operation_result(
            lambda: operation_result,
            test_case,
        )

    @pytest.mark.parametrize(
        "test_case",
        create_hash_entity_cases(),
        ids=lambda case: f"hash_entity_{case['description']}",
    )
    def test_hash_entity_by_id(self, test_case: m.Core.Tests.TestCaseMap) -> None:
        """Test hash_entity_by_id using u."""
        operation_result = u.Core.Tests.execute_domain_operation(
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
        self,
        test_case: m.Core.Tests.TestCaseMap,
    ) -> None:
        """Test compare_value_objects_by_value using u."""
        operation_result = u.Core.Tests.execute_domain_operation(
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
        self,
        test_case: m.Core.Tests.TestCaseMap,
    ) -> None:
        """Test hash_value_object_by_value using u."""
        operation_result = u.Core.Tests.execute_domain_operation(
            _require_payload_str(test_case["operation"]),
            _require_payload_mapping(test_case["input_data"]),
        )
        assert isinstance(operation_result, int), (
            f"Expected int, got {type(operation_result)}: {operation_result}"
        )
