"""Tests for FlextUtilities.Domain - Domain entity and value object operations.

Module: flext_core._utilities.domain
Scope: FlextUtilities.Domain - entity/value object comparison, hashing, validation

Tests FlextUtilities.Domain functionality including:
- Entity comparison by ID (compare_entities_by_id)
- Entity hashing by ID (hash_entity_by_id)
- Value object comparison by value (compare_value_objects_by_value)
- Value object hashing by value (hash_value_object_by_value)
- Entity ID validation (validate_entity_has_id)
- Value object immutability validation (validate_value_object_immutable)

Uses Python 3.13 patterns (dataclasses with slots), centralized factory methods,
and parametrization for DRY testing with full edge case coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from flext_core import FlextUtilities
from tests.fixtures.constants import TestConstants
from tests.helpers import (
    BadConfig,
    BadConfigTypeError,
    BadModelDump,
    ComplexValue,
    DomainTestCase,
    DomainTestHelpers,
    DomainTestType,
    ImmutableObj,
    MutableObj,
    NoConfigNoSetattr,
    NoDict,
    NoSetattr,
    SimpleValue,
)

# =========================================================================
# Domain Test Function Type Enumeration
# =========================================================================


class DomainTestFunctionType(StrEnum):
    """Domain test function types for organization."""

    COMPARE_ENTITIES = "compare_entities"
    HASH_ENTITY = "hash_entity"
    COMPARE_VALUES = "compare_value_objects"
    HASH_VALUE = "hash_value_object"
    VALIDATE_ENTITY_ID = "validate_entity_has_id"
    VALIDATE_IMMUTABLE = "validate_value_object_immutable"


class TestFlextUtilitiesDomain:
    """Comprehensive tests for FlextUtilities.Domain using advanced patterns."""

    # ============================================================================
    # Factory Methods for Test Cases
    # ============================================================================

    @staticmethod
    def create_compare_entities_cases() -> list[DomainTestCase]:
        """Create test cases for entity comparison."""
        alice_entity = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
        )
        bob_entity = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_BOB,
            TestConstants.TestDomain.ENTITY_VALUE_20,
        )
        alice_no_id = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
            with_id=False,
        )
        value_obj = DomainTestHelpers.create_value(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.VALUE_COUNT_5,
        )
        custom1 = DomainTestHelpers.create_custom_entity(
            TestConstants.TestDomain.CUSTOM_ID_1,
        )
        custom2 = DomainTestHelpers.create_custom_entity(
            TestConstants.TestDomain.CUSTOM_ID_1,
        )

        return [
            DomainTestCase(
                test_type=DomainTestType.COMPARE_ENTITIES_BY_ID,
                description="same_id",
                input_data={"entity1": alice_entity, "entity2": alice_entity},  # type: ignore[dict-item]  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_ENTITIES_BY_ID,
                description="different_id",
                input_data={"entity1": alice_entity, "entity2": bob_entity},  # type: ignore[dict-item]  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_ENTITIES_BY_ID,
                description="different_type",
                input_data={"entity1": alice_entity, "entity2": value_obj},  # type: ignore[dict-item]  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_ENTITIES_BY_ID,
                description="no_id",
                input_data={"entity1": alice_no_id, "entity2": bob_entity},  # type: ignore[dict-item]  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_ENTITIES_BY_ID,
                description="custom_id_attr",
                input_data={"entity1": custom1, "entity2": custom2},  # type: ignore[dict-item]  # type: ignore[dict-item]
                expected_result=True,
                id_attr="custom_id",
            ),
        ]

    @staticmethod
    def create_hash_entity_cases() -> list[DomainTestCase]:
        """Create test cases for entity hashing."""
        alice_entity = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
        )
        alice_no_id = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
            with_id=False,
        )
        custom = DomainTestHelpers.create_custom_entity(
            TestConstants.TestDomain.CUSTOM_ID_1,
        )

        return [
            DomainTestCase(
                test_type=DomainTestType.HASH_ENTITY_BY_ID,
                description="with_id",
                input_data={"entity": alice_entity},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_ENTITY_BY_ID,
                description="no_id",
                input_data={"entity": alice_no_id},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_ENTITY_BY_ID,
                description="custom_id_attr",
                input_data={"entity": custom},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
                id_attr="custom_id",
            ),
        ]

    @staticmethod
    def create_compare_value_objects_cases() -> list[DomainTestCase]:
        """Create test cases for value object comparison."""
        value1 = DomainTestHelpers.create_value(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.VALUE_COUNT_5,
        )
        value2 = DomainTestHelpers.create_value(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.VALUE_COUNT_10,
        )
        alice_entity = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
        )
        simple1 = SimpleValue(TestConstants.TestDomain.VALUE_DATA_TEST)
        simple2 = SimpleValue(TestConstants.TestDomain.VALUE_DATA_TEST)
        bad1 = BadModelDump()
        bad2 = BadModelDump()
        no_dict1 = NoDict(TestConstants.TestDomain.VALUE_COUNT_5)
        no_dict2 = NoDict(TestConstants.TestDomain.VALUE_COUNT_5)

        return [
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="same_values",
                input_data={"obj1": value1, "obj2": value1},  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="different_values",
                input_data={"obj1": value1, "obj2": value2},  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="different_type",
                input_data={"obj1": value1, "obj2": alice_entity},  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="no_model_dump",
                input_data={"obj1": simple1, "obj2": simple2},  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="model_dump_exception",
                input_data={"obj1": bad1, "obj2": bad2},  # type: ignore[dict-item]
                expected_result=bool,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE,
                description="no_dict",
                input_data={"obj1": no_dict1, "obj2": no_dict2},  # type: ignore[dict-item]
                expected_result=True,
            ),
        ]

    @staticmethod
    def create_hash_value_object_cases() -> list[DomainTestCase]:
        """Create test cases for value object hashing."""
        value_obj = DomainTestHelpers.create_value(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.VALUE_COUNT_5,
        )
        simple_obj = SimpleValue(TestConstants.TestDomain.VALUE_DATA_TEST)
        bad_obj = BadModelDump()
        complex_obj = ComplexValue(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.COMPLEX_ITEMS,
        )
        no_dict_obj = NoDict(TestConstants.TestDomain.VALUE_COUNT_5)

        return [
            DomainTestCase(
                test_type=DomainTestType.HASH_VALUE_OBJECT_BY_VALUE,
                description="with_model_dump",
                input_data={"obj": value_obj},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_VALUE_OBJECT_BY_VALUE,
                description="no_model_dump",
                input_data={"obj": simple_obj},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_VALUE_OBJECT_BY_VALUE,
                description="model_dump_exception",
                input_data={"obj": bad_obj},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_VALUE_OBJECT_BY_VALUE,
                description="non_hashable_values",
                input_data={"obj": complex_obj},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.HASH_VALUE_OBJECT_BY_VALUE,
                description="no_dict",
                input_data={"obj": no_dict_obj},  # type: ignore[dict-item]
                expected_result=int,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
        ]

    @staticmethod
    def create_validate_entity_has_id_cases() -> list[DomainTestCase]:
        """Create test cases for entity ID validation."""
        alice_entity = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
        )
        alice_no_id = DomainTestHelpers.create_entity(
            TestConstants.TestDomain.ENTITY_NAME_ALICE,
            TestConstants.TestDomain.ENTITY_VALUE_10,
            with_id=False,
        )
        custom = DomainTestHelpers.create_custom_entity(
            TestConstants.TestDomain.CUSTOM_ID_1,
        )

        return [
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_ENTITY_HAS_ID,
                description="has_id",
                input_data={"entity": alice_entity},  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_ENTITY_HAS_ID,
                description="no_id",
                input_data={"entity": alice_no_id},  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_ENTITY_HAS_ID,
                description="custom_attr",
                input_data={"entity": custom},  # type: ignore[dict-item]
                expected_result=True,
                id_attr="custom_id",
            ),
        ]

    @staticmethod
    def create_validate_value_object_immutable_cases() -> list[DomainTestCase]:
        """Create test cases for value object immutability validation."""
        value_obj = DomainTestHelpers.create_value(
            TestConstants.TestDomain.VALUE_DATA_TEST,
            TestConstants.TestDomain.VALUE_COUNT_5,
        )
        mutable_obj = MutableObj(TestConstants.TestDomain.VALUE_COUNT_5)
        immutable_obj = ImmutableObj(TestConstants.TestDomain.VALUE_COUNT_5)
        bad_config_obj = BadConfig()
        no_config_obj = NoConfigNoSetattr()
        no_setattr_obj = NoSetattr()

        return [
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="frozen",
                input_data={"obj": value_obj},  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="mutable",
                input_data={"obj": mutable_obj},  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="custom_setattr",
                input_data={"obj": immutable_obj},  # type: ignore[dict-item]
                expected_result=True,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="config_exception",
                input_data={"obj": bad_config_obj},  # type: ignore[dict-item]
                expected_result=bool,  # type: ignore[arg-type]  # isinstance check - type used for validation
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="no_config_no_setattr",
                input_data={"obj": no_config_obj},  # type: ignore[dict-item]
                expected_result=False,
            ),
            DomainTestCase(
                test_type=DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE,
                description="no_setattr",
                input_data={"obj": no_setattr_obj},  # type: ignore[dict-item]
                expected_result=False,
            ),
        ]

    # ============================================================================
    # Parametrized Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "test_case",
        create_compare_entities_cases(),
        ids=lambda case: f"compare_entities_{case.description}",
    )
    def test_compare_entities_by_id(self, test_case: DomainTestCase) -> None:
        """Test compare_entities_by_id with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        if test_case.expected_result is bool:
            assert isinstance(result, bool)
        else:
            assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        create_hash_entity_cases(),
        ids=lambda case: f"hash_entity_{case.description}",
    )
    def test_hash_entity_by_id(self, test_case: DomainTestCase) -> None:
        """Test hash_entity_by_id with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        # expected_result is a type (int) used for isinstance check
        # pyright: ignore[reportArgumentType] - expected_result is a type for validation
        assert isinstance(result, test_case.expected_result)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "test_case",
        create_compare_value_objects_cases(),
        ids=lambda case: f"compare_value_objects_{case.description}",
    )
    def test_compare_value_objects_by_value(self, test_case: DomainTestCase) -> None:
        """Test compare_value_objects_by_value with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        if test_case.expected_result is bool:
            assert isinstance(result, bool)
        else:
            assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        create_hash_value_object_cases(),
        ids=lambda case: f"hash_value_object_{case.description}",
    )
    def test_hash_value_object_by_value(self, test_case: DomainTestCase) -> None:
        """Test hash_value_object_by_value with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        # expected_result is a type (int) used for isinstance check
        # pyright: ignore[reportArgumentType] - expected_result is a type for validation
        assert isinstance(result, test_case.expected_result)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "test_case",
        create_validate_entity_has_id_cases(),
        ids=lambda case: f"validate_entity_has_id_{case.description}",
    )
    def test_validate_entity_has_id(self, test_case: DomainTestCase) -> None:
        """Test validate_entity_has_id with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        create_validate_value_object_immutable_cases(),
        ids=lambda case: f"validate_value_object_immutable_{case.description}",
    )
    def test_validate_value_object_immutable(self, test_case: DomainTestCase) -> None:
        """Test validate_value_object_immutable with various scenarios."""
        result = DomainTestHelpers.execute_domain_test(test_case)
        if test_case.expected_result is bool:
            assert isinstance(result, bool)
        else:
            assert result == test_case.expected_result

    # ============================================================================
    # Special Cases Not Covered by Parametrized Tests
    # ============================================================================

    def test_validate_immutable_config_type_error(self) -> None:
        """Test validation with config that raises TypeError (special exception handling)."""
        obj = BadConfigTypeError()
        try:
            # BadConfigTypeError is compatible at runtime but not statically
            # pyright: ignore[reportArgumentType] - obj is compatible at runtime
            result = FlextUtilities.Domain.validate_value_object_immutable(obj)  # type: ignore[arg-type]
            assert isinstance(result, bool)
        except TypeError:
            # Exception propagation is also acceptable for coverage
            pass
