"""Targeted tests for missing coverage in flext_core.exceptions module.

This test file specifically targets the 41 missing lines identified in coverage analysis
to improve exceptions.py from 82% to 95%+ coverage.

Missing lines targeted: 398-440, 452-463, 498

These focus on:
- TypeMismatchError type conversion logic (398-440)
- _CriticalError context handling (452-463)
- TypeError constructor coverage (498)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextConstants
from flext_core.exceptions import FlextExceptions

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestTypeErrorTypeConversionMissingCoverage:
    """Test coverage for _TypeError.__init__ type conversion logic - Lines 398-440."""

    def test_type_error_with_str_types(self) -> None:
        """Test _TypeError with string type conversion - Lines 408-409, 421-422."""
        # Test expected_type="str" conversion - Lines 408-409
        error = FlextExceptions.TypeError(
            "String type mismatch", expected_type="str", actual_type="str"
        )

        assert error.expected_type == "str"
        assert error.actual_type == "str"
        assert error.code == FlextConstants.Errors.TYPE_ERROR

        # Check context contains converted types
        assert "expected_type" in error.context
        assert "actual_type" in error.context
        assert (
            error.context["expected_type"] is str
        )  # Lines 408-409: converted to str type
        assert (
            error.context["actual_type"] is str
        )  # Lines 421-422: converted to str type

    def test_type_error_with_int_types(self) -> None:
        """Test _TypeError with int type conversion - Lines 410-411, 423-424."""
        # Test expected_type="int" and actual_type="int" conversion
        error = FlextExceptions.TypeError(
            "Integer type mismatch", expected_type="int", actual_type="int"
        )

        assert error.expected_type == "int"
        assert error.actual_type == "int"
        assert (
            error.context["expected_type"] is int
        )  # Lines 410-411: converted to int type
        assert (
            error.context["actual_type"] is int
        )  # Lines 423-424: converted to int type

    def test_type_error_with_float_types(self) -> None:
        """Test _TypeError with float type conversion - Lines 412-413, 425-426."""
        # Test expected_type="float" and actual_type="float" conversion
        error = FlextExceptions.TypeError(
            "Float type mismatch", expected_type="float", actual_type="float"
        )

        assert error.expected_type == "float"
        assert error.actual_type == "float"
        assert (
            error.context["expected_type"] is float
        )  # Lines 412-413: converted to float type
        assert (
            error.context["actual_type"] is float
        )  # Lines 425-426: converted to float type

    def test_type_error_with_bool_types(self) -> None:
        """Test _TypeError with bool type conversion - Lines 414-415, 427-428."""
        # Test expected_type="bool" and actual_type="bool" conversion
        error = FlextExceptions.TypeError(
            "Boolean type mismatch", expected_type="bool", actual_type="bool"
        )

        assert error.expected_type == "bool"
        assert error.actual_type == "bool"
        assert (
            error.context["expected_type"] is bool
        )  # Lines 414-415: converted to bool type
        assert (
            error.context["actual_type"] is bool
        )  # Lines 427-428: converted to bool type

    def test_type_error_with_list_types(self) -> None:
        """Test _TypeError with list type conversion - Lines 416-417, 429-430."""
        # Test expected_type="list" and actual_type="list" conversion
        error = FlextExceptions.TypeError(
            "List type mismatch", expected_type="list", actual_type="list"
        )

        assert error.expected_type == "list"
        assert error.actual_type == "list"
        assert (
            error.context["expected_type"] is list
        )  # Lines 416-417: converted to list type
        assert (
            error.context["actual_type"] is list
        )  # Lines 429-430: converted to list type

    def test_type_error_with_dict_types(self) -> None:
        """Test _TypeError with dict type conversion - Lines 418-419, 431-432."""
        # Test expected_type="dict" and actual_type="dict" conversion
        error = FlextExceptions.TypeError(
            "Dict type mismatch", expected_type="dict", actual_type="dict"
        )

        assert error.expected_type == "dict"
        assert error.actual_type == "dict"
        assert (
            error.context["expected_type"] is dict
        )  # Lines 418-419: converted to dict type
        assert (
            error.context["actual_type"] is dict
        )  # Lines 431-432: converted to dict type

    def test_type_error_with_none_types(self) -> None:
        """Test _TypeError with None types - Lines 398-399, 405-406."""
        # Test with None types to cover default case - Lines 398-399
        error = FlextExceptions.TypeError(
            "Type mismatch with None", expected_type=None, actual_type=None
        )

        assert error.expected_type is None  # Line 398: stored as None
        assert error.actual_type is None  # Line 399: stored as None
        assert (
            error.context["expected_type"] == ""
        )  # Lines 405-406: default to empty string
        assert (
            error.context["actual_type"] == ""
        )  # Lines 405-406: default to empty string

    def test_type_error_with_unknown_types(self) -> None:
        """Test _TypeError with unknown type strings - Lines 405-406."""
        # Test with types not in the conversion logic
        error = FlextExceptions.TypeError(
            "Unknown type mismatch",
            expected_type="custom_type",
            actual_type="another_type",
        )

        assert error.expected_type == "custom_type"
        assert error.actual_type == "another_type"
        # Lines 405-406: Unknown types remain as strings
        assert error.context["expected_type"] == "custom_type"
        assert error.context["actual_type"] == "another_type"

    def test_type_error_context_and_kwargs(self) -> None:
        """Test _TypeError context handling - Lines 400-402, 434-439."""
        # Test with existing context and additional kwargs
        existing_context = {"user_id": "123", "operation": "validation"}

        error = FlextExceptions.TypeError(
            "Type mismatch with context",
            expected_type="str",
            actual_type="int",
            context=existing_context,
            correlation_id="corr-123",
        )

        # Lines 400-402: Context extraction and dict conversion
        assert "user_id" in error.context
        assert "operation" in error.context
        assert error.context["user_id"] == "123"
        assert error.context["operation"] == "validation"

        # Lines 434-439: Type objects added to context
        assert error.context["expected_type"] is str
        assert error.context["actual_type"] is int

        # Lines 440-444: Super().__init__ call with correlation_id
        assert error.correlation_id == "corr-123"

    def test_type_error_empty_context(self) -> None:
        """Test _TypeError with empty context - Lines 400-402."""
        # Test with empty context dict - Lines 400-402
        error = FlextExceptions.TypeError(
            "Type mismatch empty context",
            expected_type="str",
            actual_type="int",
            context={},
        )

        # Lines 400-402: Empty context handling
        assert isinstance(error.context, dict)
        assert "expected_type" in error.context
        assert "actual_type" in error.context


class TestCriticalErrorMissingCoverage:
    """Test coverage for _CriticalError context handling - Lines 452-463."""

    def test_critical_error_with_context_parameter(self) -> None:
        """Test _CriticalError with context parameter - Lines 452, 456-458."""
        # Test with explicit context parameter - Lines 452, 456-458
        original_context = {"error_code": "SYS001", "component": "database"}
        additional_kwargs = {"severity": "high", "retry_count": 3}

        error = FlextExceptions.CriticalError(
            "Critical system failure",
            context=original_context,
            correlation_id="critical-456",
            **additional_kwargs,
        )

        # Line 452: Context extraction via kwargs.pop
        # Lines 456-458: Context update with remaining kwargs
        assert "error_code" in error.context
        assert "component" in error.context
        assert "severity" in error.context
        assert "retry_count" in error.context
        assert error.context["error_code"] == "SYS001"
        assert error.context["component"] == "database"
        assert error.context["severity"] == "high"
        assert error.context["retry_count"] == 3

        # Line 453: correlation_id extraction
        assert error.correlation_id == "critical-456"
        assert error.code == FlextConstants.Errors.CRITICAL_ERROR

    def test_critical_error_with_none_context(self) -> None:
        """Test _CriticalError with None context - Lines 460-461."""
        # Test with None context and kwargs - Lines 460-461
        error = FlextExceptions.CriticalError(
            "Critical error no context",
            context=None,
            severity="critical",
            module="auth",
        )

        # Lines 460-461: None context with kwargs creates new dict
        assert error.context is not None
        assert isinstance(error.context, dict)
        assert error.context["severity"] == "critical"
        assert error.context["module"] == "auth"

    def test_critical_error_without_context_parameter(self) -> None:
        """Test _CriticalError without context parameter - Lines 460-461."""
        # Test without context parameter but with kwargs - Lines 460-461
        error = FlextExceptions.CriticalError(
            "Critical error kwargs only", priority="urgent", system="payment"
        )

        # Lines 460-461: No context parameter, kwargs become context
        assert error.context is not None
        assert isinstance(error.context, dict)
        assert error.context["priority"] == "urgent"
        assert error.context["system"] == "payment"

    def test_critical_error_no_context_no_kwargs(self) -> None:
        """Test _CriticalError with no context and no kwargs - Line 462."""
        # Test with neither context nor kwargs
        error = FlextExceptions.CriticalError("Simple critical error")

        # Line 462: Empty context when no context or kwargs
        assert error.context == {}
        assert error.code == FlextConstants.Errors.CRITICAL_ERROR

    def test_critical_error_with_correlation_id_only(self) -> None:
        """Test _CriticalError with only correlation_id - Line 453."""
        # Test with only correlation_id parameter - Line 453
        error = FlextExceptions.CriticalError(
            "Critical error with correlation", correlation_id="corr-789"
        )

        # Line 453: correlation_id extraction
        assert error.correlation_id == "corr-789"
        assert error.context == {}  # Empty context


class TestTypeErrorConstructorMissingCoverage:
    """Test coverage for TypeError.__init__ - Line 498."""

    def test_type_error_constructor(self) -> None:
        """Test TypeError constructor - Line 498."""
        # Test TypeError constructor with basic parameters - Line 498
        error = FlextExceptions.TypeError("Type validation failed")

        assert "Type validation failed" in str(error)
        assert error.code == FlextConstants.Errors.TYPE_ERROR
        assert error.context == {
            "expected_type": "",
            "actual_type": "",
        }  # Always has these fields
        assert error.correlation_id is not None  # Auto-generated correlation_id

    def test_type_error_with_context_and_correlation_id(self) -> None:
        """Test TypeError constructor with context and correlation_id - Line 498."""
        # Test TypeError with context and correlation_id - Line 498
        context_data = {"field": "user_id", "value": "invalid"}

        error = FlextExceptions.TypeError(
            "Invalid type for field",
            context=context_data,
            correlation_id="type-error-123",
        )

        # Line 498: super().__init__ call with all parameters
        assert "Invalid type for field" in str(error)
        assert error.code == FlextConstants.Errors.TYPE_ERROR
        # Context includes original data plus type fields
        assert error.context["field"] == "user_id"
        assert error.context["value"] == "invalid"
        assert "expected_type" in error.context
        assert "actual_type" in error.context
        assert error.correlation_id == "type-error-123"
