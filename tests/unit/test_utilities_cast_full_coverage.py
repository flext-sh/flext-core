"""Tests for FlextUtilitiesCast - type casting utilities.

Module: flext_core._utilities.cast
Coverage target: lines 45, 70, 72, 82-84, 87, 90-96, 156, 160, 179, 183, 189, 191-192

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
import warnings

import pytest

from pydantic import BaseModel

from flext_core import t, u


class TestCastDirect:
    """Tests for u.Cast.direct()."""

    def test_direct_returns_value_if_correct_type(self) -> None:
        """Returns value when it matches target_type."""
        assert u.Cast.direct("hello", str) == "hello"
        assert u.Cast.direct(42, int) == 42

    def test_direct_raises_if_wrong_type(self) -> None:
        """Raises TypeError when value doesn't match target_type."""
        with pytest.raises(TypeError, match="not instance of"):
            u.Cast.direct("hello", int)


class TestCastGeneralValue:
    """Tests for u.Cast.general_value() - covers type conversion branches."""

    def test_general_value_same_type_returns_as_is(self) -> None:
        """Value already of target_type is returned unchanged (line 70)."""
        assert u.Cast.general_value("hello", str) == "hello"
        assert u.Cast.general_value(42, int) == 42

    def test_general_value_to_str(self) -> None:
        """Converts to str (line 72)."""
        assert u.Cast.general_value(42, str) == "42"

    def test_general_value_to_int_from_float(self) -> None:
        """Converts float to int."""
        assert u.Cast.general_value(3.14, int) == 3

    def test_general_value_to_int_from_str(self) -> None:
        """Converts str to int."""
        assert u.Cast.general_value("42", int) == 42

    def test_general_value_to_int_from_invalid_raises(self) -> None:
        """Non-castable type to int raises TypeError (lines 82-84)."""
        with pytest.raises(TypeError, match="Cannot cast.*to int"):
            u.Cast.general_value(None, int)

    def test_general_value_to_float_from_str(self) -> None:
        """Converts str to float."""
        assert u.Cast.general_value("3.14", float) == 3.14

    def test_general_value_to_float_from_invalid_raises(self) -> None:
        """Non-castable type to float raises TypeError."""
        with pytest.raises(TypeError, match="Cannot cast.*to float"):
            u.Cast.general_value(None, float)

    def test_general_value_to_bool_already_bool(self) -> None:
        """Bool input returns as-is (line 87)."""
        assert u.Cast.general_value(True, bool) is True

    def test_general_value_to_bool_from_str_true(self) -> None:
        """String 'true' converts to True."""
        assert u.Cast.general_value("true", bool) is True
        assert u.Cast.general_value("yes", bool) is True
        assert u.Cast.general_value("1", bool) is True

    def test_general_value_to_bool_from_str_false(self) -> None:
        """String not in truthy set converts to False."""
        assert u.Cast.general_value("false", bool) is False
        assert u.Cast.general_value("no", bool) is False

    def test_general_value_to_bool_from_invalid_raises(self) -> None:
        """Non-castable type to bool raises TypeError (lines 90-92)."""
        with pytest.raises(TypeError, match="Cannot cast.*to bool"):
            u.Cast.general_value(None, bool)

    def test_general_value_unsupported_target_type_raises(self) -> None:
        """Unknown target type raises TypeError (lines 93-96)."""
        with pytest.raises(TypeError, match="Cannot cast.*to list"):
            u.Cast.general_value("hello", list)


class TestCastSafe:
    """Tests for u.Cast.safe() - mode-based dispatch."""

    def test_safe_mode_direct(self) -> None:
        """safe() with mode='direct' delegates to direct() (line 156)."""
        result = u.Cast.safe("hello", str, mode="direct")
        assert result == "hello"

    def test_safe_mode_callable(self) -> None:
        """safe() with mode='callable' delegates to callable() (line 160)."""

        def my_func() -> str:
            return "ok"

        result = u.Cast.safe(my_func, str, mode="callable")
        assert result is my_func

    def test_safe_mode_general_value(self) -> None:
        """safe() with mode='general_value' delegates to general_value()."""
        result = u.Cast.safe(42, str, mode="general_value")
        assert result == "42"

    def test_safe_unknown_mode_raises(self) -> None:
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            u.Cast.safe("hello", str, mode="invalid")


class TestCastToGeneralValueType:
    """Tests for u.Cast.to_general_value_type() - covers all isinstance branches."""

    def test_none_returns_none(self) -> None:
        """None input returns None (line 179)."""
        assert u.Cast.to_general_value_type(None) is None

    def test_primitives_returned_as_is(self) -> None:
        """str, int, float, bool returned as-is."""
        assert u.Cast.to_general_value_type("hello") == "hello"
        assert u.Cast.to_general_value_type(42) == 42
        assert u.Cast.to_general_value_type(3.14) == 3.14
        assert u.Cast.to_general_value_type(True) is True

    def test_datetime_returned_as_is(self) -> None:
        """datetime returned as-is (line 183)."""
        now = datetime.now()
        assert u.Cast.to_general_value_type(now) is now

    def test_pydantic_model_returned_as_is(self) -> None:
        """BaseModel returned as-is."""

        class TestModel(BaseModel):
            name: str = "test"

        model = TestModel()
        assert u.Cast.to_general_value_type(model) is model

    def test_path_returned_as_is(self) -> None:
        """Path returned as-is."""
        p = Path("/tmp")
        assert u.Cast.to_general_value_type(p) is p

    def test_callable_converted_to_str(self) -> None:
        """Callable is converted to string (line 189)."""
        result = u.Cast.to_general_value_type(lambda: None)
        assert isinstance(result, str)

    def test_list_returned_as_sequence(self) -> None:
        """List returned as Sequence (lines 191-192)."""
        result = u.Cast.to_general_value_type([1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple_returned_as_sequence(self) -> None:
        """Tuple returned as Sequence."""
        result = u.Cast.to_general_value_type((1, 2))
        assert result == (1, 2)

    def test_dict_returned_as_mapping(self) -> None:
        """Dict returned as Mapping."""
        result = u.Cast.to_general_value_type({"key": "value"})
        assert result == {"key": "value"}

    def test_arbitrary_object_converted_to_str(self) -> None:
        """Unknown object type converted to str."""

        class Custom:
            def __str__(self) -> str:
                return "custom_obj"

        result = u.Cast.to_general_value_type(Custom())
        assert result == "custom_obj"


def test_check_direct_access_warning_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(u.Cast, "_APPROVED_MODULES", frozenset())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u.Cast._check_direct_access()
        assert len(caught) >= 1


def test_general_value_bool_branch_after_initial_type_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flext_core import c, m, r, t, u
    import builtins

    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)

    original_isinstance = builtins.isinstance
    state = {"first_bool_check": True}

    def _patched_isinstance(
        obj: object,
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if classinfo is bool and obj is True and state["first_bool_check"]:
            state["first_bool_check"] = False
            return False
        return original_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, "isinstance", _patched_isinstance)
    assert u.Cast.general_value(True, bool) is True
