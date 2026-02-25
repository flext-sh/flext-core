# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel

from flext_core import c, m, r, t, u
from flext_core._utilities.conversion import StrictJsonValue


class _SeqLike(list[int]):
    pass


class _Model(BaseModel):
    value: int


def test_conversion_string_and_join_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"a": 1}), t.ConfigMap)

    assert u.Conversion.to_str(cast("StrictJsonValue", object())).startswith("<")
    assert u.Conversion.to_str_list(None) == []
    assert u.Conversion.to_str_list(cast("StrictJsonValue", _SeqLike([1, 2]))) == [
        "1",
        "2",
    ]
    assert u.Conversion.normalize("Ab") == "Ab"
    assert u.Conversion.join([]) == ""
    assert u.Conversion.join(["A", "B"], case="lower") == "a b"


def test_to_general_value_type_branches() -> None:
    assert u.Conversion.to_general_value_type(None) is None
    model = _Model(value=1)
    assert (
        u.Conversion.to_general_value_type(
            cast("StrictJsonValue", cast("object", model))
        )
        == model
    )
    assert isinstance(
        u.Conversion.to_general_value_type(cast("StrictJsonValue", object())), str
    )


def test_to_flexible_value_and_safe_list_branches() -> None:
    assert u.Conversion.to_flexible_value(None) is None
    assert (
        u.Conversion.to_flexible_value(
            cast("StrictJsonValue", cast("object", _Model(value=1)))
        )
        is None
    )

    mapping_value: Mapping[str, t.GeneralValueType] = {"x": 1}
    assert (
        u.Conversion.to_flexible_value(cast("StrictJsonValue", mapping_value)) is None
    )
    assert (
        u.Conversion.to_flexible_value(
            cast("StrictJsonValue", cast("object", datetime.now(UTC)))
        )
        is not None
    )

    assert u.Conversion.to_str_list_safe(None) == []
    assert u.Conversion.to_str_list_safe("abc") == ["abc"]
    assert u.Conversion.to_str_list_safe(7) == ["7"]


def test_to_flexible_value_fallback_none_branch_for_unsupported_type() -> None:
    assert (
        u.Conversion.to_flexible_value(cast("StrictJsonValue", (lambda: None))) is None
    )
