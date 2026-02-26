from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from flext_core import c, m, r, t, u


class _Obj:
    pass


def test_pagination_response_string_fallbacks() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)

    pagination_data = cast(
        Mapping[str, t.ConfigMapValue],
        {"data": _Obj(), "pagination": _Obj()},
    )
    response = u.Pagination.build_pagination_response(pagination_data, message="ok")
    assert response.is_success
    value = response.value
    assert isinstance(value["data"], str)
    assert isinstance(value["pagination"], str)
