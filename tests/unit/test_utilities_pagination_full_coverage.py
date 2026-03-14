"""Tests for FlextUtilitiesPagination to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from flext_core import c, m, r, t, u


class _Obj:
    pass


def test_pagination_response_string_fallbacks() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)
    pagination_data: Mapping[
        str,
        str | Mapping[str, t.Container] | list[t.Container],
    ] = {
        "data": "fallback-data",
        "pagination": "fallback-pagination",
    }
    response = u.build_pagination_response(pagination_data, message="ok")
    assert response.is_success
    value = response.value
    assert value["data"] == "fallback-data"
    assert value["pagination"] == "fallback-pagination"
