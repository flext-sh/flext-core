"""Tests for Deprecation utilities full coverage."""

from __future__ import annotations

import warnings

from flext_core import r
from tests import c, m, t, u


def test_deprecated_class_noop_init_branch() -> None:
    assert c.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
    legacy_base = type(
        "LegacyBase", (t.NormalizedValue,), {"__init__": lambda self: None}
    )
    legacy = u.deprecated_class("NewClass", "2.0.0")(
        legacy_base,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy()
        assert len(caught) == 1
