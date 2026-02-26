from __future__ import annotations

import warnings

from flext_core import c, m, r, t, u


def test_deprecated_class_noop_init_branch() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)

    _LegacyBase = type("_LegacyBase", (object,), {"__init__": None})
    _Legacy = u.Deprecation.deprecated_class(replacement="New", version="1.0")(_LegacyBase)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _Legacy()
        assert len(caught) == 1
