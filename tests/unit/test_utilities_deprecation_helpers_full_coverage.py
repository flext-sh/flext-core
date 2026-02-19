from __future__ import annotations

import warnings

from flext_core import c, m, r, t, u


helpers = __import__(
    "flext_core._utilities._deprecation_helpers",
    fromlist=["warn_direct_module_access", "APPROVED_MODULES"],
)


def test_warn_direct_module_access_default_facade_path(monkeypatch) -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    monkeypatch.setattr(helpers, "APPROVED_MODULES", frozenset())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        helpers.warn_direct_module_access("conversion")
        assert len(caught) == 1
        assert "u.Conversion" in str(caught[0].message)
