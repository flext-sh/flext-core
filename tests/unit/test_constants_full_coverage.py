from __future__ import annotations

from enum import auto

from flext_core import c, m, r, t, u


constants_mod = __import__(
    "flext_core.constants", fromlist=["AutoStrEnum", "BiMapping"]
)
AutoStrEnum = constants_mod.AutoStrEnum
BiMapping = constants_mod.BiMapping


class _Status(AutoStrEnum):
    READY = auto()


def test_constants_auto_enum_and_bimapping_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    assert _Status.READY.value == "ready"
    generated = AutoStrEnum._generate_next_value_("X", 1, 2, ["a"])
    assert generated == "x"

    bm = BiMapping({"a": 1})
    assert bm.forward["a"] == 1
    assert bm.inverse[1] == "a"
    assert "BiMapping" in repr(bm)
