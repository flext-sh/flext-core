"""Runtime utility smoke coverage aligned to current contracts."""

from __future__ import annotations

from collections import UserDict

from tests import m, t, u


class TestRuntimeCoverage100:
    """Smoke tests for runtime normalization and guards."""

    def test_dict_like_accepts_mapping_inputs(self) -> None:
        assert u.dict_like({"k": "v"})
        assert u.dict_like(UserDict({"k": "v"}))

    def test_dict_like_rejects_model_without_mapping_protocol(self) -> None:
        class NonMapping(m.Value):
            label: str = "x"

        assert not u.dict_like(NonMapping())

    def test_normalize_to_container_keeps_scalar(self) -> None:
        scalar: t.RuntimeData = "value"
        assert u.normalize_to_container(scalar) == "value"

    def test_normalize_to_container_flattens_config_map(self) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": "x"})
        normalized = u.normalize_to_container(cfg)
        assert isinstance(normalized, dict)
        assert normalized.get("a") == 1
