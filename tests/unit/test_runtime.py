"""Runtime facade tests for current contracts."""

from __future__ import annotations

from tests import t, u


class TestFlextRuntime:
    """Public runtime utility checks."""

    def test_dict_like_detects_mapping_inputs(self) -> None:
        payload: t.JsonPayload = {"a": 1}
        assert u.dict_like(payload)

    def test_dict_like_rejects_scalar_inputs(self) -> None:
        payload: t.JsonPayload = "value"
        assert not u.dict_like(payload)

    def test_normalize_to_container_returns_flat_scalar(self) -> None:
        payload: t.JsonPayload = 42
        assert u.normalize_to_container(payload) == 42
