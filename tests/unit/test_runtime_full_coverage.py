"""Runtime behavior tests for current utility/runtime surface."""

from __future__ import annotations

from tests import m, t, u


class TestRuntimeFullCoverage:
    """Validate deterministic runtime helper behavior."""

    def test_normalize_to_container_handles_scalar(self) -> None:
        payload: t.JsonPayload = "flext"
        result = u.normalize_to_container(payload)
        assert result == "flext"

    def test_normalize_to_container_handles_flat_mapping(self) -> None:
        payload: t.JsonPayload = {"name": "flext", "enabled": True}
        result = u.normalize_to_container(payload)
        assert isinstance(result, dict)
        assert result.get("name") == "flext"

    def test_normalize_to_container_handles_config_map(self) -> None:
        payload = m.ConfigMap(root={"count": 2})
        result = u.normalize_to_container(payload)
        assert isinstance(result, dict)
        assert result.get("count") == 2
