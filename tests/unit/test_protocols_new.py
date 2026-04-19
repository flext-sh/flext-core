"""Protocol facade smoke tests aligned to current protocol surface."""

from __future__ import annotations

from tests import p, r


class TestFlextProtocols:
    """Validate core protocol facade availability and basic compliance checks."""

    def test_protocol_facade_exports_base_protocols(self) -> None:
        assert getattr(p, "Base", None) is not None
        assert getattr(p, "Result", None) is not None
        assert getattr(p, "Container", None) is not None

    def test_result_successful_object_exposes_result_contract(self) -> None:
        result = r[str].ok("ok")
        assert result.success
        assert getattr(p, "SuccessCheckable", None) is not None
