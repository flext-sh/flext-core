from __future__ import annotations

import pytest

from flext_core import c, m, r, t, u


service_models = __import__(
    "flext_core._models.service", fromlist=["FlextModelsService"]
)
FlextModelsService = service_models.FlextModelsService


def test_service_request_timeout_validator_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    with pytest.raises(ValueError, match="greater than 0"):
        FlextModelsService.DomainServiceExecutionRequest(
            service_name="svc",
            method_name="op",
            timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="less than or equal"):
        FlextModelsService.DomainServiceExecutionRequest(
            service_name="svc",
            method_name="op",
            timeout_seconds=c.Performance.MAX_TIMEOUT_SECONDS + 1,
        )


def test_service_request_timeout_post_validator_messages() -> None:
    with pytest.raises(ValueError, match="Timeout must be positive"):
        FlextModelsService.DomainServiceExecutionRequest.validate_timeout(-1.0)

    with pytest.raises(ValueError, match="exceeds maximum"):
        FlextModelsService.DomainServiceExecutionRequest.validate_timeout(
            c.Performance.MAX_TIMEOUT_SECONDS + 10.0
        )

    assert FlextModelsService.DomainServiceExecutionRequest.validate_timeout(1.0) == 1.0
