"""Tests for FlextModelsSettings to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_core import c, m, r, t, u

settings_models = __import__(
    "flext_core._models.settings", fromlist=["FlextModelsConfig"]
)
FlextModelsConfig = settings_models.FlextModelsConfig


def test_models_settings_branch_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    assert isinstance(FlextModelsConfig._get_log_level_from_config(), int)

    with pytest.raises(ValueError, match="HTTP status code validation failed"):
        FlextModelsConfig.RetryConfiguration(retry_on_status_codes=[9999])

    with pytest.raises(ValueError, match="max_delay_seconds"):
        FlextModelsConfig.RetryConfiguration(
            initial_delay_seconds=2.0, max_delay_seconds=1.0
        )

    with pytest.raises(TypeError, match="Validator must be callable"):
        FlextModelsConfig.ValidationConfiguration(custom_validators=[1])

    with pytest.raises(ValueError, match="Batch size cannot exceed"):
        FlextModelsConfig.BatchProcessingConfig(batch_size=100000)


def test_models_settings_context_validator_and_non_standard_status_input() -> None:
    req = FlextModelsConfig.ProcessingRequest(
        context={},
    )
    assert "trace_id" in req.context

    class _CodeObj:
        def __repr__(self) -> str:
            return "503"

    converted = FlextModelsConfig.RetryConfiguration.validate_backoff_strategy(
        [_CodeObj()],
    )
    assert converted == [503]

    converted_str = FlextModelsConfig.RetryConfiguration.validate_backoff_strategy(
        ["503"],
    )
    assert converted_str == [503]
