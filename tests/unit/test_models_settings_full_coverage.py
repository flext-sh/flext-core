"""Tests for FlextModelsSettings to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import pytest

from flext_core import FlextModelsConfig, r
from tests import c, m, t, u


def test_models_settings_branch_paths() -> None:
    class _ValidatorSpecStub:
        def __call__(self, value: t.Container) -> bool:
            _ = value
            return True

        def __and__(self, other: t.NormalizedValue) -> _ValidatorSpecStub:
            _ = other
            return self

        def __invert__(self) -> _ValidatorSpecStub:
            return self

        def __or__(self, other: t.NormalizedValue) -> _ValidatorSpecStub:
            _ = other
            return self

    assert c.UNKNOWN_ERROR
    assert isinstance(m.Categories(categories={}), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
    assert u.to_str(1) == "1"
    assert isinstance(FlextModelsConfig._get_log_level_from_config(), int)
    with pytest.raises(ValueError, match="HTTP status code validation failed"):
        FlextModelsConfig.RetryConfiguration(
            retry_on_exceptions=[],
            retry_on_status_codes=[9999],
        )
    with pytest.raises(ValueError, match="max_delay_seconds"):
        FlextModelsConfig.RetryConfiguration(
            retry_on_exceptions=[],
            retry_on_status_codes=[],
            initial_delay_seconds=2.0,
            max_delay_seconds=1.0,
        )
    validator = _ValidatorSpecStub()
    validation_config = FlextModelsConfig.ValidationConfiguration(
        custom_validators=[validator]
    )
    assert validation_config.custom_validators == [validator]
    with pytest.raises(ValueError, match="less than or equal to 1000"):
        FlextModelsConfig.BatchProcessingConfig(batch_size=100000, data_items=[])


def test_models_settings_context_validator_and_non_standard_status_input() -> None:
    req = FlextModelsConfig.ProcessingRequest(
        operation_id="op-1",
        data=t.ConfigMap(root={}),
        context=t.ConfigMap(root={}),
    )
    assert "trace_id" in req.context

    class _CodeObj:
        @override
        def __repr__(self) -> str:
            return "503"

    code_str: t.Scalar = str(_CodeObj())
    converted = FlextModelsConfig.RetryConfiguration.validate_backoff_strategy([
        code_str,
    ])
    assert converted == [503]
    status_codes: Sequence[t.Scalar] = ["503"]
    converted_str = FlextModelsConfig.RetryConfiguration.validate_backoff_strategy(
        status_codes,
    )
    assert converted_str == [503]
