"""Tests for FlextModelsEntity to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from flext_core import c, m, r, u
from flext_core._models.domain_event import _ComparableConfigMap as ComparableConfigMap
from flext_core._models.entity import FlextModelsEntity


def test_entity_comparable_map_and_bulk_validation_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap.model_validate({"k": 1}), m.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    cfg = ComparableConfigMap(root={"a": 1})
    assert (cfg == 1) is False

    # Source raises TypeError for non-dict/Mapping/None data
    with pytest.raises(
        TypeError,
        match="Domain event data must be a dictionary or None",
    ):
        FlextModelsEntity.DomainEvent(
            event_type="evt",
            aggregate_id="agg",
            data=object(),
        )

    entry = FlextModelsEntity.Entry(unique_id="e1")
    bad = entry.add_domain_events_bulk(
        cast("Sequence[tuple[str, m.ConfigMap | None]]", "invalid")
    )
    assert bad.is_failure
