"""Tests for m to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests import c, m, r, t, u


class TestModelsEntityFullCoverage:
    """Tests for m to achieve full coverage."""

    def test_entity_comparable_map_and_bulk_validation_paths(self) -> None:
        assert c.ErrorCode.UNKNOWN_ERROR
        assert r[int].ok(1).success
        assert isinstance(m.ConfigMap(root={"k": 1}), m.ConfigMap)
        assert u.to_str(1) == "1"
        cfg = m.ConfigMap(root={"a": 1})
        assert (cfg == 1) is False
        with pytest.raises(ValidationError):
            m.DomainEvent.model_validate(
                {
                    "event_type": "evt",
                    "aggregate_id": "agg",
                    "data": 1,
                },
            )
        entry = m.Entity(unique_id="e1")
        assert not hasattr(entry, "add_domain_events_bulk")


__all__: t.MutableSequenceOf[str] = ["TestModelsEntityFullCoverage"]
