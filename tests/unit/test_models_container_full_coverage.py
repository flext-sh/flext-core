"""Tests for FlextModelsContainer to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import r
from tests import c, m, u


def test_container_resource_registration_metadata_normalized() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap({"k": 1}), t.ConfigMap)
    assert u.to_str(1) == "1"
    reg = m.ResourceRegistration(
        name="r1",
        factory=lambda: 1,
        metadata=m.Metadata(attributes={"value": "x"}),
    )
    assert reg.metadata is not None
    assert isinstance(reg.metadata, m.Metadata)
    assert reg.metadata.attributes["value"] == "x"
