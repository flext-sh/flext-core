"""Tests for FlextUtilitiesGenerators to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilitiesGenerators
from tests import u


class TestsFlextUtilitiesGenerators:
    def test_generate_special_paths_and_dynamic_subclass(self) -> None:
        generated = u.generate(kind="id")
        assert isinstance(generated, str)
        assert generated

        custom = u.generate(
            kind="command",
            options=FlextUtilitiesGenerators.GenerateOptions(
                include_timestamp=True,
                separator="-",
                parts=("part",),
                length=8,
            ),
        )
        assert custom.startswith("cmd-")
        assert "-part-" in custom
        fallback = u.generate(kind="aggregate")
        assert isinstance(fallback, str)

    def test_generators_additional_missed_paths(self) -> None:
        generated = u.generate(
            kind="event",
            options=FlextUtilitiesGenerators.GenerateOptions(separator="-"),
        )
        assert generated.startswith("evt-")
