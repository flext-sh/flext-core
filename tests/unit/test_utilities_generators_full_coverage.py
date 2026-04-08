"""Tests for FlextUtilitiesGenerators to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo

import pytest

from flext_core import _utilities
from tests import m, u


class TestUtilitiesGeneratorsFullCoverage:
    def test_enrich_and_ensure_trace_context_branches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ids = iter(["trace-x", "span-x", "corr-x"])
        monkeypatch.setattr(
            u,
            "generate_id",
            staticmethod(lambda: next(ids)),
        )
        monkeypatch.setattr(
            u,
            "generate_datetime_utc",
            staticmethod(lambda: datetime(2026, 1, 1, tzinfo=UTC)),
        )
        enriched = u.ensure_trace_context(
            m.Core.Unit._GoodModel(value=9),
            include_correlation_id=True,
            include_timestamp=True,
        )
        assert enriched["value"] == "9"
        assert enriched["trace_id"] == "trace-x"
        assert enriched["span_id"] == "span-x"
        assert enriched["correlation_id"] == "corr-x"
        assert enriched["timestamp"] == "2026-01-01T00:00:00+00:00"
        existing = {
            "trace_id": "already-trace",
            "span_id": "already-span",
            "correlation_id": "already-corr",
            "timestamp": "already-ts",
        }
        preserved = u.ensure_trace_context(
            existing,
            include_correlation_id=True,
            include_timestamp=True,
        )
        assert preserved == existing

    def test_generate_special_paths_and_dynamic_subclass(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generated = u.generate(kind="id")
        assert isinstance(generated, str)
        assert generated
        fixed_ts = datetime(2026, 1, 2, tzinfo=UTC)

        class _FixedDatetime:
            @staticmethod
            def now(tz: tzinfo | None = None) -> datetime:
                _ = tz
                return fixed_ts

        monkeypatch.setattr(_utilities.generators, "datetime", _FixedDatetime)
        custom = u.generate(
            kind="command",
            include_timestamp=True,
            separator="-",
            parts=("part",),
            length=8,
        )
        assert custom.startswith("cmd-")
        assert "-part-" in custom
        fallback = u.generate(kind="aggregate")
        assert isinstance(fallback, str)

    def test_generators_additional_missed_paths(self) -> None:
        generated = u.generate(kind="event", separator="-")
        assert generated.startswith("evt-")
