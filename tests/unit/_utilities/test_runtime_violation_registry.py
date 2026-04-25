"""Tests for u.runtime_violation_registry — process-local typed registry.

Lane A-CH Phase 0 Task 0.2. Validates idempotent drain, clear, and thread-safe
append under concurrent producers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests import m, u


class TestsFlextCoreUtilitiesRuntimeViolationRegistry:
    """Behavior contract for the runtime violation registry."""

    def _new_report(self, rule_id: str = "ENFORCE-099") -> m.ViolationReport:
        return m.ViolationReport(
            rule_id=rule_id,
            outcome=m.ViolationOutcome.HIT,
            file=Path("/tmp/example.py"),
            line=1,
            symbol="example",
            payload={"k": "v"},
        )

    def test_registry_accessor_returns_class(self) -> None:
        reg = u.runtime_violation_registry()
        assert hasattr(reg, "append")
        assert hasattr(reg, "drain")
        assert hasattr(reg, "clear")

    def test_append_then_drain_returns_buffered_entries(self) -> None:
        reg = u.runtime_violation_registry()
        reg.clear()
        rep = self._new_report()
        reg.append(rep)
        out = reg.drain()
        assert len(out) == 1
        assert out[0].rule_id == "ENFORCE-099"

    def test_drain_is_idempotent(self) -> None:
        reg = u.runtime_violation_registry()
        reg.clear()
        reg.append(self._new_report())
        first = reg.drain()
        second = reg.drain()
        assert len(first) == 1
        assert len(second) == 0

    def test_clear_resets_state(self) -> None:
        reg = u.runtime_violation_registry()
        reg.append(self._new_report())
        reg.clear()
        assert len(reg.drain()) == 0

    def test_append_is_thread_safe_under_concurrent_producers(self) -> None:
        """Stress test: 200 concurrent appends must all land."""
        reg = u.runtime_violation_registry()
        reg.clear()
        n_appends = 200

        def _do_append(idx: int) -> None:
            reg.append(self._new_report(f"ENFORCE-{idx % 1000:03d}"))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(_do_append, range(n_appends)))
        out = reg.drain()
        assert len(out) == n_appends
