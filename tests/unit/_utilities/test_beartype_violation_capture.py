"""Beartype hook → ViolationReport capture pipeline behavior contract.

Covers ``u.register_violation_capture`` / ``u.unregister_violation_capture`` /
``u.emit_violation_capture`` end-to-end: registering an existing
``check_<tag>`` hook against a rule_id, firing the capture against a target
that triggers the hook, and asserting a ``m.ViolationReport`` lands in
``u.drain_violation_reports()``. Verifies the unknown-rule_id path is a
no-op (no exception, no registry entry).
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests import c, m, u


class TestsFlextCoreUtilitiesBeartypeViolationCapture:
    """Behavior contract for the beartype-driven violation capture pipeline."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Iterator[None]:
        u.unregister_violation_capture("ENFORCE-905")
        u.unregister_violation_capture("ENFORCE-906")
        u.clear_violation_reports()
        yield
        u.unregister_violation_capture("ENFORCE-905")
        u.unregister_violation_capture("ENFORCE-906")
        u.clear_violation_reports()

    @staticmethod
    def _build_target_with_cast_call(tmp_path: Path) -> type:
        # Synthesize a module on disk whose top-level class triggers
        # check_cast_outside_core (typing.cast call site outside flext-core).
        # Register in sys.modules so inspect.getfile(cls) can resolve back to
        # the synthesized __file__.
        module_name = f"captured_target_fixture_{tmp_path.name}"
        module_path = tmp_path / "captured_target_fixture.py"
        module_path.write_text(
            textwrap.dedent(
                """\
                from typing import cast


                class FlextProbeCaptureFixture:
                    def value(self) -> str:
                        return cast(str, "x")
                """,
            ),
            encoding="utf-8",
        )
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            msg = "spec_from_file_location returned None"
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return module.FlextProbeCaptureFixture

    def test_register_then_emit_appends_violation_report_on_hit(
        self, tmp_path: Path
    ) -> None:
        target = self._build_target_with_cast_call(tmp_path)
        u.register_violation_capture("ENFORCE-905", "check_cast_outside_core")
        u.emit_violation_capture("ENFORCE-905", target)
        drained = u.drain_violation_reports()
        assert len(drained) == 1
        report = drained[0]
        assert isinstance(report, m.ViolationReport)
        assert report.rule_id == "ENFORCE-905"
        assert report.outcome is c.ViolationOutcome.HIT
        assert report.file.endswith("captured_target_fixture.py")

    def test_emit_appends_nothing_when_hook_misses(self) -> None:
        # Use a dynamically created class — has no source file so every
        # A-PT hook source-skips → no payload, no registry entry.
        dyn = type("FlextProbeNoSourceFixture", (), {})
        u.register_violation_capture("ENFORCE-905", "check_cast_outside_core")
        u.emit_violation_capture("ENFORCE-905", dyn)
        assert u.drain_violation_reports() == ()

    def test_emit_with_unknown_rule_id_is_silent_noop(self) -> None:
        dyn = type("FlextProbeUnknownRuleFixture", (), {})
        # Never registered → emit must skip without raising.
        u.emit_violation_capture("ENFORCE-997", dyn)
        assert u.drain_violation_reports() == ()

    def test_unregister_removes_capture_mapping(self, tmp_path: Path) -> None:
        target = self._build_target_with_cast_call(tmp_path)
        u.register_violation_capture("ENFORCE-906", "check_cast_outside_core")
        u.unregister_violation_capture("ENFORCE-906")
        u.emit_violation_capture("ENFORCE-906", target)
        assert u.drain_violation_reports() == ()

    def test_register_with_unknown_hook_skips_emit_silently(self) -> None:
        dyn = type("FlextProbeUnknownHookFixture", (), {})
        u.register_violation_capture("ENFORCE-905", "check_does_not_exist_anywhere")
        u.emit_violation_capture("ENFORCE-905", dyn)
        assert u.drain_violation_reports() == ()
