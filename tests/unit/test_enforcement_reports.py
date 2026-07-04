"""Enforcement report model tests."""

from __future__ import annotations

import typing
import warnings
from typing import Annotated

import pytest

from flext_core import FlextMroViolation, FlextUtilitiesEnforcement
from tests.constants import c
from tests.models import m
from tests.utilities import u


class TestsFlextEnforcementReports:
    @pytest.mark.parametrize(
        "base_cls",
        [
            m.ArbitraryTypesModel,
            m.StrictBoundaryModel,
            m.FlexibleInternalModel,
            m.ImmutableValueModel,
            m.TaggedModel,
            m.ContractModel,
        ],
    )
    def test_base_model_has_enforcement_hook(self, base_cls: type) -> None:
        assert hasattr(base_cls, "__pydantic_init_subclass__")

    def test_empty_report_is_falsy(self) -> None:
        report = m.Report()
        assert not report
        assert report.empty
        assert len(report) == 0

    def test_nonempty_report_is_truthy(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="boom",
        )
        report = m.Report(violations=[v])
        assert report
        assert not report.empty
        assert len(report) == 1
        assert report[0] == "boom"
        assert "boom" in report

    def test_violation_includes_rule_metadata(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_M)
        assert any(
            v.rule_id == "ENFORCE-039" or "no_any" in v.message
            for v in report.violations
        )

    def test_violation_messages_include_rule_identifiers(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_M)
        assert any("[" in v.message and "]" in v.message for v in report.violations)

    def test_merge_reports(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="a",
        )
        w = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="b",
        )
        merged = m.Report(violations=[v, w])
        assert len(merged) == 2

    def test_function_local_class_skipped(self) -> None:
        """Classes defined inside functions carry ``<locals>`` in qualname."""

        class _OuterScope:
            def make(self) -> type:
                class Inner:  # nested inside a method → `<locals>` qualname
                    pass

                return Inner

        cls = _OuterScope().make()
        assert "<locals>" in cls.__qualname__
        # check returns empty report for function-local classes
        report = u.check(cls)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_emit_warn_mode_emits_flext_mro_violation_per_violation(self) -> None:
        violation = m.Violation(
            qualname="X.Y",
            layer="Model",
            severity="HARD rules",
            message="boom [ENFORCE-001]",
            rule_id="ENFORCE-001",
            agents_md_anchor="3.1",
        )
        second = violation.model_copy(update={"qualname": "X.Z"})
        report = m.Report(violations=[violation, second])
        with pytest.warns(FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)
        assert len(caught) == 2
        assert str(caught[0].message) == (
            "\nX.Y violates FLEXT Model HARD rules:\n  - boom [ENFORCE-001]"
            "\n\nFix: See AGENTS.md §3.1 and search for ENFORCE-001."
        )
        assert str(caught[1].message).startswith(
            "\nX.Z violates FLEXT Model HARD rules",
        )

    def test_emit_strict_mode_warns_then_raises_type_error(self) -> None:
        violation = m.Violation(
            qualname="X.Y",
            layer="Model",
            severity="HARD rules",
            message="boom [ENFORCE-001]",
            rule_id="ENFORCE-001",
            agents_md_anchor="3.1",
        )
        second = violation.model_copy(update={"qualname": "X.Z"})
        report = m.Report(violations=[violation, second])
        expected = (
            "\nX.Y violates FLEXT Model HARD rules:\n  - boom [ENFORCE-001]"
            "\n\nFix: See AGENTS.md §3.1 and search for ENFORCE-001."
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(TypeError) as excinfo:
                FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.STRICT)
        assert str(excinfo.value) == expected
        # STRICT warns first, then raises on the FIRST violation — the
        # second violation is never reached, so exactly one warning exists.
        assert len(recorded) == 1
        assert recorded[0].category is FlextMroViolation
        assert str(recorded[0].message) == expected

    def test_emit_off_mode_is_silent(self) -> None:
        violation = m.Violation(
            qualname="X.Y",
            layer="Model",
            severity="HARD rules",
            message="boom",
        )
        report = m.Report(violations=[violation])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.OFF)
        assert recorded == []

    @pytest.mark.parametrize(
        "mode",
        [
            c.EnforcementMode.OFF,
            c.EnforcementMode.WARN,
            c.EnforcementMode.STRICT,
        ],
    )
    def test_emit_empty_report_is_silent_in_every_mode(
        self,
        mode: c.EnforcementMode,
    ) -> None:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(m.Report(), mode=mode)
        assert recorded == []

    @pytest.mark.parametrize(
        ("rule_id", "anchor", "expected_fix"),
        [
            (
                "ENFORCE-001",
                "3.1",
                "Fix: See AGENTS.md §3.1 and search for ENFORCE-001.",
            ),
            ("ENFORCE-001", "", "Fix: Search for enforcement rule ENFORCE-001."),
            ("", "3.1", "Fix: See AGENTS.md §3.1."),
            ("", "", "Fix: See AGENTS.md § Model governance."),
        ],
    )
    def test_emit_fix_guidance_covers_rule_id_and_anchor_fallbacks(
        self,
        rule_id: str,
        anchor: str,
        expected_fix: str,
    ) -> None:
        violation = m.Violation(
            qualname="X.Y",
            layer="Model",
            severity="HARD rules",
            message="boom",
            rule_id=rule_id,
            agents_md_anchor=anchor,
        )
        report = m.Report(violations=[violation])
        with pytest.warns(FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)
        assert str(caught[0].message).endswith(expected_fix)

    def test_emit_check_report_carries_tag_layer_severity_and_fix(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_M)
        assert report.violations
        with pytest.warns(FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)
        texts = [str(entry.message) for entry in caught]
        assert any(
            "violates FLEXT Model HARD rules" in text
            and "[no_any]" in text
            and "Fix: See AGENTS.md § Model governance." in text
            for text in texts
        )
