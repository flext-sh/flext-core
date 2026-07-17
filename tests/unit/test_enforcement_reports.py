"""Behavioral tests for the enforcement Report/Violation models and emit engine.

Every test asserts an observable public contract: model field/state via the
public API, ``check``/``check_model_construction`` report contents, and the
warnings/exceptions ``emit`` produces for a caller. No private attribute,
internal helper, or implementation hook is inspected.
"""

from __future__ import annotations

import typing
import warnings
from typing import Annotated

import pytest

from flext_core._constants.enforcement import FlextMroViolation, FlextSmellViolation
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests import c
from tests import m
from tests import u


def _hard_violation(
    *,
    qualname: str = "X.Y",
    message: str = "boom [ENFORCE-001]",
    rule_id: str = "",
    anchor: str = "",
) -> p.Violation:
    """Build a Model-layer HARD-rules violation for emit-focused tests."""
    return m.Violation(
        qualname=qualname,
        layer="Model",
        severity="HARD rules",
        message=message,
        rule_id=rule_id,
        agents_md_anchor=anchor,
    )


class TestsFlextCoreEnforcementReports:
    # --- Report container contract -------------------------------------

    def test_empty_report_is_falsy_and_reports_zero_length(self) -> None:
        report = m.Report()

        assert not report
        assert report.empty
        assert len(report) == 0
        assert report.messages == []

    def test_nonempty_report_exposes_messages_via_public_protocol(self) -> None:
        violation = _hard_violation(message="boom")
        report = m.Report(violations=[violation])

        assert report
        assert not report.empty
        assert len(report) == 1
        assert report[0] == "boom"
        assert report.messages == ["boom"]
        assert "boom" in report

    def test_report_membership_ignores_non_string_fragments(self) -> None:
        report = m.Report(violations=[_hard_violation(message="boom")])

        assert 123 not in report
        assert None not in report

    def test_report_aggregates_all_violation_messages_in_order(self) -> None:
        first = _hard_violation(message="a")
        second = _hard_violation(message="b")
        report = m.Report(violations=[first, second])

        assert len(report) == 2
        assert report.messages == ["a", "b"]

    # --- Violation model contract --------------------------------------

    def test_violation_optional_fields_default_to_empty(self) -> None:
        violation = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="m",
        )

        assert violation.rule_id == ""
        assert violation.agents_md_anchor == ""
        assert violation.file_path == ""
        assert violation.line_number == 0

    def test_violation_is_frozen_and_rejects_mutation(self) -> None:
        violation = _hard_violation()

        # Pydantic's frozen ValidationError subclasses ValueError.
        with pytest.raises(ValueError):
            violation.qualname = "other"

    def test_violation_model_dump_exposes_public_fields(self) -> None:
        violation = m.Violation(
            qualname="Pkg.Cls",
            layer="Model",
            severity="HARD rules",
            message="msg",
            rule_id="ENFORCE-001",
            agents_md_anchor="3.1",
        )

        dumped = violation.model_dump()

        assert dumped["qualname"] == "Pkg.Cls"
        assert dumped["rule_id"] == "ENFORCE-001"
        assert dumped["agents_md_anchor"] == "3.1"
        assert dumped["message"] == "msg"

    # --- check() report contents ---------------------------------------

    def test_check_flags_any_typed_field_with_rule_metadata(self) -> None:
        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)

        assert any(
            violation.rule_id == "ENFORCE-039" or "no_any" in violation.message
            for violation in report.violations
        )

    def test_check_messages_embed_bracketed_rule_identifiers(self) -> None:
        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)

        assert report.violations
        assert all(
            "[" in violation.message and "]" in violation.message
            for violation in report.violations
        )

    def test_check_skips_function_local_classes(self) -> None:
        def _make() -> type:
            class Inner:
                pass

            return Inner

        report = u.check(_make())

        assert all(violation.layer != "namespace" for violation in report.violations)

    def test_check_model_construction_flags_any_field(self) -> None:
        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = FlextUtilitiesEnforcement.check_model_construction(_WithAny)

        assert report.violations
        assert any("no_any" in violation.message for violation in report.violations)

    # --- emit() warning/exception behaviour ----------------------------

    def test_emit_warn_mode_raises_one_warning_per_violation(self) -> None:
        report = m.Report(
            violations=[
                _hard_violation(qualname="X.Y", rule_id="ENFORCE-001", anchor="3.1"),
                _hard_violation(qualname="X.Z", rule_id="ENFORCE-001", anchor="3.1"),
            ],
        )

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

    def test_emit_strict_mode_warns_then_raises_on_first_violation(self) -> None:
        report = m.Report(
            violations=[
                _hard_violation(qualname="X.Y", rule_id="ENFORCE-001", anchor="3.1"),
                _hard_violation(qualname="X.Z", rule_id="ENFORCE-001", anchor="3.1"),
            ],
        )
        expected = (
            "\nX.Y violates FLEXT Model HARD rules:\n  - boom [ENFORCE-001]"
            "\n\nFix: See AGENTS.md §3.1 and search for ENFORCE-001."
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(TypeError) as excinfo:
                FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.STRICT)

        assert str(excinfo.value) == expected
        assert len(recorded) == 1
        assert recorded[0].category is FlextMroViolation
        assert str(recorded[0].message) == expected

    def test_emit_off_mode_is_silent(self) -> None:
        report = m.Report(violations=[_hard_violation(message="boom")])

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
    def test_emit_fix_guidance_falls_back_by_rule_id_and_anchor(
        self,
        rule_id: str,
        anchor: str,
        expected_fix: str,
    ) -> None:
        report = m.Report(
            violations=[
                _hard_violation(message="boom", rule_id=rule_id, anchor=anchor),
            ],
        )

        with pytest.warns(FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        assert str(caught[0].message).endswith(expected_fix)

    def test_emit_uses_smell_category_for_code_smell_rules(self) -> None:
        report = m.Report(
            violations=[
                _hard_violation(message="smell [ENFORCE-071]", rule_id="ENFORCE-071"),
            ],
        )

        with pytest.warns(FlextSmellViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        assert caught[0].category is FlextSmellViolation

    def test_emit_of_checked_report_carries_layer_tag_and_fix(self) -> None:
        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)
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
