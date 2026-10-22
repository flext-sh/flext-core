"""Behavioral tests for the enforcement Report/Violation models and emit engine.

Every test asserts an observable public contract: model field/state via the
public API, ``check``/``check_model_construction`` report contents, and the
warnings/exceptions ``emit`` produces for a caller. No private attribute,
internal helper, or implementation hook is inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from flext_core.utilities import FlextUtilitiesEnforcement
from tests import c, m, tm, u
from tests.unit._enforcement_support import invalid_annotations


def _hard_violation(
    *,
    qualname: str = "X.Y",
    message: str = "boom [ENFORCE-001]",
    rule_id: str = "",
    anchor: str = "",
) -> m.Violation:
    """Build a Model-layer HARD-rules violation for emit-focused tests.

    Returns:
        The resulting ``m.Violation``.

    """
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
    """Tests for ``FlextCoreEnforcementReports``."""

    @staticmethod
    def test_empty_report_is_falsy_and_reports_zero_length() -> None:
        """Test empty report is falsy and reports zero length."""
        report = m.Report()

        tm.that(report, truth=False)
        tm.that(report.empty, truth=True)
        tm.that(len(report), eq=0)
        tm.that(report.messages, eq=[])

    @staticmethod
    def test_report_validates_round_trip_and_rejects_invalid_or_mutated_state() -> None:
        """Public reports validate their payload and preserve frozen state."""
        report = m.Report(violations=(_hard_violation(message="validated"),))
        payload = report.model_dump_json(round_trip=True)
        restored = m.Report.model_validate_json(payload)
        tm.that(restored.model_dump_json(round_trip=True), eq=payload)
        tm.that(restored.messages, eq=["validated"])
        with pytest.raises(m.ValidationError):
            m.Report.model_validate({"violations": ({"message": "incomplete"},)})
        with pytest.raises(m.ValidationError):
            m.Report.model_validate({"unexpected": True})
        field_name = "violations"
        with pytest.raises(m.ValidationError):
            setattr(report, field_name, ())

    @staticmethod
    def test_nonempty_report_exposes_messages_via_public_protocol() -> None:
        """Test nonempty report exposes messages via public protocol."""
        violation = _hard_violation(message="boom")
        report = m.Report(violations=[violation])

        tm.that(report, truth=True)
        tm.that(report.empty, truth=False)
        tm.that(len(report), eq=1)
        tm.that(report[0], eq="boom")
        tm.that(report.messages, eq=["boom"])
        tm.that(report, where=lambda value: "boom" in value)

    @staticmethod
    def test_report_membership_ignores_non_string_fragments() -> None:
        """Test report membership ignores non string fragments."""
        report = m.Report(violations=[_hard_violation(message="boom")])

        tm.that(report, where=lambda value: 123 not in value)
        tm.that(report, where=lambda value: None not in value)

    @staticmethod
    def test_report_aggregates_all_violation_messages_in_order() -> None:
        """Test report aggregates all violation messages in order."""
        first = _hard_violation(message="a")
        second = _hard_violation(message="b")
        report = m.Report(violations=[first, second])

        tm.that(len(report), eq=2)
        tm.that(report.messages, eq=["a", "b"])

    # --- Violation model contract --------------------------------------

    @staticmethod
    def test_violation_optional_fields_default_to_empty() -> None:
        """Test violation optional fields default to empty."""
        violation = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="m",
        )

        tm.that(violation.rule_id, truth=False)
        tm.that(violation.agents_md_anchor, truth=False)
        tm.that(violation.file_path, truth=False)
        tm.that(violation.line_number, eq=0)

    @staticmethod
    def test_violation_is_frozen_and_rejects_mutation() -> None:
        """Test violation is frozen and rejects mutation."""
        violation = _hard_violation()

        # Pydantic's frozen ValidationError subclasses ValueError.
        field_name = "qualname"
        with pytest.raises(m.ValidationError):
            setattr(violation, field_name, "other")

    @staticmethod
    def test_violation_model_dump_exposes_public_fields() -> None:
        """Test violation model dump exposes public fields."""
        violation = m.Violation(
            qualname="Pkg.Cls",
            layer="Model",
            severity="HARD rules",
            message="msg",
            rule_id="ENFORCE-001",
            agents_md_anchor="3.1",
        )

        dumped = violation.model_dump()

        tm.that(dumped["qualname"], eq="Pkg.Cls")
        tm.that(dumped["rule_id"], eq="ENFORCE-001")
        tm.that(dumped["agents_md_anchor"], eq="3.1")
        tm.that(dumped["message"], eq="msg")

    # --- check() report contents ---------------------------------------

    @staticmethod
    def test_check_flags_any_typed_field_with_rule_metadata(tmp_path: Path) -> None:
        """Test check flags any typed field with rule metadata."""
        with invalid_annotations(tmp_path) as source:
            target = source.BadModel
            if not isinstance(target, type):
                pytest.fail("Invalid source fixture did not declare a model class")
            report = u.check(target)

        tm.that(
            any(
                violation.rule_id == "ENFORCE-039" or "no_any" in violation.message
                for violation in report.violations
            ),
            truth=True,
        )

    @staticmethod
    def test_check_messages_embed_bracketed_rule_identifiers(tmp_path: Path) -> None:
        """Test check messages embed bracketed rule identifiers."""
        with invalid_annotations(tmp_path) as source:
            target = source.BadModel
            if not isinstance(target, type):
                pytest.fail("Invalid source fixture did not declare a model class")
            report = u.check(target)

        tm.that(report.violations, truth=True)
        tm.that(
            all(
                "[" in violation.message and "]" in violation.message
                for violation in report.violations
            ),
            truth=True,
        )

    @staticmethod
    def test_check_skips_function_local_classes() -> None:
        """Test check skips function local classes."""

        def _make() -> type:
            class Inner:
                pass

            return Inner

        report = u.check(_make())

        tm.that(
            all(violation.layer != "namespace" for violation in report.violations),
            truth=True,
        )

    @staticmethod
    def test_check_model_construction_flags_any_field(tmp_path: Path) -> None:
        """Test check model construction flags any field."""
        with invalid_annotations(tmp_path) as source:
            target = source.BadModel
            if not isinstance(target, type) or not issubclass(target, m.BaseModel):
                pytest.fail("Invalid source fixture did not declare a native model")
            report = FlextUtilitiesEnforcement.check_model_construction(target)

        tm.that(report.violations, truth=True)
        tm.that(
            any("no_any" in violation.message for violation in report.violations),
            truth=True,
        )

    # --- emit() warning/exception behaviour ----------------------------

    @staticmethod
    def test_emit_warn_mode_raises_one_warning_per_violation() -> None:
        """Test emit warn mode raises one warning per violation."""
        report = m.Report(
            violations=[
                _hard_violation(qualname="X.Y", rule_id="ENFORCE-001", anchor="3.1"),
                _hard_violation(qualname="X.Z", rule_id="ENFORCE-001", anchor="3.1"),
            ],
        )

        with pytest.warns(c.FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        tm.that(len(caught), eq=2)
        tm.that(
            str(caught[0].message),
            eq="\nX.Y violates FLEXT Model HARD rules:\n  - boom [ENFORCE-001]\n\nFix: See AGENTS.md §3.1 and search for ENFORCE-001.",
        )
        tm.that(
            str(caught[1].message).startswith("\nX.Z violates FLEXT Model HARD rules"),
            truth=True,
        )

    @staticmethod
    def test_emit_strict_mode_warns_then_raises_on_first_violation() -> None:
        """Test emit strict mode warns then raises on first violation."""
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

        tm.that(str(excinfo.value), eq=expected)
        tm.that(len(recorded), eq=1)
        tm.that(
            recorded[0].category, where=lambda category: category is c.FlextMroViolation
        )
        tm.that(str(recorded[0].message), eq=expected)

    @staticmethod
    def test_emit_off_mode_is_silent() -> None:
        """Test emit off mode is silent."""
        report = m.Report(violations=[_hard_violation(message="boom")])

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.OFF)

        tm.that(recorded, eq=[])

    @staticmethod
    @pytest.mark.parametrize(
        "mode",
        [c.EnforcementMode.OFF, c.EnforcementMode.WARN, c.EnforcementMode.STRICT],
    )
    def test_emit_empty_report_is_silent_in_every_mode(
        mode: c.EnforcementMode,
    ) -> None:
        """Test emit empty report is silent in every mode."""
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(m.Report(), mode=mode)

        tm.that(recorded, eq=[])

    @staticmethod
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
        rule_id: str,
        anchor: str,
        expected_fix: str,
    ) -> None:
        """Test emit fix guidance falls back by rule id and anchor."""
        report = m.Report(
            violations=[
                _hard_violation(message="boom", rule_id=rule_id, anchor=anchor),
            ],
        )

        with pytest.warns(c.FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        tm.that(str(caught[0].message).endswith(expected_fix), truth=True)

    @staticmethod
    def test_emit_uses_smell_category_for_code_smell_rules() -> None:
        """Test emit uses smell category for code smell rules."""
        report = m.Report(
            violations=[
                _hard_violation(message="smell [ENFORCE-071]", rule_id="ENFORCE-071"),
            ],
        )

        with pytest.warns(c.FlextSmellViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        tm.that(
            caught[0].category, where=lambda category: category is c.FlextSmellViolation
        )

    @staticmethod
    def test_emit_of_checked_report_carries_layer_tag_and_fix(tmp_path: Path) -> None:
        """Test emit of checked report carries layer tag and fix."""
        with invalid_annotations(tmp_path) as source:
            target = source.BadModel
            if not isinstance(target, type):
                pytest.fail("Invalid source fixture did not declare a model class")
            report = u.check(target)
        tm.that(report.violations, truth=True)

        with pytest.warns(c.FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        texts = [str(entry.message) for entry in caught]
        tm.that(
            any(
                "violates FLEXT Model HARD rules" in text
                and "[no_any]" in text
                and ("Fix: See AGENTS.md § Model governance." in text)
                for text in texts
            ),
            truth=True,
        )
