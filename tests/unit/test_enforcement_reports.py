"""Behavioral tests for the enforcement Report/Violation models and emit engine.

Every test asserts an observable public contract: model field/state via the
public API, ``check``/``check_model_construction`` report contents, and the
warnings/exceptions ``emit`` produces for a caller. No private attribute,
internal helper, or implementation hook is inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
import warnings
from typing import Annotated

import pytest

from flext_core.utilities import FlextUtilitiesEnforcement
from tests.constants import c
from tests.models import m
from tests.utilities import u


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

        assert not report
        assert report.empty
        assert len(report) == 0
        assert report.messages == []

    @staticmethod
    def test_nonempty_report_exposes_messages_via_public_protocol() -> None:
        """Test nonempty report exposes messages via public protocol."""
        violation = _hard_violation(message="boom")
        report = m.Report(violations=[violation])

        assert report
        assert not report.empty
        assert len(report) == 1
        assert report[0] == "boom"
        assert report.messages == ["boom"]
        assert "boom" in report

    @staticmethod
    def test_report_membership_ignores_non_string_fragments() -> None:
        """Test report membership ignores non string fragments."""
        report = m.Report(violations=[_hard_violation(message="boom")])

        assert 123 not in report
        assert None not in report

    @staticmethod
    def test_report_aggregates_all_violation_messages_in_order() -> None:
        """Test report aggregates all violation messages in order."""
        first = _hard_violation(message="a")
        second = _hard_violation(message="b")
        report = m.Report(violations=[first, second])

        assert len(report) == 2
        assert report.messages == ["a", "b"]

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

        assert not violation.rule_id
        assert not violation.agents_md_anchor
        assert not violation.file_path
        assert violation.line_number == 0

    @staticmethod
    def test_violation_is_frozen_and_rejects_mutation() -> None:
        """Test violation is frozen and rejects mutation."""
        violation = _hard_violation()

        # Pydantic's frozen ValidationError subclasses ValueError.
        with pytest.raises(m.ValidationError):
            violation.qualname = "other"

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

        assert dumped["qualname"] == "Pkg.Cls"
        assert dumped["rule_id"] == "ENFORCE-001"
        assert dumped["agents_md_anchor"] == "3.1"
        assert dumped["message"] == "msg"

    # --- check() report contents ---------------------------------------

    @staticmethod
    def test_check_flags_any_typed_field_with_rule_metadata() -> None:
        """Test check flags any typed field with rule metadata."""

        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)

        assert any(
            violation.rule_id == "ENFORCE-039" or "no_any" in violation.message
            for violation in report.violations
        )

    @staticmethod
    def test_check_messages_embed_bracketed_rule_identifiers() -> None:
        """Test check messages embed bracketed rule identifiers."""

        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)

        assert report.violations
        assert all(
            "[" in violation.message and "]" in violation.message
            for violation in report.violations
        )

    @staticmethod
    def test_check_skips_function_local_classes() -> None:
        """Test check skips function local classes."""

        def _make() -> type:
            class Inner:
                pass

            return Inner

        report = u.check(_make())

        assert all(violation.layer != "namespace" for violation in report.violations)

    @staticmethod
    def test_check_model_construction_flags_any_field() -> None:
        """Test check model construction flags any field."""

        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = FlextUtilitiesEnforcement.check_model_construction(_WithAny)

        assert report.violations
        assert any("no_any" in violation.message for violation in report.violations)

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

        assert len(caught) == 2
        assert str(caught[0].message) == (
            "\nX.Y violates FLEXT Model HARD rules:\n  - boom [ENFORCE-001]"
            "\n\nFix: See AGENTS.md §3.1 and search for ENFORCE-001."
        )
        assert str(caught[1].message).startswith(
            "\nX.Z violates FLEXT Model HARD rules",
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

        assert str(excinfo.value) == expected
        assert len(recorded) == 1
        assert recorded[0].category is c.FlextMroViolation
        assert str(recorded[0].message) == expected

    @staticmethod
    def test_emit_off_mode_is_silent() -> None:
        """Test emit off mode is silent."""
        report = m.Report(violations=[_hard_violation(message="boom")])

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.OFF)

        assert recorded == []

    @pytest.mark.parametrize(
        "mode",
        [c.EnforcementMode.OFF, c.EnforcementMode.WARN, c.EnforcementMode.STRICT],
    )
    @staticmethod
    def test_emit_empty_report_is_silent_in_every_mode(
        mode: c.EnforcementMode,
    ) -> None:
        """Test emit empty report is silent in every mode."""
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
    @staticmethod
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

        assert str(caught[0].message).endswith(expected_fix)

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

        assert caught[0].category is c.FlextSmellViolation

    @staticmethod
    def test_emit_of_checked_report_carries_layer_tag_and_fix() -> None:
        """Test emit of checked report carries layer tag and fix."""

        class _WithAny(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_WithAny)
        assert report.violations

        with pytest.warns(c.FlextMroViolation) as caught:
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

        texts = [str(entry.message) for entry in caught]
        assert any(
            "violates FLEXT Model HARD rules" in text
            and "[no_any]" in text
            and "Fix: See AGENTS.md § Model governance." in text
            for text in texts
        )
