"""Behavioral tests for the runtime enforcement query ``u.check(target)``.

The public contract is a single query: ``u.check(cls)`` returns an
``m.Report`` whose ``violations`` are filtered by ``layer`` / ``severity`` /
message fragment. These tests assert only that observable contract — never the
collector internals, per-rule private helpers, or emit machinery.
"""

from __future__ import annotations

import pytest

from tests import p, u
from tests.unit._enforcement_support import make_class, messages


class TestsFlextCoreEnforcement:
    """Public behavior of ``u.check`` and the ``m.Report`` it returns."""

    def test_check_returns_report_with_violations_collection(self) -> None:
        """``u.check`` yields an ``m.Report`` exposing a ``violations`` list."""
        report: p.Report = u.check(int)

        dumped = report.model_dump()
        assert "violations" in dumped
        assert isinstance(report.violations, list)
        assert report.violations, "a builtin type must produce at least one finding"

    def test_compliant_flext_class_produces_no_violations(self) -> None:
        """A correctly prefixed, empty facade class is fully compliant."""
        compliant = make_class("FlextCoreWidget", {})

        assert u.check(compliant).violations == []

    @pytest.mark.parametrize("builtin", [int, str, dict])
    def test_builtin_class_flagged_missing_project_prefix(self, builtin: type) -> None:
        """Types without a project prefix raise a Namespace class-prefix finding."""
        report = u.check(builtin)

        prefix_findings = [
            v
            for v in report.violations
            if v.layer == "Namespace" and "missing project prefix" in v.message
        ]
        assert prefix_findings

    def test_private_underscore_class_exempt_from_namespace_layer(self) -> None:
        """Underscore-prefixed classes are implementation details, not facades."""

        class _PrivateHelper:
            pass

        assert all(v.layer != "Namespace" for v in u.check(_PrivateHelper).violations)

    @pytest.mark.parametrize(
        ("class_name", "member"),
        [
            ("FlextCoreAccessedGet", "get_user"),
            ("FlextCoreAccessedSet", "set_config"),
            ("FlextCoreAccessedIs", "is_ready"),
        ],
    )
    def test_accessor_prefix_method_is_flagged(
        self, class_name: str, member: str
    ) -> None:
        """``get_``/``set_``/``is_`` methods violate the accessor contract."""
        cls = make_class(class_name, {member: lambda _self: None})

        report = u.check(cls)
        assert messages(report, fragment=f'accessor method "{member}"')

    @pytest.mark.parametrize("member", ["fetch_remote", "build_widget"])
    def test_non_accessor_prefix_method_allowed(self, member: str) -> None:
        """Verb-prefixed methods that are not accessors raise no accessor finding."""
        cls = make_class("FlextCoreAccessedOk", {member: lambda _self: None})

        report = u.check(cls)
        assert not messages(report, fragment="accessor method")

    @pytest.mark.parametrize(
        ("class_name", "expect_finding"),
        [("FlextWorkerSettings", True), ("FlextCoreService", False)],
    )
    def test_settings_named_class_requires_inheritance(
        self, class_name: str, expect_finding: bool
    ) -> None:
        """Only ``*Settings`` classes must inherit ``FlextSettings``."""
        cls = make_class(class_name, {})

        report = u.check(cls)
        found = bool(messages(report, fragment="must inherit FlextSettings"))
        assert found is expect_finding

    def test_every_violation_carries_public_metadata(self) -> None:
        """Each finding exposes non-empty qualname, layer, severity and message."""
        report = u.check(int)

        assert report.violations
        for violation in report.violations:
            assert violation.qualname
            assert violation.layer
            assert violation.severity
            assert violation.message

    def test_check_is_idempotent_for_same_target(self) -> None:
        """Repeated checks of the same target yield identical findings."""
        first = [v.message for v in u.check(str).violations]
        second = [v.message for v in u.check(str).violations]

        assert first == second
