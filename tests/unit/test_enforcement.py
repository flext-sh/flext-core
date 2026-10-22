"""Behavioral tests for the runtime enforcement query ``u.check(target)``.

The public contract is a single query: ``u.check(cls)`` returns an
``m.Report`` whose ``violations`` are filtered by ``layer`` / ``severity`` /
message fragment. These tests assert only that observable contract — never the
collector internals, per-rule private helpers, or emit machinery.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import m, tm, u
from tests.unit._enforcement_support import make_class, messages, synthetic_method


class TestsFlextCoreEnforcement:
    """Public behavior of ``u.check`` and the ``m.Report`` it returns."""

    @staticmethod
    def test_check_returns_report_with_violations_collection() -> None:
        """``u.check`` yields an ``m.Report`` exposing a ``violations`` list."""
        report: m.Report = u.check(int)

        dumped = report.model_dump()
        tm.that("violations" in dumped, truth=True)
        tm.that(isinstance(report.violations, list), truth=True)
        tm.that(
            report.violations,
            truth=True,
            msg="a builtin type must produce at least one finding",
        )

    @staticmethod
    def test_compliant_flext_class_produces_no_violations() -> None:
        """A correctly prefixed, empty facade class is fully compliant."""
        compliant = make_class("FlextCoreWidget", {})

        tm.that(u.check(compliant).violations, eq=[])

    @staticmethod
    @pytest.mark.parametrize("builtin", [int, str, dict])
    def test_builtin_class_flagged_missing_project_prefix(builtin: type) -> None:
        """Types without a project prefix raise a Namespace class-prefix finding."""
        report = u.check(builtin)

        prefix_findings = [
            v
            for v in report.violations
            if v.layer == "Namespace" and "missing project prefix" in v.message
        ]
        tm.that(prefix_findings, truth=True)

    @staticmethod
    def test_private_underscore_class_exempt_from_namespace_layer() -> None:
        """Underscore-prefixed classes are implementation details, not facades."""

        class _PrivateHelper:
            pass

        tm.that(
            all(v.layer != "Namespace" for v in u.check(_PrivateHelper).violations),
            eq=True,
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("class_name", "member"),
        [
            ("FlextCoreAccessedGet", "get_user"),
            ("FlextCoreAccessedSet", "set_config"),
            ("FlextCoreAccessedIs", "is_ready"),
        ],
    )
    def test_accessor_prefix_method_is_flagged(
        class_name: str,
        member: str,
    ) -> None:
        """``get_``/``set_``/``is_`` methods violate the accessor contract."""
        cls = make_class(class_name, {member: synthetic_method})

        report = u.check(cls)
        tm.that(messages(report, fragment=f'accessor method "{member}"'), truth=True)

    @staticmethod
    @pytest.mark.parametrize("member", ["fetch_remote", "build_widget"])
    def test_non_accessor_prefix_method_allowed(member: str) -> None:
        """Verb-prefixed methods that are not accessors raise no accessor finding."""
        cls = make_class("FlextCoreAccessedOk", {member: synthetic_method})

        report = u.check(cls)
        tm.that(messages(report, fragment="accessor method"), truth=False)

    @staticmethod
    @pytest.mark.parametrize(
        ("class_name", "declares_settings_base", "expect_finding"),
        [
            ("FlextWorkerSettings", True, True),
            ("FlextWorkerSettings", False, False),
            ("FlextCoreService", False, False),
        ],
    )
    def test_declared_settings_model_requires_flext_settings(
        class_name: str,
        *,
        declares_settings_base: bool,
        expect_finding: bool,
    ) -> None:
        """Route a declared pydantic-settings base through ``FlextSettings``.

        A class that DECLARES such a base must use it; a name ending in
        ``Settings`` alone proves nothing.
        """
        bases = (m.BaseSettings,) if declares_settings_base else ()
        cls = type(class_name, bases, {})
        cls.__qualname__ = class_name
        cls.__module__ = "flext_core.synthetic"

        report = u.check(cls)
        found = bool(messages(report, fragment="must inherit FlextSettings"))
        tm.that(found is expect_finding, truth=True)

    @staticmethod
    def test_every_violation_carries_public_metadata() -> None:
        """Each finding exposes non-empty qualname, layer, severity and message."""
        report = u.check(int)

        tm.that(report.violations, truth=True)
        for violation in report.violations:
            tm.that(violation.qualname, truth=True)
            tm.that(violation.layer, truth=True)
            tm.that(violation.severity, truth=True)
            tm.that(violation.message, truth=True)

    @staticmethod
    def test_check_is_idempotent_for_same_target() -> None:
        """Repeated checks of the same target yield identical findings."""
        first = [v.message for v in u.check(str).violations]
        second = [v.message for v in u.check(str).violations]

        tm.that(first, eq=second)
