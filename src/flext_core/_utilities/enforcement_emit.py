"""Enforcement emission primitives: violation assembly, emit, exemptions."""

from __future__ import annotations

import warnings

from flext_core._constants.enforcement import (
    FlextConstantsEnforcement as c,
    FlextMroViolation,
)
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcementEmit:
    """Violation factory + warning/strict emission + exemption rules."""

    @staticmethod
    def _violation(
        tag: str,
        location: str,
        qualname: str,
        detail: t.StrMapping | None = None,
    ) -> me.Violation:
        # Look up problem/fix from legacy text mapping (derived from rows)
        problem, fix = c._ENFORCEMENT_RULES_TEXT[tag]
        subs = detail or {}
        return me.Violation(
            qualname=qualname,
            layer="Model",  # Default layer; refined by callers based on context
            severity="HARD rules",  # Default; callers override via report context
            message=c.ENFORCEMENT_MSG_VIOLATION.format(
                location=location,
                problem=problem.format(**subs) if subs else problem,
                fix=fix.format(**subs) if subs else fix,
            ),
        )

    @staticmethod
    def emit(report: me.Report, *, mode: c.EnforcementMode | None = None) -> None:
        """Emit violations as warnings (or raise TypeError in STRICT mode)."""
        if report.empty:
            return
        active = mode or c.ENFORCEMENT_MODE
        if active is c.EnforcementMode.OFF:
            return
        for v in report.violations:
            msg = (
                f"\n{v.qualname} violates FLEXT {v.layer} {v.severity}:\n  - "
                f"{v.message}\n\nFix: See AGENTS.md § {v.layer} governance."
            )
            warnings.warn(msg, FlextMroViolation, stacklevel=4)
            if active is c.EnforcementMode.STRICT:
                raise TypeError(msg)

    @staticmethod
    def detect_layer(target: type) -> str | None:
        """Infer the facade layer from the class name.

        Matches the layer keyword (``Constants`` / ``Models`` / ``Protocols``
        / ``Types`` / ``Utilities``) anywhere in the class name — not only
        at the end — so composed facades such as ``FooConstantsSettings``
        or ``FooProtocolsBase`` still get the correct layer routing.
        Generic-specialization brackets (``Foo[Bar]``) are stripped first
        so the search ignores type-parameter noise.
        """
        name = target.__name__.partition("[")[0]
        for suffix, layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
            if suffix in name:
                return layer
        return None

    @staticmethod
    def _is_exempt(target: type) -> bool:
        """Test-fixture exemption only — runtime opt-out attribute deleted.

        The legacy ``_flext_enforcement_exempt = True`` ClassVar escape hatch
        is gone. Test fixtures (modules whose path starts with ``tests.`` /
        ``tests_`` AND whose qualname contains a ``Tests``/``Test`` segment)
        remain exempt because they exercise production APIs intentionally
        outside production model governance (mutable defaults, accessor
        names, field-description requirements). Production code has no
        opt-out — fix the violation instead.
        """
        module = getattr(target, "__module__", "") or ""
        qualname = getattr(target, "__qualname__", "") or ""
        if not module.startswith(("tests.", "tests_")):
            return False
        if "_enforcement_integration_fixtures" in module:
            return False
        segments = qualname.split(".")
        return any(seg.startswith(("Tests", "Test")) for seg in segments)


__all__: list[str] = ["FlextUtilitiesEnforcementEmit"]
