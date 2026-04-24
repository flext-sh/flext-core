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
        _cat, layer, severity, problem, fix = c.ENFORCEMENT_RULES[tag]
        subs = detail or {}
        return me.Violation(
            qualname=qualname,
            layer=str(layer),
            severity=str(severity),
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
        """Honour explicit ``_flext_enforcement_exempt`` opt-out.

        Exemption applies when either:

        1. ``target.__dict__`` carries ``_flext_enforcement_exempt = True``
           on itself (not inherited — opt-out is explicit per subclass).
        2. The class lives inside a test fixture scope: the defining module
           path begins with ``tests.`` (or ``tests`` is a top-level package
           segment) AND the qualname contains a ``Tests`` container segment.
           Test fixtures exercise production APIs but are not subject to
           production model governance (mutable defaults, accessor method
           naming, field-description requirements).
        """
        if target.__dict__.get("_flext_enforcement_exempt", False):
            return True
        module = getattr(target, "__module__", "") or ""
        qualname = getattr(target, "__qualname__", "") or ""
        if not module.startswith(("tests.", "tests_")):
            return False
        if "_enforcement_integration_fixtures" in module:
            return False
        segments = qualname.split(".")
        return any(seg.startswith(("Tests", "Test")) for seg in segments)


__all__: list[str] = ["FlextUtilitiesEnforcementEmit"]
