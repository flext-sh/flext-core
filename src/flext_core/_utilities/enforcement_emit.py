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
        problem, fix = c.ENFORCEMENT_RULES_TEXT[tag]
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
        / ``Types`` / ``Utilities``) only when it appears as a standalone
        PascalCase segment — at the end of the name or immediately followed
        by another capitalised word (e.g. ``FooConstantsSettings``).
        This prevents false positives such as ``BadConstants`` where the
        layer keyword is embedded inside a larger word.
        Generic-specialization brackets (``Foo[Bar]``) are stripped first
        so the search ignores type-parameter noise.
        """
        name = target.__name__.partition("[")[0]
        for suffix, layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
            idx = name.find(suffix)
            if idx == -1:
                continue
            end = idx + len(suffix)
            # Suffix must be at the very end, or followed by an uppercase
            # letter (start of next PascalCase word).
            if end == len(name) or (end < len(name) and name[end].isupper()):
                return layer
        return None


__all__: list[str] = ["FlextUtilitiesEnforcementEmit"]
