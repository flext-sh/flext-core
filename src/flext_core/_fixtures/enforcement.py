"""Pytest plugin for flext-core runtime enforcement warnings.

This module is loaded automatically via the ``flext_core_enforcement`` pytest11
entry-point. It registers the runtime-warning contribution with the central
``flext_tests`` enforcement dispatcher so that ``FlextMroViolation`` and
``FlextSmellViolation`` warnings are tracked during test sessions.
"""

from __future__ import annotations

from flext_core._constants.enforcement import (
    FlextMroViolation,
    FlextSmellViolation,
)


def _register() -> None:
    """Register flext-core enforcement contribution when flext-tests is present."""
    try:
        from flext_tests._fixtures._enforcement_parts.registry import (  # noqa: PLC0415, PLC2701
            EnforcementContribution,
            register,
        )
    except ImportError:
        return

    register(
        "flext_core_runtime_warning",
        EnforcementContribution(
            source_kind="runtime_warning",
            warning_categories=(FlextMroViolation, FlextSmellViolation),
        ),
    )


_register()


__all__: list[str] = []
