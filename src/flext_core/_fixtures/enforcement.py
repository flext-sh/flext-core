"""Pytest plugin for flext-core runtime enforcement warnings.

This module is loaded automatically via the ``flext_core_enforcement`` pytest11
entry-point. It registers the runtime-warning contribution with the central
``flext_tests`` enforcement dispatcher so that ``FlextMroViolation`` and
``FlextSmellViolation`` warnings are tracked during test sessions.
"""

from __future__ import annotations

from flext_core import (
    FlextMroViolation,
    FlextSmellViolation,
)
from flext_tests.enforcement import (
    EnforcementContribution,
    register as register_enforcement_contribution,
)


class FlextCoreEnforcementPytestPlugin:
    """Class-owned pytest contribution for flext-core runtime warnings."""

    _SOURCE_KIND: str = "runtime_warning"

    @classmethod
    def source_kind(cls) -> str:
        """Return the catalog source kind owned by this plugin."""
        return cls._SOURCE_KIND

    @classmethod
    def contribution(cls) -> EnforcementContribution:
        """Return the registry contribution for the flext-tests dispatcher."""
        return EnforcementContribution(
            source_kind=cls.source_kind(),
            warning_categories=(FlextMroViolation, FlextSmellViolation),
        )

    @classmethod
    def register(cls) -> None:
        """Register this plugin in the flext-tests enforcement registry."""
        register_enforcement_contribution(cls.source_kind(), cls.contribution())


def _register() -> None:
    """Module-level registration helper used by tests."""
    FlextCoreEnforcementPytestPlugin.register()


FlextCoreEnforcementPytestPlugin.register()


__all__: list[str] = [
    "FlextCoreEnforcementPytestPlugin",
    "_register",
]
