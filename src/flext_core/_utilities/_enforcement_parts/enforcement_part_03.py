"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub

from .enforcement_part_02 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart02,
)

if TYPE_CHECKING:
    from flext_core._models.pydantic import FlextModelsPydantic as mp


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart02):
    @staticmethod
    def _is_exempt(target: type) -> bool:
        """Return True if the target is exempt from enforcement.

        Exemptions:
        - Normal test classes (``Tests*`` / ``Test*`` in qualname) so that
          pytest collection does not trigger enforcement noise.
        - Intentional enforcement fixtures (``_enforcement_integration_fixtures``)
          are NOT exempt — they are designed to exercise rule firing.
        """
        module = getattr(target, "__module__", "") or ""
        qualname = getattr(target, "__qualname__", "") or ""
        if "_enforcement_integration_fixtures" in module or "tests.fixtures" in module:
            return False
        if not module.startswith(("tests.", "tests_")):
            return False
        segments = qualname.split(".")
        return any(seg.startswith(("Tests", "Test")) for seg in segments)

    @staticmethod
    def run(model_type: type[mp.BaseModel]) -> None:
        """Pydantic ``__pydantic_init_subclass__`` hook.

        Function-local classes (Python's ``<locals>`` qualname marker)
        are ephemeral fixtures — validated on demand via ``check()`` but
        never emitted during class construction.
        """
        if c.ENFORCEMENT_MODE is c.EnforcementMode.OFF:
            return
        if ub.defined_in_function_scope(model_type):
            return
        if FlextUtilitiesEnforcement._is_exempt(model_type):
            return
        report = FlextUtilitiesEnforcement.check_model_construction(model_type)
        FlextUtilitiesEnforcement.emit(report)

    @staticmethod
    def run_layer(target: type, layer: str) -> None:
        """Namespace ``__init_subclass__`` hook — run layer + namespace rules.

        Function-local classes skip emission for the same reason as ``run``.
        """
        if c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.OFF:
            return
        if ub.defined_in_function_scope(target):
            return
        if FlextUtilitiesEnforcement._is_exempt(target):
            return
        report = FlextUtilitiesEnforcement.check(target, layer=layer)
        FlextUtilitiesEnforcement.emit(
            report,
            mode=c.ENFORCEMENT_NAMESPACE_MODE,
        )


__all__: list[str] = ["FlextUtilitiesEnforcement"]
