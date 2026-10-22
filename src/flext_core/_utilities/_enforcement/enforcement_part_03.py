"""Runtime enforcement engine MRO part.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c
from flext_core._models import FlextModelsPydantic
from flext_core._utilities import FlextUtilitiesBeartypeEngine
from flext_core._utilities._enforcement_parts import enforcement_part_02


class FlextUtilitiesEnforcementPart03(
    enforcement_part_02.FlextUtilitiesEnforcementPart02
):
    @staticmethod
    def _is_exempt(target: type) -> bool:
        """Return True if the target is exempt from enforcement.

        Exemptions:
        - Normal test classes (``Tests*`` / ``Test*`` in qualname) so that
          pytest collection does not trigger enforcement noise.
        - Intentional enforcement fixtures (``_enforcement_integration_fixtures``)
          are NOT exempt — they are designed to exercise rule firing.

        Returns:
            True if the target is exempt from enforcement.

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
    def run(model_type: type[FlextModelsPydantic.BaseModel]) -> None:
        """Pydantic ``__pydantic_init_subclass__`` hook.

        Function-local classes (Python's ``<locals>`` qualname marker)
        are ephemeral fixtures — validated on demand via ``check()`` but
        never emitted during class construction.
        """
        if c.ENFORCEMENT_MODE is c.EnforcementMode.OFF:
            return
        if FlextUtilitiesBeartypeEngine.defined_in_function_scope(model_type):
            return
        if FlextUtilitiesEnforcementPart03._is_exempt(model_type):
            return
        report = FlextUtilitiesEnforcementPart03.check_model_construction(model_type)
        FlextUtilitiesEnforcementPart03.emit(report)

    @staticmethod
    def run_layer(target: type, layer: str) -> None:
        """Namespace ``__init_subclass__`` hook — run layer + namespace rules.

        Function-local classes skip emission for the same reason as ``run``.
        """
        if c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.OFF:
            return
        if FlextUtilitiesBeartypeEngine.defined_in_function_scope(target):
            return
        if FlextUtilitiesEnforcementPart03._is_exempt(target):
            return
        report = FlextUtilitiesEnforcementPart03.check(target, layer=layer)
        FlextUtilitiesEnforcementPart03.emit(report, mode=c.ENFORCEMENT_NAMESPACE_MODE)


__all__: list[str] = ["FlextUtilitiesEnforcementPart03"]
