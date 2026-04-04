"""FlextModelsNamespace — base class for ALL facade namespace enforcement.

Inherit from this class in any facade (c, p, t, m, u) to get automatic
MRO namespace governance via __init_subclass__. Layer detection is automatic
from the class name suffix (Constants, Protocols, Types, Utilities, Models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConstantsEnforcement as c, FlextUtilitiesEnforcement


class FlextModelsNamespace:
    """Base class that enforces MRO namespace rules on all facade subclasses.

    Add this to ANY facade's MRO to get automatic enforcement:
    - Class prefix must match project (flext_cli → FlextCli*)
    - Inner namespace class must match project (flext_cli → class Cli:)
    - Cross-layer violations (StrEnum in models, Protocol in constants, etc.)
    - Layer-specific checks (Final hints for constants, static methods for utilities, etc.)

    Layer is auto-detected from the class name suffix.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce namespace governance on every facade subclass."""
        super().__init_subclass__(**kwargs)

        if c.ENFORCEMENT_NAMESPACE_MODE == "off":
            return

        layer = FlextUtilitiesEnforcement.detect_layer(cls)
        if layer is None:
            return

        match layer:
            case "constants":
                FlextUtilitiesEnforcement.run_constants(cls)
            case "protocols":
                FlextUtilitiesEnforcement.run_protocols(cls)
            case "types":
                FlextUtilitiesEnforcement.run_types(cls)
            case "utilities":
                FlextUtilitiesEnforcement.run_utilities(cls)
            case "models":
                FlextUtilitiesEnforcement.run_namespace_checks(cls, "models")
            case _:
                return


__all__ = ["FlextModelsNamespace"]
