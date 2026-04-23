"""FlextModelsNamespace — base class for ALL facade namespace enforcement.

Inherit from this class in any facade (c, p, t, m, u) to get automatic
MRO namespace governance via __init_subclass__. Layer detection is automatic
from the class name suffix (Constants, Protocols, Types, Utilities, Models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement as ue


class FlextModelsNamespace:
    """Base class that enforces MRO namespace rules on all facade subclasses.

    Layer is auto-detected from the class name suffix. The engine reads
    ``c.ENFORCEMENT_RULES`` and dispatches every applicable rule — no
    per-layer wrapper methods, no hardcoded dispatch.
    """

    def __init_subclass__(cls, **kwargs: tp.JsonValue) -> None:
        """Enforce namespace governance on every facade subclass.

        Layer-independent rules (``class_prefix``, ``no_accessor_methods``,
        ``settings_inheritance``, ``nested_mro``, ...) run for EVERY
        subclass; layer-specific rules (``const_*``, ``alias_*``,
        ``utility_not_static``, ``proto_*``) additionally fire when the
        class name ends in a recognised layer suffix.
        """
        super().__init_subclass__(**kwargs)
        if c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.OFF:
            return
        layer = ue.detect_layer(cls) or ""
        ue.run_layer(cls, layer)


__all__: t.MutableSequenceOf[str] = ["FlextModelsNamespace"]
