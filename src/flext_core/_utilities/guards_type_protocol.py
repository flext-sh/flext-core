"""Protocol-based guards utilities for Flext type checking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._utilities._guards_type_protocol_specs import (
    FlextUtilitiesGuardsTypeProtocolSpecsMixin,
)
from flext_core._utilities._guards_type_protocol_string import (
    FlextUtilitiesGuardsTypeProtocolStringMixin,
)
from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput
from flext_core.constants import c
from flext_core.typings import t


class FlextUtilitiesGuardsTypeProtocol(
    FlextUtilitiesGuardsTypeProtocolStringMixin,
    FlextUtilitiesGuardsTypeProtocolSpecsMixin,
):
    """Protocol-based type guards for flext type checking.

    Provides static methods for checking whether values conform to flext framework
    protocols (Context, Handler, Service, etc.) and Python type specifications.
    Uses caching for performance-critical protocol lookups.
    """

    @staticmethod
    def matches_type(
        value: ProtocolGuardInput,
        type_spec: str
        | type
        | tuple[type, ...]
        | t.Scalar,  # Scalar arm handles invalid spec at runtime
    ) -> bool:
        """Check if value matches a type spec (string name, type, or tuple of types)."""
        matched = False
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            protocol_specs = FlextUtilitiesGuardsTypeProtocol._get_protocol_specs()
            if type_name in protocol_specs:
                matched = FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value, type_name
                )
            elif type_name in c.STRING_METHOD_MAP:
                matched = not (
                    type_name
                    in {"string_non_empty", "dict_non_empty", "list_non_empty"}
                    and isinstance(value, (mp.BaseModel, mp.RootModel))
                ) and FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                    type_name,
                    value,
                )
        elif isinstance(type_spec, tuple):
            matched = isinstance(value, type_spec)
        elif isinstance(type_spec, type):
            protocol_name = (
                FlextUtilitiesGuardsTypeProtocol._get_protocol_type_map().get(
                    type_spec,
                )
            )
            if protocol_name is not None:
                matched = FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value,
                    protocol_name,
                )
            else:
                runtime_type = getattr(type_spec, "__origin__", None) or type_spec
                try:
                    matched = isinstance(value, runtime_type)
                except TypeError:
                    matched = False
        return matched


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocol"]
