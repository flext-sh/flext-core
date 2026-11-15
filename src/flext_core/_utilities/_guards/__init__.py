# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Guards package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._guards.type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecs,
    )
    from flext_core._utilities._guards.type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolString,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesGuardsTypeProtocolSpecs",
    "FlextUtilitiesGuardsTypeProtocolString",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextUtilitiesGuardsTypeProtocolSpecs": ".type_protocol_specs",
        "FlextUtilitiesGuardsTypeProtocolString": ".type_protocol_string",
    },
    public_exports=__all__,
    deferred_members={
        "FlextUtilitiesGuardsTypeProtocolSpecs": (
            "_get_protocol_specs",
            "_get_protocol_type_map",
            "context",
            "result_like",
            "result_view",
            "_check_protocol",
        ),
        "FlextUtilitiesGuardsTypeProtocolString": (
            "_has_len",
            "_is_none",
            "_is_non_empty_mapping",
            "_is_non_empty_sequence",
            "_string_predicates",
            "_run_string_type_check",
        ),
    },
)
