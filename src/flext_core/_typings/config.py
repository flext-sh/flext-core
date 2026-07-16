"""FlextTypingConfig - declarative config type aliases (ADR-005).

Type-only contracts for the minimal, runtime-safe config layer owned by
flext-core. Advanced multi-format loading / templating / schema validation are
typed and implemented in ``flext-cli`` on top of these primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.containers import FlextTypingContainers as tc


class FlextTypingConfig:
    """Type aliases for declarative config loading and env override."""

    type ConfigValue = t.JsonValue
    type ConfigMapping = t.MappingKV[str, t.JsonValue]
    type ConfigFileDict = dict[str, t.JsonValue]
    type ConfigOverrideMapping = t.MappingKV[str, str]
    type UserOverridesMapping = t.MappingKV[str, tc.JsonPayload]


__all__: list[str] = ["FlextTypingConfig"]
