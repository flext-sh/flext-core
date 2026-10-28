"""FlextTypingConfig - declarative config type aliases (ADR-005).

Type-only contracts for the minimal, runtime-safe config layer owned by
flext-core. Advanced multi-format loading / templating / schema validation are
typed and implemented in ``flext-cli`` on top of these primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from flext_core._typings.base import FlextTypingBase


class FlextTypingConfig:
    """Type aliases for declarative config loading and env override."""

    type DocumentationLanguage = Literal["en", "pt-BR"]
    type ConfigValue = FlextTypingBase.JsonValue
    type ConfigMapping = Mapping[str, FlextTypingBase.JsonValue]
    type ConfigOverrideMapping = Mapping[str, str]


__all__: list[str] = ["FlextTypingConfig"]
