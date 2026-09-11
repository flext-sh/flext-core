"""FlextCliConstantsConfig - CLI config/template/schema constants (ADR-005).

Increments the core ``c.CONFIG_*`` defaults with CLI-specific template and
schema constants. flext-cli owns the advanced multi-format loader, Jinja2
templating, and JSON-Schema validation on top of flext-core's minimal layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextCliConstantsConfig:
    """SSOT for CLI config, template, and schema constants (ADR-005)."""

    TEMPLATE_SUFFIX: Final[str] = ".j2"
    TEMPLATE_TRIM_BLOCKS: Final[bool] = False
    TEMPLATE_LSTRIP_BLOCKS: Final[bool] = False
    TEMPLATE_KEEP_TRAILING_NEWLINE: Final[bool] = True
    ERR_TEMPLATE_RENDER_FAILED: Final[str] = "template: render failed"
    ERR_TEMPLATE_NOT_FOUND: Final[str] = "template: source not found"
    ERR_TEMPLATE_OUTPUT_ESCAPE: Final[str] = "template: output path escapes output_root"
    ERR_SCHEMA_INVALID: Final[str] = "schema: document failed validation"
    ERR_SCHEMA_READ_FAILED: Final[str] = "schema: cannot read schema file"
    ERR_CONFIG_UNSUPPORTED_FORMAT: Final[str] = "config: unsupported source format"
