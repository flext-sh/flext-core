"""FlextConstantsConfig - declarative config loading constants (SSOT).

Minimal, runtime-safe defaults for the ADR-005 config layer. flext-core stays
runtime-minimal: only stdlib-backed config primitives live here. The advanced
multi-format loader, Jinja2 templating, and JSON-Schema validation are owned by
``flext-cli`` (``u.Cli.config_load`` / ``u.Cli.render_template`` /
``u.Cli.yaml_validate_schema``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsConfig:
    """SSOT for declarative config loading defaults (ADR-005)."""

    CONFIG_DIR_NAME: Final[str] = "config"
    CONFIG_SCHEMAS_DIR_NAME: Final[str] = "schemas"
    CONFIG_TEMPLATES_DIR_NAME: Final[str] = "templates"
    CONFIG_SETTINGS_FILE_NAME: Final[str] = "settings.yaml"
    CONFIG_SCHEMA_SUFFIX: Final[str] = ".schema.json"
    CONFIG_DEFAULT_ENCODING: Final[str] = "utf-8"
    CONFIG_TOML_SUFFIX: Final[str] = ".toml"
    CONFIG_YAML_SUFFIX: Final[str] = ".yaml"
    CONFIG_JSON_SUFFIX: Final[str] = ".json"
    ERR_CONFIG_READ_FAILED: Final[str] = "config: cannot read source"
    ERR_CONFIG_PARSE_FAILED: Final[str] = "config: cannot parse source"
    ERR_CONFIG_NOT_MAPPING: Final[str] = "config: expected a mapping object"
