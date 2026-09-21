"""FlextConstantsConfig - declarative config loading constants (SSOT).

Minimal, runtime-safe defaults for the ADR-005 config layer. flext-core stays
runtime-minimal: only stdlib-backed config primitives live here. The advanced
multi-format loader, Jinja2 templating, and JSON-Schema validation are owned by
``flext-cli`` (``u.Cli.config_load`` / ``u.Cli.template_render`` /
``u.Cli.schema_validate``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class FlextConstantsConfig:
    """SSOT for declarative config loading defaults (ADR-005)."""

    CONFIG_DIR_NAME: ClassVar[str] = "config"
    CONFIG_SCHEMAS_DIR_NAME: ClassVar[str] = "schemas"
    CONFIG_TEMPLATES_DIR_NAME: ClassVar[str] = "templates"
    CONFIG_SETTINGS_FILE_NAME: ClassVar[str] = "settings.yaml"
    CONFIG_SCHEMA_SUFFIX: ClassVar[str] = ".schema.json"
    CONFIG_DEFAULT_ENCODING: ClassVar[str] = "utf-8"
    CONFIG_TOML_SUFFIX: ClassVar[str] = ".toml"
    CONFIG_YAML_SUFFIX: ClassVar[str] = ".yaml"
    CONFIG_JSON_SUFFIX: ClassVar[str] = ".json"
    CONFIG_EXPAND_MAX_PASSES: ClassVar[int] = 10
    ERR_CONFIG_READ_FAILED: ClassVar[str] = "config: cannot read source"
    ERR_CONFIG_PARSE_FAILED: ClassVar[str] = "config: cannot parse source"
    ERR_CONFIG_NOT_MAPPING: ClassVar[str] = "config: expected a mapping object"
