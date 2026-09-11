"""Universal multi-format config loader shared through ``u.Cli.config_*``.

flext-cli **amplifies** the flext-core minimal config layer (ADR-005): it reuses
``u.config_merge`` / ``u.config_env_override`` (core, stdlib) and the existing
``u.Cli.yaml_safe_load`` / ``json_read`` / ``toml_read_json`` readers, adding
multi-format dispatch, directory loading, and JSON-Schema validation. It does
**not** re-implement merge, env-expansion, or the ``m.ConfigDocument`` record.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from flext_cli import c, m, p, r, t
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_06 import (
    FlextCliUtilitiesToml as _TomlRead,
)
from flext_cli._utilities.json import FlextCliUtilitiesJson
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml
from flext_core import u


class FlextCliUtilitiesConfig:
    """Universal multi-format config load + schema validation (ADR-005)."""

    @staticmethod
    def _read_by_suffix(path: Path) -> p.Result[t.JsonMapping]:
        """Dispatch to the reader matching ``path`` suffix; reuse core cli readers."""
        suffix = path.suffix.lower()
        if suffix in {c.CONFIG_YAML_SUFFIX, ".yml"}:
            return FlextCliUtilitiesYaml.yaml_safe_load(path)
        if suffix == c.CONFIG_JSON_SUFFIX:
            return FlextCliUtilitiesJson.json_read(path)
        if suffix == c.CONFIG_TOML_SUFFIX:
            return _TomlRead.toml_read_json(path)
        return r[t.JsonMapping].fail(f"{c.Cli.ERR_CONFIG_UNSUPPORTED_FORMAT}: {path}")

    @staticmethod
    def config_load(
        path: Path, *, schema_path: Path | None = None, expand_env: bool = True
    ) -> p.Result[m.ConfigDocument]:
        """Load a YAML/JSON/TOML config into a validated ``m.ConfigDocument``.

        Reuses core ``u.config_env_override`` for ``${VAR}`` expansion and
        ``u.Cli.schema_validate`` for optional JSON-Schema validation.
        """
        read = FlextCliUtilitiesConfig._read_by_suffix(path)
        if read.failure:
            return r[m.ConfigDocument].fail(
                read.error or f"{c.ERR_CONFIG_READ_FAILED}: {path}"
            )
        data: t.JsonValue = dict(read.value)
        if expand_env:
            data = u.config_env_override(data, dict(os.environ))
        if not isinstance(data, dict):
            return r[m.ConfigDocument].fail(f"{c.ERR_CONFIG_NOT_MAPPING}: {path}")
        if schema_path is not None:
            validated = FlextCliUtilitiesConfig.schema_validate(data, schema_path)
            if validated.failure:
                return r[m.ConfigDocument].fail(
                    validated.error or c.Cli.ERR_SCHEMA_INVALID
                )
        return r[m.ConfigDocument].ok(
            m.ConfigDocument(
                data=data,
                source_path=str(path),
                schema_ref=str(schema_path) if schema_path is not None else None,
            )
        )

    @staticmethod
    def config_load_dir(
        config_dir: Path,
    ) -> p.Result[t.MappingKV[str, m.ConfigDocument]]:
        """Load every ``config/*.yaml`` file, pairing each with its ``schemas/`` schema.

        The schema for ``config/<name>.yaml`` is
        ``<config_dir>/../schemas/<name>.schema.json`` when present.
        """
        if not config_dir.is_dir():
            return r[t.MappingKV[str, m.ConfigDocument]].fail(
                f"{c.ERR_CONFIG_READ_FAILED}: {config_dir}"
            )
        schemas_dir = config_dir.parent / c.CONFIG_SCHEMAS_DIR_NAME
        documents: dict[str, m.ConfigDocument] = {}
        for source in sorted(config_dir.glob(f"*{c.CONFIG_YAML_SUFFIX}")):
            schema = schemas_dir / f"{source.stem}{c.CONFIG_SCHEMA_SUFFIX}"
            loaded = FlextCliUtilitiesConfig.config_load(
                source, schema_path=schema if schema.is_file() else None
            )
            if loaded.failure:
                return r[t.MappingKV[str, m.ConfigDocument]].fail(
                    loaded.error or f"{c.ERR_CONFIG_PARSE_FAILED}: {source}"
                )
            documents[source.stem] = loaded.value
        return r[t.MappingKV[str, m.ConfigDocument]].ok(documents)

    @staticmethod
    def schema_validate(data: t.JsonMapping, schema_path: Path) -> p.Result[bool]:
        """Validate ``data`` against the JSON Schema at ``schema_path`` → ``r[bool]``."""
        schema_read = FlextCliUtilitiesJson.json_read(schema_path)
        if schema_read.failure:
            return r[bool].fail(
                schema_read.error or f"{c.Cli.ERR_SCHEMA_READ_FAILED}: {schema_path}"
            )
        return u.try_(
            lambda: FlextCliUtilitiesConfig._run_validator(schema_read.value, data),
            catch=(ValidationError, SchemaError),
            op_name="schema_validate",
        )

    @staticmethod
    def _run_validator(schema: t.JsonMapping, data: t.JsonMapping) -> bool:
        """Run a Draft 2020-12 validator; raise on first violation, else ``True``."""
        Draft202012Validator(dict(schema)).validate(dict(data))
        return True


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesConfig"]
