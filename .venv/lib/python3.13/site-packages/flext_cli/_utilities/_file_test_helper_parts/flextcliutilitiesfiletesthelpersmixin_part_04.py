"""Test-oriented file helpers generalized for reuse through ``u.Cli``.

These operations are generic enough to be used by tests, examples, and
maintenance scripts, but were originally duplicated in ``flext-tests``.
They live here so ``flext-tests`` can delegate to ``u.Cli`` instead of
reimplementing them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_cli import c, p, r, t
from flext_cli._utilities.json import FlextCliUtilitiesJson as uj
from flext_cli._utilities.toml import FlextCliUtilitiesToml as ut
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml as uy

if TYPE_CHECKING:
    from pathlib import Path


class FlextCliUtilitiesFileTestHelpersMixin:
    """Implementation part for FlextCliUtilitiesFileTestHelpersMixin."""

    @staticmethod
    def files_parse_content(path: Path, fmt: str) -> p.Result[t.JsonMapping]:
        """Parse JSON/YAML/TOML file content generically by format token."""
        if fmt == c.Cli.FILE_FORMAT_JSON:
            result = uj.json_read(path)
            if result.failure:
                return r[t.JsonMapping].fail(result.error or "json_read failed")
            return r[t.JsonMapping].ok(result.value)
        if fmt == c.Cli.FILE_FORMAT_YAML:
            result = uy.yaml_safe_load(path)
            if result.failure:
                return r[t.JsonMapping].fail(result.error or "yaml_safe_load failed")
            return r[t.JsonMapping].ok(result.value)
        if fmt == c.Cli.FILE_FORMAT_TOML:
            toml_result = ut.toml_read_json(path)
            if toml_result.failure:
                return r[t.JsonMapping].fail(
                    toml_result.error or "toml_read_json failed"
                )
            return r[t.JsonMapping].ok(toml_result.value)
        msg = f"Cannot parse format: {fmt}"
        return r[t.JsonMapping].fail(msg)


__all__: list[str] = ["FlextCliUtilitiesFileTestHelpersMixin"]
