"""Phase: Ensure standard Ruff configuration inline with known-first-party overlay."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse as _TomlParse
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument

array = _Toml.array
as_string_list = _Toml.as_string_list
ensure_table = _Toml.ensure_table
table_string_keys = _Toml.table_string_keys
toml_get = _Toml.get
unwrap_item = _Toml.unwrap_item
discover_first_party_namespaces = _TomlParse.discover_first_party_namespaces


class EnsureRuffConfigPhase:
    """Ensure standard Ruff configuration inline with known-first-party overlay."""

    def __init__(self, tool_config: FlextInfraToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        path: Path,
        workspace_root: Path,
    ) -> list[str]:
        _ = workspace_root
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        ruff = ensure_table(tool, c.Infra.Toml.RUFF)
        if c.Infra.Toml.EXTEND in ruff:
            del ruff[c.Infra.Toml.EXTEND]
            changes.append("tool.ruff.extend removed")

        ruff_cfg = self._tool_config.tools.ruff
        if sorted(as_string_list(toml_get(ruff, c.Infra.Toml.EXCLUDE))) != sorted(
            ruff_cfg.exclude,
        ):
            ruff[c.Infra.Toml.EXCLUDE] = array(sorted(ruff_cfg.exclude))
            changes.append("tool.ruff.exclude set")
        for key, value in {
            "fix": ruff_cfg.fix,
            "line-length": ruff_cfg.line_length,
            "preview": ruff_cfg.preview,
            "respect-gitignore": ruff_cfg.respect_gitignore,
            "show-fixes": ruff_cfg.show_fixes,
            "target-version": ruff_cfg.target_version,
        }.items():
            if unwrap_item(toml_get(ruff, key)) != value:
                ruff[key] = value
                changes.append(f"tool.ruff.{key} set")
        if sorted(as_string_list(toml_get(ruff, "src"))) != sorted(ruff_cfg.src):
            ruff["src"] = array(sorted(ruff_cfg.src))
            changes.append("tool.ruff.src set")

        ruff_format = ensure_table(ruff, "format")
        for key, value in {
            "docstring-code-format": ruff_cfg.format.docstring_code_format,
            "indent-style": ruff_cfg.format.indent_style,
            "line-ending": ruff_cfg.format.line_ending,
            "quote-style": ruff_cfg.format.quote_style,
        }.items():
            if unwrap_item(toml_get(ruff_format, key)) != value:
                ruff_format[key] = value
                changes.append(f"tool.ruff.format.{key} set")

        lint = ensure_table(ruff, c.Infra.Toml.LINT_SECTION)
        if sorted(as_string_list(toml_get(lint, "select"))) != sorted(
            ruff_cfg.lint.select,
        ):
            lint["select"] = array(sorted(ruff_cfg.lint.select))
            changes.append("tool.ruff.lint.select set")
        if sorted(as_string_list(toml_get(lint, c.Infra.Toml.IGNORE))) != sorted(
            ruff_cfg.lint.ignore,
        ):
            lint[c.Infra.Toml.IGNORE] = array(sorted(ruff_cfg.lint.ignore))
            changes.append("tool.ruff.lint.ignore set")

        isort = ensure_table(lint, c.Infra.Toml.ISORT)
        for key, value in {
            "combine-as-imports": ruff_cfg.lint.isort.combine_as_imports,
            "force-single-line": ruff_cfg.lint.isort.force_single_line,
            "split-on-trailing-comma": ruff_cfg.lint.isort.split_on_trailing_comma,
        }.items():
            if unwrap_item(toml_get(isort, key)) != value:
                isort[key] = value
                changes.append(f"tool.ruff.lint.isort.{key} set")

        per_file_ignores = ensure_table(lint, "per-file-ignores")
        for pattern in table_string_keys(per_file_ignores):
            if pattern not in ruff_cfg.lint.per_file_ignores:
                del per_file_ignores[pattern]
                changes.append(f"tool.ruff.lint.per-file-ignores.{pattern} removed")
        for pattern, rules in ruff_cfg.lint.per_file_ignores.items():
            if sorted(as_string_list(toml_get(per_file_ignores, pattern))) != sorted(
                rules,
            ):
                per_file_ignores[pattern] = array(sorted(rules))
                changes.append(f"tool.ruff.lint.per-file-ignores.{pattern} set")

        detected_packages = sorted(discover_first_party_namespaces(path.parent))
        if detected_packages:
            current_kfp = sorted(
                as_string_list(toml_get(isort, c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN)),
            )
            if current_kfp != detected_packages:
                isort[c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN] = array(detected_packages)
                changes.append(
                    f"tool.ruff.lint.isort.known-first-party set to {detected_packages}",
                )
        if c.Infra.Toml.LINT_SECTION in doc:
            del doc[c.Infra.Toml.LINT_SECTION]
            changes.append("removed stale top-level [lint] section")
        return changes
