"""Phase: Ensure standard Ruff configuration inline with known-first-party overlay."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsureRuffConfigPhase:
    """Ensure standard Ruff configuration inline with known-first-party overlay."""

    def __init__(self, tool_config: FlextInfraDepsModels.ToolConfigDocument) -> None:
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
        tool: Item | Container | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        ruff = u.Infra.ensure_table(tool, c.Infra.Toml.RUFF)
        if c.Infra.Toml.EXTEND in ruff:
            del ruff[c.Infra.Toml.EXTEND]
            changes.append("tool.ruff.extend removed")

        ruff_cfg = self._tool_config.tools.ruff
        if sorted(
            u.Infra.as_string_list(u.Infra.get(ruff, c.Infra.Toml.EXCLUDE))
        ) != sorted(ruff_cfg.exclude):
            ruff[c.Infra.Toml.EXCLUDE] = u.Infra.array(sorted(ruff_cfg.exclude))
            changes.append("tool.ruff.exclude set")
        for key, value in {
            "fix": ruff_cfg.fix,
            "line-length": ruff_cfg.line_length,
            "preview": ruff_cfg.preview,
            "respect-gitignore": ruff_cfg.respect_gitignore,
            "show-fixes": ruff_cfg.show_fixes,
            "target-version": ruff_cfg.target_version,
        }.items():
            if u.Infra.unwrap_item(u.Infra.get(ruff, key)) != value:
                ruff[key] = value
                changes.append(f"tool.ruff.{key} set")
        if sorted(u.Infra.as_string_list(u.Infra.get(ruff, "src"))) != sorted(
            ruff_cfg.src
        ):
            ruff["src"] = u.Infra.array(sorted(ruff_cfg.src))
            changes.append("tool.ruff.src set")

        ruff_format = u.Infra.ensure_table(ruff, "format")
        for key, value in {
            "docstring-code-format": ruff_cfg.format.docstring_code_format,
            "indent-style": ruff_cfg.format.indent_style,
            "line-ending": ruff_cfg.format.line_ending,
            "quote-style": ruff_cfg.format.quote_style,
        }.items():
            if u.Infra.unwrap_item(u.Infra.get(ruff_format, key)) != value:
                ruff_format[key] = value
                changes.append(f"tool.ruff.format.{key} set")

        lint = u.Infra.ensure_table(ruff, c.Infra.Toml.LINT_SECTION)
        if sorted(u.Infra.as_string_list(u.Infra.get(lint, "select"))) != sorted(
            ruff_cfg.lint.select,
        ):
            lint["select"] = u.Infra.array(sorted(ruff_cfg.lint.select))
            changes.append("tool.ruff.lint.select set")
        if sorted(
            u.Infra.as_string_list(u.Infra.get(lint, c.Infra.Toml.IGNORE))
        ) != sorted(ruff_cfg.lint.ignore):
            lint[c.Infra.Toml.IGNORE] = u.Infra.array(sorted(ruff_cfg.lint.ignore))
            changes.append("tool.ruff.lint.ignore set")

        isort = u.Infra.ensure_table(lint, c.Infra.Toml.ISORT)
        for key, value in {
            "combine-as-imports": ruff_cfg.lint.isort.combine_as_imports,
            "force-single-line": ruff_cfg.lint.isort.force_single_line,
            "split-on-trailing-comma": ruff_cfg.lint.isort.split_on_trailing_comma,
        }.items():
            if u.Infra.unwrap_item(u.Infra.get(isort, key)) != value:
                isort[key] = value
                changes.append(f"tool.ruff.lint.isort.{key} set")

        per_file_ignores = u.Infra.ensure_table(lint, "per-file-ignores")
        for pattern in u.Infra.table_string_keys(per_file_ignores):
            if pattern not in ruff_cfg.lint.per_file_ignores:
                del per_file_ignores[pattern]
                changes.append(f"tool.ruff.lint.per-file-ignores.{pattern} removed")
        for pattern, rules in ruff_cfg.lint.per_file_ignores.items():
            if sorted(
                u.Infra.as_string_list(u.Infra.get(per_file_ignores, pattern))
            ) != sorted(rules):
                per_file_ignores[pattern] = u.Infra.array(sorted(rules))
                changes.append(f"tool.ruff.lint.per-file-ignores.{pattern} set")

        detected_packages = sorted(u.Infra.discover_first_party_namespaces(path.parent))
        if detected_packages:
            current_kfp = sorted(
                u.Infra.as_string_list(
                    u.Infra.get(isort, c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN)
                ),
            )
            if current_kfp != detected_packages:
                isort[c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN] = u.Infra.array(
                    detected_packages
                )
                changes.append(
                    f"tool.ruff.lint.isort.known-first-party set to {detected_packages}",
                )
        if c.Infra.Toml.LINT_SECTION in doc:
            del doc[c.Infra.Toml.LINT_SECTION]
            changes.append("removed stale top-level [lint] section")
        return changes
