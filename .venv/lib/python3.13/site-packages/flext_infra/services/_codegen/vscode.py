"""VS Code settings codegen owner — the single canonical merge instrumentation.

This is the only place that knows how ``.vscode/settings.json`` is produced.
It parses that explicitly JSONC document through the canonical string-aware
normalizer, validates the resulting mapping, merges the config-driven canonical
keys from ``config.Infra.codegen.vscode`` plus the artifact-derived exclude maps,
and serializes the result with ``u.Cli.json_dumps``. Rendering is deliberately
independent of repository topology so opening a root or declared_repository produces the
same document. Planning, atomic writes, and fixed-point verification stay owned
by ``FlextInfraCodegenConform``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, t, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraCodegenVscodeMixin:
    """Produce the canonical ``.vscode/settings.json`` document for one root."""

    @classmethod
    def render_vscode_settings(cls, repository_root: Path) -> p.Result[str]:
        """Return the canonical-merged ``settings.json`` document for one root."""
        settings_path = (
            repository_root / c.Infra.VSCODE_DIRNAME / c.Infra.VSCODE_SETTINGS_FILENAME
        )
        read_result = cls._read_existing_settings(settings_path)
        if read_result.failure:
            return r[str].fail(read_result.error or "VS Code settings read failed")
        settings: t.MutableJsonMapping = {
            key: u.normalize_to_json_value(value)
            for key, value in read_result.value.items()
        }
        applied = cls._apply_canonical_settings(settings, repository_root)
        if applied.failure:
            return r[str].fail(applied.error or "VS Code settings merge failed")
        serialized = u.Cli.json_dumps(dict(settings), indent=2)
        if serialized.failure:
            return r[str].fail(serialized.error or "VS Code settings serialize failed")
        return r[str].ok(serialized.value + "\n")

    @classmethod
    def _read_existing_settings(cls, settings_path: Path) -> p.Result[t.JsonMapping]:
        """Read one VS Code JSONC document into a validated JSON mapping."""
        if not settings_path.exists():
            empty_settings: t.JsonMapping = {}
            return r[t.JsonMapping].ok(empty_settings)
        read_result = u.Cli.files_read_text(settings_path)
        if read_result.failure:
            return r[t.JsonMapping].fail(
                read_result.error or "VS Code settings read failed"
            )
        parsed = u.Cli.json_parse(cls._normalize_jsonc(read_result.value))
        if parsed.failure:
            return r[t.JsonMapping].fail(
                parsed.error or "VS Code settings JSONC parse failed"
            )
        if not isinstance(parsed.value, Mapping):
            return r[t.JsonMapping].fail("VS Code settings root must be an object")
        return r[t.JsonMapping].ok(
            t.Cli.JSON_MAPPING_ADAPTER.validate_python(parsed.value)
        )

    @classmethod
    def _normalize_jsonc(cls, content: str) -> str:
        """Return strict JSON text from VS Code JSONC content."""
        return cls._remove_trailing_commas(cls._remove_jsonc_comments(content))

    @staticmethod
    def _remove_jsonc_comments(content: str) -> str:
        """Remove JSONC comments while preserving comment markers in strings."""
        output: list[str] = []
        in_string = False
        escaped = False
        in_line_comment = False
        in_block_comment = False
        index = 0
        while index < len(content):
            char = content[index]
            next_char = content[index + 1] if index + 1 < len(content) else ""
            if in_line_comment:
                if char in "\r\n":
                    in_line_comment = False
                    output.append(char)
                index += 1
                continue
            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    index += 2
                    continue
                index += 1
                continue
            if in_string:
                output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                output.append(char)
                index += 1
                continue
            if char == "/" and next_char == "/":
                in_line_comment = True
                index += 2
                continue
            if char == "/" and next_char == "*":
                in_block_comment = True
                index += 2
                continue
            output.append(char)
            index += 1
        return "".join(output)

    @staticmethod
    def _remove_trailing_commas(content: str) -> str:
        """Remove commas before object or array closers outside strings."""
        output: list[str] = []
        in_string = False
        escaped = False
        index = 0
        while index < len(content):
            char = content[index]
            if in_string:
                output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                output.append(char)
                index += 1
                continue
            if char == ",":
                next_index = index + 1
                while next_index < len(content) and content[next_index].isspace():
                    next_index += 1
                if next_index < len(content) and content[next_index] in "}]":
                    index += 1
                    continue
            output.append(char)
            index += 1
        return "".join(output)

    @classmethod
    def _apply_canonical_settings(
        cls, settings: t.MutableJsonMapping, repository_root: Path
    ) -> p.Result[bool]:
        """Merge canonical codegen VS Code settings into one settings mapping."""
        spec = config.Infra.codegen.vscode
        changed = cls._apply_enforced_settings(
            settings,
            scalar_settings=spec.scalar_settings,
            list_settings=spec.list_settings,
            repository_root=repository_root,
        )
        if changed.failure:
            return r[bool].fail(changed.error)
        # The three exclude maps derive from the codegen artifact SSOT;
        # map_union_settings keeps only the remaining non-artifact keys.
        codegen = config.Infra.codegen
        map_union_settings: dict[str, Mapping[str, str | bool]] = {
            "files.exclude": dict(codegen.vscode_files_exclude_map),
            "files.watcherExclude": dict(codegen.vscode_watcher_exclude_map),
            "search.exclude": dict(codegen.vscode_search_exclude_map),
            **spec.map_union_settings,
        }
        return r[bool].ok(
            cls._apply_union_settings(settings, map_union_settings) or changed.value
        )

    @classmethod
    def _apply_enforced_settings(
        cls,
        settings: t.MutableJsonMapping,
        *,
        scalar_settings: Mapping[str, str | bool],
        list_settings: Mapping[str, tuple[str, ...]],
        repository_root: Path,
    ) -> p.Result[bool]:
        """Enforce exact scalar and list VS Code keys from the codegen config."""
        changed = False
        for key, value in scalar_settings.items():
            normalized = u.normalize_to_json_value(value)
            if settings.get(key) == normalized:
                continue
            settings[key] = normalized
            changed = True
        for key, list_value in list_settings.items():
            entries = cls._resolve_list_setting(
                key, list_value, repository_root=repository_root
            )
            if entries.failure:
                return r[bool].fail(entries.error)
            canonical: list[t.JsonValue] = [
                u.normalize_to_json_value(entry) for entry in entries.value
            ]
            if settings.get(key) == canonical:
                continue
            settings[key] = canonical
            changed = True
        return r[bool].ok(changed)

    @staticmethod
    def _apply_union_settings(
        settings: t.MutableJsonMapping,
        map_union_settings: Mapping[str, Mapping[str, str | bool]],
    ) -> bool:
        """Union-merge canonical map keys over existing project entries."""
        changed = False
        for key, canonical_map in map_union_settings.items():
            current = settings.get(key)
            existing: dict[str, t.JsonValue] = (
                {
                    name: u.normalize_to_json_value(value)
                    for name, value in current.items()
                }
                if isinstance(current, Mapping)
                else {}
            )
            merged: dict[str, t.JsonValue] = existing | {
                name: u.normalize_to_json_value(value)
                for name, value in canonical_map.items()
            }
            if settings.get(key) == merged:
                continue
            settings[key] = merged
            changed = True
        return changed

    @staticmethod
    def _resolve_list_setting(
        key: str, base_entries: tuple[str, ...], *, repository_root: Path
    ) -> p.Result[tuple[str, ...]]:
        """Return one canonical list without consulting repository topology."""
        del key, repository_root
        return r[tuple[str, ...]].ok(base_entries)


__all__: list[str] = ["FlextInfraCodegenVscodeMixin"]
