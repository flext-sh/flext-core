"""Direct legacy-removal execution for the refactor service.

Pure rope + ``c.Infra.*_RE`` constants — no direct ``import ast`` or
``import re``. Wrapper detection uses rope's
``parse_string_module`` + ``walk_ast_nodes`` to locate passthrough
function definitions; pattern matching of node shapes uses
``node_kind`` and plain attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorLegacyTextOps:
    """Execute legacy-removal text operations directly from declarative settings."""

    def _apply_legacy_removal(
        self, settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Apply legacy removal."""
        changes: t.MutableSequenceOf[str] = []
        rule_id = str(settings.get(c.Infra.RK_ID, c.Infra.DEFAULT_UNKNOWN))
        fix_action = u.Cli.json_get_str_key(
            settings, c.Infra.RK_FIX_ACTION, case="lower"
        )
        updated = source
        if "alias" in rule_id or fix_action == "remove":
            updated, alias_changes = self._remove_aliases(settings, updated)
            changes.extend(alias_changes)
        if "deprecated" in rule_id or fix_action == "remove_and_update_refs":
            updated, deprecated_changes = self._remove_deprecated(updated)
            changes.extend(deprecated_changes)
        if "wrapper" in rule_id or fix_action == "inline_and_remove":
            updated, wrapper_changes = self._remove_wrappers(updated)
            changes.extend(wrapper_changes)
        if "bypass" in rule_id or fix_action == "keep_try_only":
            updated, bypass_changes = self._remove_import_bypasses(updated)
            changes.extend(bypass_changes)
        return (updated, changes)

    @staticmethod
    def _remove_aliases(
        settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Remove module-level ``X = Y`` aliases unless allow-listed."""
        allow_aliases = set(
            u.Infra.string_list(settings.get(c.Infra.RK_ALLOW_ALIASES, []))
        )
        allow_target_suffixes = tuple(
            u.Infra.string_list(settings.get(c.Infra.RK_ALLOW_TARGET_SUFFIXES, []))
        )
        changes: t.MutableSequenceOf[str] = []
        kept: t.MutableSequenceOf[str] = []
        for line in source.splitlines(keepends=True):
            if line != line.lstrip():
                kept.append(line)
                continue
            match = c.Infra.BARE_ALIAS_LINE_RE.match(line.strip())
            if match is None:
                kept.append(line)
                continue
            target, value = match.groups()
            if (
                target in allow_aliases
                or target in {c.Infra.DUNDER_VERSION, c.Infra.DUNDER_ALL}
                or (allow_target_suffixes and target.endswith(allow_target_suffixes))
            ):
                kept.append(line)
                continue
            changes.append(f"Removed alias: {target} = {value}")
        return ("".join(kept), changes) if changes else (source, list[str]())

    @staticmethod
    def _remove_deprecated(source: str) -> t.Infra.TransformResult:
        """Strip ``@deprecated``-decorated class/function blocks via constants regex."""
        changes = [
            f"Removed deprecated: {match.group(1)}"
            for match in c.Infra.DEPRECATED_RE.finditer(source)
        ]
        return (
            (c.Infra.DEPRECATED_RE.sub("", source), changes)
            if changes
            else (source, list[str]())
        )

    @classmethod
    def _remove_wrappers(cls, source: str) -> t.Infra.TransformResult:
        """Inline passthrough function wrappers via rope-located ranges."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return (source, list[str]())
        ast_root = pymodule.get_ast()
        body = getattr(ast_root, "body", []) or []
        lines = source.splitlines(keepends=True)
        changes: t.MutableSequenceOf[str] = []
        replacements: list[tuple[int, int, str]] = []
        for node in body:
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "FunctionDef":
                continue
            node_body = getattr(node, "body", []) or []
            if len(node_body) != 1:
                continue
            return_stmt = node_body[0]
            if FlextInfraUtilitiesRopeAnalysis.node_kind(return_stmt) != "Return":
                continue
            returned = getattr(return_stmt, "value", None)
            if FlextInfraUtilitiesRopeAnalysis.node_kind(returned) != "Call":
                continue
            call_func = getattr(returned, "func", None)
            if FlextInfraUtilitiesRopeAnalysis.node_kind(call_func) != "Name":
                continue
            if not cls._is_passthrough_wrapper(node, returned):
                continue
            target_name = getattr(call_func, "id", "")
            node_name = getattr(node, "name", "")
            lineno = getattr(node, "lineno", 0)
            end_lineno = getattr(node, "end_lineno", None) or lineno
            replacements.append((
                lineno - 1,
                end_lineno,
                f"{node_name} = {target_name}\n",
            ))
            changes.append(f"Inlined wrapper: {node_name} -> {target_name}")
        for start, end, replacement in reversed(replacements):
            lines[start:end] = [replacement]
        if not changes:
            return (source, list[str]())
        return ("".join(lines).rstrip("\n") + "\n", changes)

    @staticmethod
    def _is_passthrough_wrapper(func: object, call: object) -> bool:
        """Whether ``func``'s body is exactly ``return call(*args, **kwargs)`` over its params."""
        args_obj = getattr(func, "args", None)
        if args_obj is None:
            return False
        param_names = [
            arg.arg
            for arg in (
                *(getattr(args_obj, "posonlyargs", []) or []),
                *(getattr(args_obj, "args", []) or []),
            )
        ]
        keyword_names = [arg.arg for arg in (getattr(args_obj, "kwonlyargs", []) or [])]
        call_keywords = getattr(call, "keywords", []) or []
        named_keywords = {
            kw.arg: kw.value for kw in call_keywords if getattr(kw, "arg", None)
        }
        keyword_unpack = [
            kw.value for kw in call_keywords if getattr(kw, "arg", None) is None
        ]
        call_args = list(getattr(call, "args", []) or [])
        pos_index = 0
        for name in param_names:
            if pos_index < len(call_args) and FlextInfraRefactorLegacyTextOps._is_name(
                call_args[pos_index], name
            ):
                pos_index += 1
                continue
            if name in named_keywords and FlextInfraRefactorLegacyTextOps._is_name(
                named_keywords[name], name
            ):
                continue
            return False
        remaining_args = call_args[pos_index:]
        vararg = getattr(args_obj, "vararg", None)
        if vararg is None:
            vararg_ok = not remaining_args
        else:
            vararg_ok = (
                len(remaining_args) == 1
                and FlextInfraUtilitiesRopeAnalysis.node_kind(remaining_args[0])
                == "Starred"
                and FlextInfraRefactorLegacyTextOps._is_name(
                    getattr(remaining_args[0], "value", None), vararg.arg
                )
            )
        keywords_ok = all(
            name in named_keywords
            and FlextInfraRefactorLegacyTextOps._is_name(named_keywords[name], name)
            for name in keyword_names
        )
        kwarg = getattr(args_obj, "kwarg", None)
        if kwarg is None:
            kwarg_ok = not keyword_unpack
        else:
            kwarg_ok = len(keyword_unpack) == 1 and (
                FlextInfraRefactorLegacyTextOps._is_name(keyword_unpack[0], kwarg.arg)
            )
        return vararg_ok and keywords_ok and kwarg_ok

    @staticmethod
    def _is_name(node: object | None, name: str) -> bool:
        """Whether ``node`` is an ``ast.Name`` whose ``id`` is ``name``."""
        return (
            FlextInfraUtilitiesRopeAnalysis.node_kind(node) == "Name"
            and getattr(node, "id", None) == name
        )

    @staticmethod
    def _remove_import_bypasses(source: str) -> t.Infra.TransformResult:
        """Strip ``try/except ImportError`` import bypass blocks via constants regex."""
        changes = [
            "Removed import bypass block"
            for _ in c.Infra.IMPORT_BYPASS_RE.finditer(source)
        ]
        if not changes:
            return (source, list[str]())
        return (c.Infra.IMPORT_BYPASS_RE.sub(r"\1", source), changes)


__all__: list[str] = ["FlextInfraRefactorLegacyTextOps"]
