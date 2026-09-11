"""Detect inline/lazy imports inside function bodies and importlib calls.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
import sys
from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraInlineImportDetector:
    """Detect function imports and dynamic imports through Rope facts."""

    @staticmethod
    def fix_action_for(*, module_name: str, is_importlib: bool) -> str:
        """Return the configured action class for one violation."""
        if is_importlib:
            return "manual"
        top_level = module_name.split(".", maxsplit=1)[0] if module_name else ""
        if top_level in sys.stdlib_module_names:
            return "hoist_inline_import"
        if top_level in c.ENFORCEMENT_LIBRARY_OWNERS:
            return "rewrite_library_abstraction"
        return "manual"

    @classmethod
    def detect_file(
        cls, ctx: m.Infra.DetectorContext
    ) -> t.SequenceOf[m.Infra.InlineImportViolation]:
        """Return Rope-resolved inline imports and dynamic import calls."""
        file_path = ctx.file_path
        parts = file_path.parts
        if (
            file_path.name in c.Infra.INLINE_IMPORT_EXEMPT_FILE_NAMES
            or any(part in c.Infra.INLINE_IMPORT_EXEMPT_PATH_PARTS for part in parts)
            or (file_path.name == "lazy.py" and c.Infra.PKG_CORE_UNDERSCORE in parts)
        ):
            return ()
        resource = u.Infra.fetch_python_resource(
            ctx.rope_project, file_path, skip_init_py=False
        )
        if resource is None:
            return ()
        source = resource.read()
        statements = u.Infra.logical_statements(source)
        source_lines = source.splitlines(keepends=True)
        module, _, name = c.Infra.IMPORTLIB_IMPORT_MODULE.rpartition(".")
        bindings: t.MutableSequenceOf[t.Triple[str, int, int]] = []
        violations: t.MutableSequenceOf[m.Infra.InlineImportViolation] = []
        for statement in statements:
            if statement.type_checking_guarded or statement.category not in {
                c.Infra.StatementCategory.IMPORT,
                c.Infra.StatementCategory.FROM_IMPORT,
            }:
                continue
            in_function = statement.enclosing_kind == c.Infra.RopeScopeKind.FUNCTION
            at_module = statement.enclosing_kind == c.Infra.RopeScopeKind.MODULE
            pymodule = u.Infra.get_string_module(
                ctx.rope_project, statement.text.strip()
            )
            module_imports = u.Infra.module_imports_for_pymodule(
                ctx.rope_project, pymodule
            )
            for import_statement in u.Infra.import_statements(module_imports):
                import_info = import_statement.import_info
                if u.Infra.is_normal_import(import_info):
                    for imported_name, alias_name in import_info.names_and_aliases:
                        if in_function:
                            current_import = f"import {imported_name}"
                            violations.append(
                                m.Infra.InlineImportViolation(
                                    file=str(file_path),
                                    line=statement.line,
                                    current_import=current_import,
                                    detail=(
                                        f"Inline import '{current_import}' inside "
                                        "function body"
                                    ),
                                    module_name=imported_name.split(".", maxsplit=1)[0],
                                    imported_symbols=(alias_name or imported_name,),
                                )
                            )
                        elif at_module and imported_name == module:
                            local_name = alias_name or imported_name
                            bindings.append((
                                f"{local_name}.{name}",
                                len(local_name) + 1,
                                cls._binding_offset(
                                    source=source,
                                    source_lines=source_lines,
                                    statement=statement,
                                    local_name=local_name,
                                ),
                            ))
                    continue
                if not u.Infra.is_from_import(import_info):
                    continue
                if in_function:
                    names = ", ".join(
                        f"{name} as {alias}" if alias else name
                        for name, alias in import_info.names_and_aliases
                    )
                    current_import = f"from {import_info.module_name} import {names}"
                    violations.append(
                        m.Infra.InlineImportViolation(
                            file=str(file_path),
                            line=statement.line,
                            current_import=current_import,
                            detail=(
                                f"Inline import '{current_import}' inside function body"
                            ),
                            module_name=import_info.module_name,
                            imported_symbols=tuple(
                                name for name, _alias in import_info.names_and_aliases
                            ),
                        )
                    )
                    continue
                if at_module and import_info.module_name == module:
                    for imported_name, alias_name in import_info.names_and_aliases:
                        if imported_name != name:
                            continue
                        local_name = alias_name or imported_name
                        bindings.append((
                            local_name,
                            0,
                            cls._binding_offset(
                                source=source,
                                source_lines=source_lines,
                                statement=statement,
                                local_name=local_name,
                            ),
                        ))
        for primary, suffix_offset, binding_offset in bindings:
            for location in u.Infra.find_occurrences(
                ctx.rope_project, resource, binding_offset, resources=(resource,)
            ):
                primary_offset = location.offset + suffix_offset
                region = next(
                    (
                        item
                        for item in statements
                        if item.line <= location.lineno <= item.end_line
                    ),
                    None,
                )
                if (
                    region is not None
                    and not region.type_checking_guarded
                    and u.Infra.word_primary_at(source, primary_offset) == primary
                    and u.Infra.word_is_function_call(source, primary_offset)
                ):
                    violations.append(
                        m.Infra.InlineImportViolation(
                            file=str(file_path),
                            line=location.lineno,
                            current_import="importlib.import_module(...)",
                            detail=(
                                "Dynamic import via importlib.import_module outside "
                                "flext_core/lazy.py"
                            ),
                            is_importlib=True,
                        )
                    )
        return tuple(sorted(violations, key=operator.attrgetter("line")))

    @staticmethod
    def _binding_offset(
        *,
        source: str,
        source_lines: t.StrSequence,
        statement: m.Infra.LogicalStatement,
        local_name: str,
    ) -> int:
        """Return a Rope-recognized import binding offset."""
        region_start = sum(
            len(source_line) for source_line in source_lines[: statement.line - 1]
        )
        relative_offset = 0
        while (
            relative_offset := statement.text.find(local_name, relative_offset)
        ) >= 0:
            candidate = region_start + relative_offset
            if u.Infra.word_primary_at(source, candidate) == local_name:
                return candidate
            relative_offset += len(local_name)
        msg = f"Rope import binding offset unavailable for {local_name}"
        raise RuntimeError(msg)


__all__: list[str] = ["FlextInfraInlineImportDetector"]
