"""Rope-backed autofix for ENFORCE-079 ClassVar constants outside _constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import m, u
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore

if TYPE_CHECKING:
    from flext_infra import t


_DOCSTRING_DELIMITER_COUNT = 2
_CLASSVAR_DECLARATION_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9_]*:\s*)ClassVar\[(.*)\](\s*=)", re.DOTALL
)


@dataclass(frozen=True)
class ClassvarConstantAutofixPlan:
    """Planned edits for one ENFORCE-079 autofix."""

    class_module: str
    class_name: str
    constant_name: str
    constants_module: str
    source_resource: t.Infra.RopeResource
    target_resource: t.Infra.RopeResource
    declaration_line: str
    class_lineno: int


class FlextInfraRefactorClassvarConstantAutofix:
    """Move a ClassVar constant from a class body to a _constants module.

    The fix keeps the public facade contract intact: consumers inside the
    package access the constant through the canonical ``_constants`` module,
    while external consumers continue to reach it via ``c.Infra.*``.
    """

    @staticmethod
    def plan(
        repository_root: Path,
        class_full_name: str,
        constant_name: str,
        constants_module: str,
    ) -> ClassvarConstantAutofixPlan:
        """Build an autofix plan without touching disk."""
        with FlextInfraUtilitiesRopeCore.open_project(repository_root) as project:
            return FlextInfraRefactorClassvarConstantAutofix._plan_with_project(
                project, class_full_name, constant_name, constants_module
            )

    @staticmethod
    def _plan_with_project(
        project: t.Infra.RopeProject,
        class_full_name: str,
        constant_name: str,
        constants_module: str,
    ) -> ClassvarConstantAutofixPlan:
        class_module, class_name = class_full_name.rsplit(".", maxsplit=1)
        source_mod = project.get_module(class_module, project.root)
        source_resource = source_mod.get_resource()
        if not u.Infra.is_resource(source_resource):
            msg = f"{class_module} did not resolve to a file resource"
            raise TypeError(msg)
        pyclass = source_mod.get_attribute(class_name).get_object()
        if not u.Infra.is_runtime_pyclass(pyclass):
            msg = f"{class_full_name} did not resolve to a class"
            raise TypeError(msg)
        source_text = source_resource.read()
        class_lineno = _class_start_lineno(source_text, class_name)
        declaration_line = _extract_declaration_line(
            source_text, class_name, constant_name, class_lineno
        )
        target_resource = _target_resource_for_module(
            project,
            constants_module,
            class_module=class_module,
            source_resource=source_resource,
        )
        return ClassvarConstantAutofixPlan(
            class_module=class_module,
            class_name=class_name,
            constant_name=constant_name,
            constants_module=constants_module,
            source_resource=source_resource,
            target_resource=target_resource,
            declaration_line=declaration_line,
            class_lineno=class_lineno,
        )

    @staticmethod
    def apply(
        repository_root: Path,
        class_full_name: str,
        constant_name: str,
        constants_module: str,
        *,
        dry_run: bool = False,
    ) -> m.Infra.ClassvarConstantAutofixResult:
        """Move the constant and rewrite all internal references."""
        with FlextInfraUtilitiesRopeCore.open_project(repository_root) as project:
            plan = FlextInfraRefactorClassvarConstantAutofix._plan_with_project(
                project, class_full_name, constant_name, constants_module
            )
            return FlextInfraRefactorClassvarConstantAutofix._apply_with_project(
                project, plan, dry_run=dry_run
            )

    @staticmethod
    def _apply_with_project(
        project: t.Infra.RopeProject,
        plan: ClassvarConstantAutofixPlan,
        *,
        dry_run: bool,
    ) -> m.Infra.ClassvarConstantAutofixResult:
        source_text = plan.source_resource.read()
        target_path = Path(plan.target_resource.real_path)
        target_text = plan.target_resource.read()
        touched: set[str] = {plan.source_resource.path, plan.target_resource.path}
        constants_alias = plan.constants_module.split(".")[-1]

        # 1. Remove the ClassVar declaration from the class body.
        new_source = _remove_declaration_line(
            source_text,
            plan.declaration_line,
            plan.class_lineno,
            plan.class_name,
            plan.constant_name,
        )

        # 2. Append the constant to the target _constants module, unless the
        # class declaration was only a typed alias to an existing owner.
        if _module_declares_name(target_text, plan.constant_name):
            if not _declaration_aliases_target_constant(
                plan.declaration_line, constants_alias, plan.constant_name
            ):
                msg = (
                    f"{plan.constants_module} already declares "
                    f"{plan.constant_name}; class declaration is not an owner alias"
                )
                raise TypeError(msg)
            new_target = target_text
        else:
            new_target = _append_constant(
                target_text, plan.declaration_line, plan.constants_module
            )

        # 3. Rewrite internal references from ClassName.NAME / cls.NAME /
        # self.__class__.NAME to the canonical constants module access.
        class_module_obj = project.get_module(plan.class_module, project.root)
        pyclass = class_module_obj.get_attribute(plan.class_name).get_object()
        pyname = pyclass.get_attribute(plan.constant_name)
        finder = u.Infra.create_occurrence_finder(
            project, plan.constant_name, pyname, imports=True, in_hierarchy=False
        )
        rewrites: dict[str, list[tuple[int, int, str]]] = {}
        # Iterate over concrete project resources to avoid rope crashing when
        # an occurrence cannot be resolved to a resource (resource=None).
        for resource in project.get_python_files():
            for occurrence in finder.find_occurrences(resource=resource):
                offset = occurrence.offset
                source = resource.read()
                prefix = _attribute_prefix(source, offset)
                if _should_rewrite_prefix(prefix, plan.class_name):
                    try:
                        start, end = u.Infra.word_primary_range(source, offset)
                    except (*u.Infra.rope_runtime_errors(), TypeError, ValueError):
                        start, end = occurrence.get_word_range()
                    replacement = (
                        f"{plan.constants_module.split('.')[-1]}.{plan.constant_name}"
                    )
                    rewrites.setdefault(resource.path, []).append((
                        start,
                        end,
                        replacement,
                    ))
                    touched.add(resource.path)

        if dry_run:
            source_preview = _rewrite_class_access_in_source(
                new_source,
                plan.class_name,
                plan.constant_name,
                plan.constants_module.split(".")[-1],
            )
            return m.Infra.ClassvarConstantAutofixResult(
                touched_files=tuple(sorted(touched)),
                source_text=source_preview,
                target_text=new_target,
                rewrites=dict(rewrites),
            )

        # 4. Rewrite source-module references textually. Rope occurrence offsets
        # target the original source and become stale after the declaration is
        # removed, so the modified source uses the known class-access forms.
        rewrites.pop(plan.source_resource.path, None)
        new_source = _rewrite_class_access_in_source(
            new_source, plan.class_name, plan.constant_name, constants_alias
        )
        # 5. Inject ``from . import _constants`` (or equivalent) into the source
        # module so the rewritten references resolve.
        new_source = _ensure_constants_import(
            new_source, constants_alias, plan.class_module, plan.constants_module
        )
        Path(plan.source_resource.real_path).write_text(new_source, encoding="utf-8")
        if new_target != target_text:
            target_path.write_text(new_target, encoding="utf-8")

        # 6. Apply remaining rewrites to other resources.
        for resource in project.get_python_files():
            edits = rewrites.get(resource.path)
            if not edits:
                continue
            text = resource.read()
            text = _apply_edits(text, edits)
            Path(resource.real_path).write_text(text, encoding="utf-8")

        return m.Infra.ClassvarConstantAutofixResult(
            touched_files=tuple(sorted(touched)), constant_module=plan.constants_module
        )


def _class_start_lineno(source: str, class_name: str) -> int:
    """Return the 1-based line where ``class class_name`` starts."""
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("class ") and class_name in stripped:
            return lineno
    msg = f"Could not locate class {class_name}"
    raise ValueError(msg)


def _target_resource_for_module(
    project: t.Infra.RopeProject,
    constants_module: str,
    *,
    class_module: str,
    source_resource: t.Infra.RopeResource,
) -> t.Infra.RopeResource:
    """Return the existing canonical constants module resource."""
    source_path = Path(source_resource.real_path).resolve()
    class_path = Path(*class_module.split("."))
    source_suffix = next(
        (
            suffix
            for suffix in (class_path.with_suffix(".py"), class_path / "__init__.py")
            if source_path.parts[-len(suffix.parts) :] == suffix.parts
        ),
        None,
    )
    if source_suffix is None:
        msg = f"{class_module} did not resolve inside its import root"
        raise TypeError(msg)
    import_root = source_path
    for _part in source_suffix.parts:
        import_root = import_root.parent
    constants_path = Path(*constants_module.split("."))
    candidates = tuple(
        candidate
        for candidate in (
            import_root / constants_path.with_suffix(".py"),
            import_root / constants_path / "__init__.py",
        )
        if candidate.is_file()
    )
    if not candidates:
        msg = f"constants module {constants_module} does not exist"
        raise TypeError(msg)
    if len(candidates) != 1:
        msg = f"constants module {constants_module} has multiple canonical owners"
        raise TypeError(msg)
    target_resource = u.Infra.get_resource_from_path(project, candidates[0])
    if target_resource is None:
        msg = f"constants module {constants_module} is outside the Rope project"
        raise TypeError(msg)
    resource: t.Infra.RopeResource = target_resource
    return resource


def _extract_declaration_line(
    source: str, class_name: str, constant_name: str, class_lineno: int
) -> str:
    """Return the exact source line that declares the class-level constant."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for statement in node.body:
                if not _statement_declares_name(statement, constant_name):
                    continue
                segment = ast.get_source_segment(source, statement)
                if segment is not None:
                    return _normalize_declaration_source(segment)
                return _normalize_declaration_source(
                    _statement_source_by_lines(source, statement)
                )
            break
    lines = source.splitlines()
    for idx in range(class_lineno, len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("class "):
            break
        if stripped.startswith((f"{constant_name}:", f"{constant_name} =")):
            return line.rstrip()
    msg = f"Could not find constant declaration for {constant_name} in {class_name}"
    raise ValueError(msg)


def _statement_declares_name(statement: ast.stmt, constant_name: str) -> bool:
    """Return whether ``statement`` declares ``constant_name``."""
    if isinstance(statement, ast.AnnAssign):
        return (
            isinstance(statement.target, ast.Name)
            and statement.target.id == constant_name
        )
    if isinstance(statement, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in statement.targets
        )
    return False


def _module_declares_name(source: str, constant_name: str) -> bool:
    """Return whether module source already declares ``constant_name``."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        msg = f"target constants module is not parseable: {exc.msg}"
        raise TypeError(msg) from exc
    return any(
        _statement_declares_name(statement, constant_name) for statement in tree.body
    )


def _declaration_aliases_target_constant(
    declaration_line: str, constants_alias: str, constant_name: str
) -> bool:
    """Return whether a class declaration points at the existing constants owner."""
    try:
        tree = ast.parse(textwrap.dedent(declaration_line).strip())
    except SyntaxError:
        return False
    if len(tree.body) != 1:
        return False
    value = _assignment_value(tree.body[0])
    return (
        isinstance(value, ast.Attribute)
        and value.attr == constant_name
        and isinstance(value.value, ast.Name)
        and value.value.id == constants_alias
    )


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    """Return the right-hand side expression for supported assignments."""
    if isinstance(statement, ast.AnnAssign):
        return statement.value
    if isinstance(statement, ast.Assign):
        return statement.value
    return None


def _statement_source_by_lines(source: str, statement: ast.stmt) -> str:
    """Return statement source using AST line metadata."""
    lines = source.splitlines()
    start = max(getattr(statement, "lineno", 1), 1)
    end = max(getattr(statement, "end_lineno", start), start)
    return "\n".join(lines[start - 1 : end])


def _normalize_declaration_source(source: str) -> str:
    """Return declaration source normalized for module-level placement."""
    lines = textwrap.dedent(source.rstrip()).splitlines()
    if len(lines) <= 1 or lines[0].startswith((" ", "\t")):
        return "\n".join(lines)
    tail = lines[1:]
    indents = [len(line) - len(line.lstrip()) for line in tail if line.strip()]
    if not indents:
        return "\n".join(lines)
    trim = min(indents)
    if trim <= 0:
        return "\n".join(lines)
    return "\n".join((
        lines[0],
        *(line[trim:] if line.strip() else line for line in tail),
    ))


def _remove_declaration_line(
    source: str,
    declaration_line: str,
    class_lineno: int,
    class_name: str,
    constant_name: str,
) -> str:
    """Remove the constant declaration from the class body, preserving layout."""
    lines = source.splitlines(keepends=True)
    ast_removed = _remove_declaration_by_ast(source, lines, class_name, constant_name)
    if ast_removed is not None:
        return ast_removed
    for idx in range(class_lineno - 1, len(lines)):
        if _declaration_block_matches(lines, declaration_line, idx):
            # Also remove the blank line that typically precedes it, if any.
            end_idx = idx + len(declaration_line.splitlines())
            if idx > 0 and not lines[idx - 1].strip():
                del lines[idx - 1 : end_idx]
            else:
                del lines[idx:end_idx]
            return "".join(lines)
    return source


def _remove_declaration_by_ast(
    source: str, lines: list[str], class_name: str, constant_name: str
) -> str | None:
    """Remove a class-body declaration using AST line metadata."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not _statement_declares_name(statement, constant_name):
                continue
            start = max(statement.lineno - 1, 0)
            end = max(
                getattr(statement, "end_lineno", statement.lineno), statement.lineno
            )
            delete_start = (
                start - 1 if start > 0 and not lines[start - 1].strip() else start
            )
            del lines[delete_start:end]
            return "".join(lines)
    return None


def _declaration_block_matches(
    lines: t.SequenceOf[str], declaration_line: str, start: int
) -> bool:
    """Return whether source lines at ``start`` match the declaration block."""
    declaration_lines = declaration_line.splitlines()
    if not declaration_lines or start + len(declaration_lines) > len(lines):
        return False
    source_block = "".join(lines[start : start + len(declaration_lines)]).rstrip()
    return source_block == declaration_line or (
        textwrap.dedent(source_block).rstrip()
        == textwrap.dedent(declaration_line).rstrip()
    )


def _apply_edits(text: str, edits: t.SequenceOf[tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) edits to ``text`` in reverse order."""
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _append_constant(
    target_text: str, declaration_line: str, constants_module: str
) -> str:
    """Append the constant declaration to the target module source."""
    module_declaration = _module_level_declaration_source(declaration_line)
    with_imports = _ensure_declaration_imports(
        target_text, module_declaration, constants_module
    )
    trimmed = with_imports.rstrip()
    dedented = textwrap.dedent(module_declaration).strip()
    return f"{trimmed}\n\n{dedented}\n"


def _module_level_declaration_source(declaration_line: str) -> str:
    """Return a declaration valid at module level."""
    return _CLASSVAR_DECLARATION_PATTERN.sub(r"\1\2\3", declaration_line, count=1)


def _ensure_declaration_imports(
    target_text: str, declaration_line: str, constants_module: str
) -> str:
    """Add imports needed by the moved declaration."""
    stdlib_imports: list[str] = []
    typing_names: list[str] = []
    project_imports: list[str] = []
    if "Mapping[" in declaration_line:
        stdlib_imports.append("from collections.abc import Mapping\n")
    if "MappingProxyType(" in declaration_line:
        stdlib_imports.append("from types import MappingProxyType\n")
    if "ClassVar[" in declaration_line:
        typing_names.append("ClassVar")
    if "Final[" in declaration_line:
        typing_names.append("Final")
    if "t." in declaration_line:
        package_name = constants_module.split(".", maxsplit=1)[0]
        project_imports.append(f"from {package_name}.typings import t\n")
    if typing_names:
        names = ", ".join(sorted(set(typing_names)))
        stdlib_imports.append(f"from typing import {names}\n")
    return _insert_missing_import_lines(
        target_text, (*sorted(stdlib_imports), *sorted(project_imports))
    )


def _insert_missing_import_lines(source: str, import_lines: t.StrSequence) -> str:
    """Insert missing import lines after the module header/future imports."""
    missing = [line for line in import_lines if line.strip() not in source]
    if not missing:
        return source
    lines = source.splitlines(keepends=True)
    insert_after = _module_import_insert_after(lines)
    stdlib = [line for line in missing if not _is_project_import_line(line)]
    project = [line for line in missing if _is_project_import_line(line)]
    block: list[str] = ["\n"]
    block.extend(stdlib)
    if project:
        block.append("\n")
        block.extend(project)
    block.append("\n")
    if insert_after >= 0:
        lines[insert_after + 1 : insert_after + 1] = block
    else:
        lines[0:0] = block
    return "".join(lines)


def _is_project_import_line(import_line: str) -> bool:
    """Return whether an import line targets the moved constant package."""
    return ".typings import t" in import_line


def _attribute_prefix(source: str, offset: int) -> str:
    """Return the primary expression for an attribute access at ``offset``.

    For ``Foo.BAR`` at the position of ``BAR`` this returns ``"Foo"``.
    For ``cls.BAR`` returns ``"cls"``.  For ``self.__class__.BAR`` returns
    ``"self.__class__"``.  For bare ``BAR`` returns the empty string.
    """
    try:
        return u.Infra.word_primary_at(source, offset)
    except (*u.Infra.rope_runtime_errors(), TypeError, ValueError):
        return ""


def _should_rewrite_prefix(prefix: str, class_name: str) -> bool:
    """Return whether an attribute prefix should be rewritten to _constants."""
    return prefix in {class_name, "cls"} or prefix.endswith(".__class__")


def _ensure_constants_import(
    source: str, constants_alias: str, class_module: str, constants_module: str
) -> str:
    """Add an import for the canonical _constants module if absent.

    Computes a relative import from ``class_module`` to ``constants_module``.
    The import binds the module itself so consumers write
    ``_constants.NAME``.
    """
    if constants_module == class_module:
        return source
    class_parts = class_module.split(".")
    constants_parts = constants_module.split(".")
    common = 0
    for a, b in zip(class_parts, constants_parts, strict=False):
        if a != b:
            break
        common += 1
    ups = len(class_parts) - common - 1  # minus the source module itself
    rel_parts = constants_parts[common:]
    if ups == 0 and rel_parts == [constants_alias]:
        import_line = f"from . import {constants_alias}\n"
    elif ups == 0 and len(rel_parts) > 1:
        import_line = f"from .{'.'.join(rel_parts[:-1])} import {constants_alias}\n"
    elif ups == 0 and rel_parts:
        import_line = f"from .{'.'.join(rel_parts)} import {constants_alias}\n"
    elif ups > 0 and len(rel_parts) > 1:
        import_line = (
            f"from {'.' * ups}{'.'.join(rel_parts[:-1])} import {constants_alias}\n"
        )
    elif ups > 0 and rel_parts:
        import_line = (
            f"from {'.' * ups}{'.'.join(rel_parts)} import {constants_alias}\n"
        )
    elif ups > 0:
        import_line = f"from {'.' * ups} import {constants_alias}\n"
    else:
        import_line = f"from {constants_module} import {constants_alias}\n"

    lines = source.splitlines(keepends=True)
    existing = {line.strip() for line in lines}
    if import_line.strip() in existing:
        return source
    _insert_local_import_line(lines, import_line)
    return "".join(lines)


def _insert_local_import_line(lines: list[str], import_line: str) -> None:
    """Insert a relative import after the module import block."""
    insert_after = _module_import_block_end(lines)
    insert_at = insert_after + 1 if insert_after >= 0 else 0
    block = []
    if insert_at > 0 and lines[insert_at - 1].strip():
        block.append("\n")
    block.append(import_line)
    if insert_at < len(lines) and lines[insert_at].strip():
        block.append("\n")
    lines[insert_at:insert_at] = block


def _module_import_block_end(lines: t.StrSequence) -> int:
    """Return the last line index of the module import block."""
    idx = _module_import_insert_after(lines) + 1
    last_import = idx - 1
    paren_depth = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if paren_depth > 0:
            paren_depth += lines[idx].count("(") - lines[idx].count(")")
            last_import = idx
            idx += 1
            continue
        if not stripped:
            idx += 1
            continue
        if stripped.startswith(("import ", "from ")) and not stripped.startswith(
            "from __future__ import"
        ):
            paren_depth += lines[idx].count("(") - lines[idx].count(")")
            last_import = idx
            idx += 1
            continue
        break
    return last_import


def _module_import_insert_after(lines: t.StrSequence) -> int:
    """Return index after module docstring and future imports only."""
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    insert_after = -1
    if idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            insert_after = idx
            if stripped.count(quote) < _DOCSTRING_DELIMITER_COUNT:
                idx += 1
                while idx < len(lines):
                    insert_after = idx
                    if quote in lines[idx]:
                        idx += 1
                        break
                    idx += 1
            else:
                idx += 1
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("from __future__ import"):
            insert_after = idx
            idx += 1
            continue
        break
    return insert_after


def _rewrite_class_access_in_source(
    source: str, class_name: str, constant_name: str, constants_alias: str
) -> str:
    """Rewrite class-qualified constant access inside the source module itself.

    Rope's occurrence finder binds ``ClassName.NAME`` and ``cls.NAME`` but not
    ``self.__class__.NAME``.  We perform a fast textual pass over the source
    (after the declaration was removed) and rewrite the known patterns.
    """
    patterns = (
        f"{class_name}.{constant_name}",
        f"cls.{constant_name}",
        f"self.{constant_name}",
        f"self.__class__.{constant_name}",
    )
    replacement = f"{constants_alias}.{constant_name}"
    edits: list[tuple[int, int, str]] = []
    for pattern in patterns:
        start = 0
        while True:
            idx = source.find(pattern, start)
            if idx < 0:
                break
            edits.append((idx, idx + len(pattern), replacement))
            start = idx + len(replacement)
    return _apply_edits(source, edits)


__all__: list[str] = ["FlextInfraRefactorClassvarConstantAutofix"]
