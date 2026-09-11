"""AST-only silent-failure detection helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import NamedTuple, override


class _SilentFailureFinding(NamedTuple):
    line: int
    column: int
    kind: str
    detail: str
    fix_action: str
    replacement: tuple[int, int, str] | None = None


class _SilentFailureAstVisitor(ast.NodeVisitor):
    """AST visitor collecting exception-silencing patterns.

    Walks the rope-backed AST (``pymodule.get_ast()``).  No regex is used;
    all findings are derived from structural AST nodes.
    """

    _SENTINEL_CONSTANTS: frozenset[object] = frozenset({False, None})
    _BROAD_EXCEPTION_NAMES: frozenset[str] = frozenset({"Exception", "BaseException"})

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.findings: list[_SilentFailureFinding] = []
        self._import_aliases: dict[str, str] = {}
        self._parents: dict[ast.AST, ast.AST] = {}

    def analyze(self, tree: ast.Module) -> list[_SilentFailureFinding]:
        """Build parent map and walk the rope-backed module AST."""
        self._parents.clear()
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent
        self.visit(tree)
        return self.findings

    def _enclosing_function(
        self, node: ast.AST
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
                return current
            current = self._parents.get(current)
        return None

    def _result_inner_type(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str | None:
        returns = func.returns
        if not isinstance(returns, ast.Subscript):
            return None
        value = returns.value
        is_result_shape = (
            isinstance(value, ast.Name) and value.id in {"r", "Result"}
        ) or (isinstance(value, ast.Attribute) and value.attr == "Result")
        if not is_result_shape:
            return None
        inner_type: str = ast.unparse(returns.slice)
        return inner_type

    def _line_offsets(self, lineno: int) -> tuple[int, int]:
        start = sum(len(self.lines[i]) for i in range(lineno - 1))
        end = start + len(self.lines[lineno - 1])
        return start, end

    def _indent_of(self, node: ast.Return) -> str:
        line = self.lines[node.lineno - 1]
        indent: str = line[: len(line) - len(line.lstrip())]
        return indent

    def _add_finding(
        self,
        *,
        line: int,
        column: int,
        kind: str,
        detail: str,
        fix_action: str = "manual",
        replacement: tuple[int, int, str] | None = None,
    ) -> None:
        self.findings.append(
            _SilentFailureFinding(
                line=line,
                column=column,
                kind=kind,
                detail=detail,
                fix_action=fix_action,
                replacement=replacement,
            )
        )

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            self._import_aliases[bound_name] = alias.name
        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        for alias in node.names:
            bound_name = alias.asname or alias.name
            self._import_aliases[bound_name] = (
                f"{module_name}.{alias.name}" if module_name else alias.name
            )
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        call_name = _resolve_call_name(node, self._import_aliases)
        if call_name == "contextlib.suppress":
            self._add_finding(
                line=node.lineno,
                column=node.col_offset,
                kind="silent-failure-suppress",
                detail=(
                    "contextlib.suppress(...) silences exceptions without logging "
                    "or propagation"
                ),
            )
        elif _is_unwrap_or_call(node):
            self._add_finding(
                line=node.lineno,
                column=node.col_offset,
                kind="silent-failure-unwrap-or",
                detail="unwrap_or(sentinel) hides a failure path",
            )
        self.generic_visit(node)

    @override
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._is_except_pass(node):
            self._add_finding(
                line=node.lineno,
                column=node.col_offset,
                kind="silent-failure-except-pass",
                detail="except handler with empty pass swallows the exception",
            )
        elif self._is_broad_unhandled_except(node):
            self._add_finding(
                line=node.lineno,
                column=node.col_offset,
                kind="silent-failure-broad-except",
                detail=(
                    "broad except catches Exception/BaseException without re-raise "
                    "or r[...].fail(...) propagation"
                ),
            )
        elif self._is_except_sentinel(node):
            self._add_except_sentinel_finding(node)
        self.generic_visit(node)

    @override
    def visit_If(self, node: ast.If) -> None:
        guard_info = self._guard_info(node)
        if guard_info is not None:
            result_name, _success_branch = guard_info
            self._add_guard_finding(node, result_name)
        self.generic_visit(node)

    def _is_except_pass(self, node: ast.ExceptHandler) -> bool:
        return all(
            isinstance(stmt, ast.Pass)
            or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
            for stmt in node.body
        ) and any(isinstance(stmt, ast.Pass) for stmt in node.body)

    def _is_broad_unhandled_except(self, node: ast.ExceptHandler) -> bool:
        if self._body_has_raise_or_fail(node.body):
            return False
        type_name = _expression_name(node.type, self._import_aliases)
        if not type_name:
            return True  # bare ``except:`` is equivalent to BaseException
        return type_name in self._BROAD_EXCEPTION_NAMES

    def _is_except_sentinel(self, node: ast.ExceptHandler) -> bool:
        if node.type is not None:
            type_name = _expression_name(node.type, self._import_aliases)
            if type_name in self._BROAD_EXCEPTION_NAMES or not type_name:
                return False
        return not self._body_has_raise_or_fail(
            node.body
        ) and self._body_has_sentinel_return(node.body)

    def _body_has_sentinel_return(self, body: list[ast.stmt]) -> bool:
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Return) and self._is_sentinel_value(
                    child.value
                ):
                    return True
        return False

    def _is_sentinel_value(self, node: ast.expr | None) -> bool:
        if node is None:
            return True
        if isinstance(node, ast.Constant) and node.value in self._SENTINEL_CONSTANTS:
            return True
        if isinstance(node, ast.List) and not node.elts:
            return True
        return isinstance(node, ast.Dict) and not node.keys

    @staticmethod
    def _body_has_raise_or_fail(body: list[ast.stmt]) -> bool:
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Raise):
                    return True
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "fail"
                ):
                    return True
        return False

    def _guard_info(self, node: ast.If) -> tuple[str, bool] | None:
        test = node.test
        if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
            if test.attr == "failure":
                return test.value.id, False
            if test.attr == "success":
                return test.value.id, True
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Attribute)
            and isinstance(test.operand.value, ast.Name)
        ):
            if test.operand.attr == "success":
                return test.operand.value.id, False
            if test.operand.attr == "failure":
                return test.operand.value.id, True
        return None

    def _add_guard_finding(self, node: ast.If, result_name: str) -> None:
        return_node = self._first_sentinel_return(node.body)
        if return_node is None:
            return
        func = self._enclosing_function(node)
        inner_type = self._result_inner_type(func) if func is not None else None
        replacement: tuple[int, int, str] | None = None
        fix_action = "manual"
        if inner_type is not None:
            lbl = result_name.removesuffix("_result").replace("_", " ").strip()
            failure_label = f"{lbl} failed" if lbl else "operation failed"
            indent = self._indent_of(return_node)
            start, end = self._line_offsets(return_node.lineno)
            replacement = (
                start,
                end,
                (
                    f"{indent}return r[{inner_type}].fail("
                    f"{result_name}.error or {failure_label!r})\n"
                ),
            )
            fix_action = "fix_silent_failure_sentinels"
        self._add_finding(
            line=return_node.lineno,
            column=return_node.col_offset,
            kind="silent-failure-guard",
            detail=(
                f"failure branch for '{result_name}' returns sentinel "
                "instead of propagating the error"
            ),
            fix_action=fix_action,
            replacement=replacement,
        )

    def _add_except_sentinel_finding(self, node: ast.ExceptHandler) -> None:
        return_node = self._first_sentinel_return(node.body)
        if return_node is None:
            return
        exception_name = node.name
        func = self._enclosing_function(node)
        inner_type = self._result_inner_type(func) if func is not None else None
        replacement: tuple[int, int, str] | None = None
        fix_action = "manual"
        if inner_type is not None and exception_name is not None:
            indent = self._indent_of(return_node)
            start, end = self._line_offsets(return_node.lineno)
            replacement = (
                start,
                end,
                (
                    f"{indent}return r[{inner_type}].fail("
                    f"str({exception_name}), exception={exception_name})\n"
                ),
            )
            fix_action = "fix_silent_failure_sentinels"
        self._add_finding(
            line=return_node.lineno,
            column=return_node.col_offset,
            kind="silent-failure-except",
            detail=(
                "exception branch returns sentinel instead of "
                "propagating the caught error"
            ),
            fix_action=fix_action,
            replacement=replacement,
        )

    def _first_sentinel_return(self, body: list[ast.stmt]) -> ast.Return | None:
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Return) and self._is_sentinel_value(
                    child.value
                ):
                    return child
        return None


def _resolve_call_name(node: ast.Call, import_aliases: Mapping[str, str]) -> str:
    """Resolve a call expression to a dotted name using alias context."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = import_aliases.get(func.value.id, func.value.id)
        return f"{base}.{func.attr}"
    if isinstance(func, ast.Name):
        return import_aliases.get(func.id, func.id)
    return ""


def _is_unwrap_or_call(node: ast.Call) -> bool:
    """Return True for ``<something>.unwrap_or(<sentinel>)``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "unwrap_or":
        return False
    if not node.args:
        return False
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and first_arg.value in {False, None}:
        return True
    if isinstance(first_arg, ast.List) and not first_arg.elts:
        return True
    return isinstance(first_arg, ast.Dict) and not first_arg.keys


def _expression_name(node: ast.expr | None, import_aliases: Mapping[str, str]) -> str:
    """Resolve a bare expression to a dotted name."""
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return import_aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        base = import_aliases.get(node.value.id, node.value.id)
        return f"{base}.{node.attr}"
    return ""


def collect_silent_failure_findings(
    tree: ast.Module, source: str
) -> list[_SilentFailureFinding]:
    """Collect all silent-failure findings from a rope-backed module AST."""
    return _SilentFailureAstVisitor(source).analyze(tree)


def collect_silent_failure_fixes(
    tree: ast.Module, source: str, *, kinds: set[str] | frozenset[str] | None = None
) -> list[tuple[int, int, str]]:
    """Return deterministic auto-fix replacements for silent-failure sentinels."""
    allowed = kinds if kinds is not None else frozenset()
    return [
        finding.replacement
        for finding in _SilentFailureAstVisitor(source).analyze(tree)
        if finding.replacement is not None and (not allowed or finding.kind in allowed)
    ]


__all__: list[str] = ["collect_silent_failure_findings", "collect_silent_failure_fixes"]
