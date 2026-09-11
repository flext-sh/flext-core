"""Pattern modernizer transformer — AST-based fixes for FLEXT code hygiene.

Applies safe, syntactic transformations that do not require type inference:

- ``u.Cli.print(...)`` → ``logger.info(...)`` and injects a module-level logger.
- ``breakpoint()`` / ``import pdb; pdb.set_trace()`` → removed.
- Bare ``except:`` → ``except Exception:``.
- ``open(path, mode)`` without ``encoding`` → ``open(path, mode, encoding="utf-8")``.

The transformer is intentionally conservative: it only rewrites code when the
result is unambiguous and records every change.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import u
from flext_infra.transformers._rewrite import (
    FlextInfraSourceRewrite,
    FlextInfraSourceRewriter,
)
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorPatternModernizer(FlextInfraRopeTransformer):
    """AST-driven transformer for common FLEXT anti-patterns."""

    _description = (
        "fix common FLEXT anti-patterns (print, pdb, bare except, open encoding)"
    )

    _LOGGER_NAME: ClassVar[str] = "logger"
    _LOGGER_LINE: ClassVar[str] = "logger = u.fetch_logger(__name__)\n"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply pattern modernizations to source text."""
        self.changes.clear()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)

        module_has_logger = self._has_logger_binding(tree)
        visitor = self._PatternVisitor(source, module_has_logger=module_has_logger)
        visitor.visit(tree)

        if not visitor.rewrites:
            return source, list(self.changes)

        updated = FlextInfraSourceRewriter.apply_rewrites(source, visitor.rewrites)

        if visitor.needs_logger and not module_has_logger:
            updated = self._ensure_u_import(updated)
            updated = self._inject_logger(updated)

        for change in visitor.changes:
            self._record_change(change)

        return updated, list(self.changes)

    @staticmethod
    def _has_logger_binding(tree: ast.AST) -> bool:
        """Return whether the module already defines a ``logger`` name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "logger":
                return True
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "logger":
                        return True
        return False

    @classmethod
    def _inject_logger(cls, source: str) -> str:
        """Inject ``logger = u.fetch_logger(__name__)`` after imports/docstring."""
        lines = source.splitlines(keepends=True)
        insert_idx = u.Infra.find_import_insert_position(lines)
        lines.insert(insert_idx, cls._LOGGER_LINE)
        return "".join(lines)

    class _PatternVisitor(FlextInfraSourceRewriter):
        """Collect rewrites for anti-patterns."""

        def __init__(self, source: str, *, module_has_logger: bool) -> None:
            super().__init__(source)
            self._module_has_logger = module_has_logger
            self.needs_logger = False

        @override
        def visit_Expr(self, node: ast.Expr) -> None:
            """Handle expression statements: print, breakpoint."""
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == "print":
                    self.needs_logger = True
                    call_text = self.node_text(value)
                    new_call = re.sub(r"\bprint\b", "logger.info", call_text, count=1)
                    self.append_rewrite(
                        node, new_call, "Replaced u.Cli.print() with logger.info()"
                    )
                    return
                if value.func.id == "breakpoint":
                    self.append_rewrite(node, "pass", "Replaced breakpoint() with pass")
                    return
            elif isinstance(value, ast.Call) and self._is_u_cli_print(value):
                self.needs_logger = True
                call_text = self.node_text(value)
                new_call = re.sub(
                    r"\bu\.Cli\.print\b", "logger.info", call_text, count=1
                )
                self.append_rewrite(
                    node, new_call, "Replaced u.Cli.print() with logger.info()"
                )
                return
            self.generic_visit(node)

        @staticmethod
        def _is_u_cli_print(value: ast.Call) -> bool:
            """Return True for ``u.Cli.print(...)`` calls."""
            func = value.func
            return (
                isinstance(func, ast.Attribute)
                and func.attr == "print"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "Cli"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "u"
            )

        @override
        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            """Fix bare ``except:`` to ``except Exception:``."""
            if node.type is None:
                lines = self._source.splitlines(keepends=True)
                lineno = getattr(node, "lineno", 1)
                line = lines[lineno - 1]
                stripped = line.lstrip()
                prefix_len = len(line) - len(stripped)
                if stripped.startswith("except:"):
                    indent = line[:prefix_len]
                    new_line = f"{indent}except Exception:\n"
                    start = self._offset(lineno, 0)
                    end = start + len(line)
                    self.rewrites.append(FlextInfraSourceRewrite(start, end, new_line))
                    self.changes.append("Fixed bare except: → except Exception:")
            self.generic_visit(node)

        @override
        def visit_Call(self, node: ast.Call) -> None:
            """Fix ``open()`` calls missing ``encoding``."""
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "open"
                and not any(kw.arg == "encoding" for kw in node.keywords)
            ):
                call_text = self.node_text(node)
                if call_text.endswith(")"):
                    new_call = call_text[:-1] + ', encoding="utf-8")'
                    self.append_rewrite(
                        node, new_call, 'Added encoding="utf-8" to open()'
                    )
            self.generic_visit(node)

        @override
        def visit_Import(self, node: ast.Import) -> None:
            """Remove ``import pdb`` statements."""
            if any(alias.name == "pdb" for alias in node.names):
                self.append_rewrite(node, "pass", "Removed import pdb")
            self.generic_visit(node)

        @override
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            """Remove ``from pdb import ...`` statements."""
            if node.module == "pdb":
                self.append_rewrite(node, "pass", "Removed from pdb import")
            self.generic_visit(node)


__all__: list[str] = ["FlextInfraRefactorPatternModernizer"]
