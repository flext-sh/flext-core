"""AST analysis for deferred self-reference and recursive model defects.

Two structural defects share one root cause: a nested model that cannot name
what it depends on at class-body execution time.

1. DEFERRED_SELF_REFERENCE — ``default_factory=lambda: Outer.Sibling()``.
   A ``default_factory`` runs while the class body executes, and the outer
   class name is still unbound there, so the author wraps the reference in a
   lambda to postpone resolution. That converts a definition-order defect
   into a runtime one and hides it from every static gate.

2. RECURSIVE_MODEL — a model whose annotation or default reaches itself,
   directly or through the outer namespace. Instantiation then depends on a
   type that is not finished being defined.

The canonical repair for both is the workspace's diamond-FLEXT composition:
hoist the referenced model into its own namespace class, inherit that
namespace, and reference the model as a resolved base-class attribute. Every
default then stays a direct callable.
"""

from __future__ import annotations

import ast
from typing import NamedTuple


class DeferredSelfReferenceFinding(NamedTuple):
    """One deferred-self-reference or recursive-model occurrence."""

    line: int
    column: int
    kind: str
    detail: str


_DEFERRED_KIND = "DEFERRED_SELF_REFERENCE"
_RECURSIVE_KIND = "RECURSIVE_MODEL"
_FACTORY_KEYWORD = "default_factory"
# Annotations whose subscript is a slot or a type table, never an instantiated
# model field: naming the owner inside them is self-typing, not recursion.
_NON_FIELD_WRAPPERS = frozenset({"ClassVar", "TypeAdapter"})


def _enclosing_class_names(stack: list[ast.ClassDef]) -> frozenset[str]:
    """Return every class name currently open in the definition stack."""
    return frozenset(node.name for node in stack)


def _attribute_root(node: ast.expr) -> str | None:
    """Return the leftmost identifier of an attribute chain, if any."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _referenced_roots(node: ast.expr) -> frozenset[str]:
    """Return every identifier the expression resolves against."""
    roots: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            root = _attribute_root(child)
            if root is not None:
                roots.add(root)
        elif isinstance(child, ast.Name):
            roots.add(child.id)
    return frozenset(roots)


def _annotation_wrappers(node: ast.expr) -> frozenset[str]:
    """Return the trailing names of every generic wrapper in an annotation."""
    wrappers: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        value = child.value
        if isinstance(value, ast.Attribute):
            wrappers.add(value.attr)
        elif isinstance(value, ast.Name):
            wrappers.add(value.id)
    return frozenset(wrappers)


def _factory_keyword(call: ast.Call) -> ast.keyword | None:
    """Return the ``default_factory`` keyword of a call, if present."""
    for keyword in call.keywords:
        if keyword.arg == _FACTORY_KEYWORD:
            return keyword
    return None


def collect_deferred_self_reference_findings(
    tree: ast.Module,
) -> tuple[DeferredSelfReferenceFinding, ...]:
    """Collect deferred-self-reference and recursive-model findings in a module."""
    findings: list[DeferredSelfReferenceFinding] = []
    stack: list[ast.ClassDef] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            stack.append(node)
            for child in node.body:
                visit(child)
            stack.pop()
            return

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # A function body is executed on call, long after the class is
            # bound, so neither defect can occur there: an annotated local is
            # not a model field and a default already resolved.
            return

        enclosing = _enclosing_class_names(stack)
        if enclosing and isinstance(node, ast.Call):
            keyword = _factory_keyword(node)
            if keyword is not None and isinstance(keyword.value, ast.Lambda):
                hit = enclosing & _referenced_roots(keyword.value.body)
                if hit:
                    name = min(hit)
                    findings.append(
                        DeferredSelfReferenceFinding(
                            line=keyword.value.lineno,
                            column=keyword.value.col_offset,
                            kind=_DEFERRED_KIND,
                            detail=(
                                f"default_factory defers resolution of {name!r} "
                                "through a lambda because the enclosing class is "
                                "still unbound. Hoist the referenced model into "
                                "its own namespace class, inherit it (diamond "
                                "FLEXT), and pass the model itself as the factory."
                            ),
                        )
                    )

        if stack and isinstance(node, ast.AnnAssign):
            owner = stack[-1].name
            annotation = node.annotation
            # ClassVar slots (singleton holders) and TypeAdapter tables are
            # never instantiated as model fields, so naming the owner there is
            # a normal self-typed reference, not a recursive model.
            annotation_roots = _referenced_roots(annotation)
            wrappers = _annotation_wrappers(annotation)
            is_field_annotation = not wrappers & _NON_FIELD_WRAPPERS
            if owner in annotation_roots and is_field_annotation:
                findings.append(
                    DeferredSelfReferenceFinding(
                        line=annotation.lineno,
                        column=annotation.col_offset,
                        kind=_RECURSIVE_KIND,
                        detail=(
                            f"model {owner!r} annotates a field with itself. A "
                            "recursive model cannot be instantiated while its own "
                            "definition is incomplete; split the recursive leg "
                            "into a separate namespace class composed by FLEXT."
                        ),
                    )
                )

        for child in ast.iter_child_nodes(node):
            visit(child)

    for child in tree.body:
        visit(child)
    return tuple(findings)


__all__: list[str] = [
    "DeferredSelfReferenceFinding",
    "collect_deferred_self_reference_findings",
]
