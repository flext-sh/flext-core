"""Local rope patch adding Python 3.13 AST awareness to Rope.

Rope 1.14 and its master branch do not implement handlers for part of the
Python 3.13 parser surface used in this workspace. In practice that currently
includes PEP 695 aliases/type parameters plus several structural pattern
matching nodes. Modules that use this syntax trigger
``MismatchedTokenError`` or noisy ``Unknown node type`` warnings during rope
analysis and abort census/codegen runs.

This module monkey-patches ``rope.refactor.patchedast._PatchingASTWalker``
once at import time to add the missing handlers. The patch is idempotent:
subsequent imports are a no-op. Semantics mirror what rope would do if it
knew about these nodes:

- ``type Name[T] = Value`` → ``["type", name, "[", type_params, "]", "=", value]``
- ``def f[T](...)`` / ``class C[T](...)`` → ``type_params`` list is rendered
  inside ``[...]`` between the name and the opening parenthesis.
- ``TypeVar``, ``ParamSpec``, ``TypeVarTuple`` emit their name (with the
    ``**`` / ``*`` prefix for ParamSpec / TypeVarTuple) plus an optional
    ``bound``.
- Pattern-matching nodes mirror the source token stream closely enough for
    Rope's patched AST region walker to keep working without upstream support.

NOTE (multi-agent, flext-f8vk / kimi): this module lives in the ``_rope``
subpackage so the root ``pyproject.toml`` can scope
``reportPrivateUsage = "none"`` to this intentional monkeypatch surface —
rope 1.14 exposes no public API to register patched-AST handlers, and the
FLEXT typing law forbids the getattr-dispatch/Any workaround.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesRopePep695Patch:
    """Idempotent monkey-patch applying PEP 695 support to rope's AST walker.

    The structural contract for rope's internal ``_PatchingASTWalker`` lives
    in ``p.Infra.PatchingASTWalker`` (canonical protocol namespace).
    """

    _applied: ClassVar[bool] = False

    @classmethod
    def apply(cls) -> None:
        """Install PEP 695 handlers on rope's ``_PatchingASTWalker`` once."""
        if cls._applied:
            return
        # Vendor boundary: rope exposes NO public API to register a walker
        # handler, so these private slots are the library's real interface.
        # The access is written LITERALLY, never routed through getattr with a
        # name constant: hiding it from the analyzer would be a disguised
        # suppression. pep695_ast_walker narrows the walker to the handler
        # contract and raises when rope changes its internals, so the typing is
        # resolved and only the ruff rule remains exempted for this directory
        # (operator authorization 2026-08-08).
        walker = FlextInfraUtilitiesRopeRuntime.pep695_ast_walker()
        original_function_def: Callable[..., None] = walker._handle_function_def_node
        original_class_def: Callable[..., None] = walker._ClassDef

        def _type_params_children(
            node: p.Infra.PatchingASTWalker.TypeParameterOwner,
        ) -> list[p.AttributeProbe]:
            """Type params children."""
            type_params = getattr(node, "type_params", None) or ()
            if not type_params:
                return []
            children: list[p.AttributeProbe] = ["["]
            for index, tp in enumerate(type_params):
                if index > 0:
                    children.append(",")
                children.append(tp)
            children.append("]")
            return children

        def _pattern_opening_token(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.PositionedNode,
        ) -> str:
            """Pattern opening token."""
            lineno = getattr(node, "lineno", None)
            col_offset = getattr(node, "col_offset", None)
            if not isinstance(lineno, int) or not isinstance(col_offset, int):
                return ""
            line_start = self.lines.get_line_start(lineno)
            source_text: str = self.source.source
            start = line_start + col_offset
            source_fragment: str = source_text[start : start + 1]
            return source_fragment

        def _patched_function_def(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.FunctionDefinitionNode,
            *,
            is_async: bool,
        ) -> None:
            """Patched function def."""
            if not getattr(node, "type_params", None):
                original_function_def(self, node, is_async=is_async)
                return
            children: list[p.AttributeProbe] = []
            for decorator in node.decorator_list:
                children.extend(("@", decorator))
            children.extend(["async", "def"] if is_async else ["def"])
            children.append(node.name)
            children.extend(_type_params_children(node))
            children.extend(["(", node.args, ")"])
            children.append(":")
            children.extend(node.body)
            self._handle(node, children)

        def _patched_class_def(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.ClassDefinitionNode,
        ) -> None:
            """Patched class def."""
            if not getattr(node, "type_params", None):
                original_class_def(self, node)
                return
            children: list[p.AttributeProbe] = []
            for decorator in node.decorator_list:
                children.extend(("@", decorator))
            children.extend(["class", node.name])
            children.extend(_type_params_children(node))
            if node.bases:
                children.append("(")
                children.extend(self._child_nodes(node.bases, ","))
                children.append(")")
            children.append(":")
            children.extend(node.body)
            self._handle(node, children)

        def _type_alias(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.TypeAliasNode,
        ) -> None:
            """Type alias."""
            children: list[p.AttributeProbe] = ["type", node.name]
            children.extend(_type_params_children(node))
            children.extend(["=", node.value])
            self._handle(node, children)

        def _type_var(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.TypeVariableNode,
        ) -> None:
            """Type var."""
            children: list[p.AttributeProbe] = [node.name]
            if getattr(node, "bound", None) is not None:
                children.extend([":", node.bound])
            self._handle(node, children)

        def _param_spec(
            self: p.Infra.PatchingASTWalker, node: p.Infra.PatchingASTWalker.NamedNode
        ) -> None:
            """Param spec."""
            self._handle(node, ["**", node.name])

        def _type_var_tuple(
            self: p.Infra.PatchingASTWalker, node: p.Infra.PatchingASTWalker.NamedNode
        ) -> None:
            """Type var tuple."""
            self._handle(node, ["*", node.name])

        def _match_sequence(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.MatchSequenceNode,
        ) -> None:
            """Match sequence."""
            children = self._child_nodes(node.patterns, ",")
            opening = _pattern_opening_token(self, node)
            if opening == "[":
                self._handle(node, ["[", *children, "]"])
                return
            if opening == "(" and not node.patterns:
                self._handle(node, [self.empty_tuple])
                return
            self._handle(node, children, eat_parens=opening == "(")

        def _match_singleton(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.MatchSingletonNode,
        ) -> None:
            """Match singleton."""
            self._handle(node, [str(node.value)])

        def _match_star(
            self: p.Infra.PatchingASTWalker,
            node: p.Infra.PatchingASTWalker.MatchStarNode,
        ) -> None:
            """Match star."""
            self._handle(node, ["*", node.name or "_"])

        def _match_or(
            self: p.Infra.PatchingASTWalker, node: p.Infra.PatchingASTWalker.MatchOrNode
        ) -> None:
            """Match or."""
            self._handle(node, self._child_nodes(node.patterns, "|"))

        walker._handle_function_def_node = _patched_function_def
        walker._ClassDef = _patched_class_def
        walker._TypeAlias = _type_alias
        walker._TypeVar = _type_var
        walker._ParamSpec = _param_spec
        walker._TypeVarTuple = _type_var_tuple
        walker._MatchSequence = _match_sequence
        walker._MatchSingleton = _match_singleton
        walker._MatchStar = _match_star
        walker._MatchOr = _match_or
        cls._applied = True


FlextInfraUtilitiesRopePep695Patch.apply()


__all__: list[str] = ["FlextInfraUtilitiesRopePep695Patch"]
