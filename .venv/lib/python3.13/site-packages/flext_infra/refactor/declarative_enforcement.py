"""Declarative enforcement engine driven by the flext-core catalog.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra.detectors.class_placement_detector import (
    FlextInfraClassPlacementDetector,
)
from flext_infra.detectors.compatibility_alias_detector import (
    FlextInfraCompatibilityAliasDetector,
)
from flext_infra.detectors.loose_test_function_detector import (
    FlextInfraLooseTestFunctionDetector,
)

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import m, p, t


class FlextInfraRefactorDeclarativeEnforcement:
    """Generic rope/beartype detector for catalog rules without a bespoke scanner.

    The engine maps an ``EnforcementRuleSpec`` to a detection strategy using
    only ``source.kind`` / ``source.violation_field`` / ``source.predicate_kind``.
    It keeps every rule-specific implementation small and delegates structural
    work to rope (``PyModule.get_ast``, ``get_attributes``) and beartype-style
    runtime contracts.
    """

    _MAGIC_NUMBER_TYPES: ClassVar[frozenset[type[int | float]]] = frozenset({
        int,
        float,
    })
    _MAGIC_STRING_TYPES: ClassVar[frozenset[type[str]]] = frozenset({str})
    _INFRA_VIOLATION_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "magic_literal_violations",
        "stub_file_violations",
        "foreign_canonical_alias_violations",
        "loose_test_function_violations",
    })
    _BEARTYPE_PREDICATES: ClassVar[frozenset[str]] = frozenset({"classvar_constant"})

    @classmethod
    def supports(cls, rule: me.EnforcementRuleSpec) -> bool:
        """Return whether this engine can evaluate ``rule`` from source metadata."""
        source = rule.source
        if source.kind == "flext_infra_detector":
            return source.violation_field in cls._INFRA_VIOLATION_FIELDS
        if source.kind == "beartype":
            predicate_kind = source.predicate_kind
            return (
                str(getattr(predicate_kind, "value", predicate_kind))
                in cls._BEARTYPE_PREDICATES
            )
        return False

    @classmethod
    def detect(
        cls, rule: me.EnforcementRuleSpec, ctx: m.Infra.DetectorContext
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Return probes for violations of ``rule`` inside ``ctx.file_path``."""
        rule_id = cls._rule_id_short(rule.id)
        source = rule.source
        if source.kind == "flext_infra_detector":
            violation_field = getattr(source, "violation_field", "")
            if violation_field == "stub_file_violations":
                return cls._detect_stub_files(ctx, rule_id=rule_id)
            if violation_field == "magic_literal_violations":
                return cls._detect_magic_literals(ctx, rule_id=rule_id)
            if violation_field == "foreign_canonical_alias_violations":
                return cls._detect_foreign_canonical_aliases(ctx, rule_id=rule_id)
            if violation_field == "loose_test_function_violations":
                return cls._detect_loose_test_functions(ctx, rule_id=rule_id)
        elif source.kind == "beartype":
            predicate_kind = getattr(source, "predicate_kind", None)
            predicate_value = getattr(predicate_kind, "value", predicate_kind)
            if predicate_value == "classvar_constant":
                return cls._detect_classvar_constants(ctx, rule_id=rule_id)
        violation_field = getattr(source, "violation_field", "")
        predicate_kind = getattr(source, "predicate_kind", "")
        msg = (
            f"unsupported declarative enforcement source for {rule.id}: "
            f"kind={source.kind!r} violation_field={violation_field!r} "
            f"predicate_kind={predicate_kind!r}"
        )
        raise ValueError(msg)

    @classmethod
    def _detect_stub_files(
        cls, ctx: m.Infra.DetectorContext, *, rule_id: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Return a probe for ``ctx.file_path`` when it is a prohibited ``.pyi``."""
        file_path = ctx.file_path
        if file_path.suffix != ".pyi":
            return ()
        return (cls._probe(file_path, line=1, rule_id=rule_id),)

    @classmethod
    def _detect_magic_literals(
        cls, ctx: m.Infra.DetectorContext, *, rule_id: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Return probes for magic numbers/strings in executable code."""
        res = FlextInfraUtilitiesRopeCore.get_resource_from_path(
            ctx.rope_project, ctx.file_path
        )
        if res is None:
            msg = (
                f"declarative enforcement {ctx.file_path} failed: "
                "unable to resolve rope resource"
            )
            raise RuntimeError(msg)
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(ctx.rope_project, res)
            tree = pymodule.get_ast()
        except u.Infra.rope_runtime_errors() as exc:
            msg = (
                f"declarative enforcement {ctx.file_path} failed: "
                f"unable to parse rope AST: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        probes: list[p.AttributeProbe] = []
        parent_map = cls._rope_parent_map(tree)
        for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(tree):
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "Constant":
                continue
            value = getattr(node, "value", None)
            if not cls._is_magic_literal(value):
                continue
            line = getattr(node, "lineno", 1)
            if not isinstance(line, int) or line <= 0:
                line = 1
            if cls._is_exempt_literal_position(node, parent_map):
                continue
            probes.append(
                cls._probe(
                    ctx.file_path, line=line, rule_id=rule_id, literal=repr(value)
                )
            )
        return tuple(probes)

    @classmethod
    def _detect_classvar_constants(
        cls, ctx: m.Infra.DetectorContext, *, rule_id: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Delegate ClassVar-outside-_constants detection to the canonical scanner."""
        try:
            violations = FlextInfraClassPlacementDetector.detect_file(ctx)
        except c.EXC_BROAD_RUNTIME as exc:
            msg = (
                f"declarative enforcement {ctx.file_path} failed: "
                f"class placement detector failed: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        return tuple(
            cls._probe(
                Path(v.file),
                line=v.line,
                rule_id=rule_id,
                object_name=v.name,
                base_class=v.base_class,
            )
            for v in violations
            if v.action == "classvar_relocation"
        )

    @classmethod
    def _detect_loose_test_functions(
        cls, ctx: m.Infra.DetectorContext, *, rule_id: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Delegate loose-test-function detection to the canonical scanner."""
        try:
            violations = FlextInfraLooseTestFunctionDetector.detect_file(ctx)
        except c.EXC_BROAD_RUNTIME as exc:
            msg = (
                f"declarative enforcement {ctx.file_path} failed: "
                f"loose test function detector failed: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        return tuple(
            cls._probe(Path(v.file), line=v.line, rule_id=rule_id, object_name=v.name)
            for v in violations
        )

    @classmethod
    def _detect_foreign_canonical_aliases(
        cls, ctx: m.Infra.DetectorContext, *, rule_id: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Delegate foreign-canonical-alias detection to the canonical scanner."""
        try:
            violations = FlextInfraCompatibilityAliasDetector.detect_file(ctx)
        except c.EXC_BROAD_RUNTIME as exc:
            msg = (
                f"declarative enforcement {ctx.file_path} failed: "
                f"compatibility alias detector failed: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        probes: list[p.AttributeProbe] = []
        for violation in violations:
            action = FlextInfraCompatibilityAliasDetector.fix_action_for(
                violation, current_project=ctx.project_name
            )
            if action != "rewrite_foreign_canonical_alias":
                continue
            probes.append(
                cls._probe(
                    ctx.file_path,
                    line=violation.line,
                    rule_id=rule_id,
                    object_name=violation.alias_name,
                    target_name=violation.target_name,
                    module_name=violation.module_name,
                )
            )
        return tuple(probes)

    @staticmethod
    def _rule_id_short(rule_id: str) -> str:
        """Return the numeric suffix of an ENFORCE-NNN identifier."""
        return rule_id.rsplit("-", maxsplit=1)[-1]

    @classmethod
    def _is_magic_literal(cls, value: t.Primitives | None) -> bool:
        """Return True for primitive literals that should be named constants."""
        value_type = type(value)
        return (
            value_type in cls._MAGIC_NUMBER_TYPES
            or value_type in cls._MAGIC_STRING_TYPES
        )

    @classmethod
    def _is_exempt_literal_position(
        cls, node: p.AttributeProbe, parent_map: dict[int, p.AttributeProbe]
    ) -> bool:
        """Return True when a Constant node lives in an exempt syntactic position."""
        parent = parent_map.get(id(node))
        if parent is None:
            return False
        parent_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(parent)
        return parent_kind in {"arguments", "arg", "keyword", "AnnAssign"} or (
            parent_kind in {"Assign", "AnnAssign"}
            and cls._is_module_level(parent, parent_map)
        )

    @classmethod
    def _is_module_level(
        cls, node: p.AttributeProbe, parent_map: dict[int, p.AttributeProbe]
    ) -> bool:
        """Return True when ``node`` is a direct child of the module body."""
        current = node
        while True:
            parent = parent_map.get(id(current))
            if parent is None:
                return False
            parent_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(parent)
            if parent_kind in {"ClassDef", "FunctionDef", "AsyncFunctionDef"}:
                return False
            if parent_kind == "Module":
                return True
            current = parent

    @staticmethod
    def _probe(
        file_path: Path, *, line: int, rule_id: str, **kwargs: t.JsonValue
    ) -> p.AttributeProbe:
        """Build a probe consumable by fixer adapters."""
        return SimpleNamespace(
            file_path=str(file_path), line=line, rule_id=rule_id, **kwargs
        )

    @staticmethod
    def _rope_parent_map(root: p.AttributeProbe) -> dict[int, p.AttributeProbe]:
        """Build a child-id -> parent map for the full rope AST."""
        parent_map: dict[int, p.AttributeProbe] = {}
        stack: list[p.AttributeProbe] = [root]
        while stack:
            parent = stack.pop()
            for field_name in getattr(parent, "_fields", ()):
                value = getattr(parent, field_name, None)
                if isinstance(value, list):
                    for child in value:
                        if hasattr(child, "_fields"):
                            parent_map[id(child)] = parent
                            stack.append(child)
                elif hasattr(value, "_fields"):
                    parent_map[id(value)] = parent
                    stack.append(value)
        return parent_map
