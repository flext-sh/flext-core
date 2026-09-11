"""Detect classes declared outside their canonical family locations via rope.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraClassPlacementDetector:
    """Detect misplaced class declarations across the five FLEXT families."""

    @staticmethod
    def detect_file(
        ctx: m.Infra.DetectorContext,
    ) -> t.SequenceOf[m.Infra.ClassPlacementViolation]:
        """Detect classes and class-level constants outside canonical families."""
        res = u.Infra.fetch_python_resource(
            ctx.rope_project, ctx.file_path, skip_protected=True, skip_settings=True
        )
        if res is None:
            return []
        file_path = ctx.file_path
        parts = file_path.parts
        violations: list[m.Infra.ClassPlacementViolation] = []

        governed_classes = (
            FlextInfraClassPlacementDetector._governed_classes_with_family(
                ctx.rope_project, res
            )
        )
        single_governed_class = len(governed_classes) == 1

        # 1. Misplaced governed classes → one_class_per_module action.
        for ci, family in governed_classes:
            if FlextInfraClassPlacementDetector._in_canonical_location(
                family, parts, file_path.name
            ):
                continue
            violations.append(
                FlextInfraClassPlacementDetector._violation_for_class(
                    ctx=ctx, ci=ci, family=family, fixable=single_governed_class
                )
            )

        # 2. Class-level constants outside _constants → classvar_relocation action.
        #    Covers explicit ClassVar annotations AND implicit UPPER_CASE
        #    constant-like assignments (no ClassVar annotation).
        if not FlextInfraClassPlacementDetector._in_canonical_location(
            "c", parts, file_path.name
        ):
            classvar_violations: list[m.Infra.ClassPlacementViolation] = []
            for ci in FlextInfraClassPlacementDetector._public_classes(
                ctx.rope_project, res
            ):
                classvar_violations.extend(
                    m.Infra.ClassPlacementViolation(
                        file=str(file_path),
                        line=constant.line,
                        name=constant.name,
                        base_class=ci.name,
                        suggestion=(
                            f"Move constant {constant.name} "
                            f"from {ci.name} to _constants"
                        ),
                        action="classvar_relocation",
                        fixable=True,
                        target_facade=FlextInfraClassPlacementDetector._target_facade(
                            ctx, "c"
                        ),
                        family="c",
                    )
                    for constant in FlextInfraClassPlacementDetector._class_constants(
                        ctx.rope_project, res, class_name=ci.name
                    )
                )
            violations.extend(classvar_violations)

        # 3. Misplaced type aliases → deep_namespace_refactor action.
        for alias_name, alias_line in FlextInfraClassPlacementDetector._type_aliases(
            ctx.rope_project, res
        ):
            if FlextInfraClassPlacementDetector._in_canonical_location(
                "t", parts, file_path.name
            ):
                continue
            violations.append(
                m.Infra.ClassPlacementViolation(
                    file=str(file_path),
                    line=alias_line,
                    name=alias_name,
                    base_class="TypeAlias",
                    suggestion=(
                        f"Move type alias {alias_name} to typings.py or _typings/"
                    ),
                    action="deep_namespace_refactor",
                    fixable=False,
                    target_facade=FlextInfraClassPlacementDetector._target_facade(
                        ctx, "t"
                    ),
                    family="t",
                )
            )

        return violations

    @staticmethod
    def _governed_classes_with_family(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> tuple[tuple[m.Infra.ClassInfo, str], ...]:
        """Return public governed classes with their family letters."""
        results: list[tuple[m.Infra.ClassInfo, str]] = []
        for ci in u.Infra.get_class_info(rope_project, resource):
            if ci.name.startswith("_"):
                continue
            family = FlextInfraClassPlacementDetector._family_for_class(ci)
            if family is None:
                continue
            results.append((ci, family))
        return tuple(results)

    @staticmethod
    def _violation_for_class(
        *,
        ctx: m.Infra.DetectorContext,
        ci: m.Infra.ClassInfo,
        family: str,
        fixable: bool,
    ) -> m.Infra.ClassPlacementViolation:
        """Build a ClassPlacementViolation for a misplaced class."""
        return m.Infra.ClassPlacementViolation(
            file=str(ctx.file_path),
            line=ci.line,
            name=ci.name,
            base_class=ci.bases[0] if ci.bases else "object",
            suggestion=FlextInfraClassPlacementDetector._suggestion_for_family(family),
            action="one_class_per_module",
            fixable=fixable,
            target_facade=FlextInfraClassPlacementDetector._target_facade(ctx, family),
            family=family,
        )

    @staticmethod
    def _family_for_class(ci: m.Infra.ClassInfo) -> str | None:
        """Return the canonical family letter for a class, or None if not governed."""
        terminal_bases = {
            base_name.rsplit(".", maxsplit=1)[-1] for base_name in ci.bases
        }
        if terminal_bases & c.Infra.PLACEMENT_PYDANTIC_BASE_NAMES:
            return "m"
        if terminal_bases & c.Infra.PLACEMENT_PROTOCOL_BASE_NAMES:
            return "p"
        if terminal_bases & c.Infra.PLACEMENT_ENUM_BASE_NAMES:
            return "c"
        if any(
            ci.name.endswith(suffix)
            for suffix in c.Infra.PLACEMENT_UTILITY_NAME_SUFFIXES
        ):
            return "u"
        return None

    @staticmethod
    def _in_canonical_location(
        family: str, parts: tuple[str, ...], file_name: str
    ) -> bool:
        """Return True when the file already lives in the canonical family area."""
        dir_sets = {
            "m": c.Infra.PLACEMENT_CANONICAL_MODEL_DIRS,
            "p": c.Infra.PLACEMENT_CANONICAL_PROTOCOL_DIRS,
            "c": c.Infra.PLACEMENT_CANONICAL_CONSTANTS_DIRS,
            "u": c.Infra.PLACEMENT_CANONICAL_UTILITY_DIRS,
            "t": c.Infra.PLACEMENT_CANONICAL_TYPING_DIRS,
        }
        file_sets = {
            "m": c.Infra.PLACEMENT_CANONICAL_MODEL_FILES,
            "p": c.Infra.PLACEMENT_CANONICAL_PROTOCOL_FILES,
            "c": c.Infra.PLACEMENT_CANONICAL_CONSTANTS_FILES,
            "u": c.Infra.PLACEMENT_CANONICAL_UTILITY_FILES,
            "t": c.Infra.PLACEMENT_CANONICAL_TYPING_FILES,
        }
        if file_name in file_sets.get(family, frozenset()):
            return True
        return bool(set(parts) & dir_sets.get(family, frozenset()))

    @staticmethod
    def _suggestion_for_family(family: str) -> str:
        """Return a human-readable relocation suggestion for a family."""
        return {
            "m": "Move Pydantic model class to models.py or _models/",
            "p": "Move Protocol class to protocols.py or _protocols/",
            "c": "Move Enum constant class to constants.py or _constants/",
            "u": "Move utility class to utilities.py or _utilities/",
            "t": "Move type alias to typings.py or _typings/",
        }.get(family, "Move class to canonical family location")

    @staticmethod
    def _target_facade(ctx: m.Infra.DetectorContext, family: str) -> str:
        """Return the canonical facade class name for a family and project."""
        stem = u.derive_class_stem(ctx.project_name) if ctx.project_name else ""
        suffix = c.Infra.FAMILY_SUFFIXES.get(family, "")
        return f"{stem}{suffix}" if stem and suffix else ""

    @staticmethod
    def _public_classes(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[m.Infra.ClassInfo]:
        """Return public top-level classes from the current Rope AST."""
        try:
            tree = FlextInfraUtilitiesRopeCore.get_pymodule(
                rope_project, resource
            ).get_ast()
        except u.Infra.rope_runtime_errors():
            return ()
        classes: list[m.Infra.ClassInfo] = []
        for node in getattr(tree, "body", ()) or ():
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "ClassDef":
                continue
            name = getattr(node, "name", "")
            if not isinstance(name, str) or not name or name.startswith("_"):
                continue
            line = getattr(node, "lineno", 1)
            classes.append(
                m.Infra.ClassInfo(
                    name=name,
                    line=line if isinstance(line, int) and line > 0 else 1,
                    bases=(),
                )
            )
        return tuple(classes)

    @staticmethod
    def _class_body_nodes(tree: object, *, class_name: str) -> t.SequenceOf[object]:
        """Return direct body nodes for the top-level class named ``class_name``."""
        module_body = getattr(tree, "body", None) or ()
        if not isinstance(module_body, (list, tuple)):
            return ()
        for node in module_body:
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "ClassDef":
                continue
            if getattr(node, "name", "") == class_name:
                class_body = getattr(node, "body", None) or ()
                return class_body if isinstance(class_body, (list, tuple)) else ()
        return ()

    @staticmethod
    def _class_constants(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        class_name: str,
    ) -> t.SequenceOf[m.Infra.ConstantInfo]:
        """Return class-level constants declared in ``class_name``'s body.

        Includes explicit ``ClassVar[...]`` annotations and implicit
        UPPER_CASE assignments whose value looks like a canonical constant.
        """
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        except u.Infra.rope_runtime_errors():
            return ()
        tree = pymodule.get_ast()
        body = FlextInfraClassPlacementDetector._class_body_nodes(
            tree, class_name=class_name
        )
        constants: list[m.Infra.ConstantInfo] = []
        for node in body:
            node_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
            if node_kind == "AnnAssign":
                constant = FlextInfraClassPlacementDetector._annassign_constant(node)
            elif node_kind == "Assign":
                constant = FlextInfraClassPlacementDetector._assign_constant(node)
            else:
                continue
            if constant is not None:
                constants.append(constant)
        return tuple(constants)

    @staticmethod
    def _annassign_constant(node: object) -> m.Infra.ConstantInfo | None:
        """Return ConstantInfo for an AnnAssign node, or None if not a violation."""
        target_name = FlextInfraUtilitiesRopeAnalysis.name_of(
            getattr(node, "target", None)
        )
        if not target_name or target_name.startswith("_"):
            return None
        if target_name in c.Infra.CLASSVAR_EXEMPT_NAMES:
            return None
        if not c.Infra.NAMESPACE_CONSTANT_PATTERN.match(target_name):
            return None
        annotation = getattr(node, "annotation", None)
        has_classvar = FlextInfraClassPlacementDetector._annotation_contains(
            annotation, "ClassVar"
        )
        value = getattr(node, "value", None)
        if (
            not has_classvar
            and not FlextInfraClassPlacementDetector._classvar_value_permitted(value)
        ):
            return None
        line = getattr(node, "lineno", 1)
        return m.Infra.ConstantInfo(
            name=target_name, line=line if isinstance(line, int) and line > 0 else 1
        )

    @staticmethod
    def _assign_constant(node: object) -> m.Infra.ConstantInfo | None:
        """Return ConstantInfo for an implicit Assign node, or None if not a violation."""
        targets = getattr(node, "targets", None)
        if not isinstance(targets, (list, tuple)) or len(targets) != 1:
            return None
        target_name = FlextInfraUtilitiesRopeAnalysis.name_of(targets[0])
        if not target_name or target_name.startswith("_"):
            return None
        if target_name in c.Infra.CLASSVAR_EXEMPT_NAMES:
            return None
        if not c.Infra.NAMESPACE_CONSTANT_PATTERN.match(target_name):
            return None
        value = getattr(node, "value", None)
        if not FlextInfraClassPlacementDetector._classvar_value_permitted(value):
            return None
        line = getattr(node, "lineno", 1)
        return m.Infra.ConstantInfo(
            name=target_name, line=line if isinstance(line, int) and line > 0 else 1
        )

    @staticmethod
    def _type_aliases(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[tuple[str, int]]:
        """Return module-level type aliases as (name, line) pairs."""
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        except u.Infra.rope_runtime_errors():
            return ()
        tree = pymodule.get_ast()
        aliases: list[tuple[str, int]] = []
        for node in getattr(tree, "body", []) or []:
            kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
            if kind == "TypeAlias":
                name = getattr(node, "name", None)
                name_str = getattr(name, "id", str(name)) if name else ""
                line = getattr(node, "lineno", 1)
                if name_str:
                    aliases.append((name_str, line))
                continue
            if kind == "AnnAssign":
                annotation = getattr(node, "annotation", None)
                if not FlextInfraClassPlacementDetector._annotation_contains(
                    annotation, "TypeAlias"
                ):
                    continue
                target_name = FlextInfraUtilitiesRopeAnalysis.name_of(
                    getattr(node, "target", None)
                )
                line = getattr(node, "lineno", 1)
                if target_name:
                    aliases.append((target_name, line))
        return tuple(aliases)

    @staticmethod
    def _annotation_contains(annotation: object | None, name: str) -> bool:
        """Return True when ``name`` appears in any sub-node identifier."""
        if annotation is None:
            return False
        for sub in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(annotation):
            if FlextInfraUtilitiesRopeAnalysis.name_of(sub) == name:
                return True
        return False

    @staticmethod
    def _classvar_value_permitted(value: object | None) -> bool:
        """Return True when a ClassVar default is a literal/canonical constant."""
        if value is None:
            return True
        kind = FlextInfraUtilitiesRopeAnalysis.node_kind(value)
        if kind in {"Constant", "Name", "Attribute", "Tuple", "List", "Set", "Dict"}:
            return True
        if kind == "Call":
            func = getattr(value, "func", None)
            func_name = FlextInfraUtilitiesRopeAnalysis.name_of(func)
            if func_name in c.Infra.CLASSVAR_ALLOWED_CALLS:
                return True
            if FlextInfraUtilitiesRopeAnalysis.node_kind(func) == "Attribute":
                base = getattr(func, "value", None)
                base_name = getattr(base, "id", "")
                return base_name in c.Infra.CLASSVAR_ALLOWED_CALLS
        return False


__all__: list[str] = ["FlextInfraClassPlacementDetector"]
