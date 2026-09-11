"""Rope-backed class and module introspection helpers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, ClassVar

from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from flext_infra.protocols import p
    from flext_infra.typings import t


class FlextInfraUtilitiesRopeAnalysisIntrospection:
    """Rope-backed class and module introspection helpers.

    Extracted mixin providing: get_class_nested_classes,
    get_module_symbols, extract_public_methods_from_dir.
    """

    _METHOD_KIND_LABELS: ClassVar[t.StrMapping] = {
        "staticmethod": "static",
        "classmethod": "class",
    }

    @staticmethod
    def get_class_nested_classes(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        class_name: str,
    ) -> t.StrSequence:
        """Return names of nested classes within a given class."""
        try:
            return FlextInfraUtilitiesRopeAnalysisIntrospection._nested_class_names(
                rope_project, resource, class_name
            )
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return ()

    @staticmethod
    def _nested_class_names(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        class_name: str,
    ) -> t.StrSequence:
        """Return nested class names from a resolved Rope class object."""
        result: list[str] = []
        pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        attributes = pymodule.get_attributes()
        if class_name not in attributes:
            return result
        obj = attributes[class_name].get_object()
        if not FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj):
            return result
        for name, pyname in obj.get_attributes().items():
            child = pyname.get_object()
            if FlextInfraUtilitiesRopeRuntime.is_abstract_class(child):
                result.append(name)
        return result

    @staticmethod
    def get_module_symbols(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[m.Infra.SymbolInfo]:
        """Return top-level symbols defined in one module through Rope metadata."""
        result: t.MutableSequenceOf[m.Infra.SymbolInfo] = []
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
            tree: p.AttributeProbe = pymodule.get_ast()
            body: p.AttributeProbe = getattr(tree, "body", ())
            if not isinstance(body, (list, tuple)):
                return result
            for node in body:
                result.extend(
                    FlextInfraUtilitiesRopeAnalysisIntrospection._module_symbols_from_node(
                        node
                    )
                )
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return result
        return sorted(result, key=operator.attrgetter("line"))

    @staticmethod
    def _module_symbols_from_node(
        node: p.AttributeProbe,
    ) -> t.SequenceOf[m.Infra.SymbolInfo]:
        """Return top-level symbol entries represented by one Rope AST node."""
        node_kind = node.__class__.__name__
        line = FlextInfraUtilitiesRopeAnalysisIntrospection._ast_line(node)
        if node_kind == "ClassDef":
            name = FlextInfraUtilitiesRopeAnalysisIntrospection._ast_named_value(node)
            return (
                (m.Infra.SymbolInfo(name=name, kind="class", line=line),)
                if name
                else ()
            )
        if node_kind in {"FunctionDef", "AsyncFunctionDef"}:
            name = FlextInfraUtilitiesRopeAnalysisIntrospection._ast_named_value(node)
            return (
                (m.Infra.SymbolInfo(name=name, kind="function", line=line),)
                if name
                else ()
            )
        return tuple(
            m.Infra.SymbolInfo(name=name, kind="assignment", line=line)
            for name in FlextInfraUtilitiesRopeAnalysisIntrospection._assignment_names(
                node, node_kind
            )
        )

    @staticmethod
    def _assignment_names(node: p.AttributeProbe, node_kind: str) -> t.StrSequence:
        """Return assignment-like target names from one top-level AST node."""
        if node_kind == "Assign":
            raw_targets: p.AttributeProbe = getattr(node, "targets", ())
            if not isinstance(raw_targets, (list, tuple)):
                return ()
            return tuple(
                name
                for target in raw_targets
                if (
                    name
                    := FlextInfraUtilitiesRopeAnalysisIntrospection._ast_named_value(
                        target
                    )
                )
            )
        if node_kind == "AnnAssign":
            target: p.AttributeProbe = getattr(node, "target", None)
            name = FlextInfraUtilitiesRopeAnalysisIntrospection._ast_named_value(target)
            return (name,) if name else ()
        if node_kind == "TypeAlias":
            name = FlextInfraUtilitiesRopeAnalysisIntrospection._ast_named_value(node)
            return (name,) if name else ()
        return ()

    @staticmethod
    def _ast_named_value(node: p.AttributeProbe | None) -> str:
        """Return ``name``/``id`` carried by a Rope AST node."""
        if node is None:
            return ""
        direct: p.AttributeProbe = getattr(node, "name", "")
        if isinstance(direct, str) and direct:
            return direct
        identifier: p.AttributeProbe = getattr(node, "id", "")
        if isinstance(identifier, str) and identifier:
            return identifier
        nested: p.AttributeProbe = getattr(direct, "id", "")
        if isinstance(nested, str):
            return nested
        return ""

    @staticmethod
    def _ast_line(node: p.AttributeProbe) -> int:
        """Return a stable one-based source line for a Rope AST node."""
        line: p.AttributeProbe = getattr(node, "lineno", 1)
        return line if isinstance(line, int) and line > 0 else 1

    @classmethod
    def extract_public_methods_from_dir(
        cls: type[p.Infra.RopeAnalysisMethods], package_dir: Path
    ) -> t.MappingKV[str, t.SequenceOf[t.Triple[str, str, str]]]:
        """Extract public methods from all Python files in a package directory."""
        result: MutableMapping[str, t.MutableSequenceOf[t.Triple[str, str, str]]] = {}
        project_root = FlextInfraUtilitiesDiscovery.project_root(package_dir / "foo.py")
        if project_root is None:
            return result
        with FlextInfraUtilitiesRopeCore.open_project(project_root.parent) as rope_proj:
            for py_file in sorted(package_dir.glob(c.Infra.EXT_PYTHON_GLOB)):
                if py_file.name == c.Infra.INIT_PY:
                    continue
                resource = cls.get_resource_from_path(rope_proj, py_file)
                if resource is None:
                    continue
                classes = cls.get_module_classes(rope_proj, resource)
                for class_name in classes:
                    class_methods = cls.get_class_methods(
                        rope_proj, resource, class_name, include_private=False
                    )
                    methods = result.setdefault(class_name, [])
                    for method_name, method_kind in class_methods.items():
                        methods.append((
                            method_name,
                            FlextInfraUtilitiesRopeAnalysisIntrospection._METHOD_KIND_LABELS.get(
                                method_kind, "instance"
                            ),
                            py_file.name,
                        ))
        return result


__all__: list[str] = ["FlextInfraUtilitiesRopeAnalysisIntrospection"]
