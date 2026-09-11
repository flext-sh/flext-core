"""Namespace validation rules (rules 0-3) for flext projects.

Rule checks operate on rope-provided AST roots without ever importing
``ast``. Type discrimination uses ``FlextInfraUtilitiesRopeAnalysis.node_kind``
(``type(node).__name__``) and field access uses plain attribute lookups,
so rope's ``pymodule.get_ast()`` output is consumed transparently.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_infra import c, t
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FlextInfraNamespaceRules:
    """Implementation of namespace rules 0-3 via rope's AST attribute walks."""

    _DIRECT_FACADE_SUFFIX_RULES = (
        ("Constants", c.Infra.CONSTANTS_PY, c.Infra.FAMILY_DIRECTORIES["c"]),
        ("Models", c.Infra.MODELS_PY, c.Infra.FAMILY_DIRECTORIES["m"]),
        ("Protocols", c.Infra.PROTOCOLS_PY, c.Infra.FAMILY_DIRECTORIES["p"]),
        ("Types", c.Infra.TYPINGS_PY, c.Infra.FAMILY_DIRECTORIES["t"]),
        ("Utilities", c.Infra.UTILITIES_PY, c.Infra.FAMILY_DIRECTORIES["u"]),
    )
    _FACADE_DAG: t.MappingKV[str, int] = MappingProxyType({
        "c": 0,
        "t": 1,
        "p": 2,
        "m": 3,
        "u": 4,
    })

    @staticmethod
    def _is_private_dir_file(filepath: Path) -> bool:
        """Return whether the file lives inside a private (``_xxx``) directory."""
        return any(
            part.startswith("_") and not part.startswith("__")
            for part in filepath.parts
            if "." not in part
        )

    @staticmethod
    def _kind(node: object) -> str:
        """Shortcut for ``FlextInfraUtilitiesRopeAnalysis.node_kind``."""
        return FlextInfraUtilitiesRopeAnalysis.node_kind(node)

    @staticmethod
    def _is_type_checking_guard(node: object) -> bool:
        """Return True if ``node`` is an ``if TYPE_CHECKING:`` block."""
        if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "If":
            return False
        test = getattr(node, "test", None)
        return (
            FlextInfraUtilitiesRopeAnalysis.node_kind(test) == "Name"
            and getattr(test, "id", "") == "TYPE_CHECKING"
        )

    @staticmethod
    def _annotation_contains(annotation: object | None, name: str) -> bool:
        """Return True when ``name`` appears in any sub-node identifier."""
        if annotation is None:
            return False
        for sub in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(annotation):
            if FlextInfraUtilitiesRopeAnalysis.name_of(sub) == name:
                return True
        return False

    def target_name(self, target: object | None) -> str:
        """Return ``target.id`` when ``target`` is an ``ast.Name``-shaped node."""
        if target is None or self._kind(target) != "Name":
            return ""
        identifier = getattr(target, "id", "")
        return identifier if isinstance(identifier, str) else ""

    def call_name(self, func: object | None) -> str:
        """Return the trailing identifier of a call's callable, recursively."""
        if func is None:
            return ""
        kind = self._kind(func)
        if kind == "Name":
            identifier = getattr(func, "id", "")
            return identifier if isinstance(identifier, str) else ""
        if kind == "Attribute":
            attribute = getattr(func, "attr", "")
            return attribute if isinstance(attribute, str) else ""
        if kind == "Call":
            return self.call_name(getattr(func, "func", None))
        return ""

    def _accumulate_violations(
        self, rule_prefix: str, messages: Iterable[str]
    ) -> t.StrSequence:
        """Render messages into ``[<prefix>-<NNN>] <message>`` lines."""
        return [
            f"[{rule_prefix}-{i:03d}] {msg}" for i, msg in enumerate(messages, start=1)
        ]

    @staticmethod
    def _outer_classes(tree: object) -> list[object]:
        """Return top-level ``ClassDef`` nodes in module ``tree``."""
        body = getattr(tree, "body", []) or []
        return [
            node
            for node in body
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) == "ClassDef"
        ]

    def check_rule_0(
        self,
        tree: object,
        filepath: Path,
        prefix: str,
        *,
        is_test_file: bool = False,
        strict_top_level: bool = True,
        strict_single_class: bool = True,
        require_public_class: bool = True,
    ) -> t.StrSequence:
        """Rule 0 — One public class per facade module + project prefix.

        Facade modules (``constants.py``, ``models.py``, ``protocols.py``,
        ``typings.py``, ``utilities.py``) are checked strictly: exactly one
        public class, correct prefix, and only allowed module-level statements.

        Family modules (files inside ``_constants/``, ``_models/``,
        ``_protocols/``, ``_typings/``, ``_utilities/``) and test files are
        validated loosely: every public class must use the right prefix, but
        multiple classes, helper functions, and module-level assignments are
        permitted.
        """
        outer_classes = self._outer_classes(tree)
        public_classes = [
            cls for cls in outer_classes if not getattr(cls, "name", "").startswith("_")
        ]
        messages: list[str] = []
        expected_prefix = f"Tests{prefix}" if is_test_file else prefix
        if require_public_class and not public_classes:
            messages.append(f"{filepath}:1 — No outer class found")
        if strict_single_class and len(public_classes) > 1:
            first_line = getattr(public_classes[0], "lineno", 1)
            names = ", ".join(getattr(cls, "name", "") for cls in public_classes)
            messages.append(
                f"{filepath}:{first_line} — Multiple outer classes found "
                f"(expected 1, got {len(public_classes)}: {names})"
            )
        for cls in public_classes:
            cls_name = getattr(cls, "name", "")
            if expected_prefix and not cls_name.startswith(expected_prefix):
                messages.append(
                    f"{filepath}:{getattr(cls, 'lineno', 0)} — Class "
                    f"{cls_name!r} does not start with prefix {expected_prefix!r}"
                )
        if strict_top_level:
            body = getattr(tree, "body", []) or []
            for node in body:
                if self._kind(node) == "ClassDef":
                    continue
                if not self._is_allowed_module_level(node, filepath):
                    messages.append(
                        f"{filepath}:{getattr(node, 'lineno', 0)} — "
                        f"Disallowed top-level statement: {self._kind(node)}"
                    )
        return self._accumulate_violations("NS-000", messages)

    def check_rule_1(self, tree: object, filepath: Path) -> t.StrSequence:
        """Rule 1 — Constants centralization."""
        if filepath.name == c.Infra.CONSTANTS_PY:
            return self._check_rule_1_canonical(tree, filepath)
        return self._check_rule_1_non_canonical(tree, filepath)

    def _base_text_set(self, cls_node: object) -> set[str]:
        """Render every base of a class as the trailing identifier text."""
        bases = getattr(cls_node, "bases", []) or []
        return {self.call_name(base) for base in bases if self.call_name(base)}

    def _check_rule_1_canonical(self, tree: object, filepath: Path) -> t.StrSequence:
        """Constants module — each class must inherit from a Constants base."""
        messages: list[str] = []
        for cls in self._outer_classes(tree):
            base_strs = self._base_text_set(cls)
            if not any("Constants" in b or b == "c" for b in base_strs):
                messages.append(
                    f"{filepath}:{getattr(cls, 'lineno', 0)} — Constants class "
                    f"{getattr(cls, 'name', '')!r} must inherit from a Constants base"
                )
            messages.extend(
                f"{filepath}:{getattr(inner, 'lineno', 0)} — Method "
                f"{getattr(inner, 'name', '')!r} found in Constants class"
                for inner in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(cls)
                if self._kind(inner) in {"FunctionDef", "AsyncFunctionDef"}
            )
        return self._accumulate_violations("NS-001", messages)

    def _check_rule_1_non_canonical(
        self, tree: object, filepath: Path
    ) -> t.StrSequence:
        """Non-constants module — flag loose Final/collection/Enum constants."""
        messages: list[str] = []
        body = getattr(tree, "body", []) or []
        for node in body:
            messages.extend(self._check_loose_final(node, filepath))
            messages.extend(self._check_loose_collection(node, filepath))
        if filepath.parent.name != "_constants":
            for cls in self._outer_classes(tree):
                messages.extend(self._check_loose_enums(cls, filepath))
        return self._accumulate_violations("NS-001", messages)

    def _check_loose_final(self, node: object, filepath: Path) -> t.StrSequence:
        """Flag bare ``X: Final = ...`` outside ``constants.py``/``_constants/``."""
        if self._kind(node) != "AnnAssign" or not self._annotation_contains(
            getattr(node, "annotation", None), "Final"
        ):
            return []
        target = self.target_name(getattr(node, "target", None))
        if target and not target.startswith("_"):
            return [
                (
                    f"{filepath}:{getattr(node, 'lineno', 0)} — Loose Final "
                    f"constant {target!r} belongs in constants.py"
                )
            ]
        return []

    def _check_loose_collection(self, node: object, filepath: Path) -> t.StrSequence:
        """Flag bare ``X = frozenset({...})``-style declarations outside constants."""
        if self._kind(node) != "Assign":
            return []
        value = getattr(node, "value", None)
        if self._kind(value) != "Call":
            return []
        func_name = self.call_name(getattr(value, "func", None))
        if func_name not in c.Infra.COLLECTION_CALLS:
            return []
        targets = getattr(node, "targets", []) or []
        target_name = self.target_name(targets[0]) if targets else ""
        if (
            target_name
            and target_name not in c.Infra.DUNDER_ALLOWED
            and target_name not in c.Infra.ALIAS_NAMES
        ):
            return [
                (
                    f"{filepath}:{getattr(node, 'lineno', 0)} — Loose collection "
                    f"constant {target_name!r} belongs in constants.py"
                )
            ]
        return []

    def _check_loose_enums(self, cls: object, filepath: Path) -> t.StrSequence:
        """Flag inner ``Enum``-derived classes outside the ``_constants`` tree."""
        messages: list[str] = []
        body = getattr(cls, "body", []) or []
        for inner in body:
            if self._kind(inner) != "ClassDef":
                continue
            inner_bases = self._base_text_set(inner)
            if any(base in c.Infra.ENUM_BASES for base in inner_bases):
                messages.append(
                    f"{filepath}:{getattr(inner, 'lineno', 0)} — Loose Enum "
                    f"{getattr(inner, 'name', '')!r} belongs in constants.py"
                )
        return messages

    def check_rule_2(self, tree: object, filepath: Path) -> t.StrSequence:
        """Rule 2 — Types centralization."""
        if filepath.name == c.Infra.TYPINGS_PY:
            return self._check_rule_2_canonical(tree, filepath)
        if filepath.parent.name == c.Infra.FAMILY_DIRECTORIES["t"]:
            return []
        return self._check_rule_2_non_canonical(tree, filepath)

    def _check_rule_2_canonical(self, tree: object, filepath: Path) -> t.StrSequence:
        """Types module — each class must inherit from a Types base; inner classes guarded."""
        messages: list[str] = []
        for cls in self._outer_classes(tree):
            base_strs = self._base_text_set(cls)
            if not any("Types" in b or b == "t" for b in base_strs):
                messages.append(
                    f"{filepath}:{getattr(cls, 'lineno', 0)} — Types class "
                    f"{getattr(cls, 'name', '')!r} must inherit from a Types base"
                )
            body = getattr(cls, "body", []) or []
            for inner in body:
                if self._kind(inner) != "ClassDef":
                    continue
                inner_bases = self._base_text_set(inner)
                if inner_bases & {"BaseModel", "Protocol"}:
                    messages.append(
                        f"{filepath}:{getattr(inner, 'lineno', 0)} — Inner class "
                        f"{getattr(inner, 'name', '')!r} in Types must not inherit "
                        f"from BaseModel/Protocol"
                    )
        return self._accumulate_violations("NS-002", messages)

    def _check_rule_2_non_canonical(
        self, tree: object, filepath: Path
    ) -> t.StrSequence:
        """Non-typings module — flag loose TypeVar/TypeAlias declarations."""
        messages: list[str] = []
        body = getattr(tree, "body", []) or []
        for node in body:
            messages.extend(self._check_loose_typevar(node, filepath))
            messages.extend(self._check_loose_typealias_annotation(node, filepath))
            messages.extend(self._check_loose_pep695_typealias(node, filepath))
        return self._accumulate_violations("NS-002", messages)

    def _check_loose_typevar(self, node: object, filepath: Path) -> t.StrSequence:
        """Flag bare ``X = TypeVar(...)`` outside typings.py."""
        if self._kind(node) != "Assign":
            return []
        value = getattr(node, "value", None)
        if self._kind(value) != "Call":
            return []
        func_name = self.call_name(getattr(value, "func", None))
        if func_name not in c.Infra.TYPEVAR_CALLABLES:
            return []
        targets = getattr(node, "targets", []) or []
        target_name = self.target_name(targets[0]) if targets else ""
        return [
            (
                f"{filepath}:{getattr(node, 'lineno', 0)} — TypeVar "
                f"{target_name!r} belongs in typings.py"
            )
        ]

    def _check_loose_typealias_annotation(
        self, node: object, filepath: Path
    ) -> t.StrSequence:
        """Flag bare ``X: TypeAlias = ...`` outside typings.py."""
        if self._kind(node) != "AnnAssign" or not self._annotation_contains(
            getattr(node, "annotation", None), "TypeAlias"
        ):
            return []
        target_name = self.target_name(getattr(node, "target", None))
        return [
            (
                f"{filepath}:{getattr(node, 'lineno', 0)} — TypeAlias "
                f"{target_name!r} belongs in typings.py"
            )
        ]

    def _check_loose_pep695_typealias(
        self, node: object, filepath: Path
    ) -> t.StrSequence:
        """Flag bare PEP 695 ``type X = ...`` outside typings.py."""
        if self._kind(node) != "TypeAlias":
            return []
        name_node = getattr(node, "name", None)
        name_str = getattr(name_node, "id", str(name_node)) if name_node else "unknown"
        return [
            (
                f"{filepath}:{getattr(node, 'lineno', 0)} — PEP 695 TypeAlias "
                f"{name_str!r} belongs in typings.py"
            )
        ]

    def check_rule_3(
        self, tree: object, filepath: Path, *, class_stem: str, package_name: str
    ) -> t.StrSequence:
        """Rule 3 — Runtime modules use namespaced FLEXT aliases (c/m/p/t/u)."""
        owner_rules = self._owner_direct_facade_rules(class_stem)
        messages: list[str] = []
        owner = self._facade_owner(filepath)
        is_test_file = "tests" in filepath.parts
        for node, type_checking_only in self._imports_with_context(tree):
            if self._kind(node) != "ImportFrom":
                continue
            module = getattr(node, "module", None)
            if not isinstance(module, str):
                continue
            imported_owner = self._module_owner(module, namespace="tests")
            if (
                is_test_file
                and owner is not None
                and imported_owner is None
                and self._is_test_support_module(module)
            ):
                imported_names = {
                    getattr(alias, "name", "")
                    for alias in getattr(node, "names", []) or []
                }
                owner_assembly = (
                    module == "tests" and imported_names <= self._FACADE_DAG.keys()
                )
                if not owner_assembly:
                    messages.append(
                        f"{filepath}:{getattr(node, 'lineno', 0)} — Test facade must "
                        f"not import test support module {module!r}"
                    )
                    continue
            namespace_module = "tests" if is_test_file else package_name
            if (
                module == namespace_module
                and owner is not None
                and not type_checking_only
            ):
                for alias in getattr(node, "names", []) or []:
                    imported_name = getattr(alias, "name", "")
                    if imported_name in self._FACADE_DAG and (
                        self._FACADE_DAG[imported_name] > self._FACADE_DAG[owner]
                    ):
                        messages.append(
                            f"{filepath}:{getattr(node, 'lineno', 0)} — Invalid runtime "
                            f"namespace import {imported_name!r}: {owner} may only import "
                            "forward through c -> t -> p -> m -> u"
                        )
            elif (
                imported_owner is not None
                and owner is not None
                and not type_checking_only
                and self._FACADE_DAG[imported_owner] > self._FACADE_DAG[owner]
            ):
                messages.append(
                    f"{filepath}:{getattr(node, 'lineno', 0)} — Invalid runtime "
                    f"namespace import from {module!r}: {owner} may only import "
                    "forward through c -> t -> p -> m -> u"
                )
            if module != package_name or self._is_private_dir_file(filepath):
                continue
            for alias in getattr(node, "names", []) or []:
                alias_name = getattr(alias, "name", "")
                if self._allows_direct_facade_import(filepath, alias_name, owner_rules):
                    continue
                if not self._is_local_facade_owner_import(alias_name, owner_rules):
                    continue
                messages.append(
                    f"{filepath}:{getattr(node, 'lineno', 0)} — Runtime module must "
                    f"use namespaced FLEXT aliases (c/m/p/t/u) instead of direct "
                    f"import {alias_name!r}"
                )
        return self._accumulate_violations("NS-003", messages)

    def _imports_with_context(self, tree: object) -> list[tuple[object, bool]]:
        """Return imports paired with whether they are TYPE_CHECKING-only."""
        guarded_import_ids = {
            id(subnode)
            for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(tree)
            if self._is_type_checking_guard(node)
            for subnode in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(node)
            if self._kind(subnode) in {"Import", "ImportFrom"}
        }
        return [
            (node, id(node) in guarded_import_ids)
            for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(tree)
            if self._kind(node) in {"Import", "ImportFrom"}
        ]

    def _facade_owner(self, filepath: Path) -> str | None:
        """Return the owning facade alias for roots and private families."""
        filename_owners: dict[str, str] = {
            c.Infra.CONSTANTS_PY: "c",
            c.Infra.TYPINGS_PY: "t",
            c.Infra.PROTOCOLS_PY: "p",
            c.Infra.MODELS_PY: "m",
            c.Infra.UTILITIES_PY: "u",
        }
        if filepath.name in filename_owners:
            owner: str = t.Infra.STR_ADAPTER.validate_python(
                filename_owners[filepath.name]
            )
            return owner
        for raw_owner, directory in c.Infra.FAMILY_DIRECTORIES.items():
            family_owner: str = t.Infra.STR_ADAPTER.validate_python(raw_owner)
            if directory in filepath.parts and family_owner in self._FACADE_DAG:
                return family_owner
        return None

    @staticmethod
    def _is_test_support_module(module: str) -> bool:
        """Return whether a module belongs to the local test-support tree."""
        return module == "tests" or module.startswith("tests.")

    def _module_owner(self, module: str, *, namespace: str) -> str | None:
        """Return the facade owner encoded by a namespace-private module."""
        for owner, directory in c.Infra.FAMILY_DIRECTORIES.items():
            if module == f"{namespace}.{directory}" or module.startswith(
                f"{namespace}.{directory}."
            ):
                return owner if owner in self._FACADE_DAG else None
        return None

    def _allows_direct_facade_import(
        self,
        filepath: Path,
        imported_name: str,
        owner_rules: tuple[tuple[str, str, str], ...],
    ) -> bool:
        """Return whether the import-site is one of the facade's declared owners."""
        for prefix, facade_filename, private_dir in owner_rules:
            if not imported_name.startswith(prefix):
                continue
            if filepath.name == facade_filename:
                return True
            if private_dir in filepath.parts:
                return True
        return False

    @staticmethod
    def _is_local_facade_owner_import(
        imported_name: str, owner_rules: tuple[tuple[str, str, str], ...]
    ) -> bool:
        """Return whether ``imported_name`` would target a local facade."""
        return any(
            imported_name.startswith(prefix)
            for prefix, _facade_filename, _private_dir in owner_rules
        )

    def _owner_direct_facade_rules(
        self, class_stem: str
    ) -> tuple[tuple[str, str, str], ...]:
        """Materialize the ``(prefix, facade_filename, private_dir)`` owner table."""
        return tuple(
            (f"{class_stem}{suffix}", facade_filename, private_dir)
            for suffix, facade_filename, private_dir in self._DIRECT_FACADE_SUFFIX_RULES
        )

    def _is_allowed_module_level(self, node: object, filepath: Path) -> bool:
        """Return whether ``node`` is permitted at module level."""
        kind = self._kind(node)
        if kind in {"Import", "ImportFrom"}:
            return True
        if self._is_module_docstring(node):
            return True
        if self._is_type_checking_guard(node):
            return True
        if kind in {"Assign", "TypeAlias", "AnnAssign"}:
            return self._is_allowed_assignment(node, filepath, kind)
        return False

    def _is_allowed_assignment(self, node: object, filepath: Path, kind: str) -> bool:
        """Return whether an ``Assign``/``TypeAlias``/``AnnAssign`` is permitted."""
        typings_dir: str = c.Infra.FAMILY_DIRECTORIES["t"]
        if kind == "Assign":
            return self._is_allowed_assign(node, filepath)
        if kind == "TypeAlias":
            return (
                filepath.name == c.Infra.TYPINGS_PY
                or filepath.parent.name == typings_dir
            )
        return self._is_allowed_ann_assign(node, filepath)

    @staticmethod
    def _is_module_docstring(node: object) -> bool:
        """Return whether ``node`` is the module-level docstring expression."""
        if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "Expr":
            return False
        value = getattr(node, "value", None)
        if FlextInfraUtilitiesRopeAnalysis.node_kind(value) != "Constant":
            return False
        return isinstance(getattr(value, "value", None), str)

    def _is_allowed_assign(self, node: object, filepath: Path) -> bool:
        """Return whether a bare ``Assign`` is allowed at module scope."""
        typings_dir: str = c.Infra.FAMILY_DIRECTORIES["t"]
        targets = getattr(node, "targets", []) or []
        for target in targets:
            target_id = self.target_name(target)
            if target_id and target_id in c.Infra.DUNDER_ALLOWED:
                return True
        if len(targets) == 1:
            target_id = self.target_name(targets[0])
            if target_id and target_id in c.Infra.ALIAS_NAMES:
                return True
        value = getattr(node, "value", None)
        if self._kind(value) == "Call":
            func_name = self.call_name(getattr(value, "func", None))
            if func_name in c.Infra.TYPEVAR_CALLABLES:
                return (
                    filepath.name == c.Infra.TYPINGS_PY
                    or filepath.parent.name == typings_dir
                )
        return False

    def _is_allowed_ann_assign(self, node: object, filepath: Path) -> bool:
        """Return whether an ``AnnAssign`` is allowed at module scope."""
        typings_dir: str = c.Infra.FAMILY_DIRECTORIES["t"]
        target_id = self.target_name(getattr(node, "target", None))
        if target_id and target_id in c.Infra.DUNDER_ALLOWED:
            return True
        if target_id and target_id in c.Infra.ALIAS_NAMES:
            return True
        if self._annotation_contains(getattr(node, "annotation", None), "TypeAlias"):
            return (
                filepath.name == c.Infra.TYPINGS_PY
                or filepath.parent.name == typings_dir
            )
        if (
            target_id
            and target_id.startswith("_")
            and self._annotation_contains(getattr(node, "annotation", None), "Final")
        ):
            return (
                filepath.name == c.Infra.CONSTANTS_PY
                or filepath.parent.name == "_constants"
            )
        return False


__all__: list[str] = ["FlextInfraNamespaceRules"]
