"""Refactor/CST helper utilities for infrastructure code analysis.

Centralizes CST (Concrete Syntax Tree) helpers previously defined as
module-level functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import ast
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import flext_infra
import libcst as cst
from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, t
from flext_infra._utilities.io import FlextInfraUtilitiesIo
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml


class FlextInfraUtilitiesRefactor:
    """CST/refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        name = u.Infra.dotted_name(cst_expr)
    """

    @staticmethod
    def capture_output(cmd: Sequence[str]) -> r[str]:
        """Run *cmd* and return stripped stdout as ``r[str]``."""
        return FlextInfraUtilitiesSubprocess().capture(cmd)

    @staticmethod
    def copy_workspace_for_dry_run(workspace: Path) -> Path:
        """Create an ephemeral deep-copy of a workspace for isolated dry-run analysis."""
        temp_root = Path(tempfile.mkdtemp(prefix="flext-refactor-dryrun-"))
        workspace_copy = temp_root / "workspace"

        def _ignore(_dir: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            for name in names:
                if name in {
                    ".git",
                    ".beads",
                    ".venv",
                    ".mypy_cache",
                    ".pytest_cache",
                    ".ruff_cache",
                    "__pycache__",
                }:
                    ignored.add(name)
                    continue
                if name.endswith(".sock"):
                    ignored.add(name)
            return ignored

        _ = shutil.copytree(workspace, workspace_copy, ignore=_ignore)
        source_venv_cfg = workspace / ".venv" / "pyvenv.cfg"
        if source_venv_cfg.exists():
            target_venv = workspace_copy / ".venv"
            target_venv.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(source_venv_cfg, target_venv / "pyvenv.cfg")
        return workspace_copy

    @staticmethod
    def create_refactor_parser(
        prog: str, description: str, *, include_apply: bool = True
    ) -> argparse.ArgumentParser:
        """Create a standard argparse.ArgumentParser with common CLI flags."""
        parser = argparse.ArgumentParser(prog=prog, description=description)
        _ = parser.add_argument(
            "--workspace",
            type=Path,
            default=Path.cwd(),
            help="Workspace root directory (default: cwd)",
        )
        if include_apply:
            mode = parser.add_mutually_exclusive_group(required=False)
            _ = mode.add_argument(
                "--dry-run", action="store_true", help="Plan/Scan only"
            )
            _ = mode.add_argument("--apply", action="store_true", help="Apply changes")
            _ = parser.add_argument(
                "--dry-run-copy-workspace",
                action="store_true",
                help="Run dry-run against a temporary full workspace copy",
            )
        return parser

    @staticmethod
    def resolve_workspace_args(args: argparse.Namespace) -> tuple[Path, bool]:
        """Resolve workspace Path and apply_changes boolean honoring dry-run and workspace copy."""
        workspace_path: Path = args.workspace.resolve()
        apply_changes: bool = bool(getattr(args, "apply", False))
        if getattr(args, "dry_run_copy_workspace", False):
            workspace_path = FlextInfraUtilitiesRefactor.copy_workspace_for_dry_run(
                workspace_path
            )
            apply_changes = False
        return workspace_path, apply_changes

    @staticmethod
    def dotted_name(expr: cst.BaseExpression) -> str:
        """Extract dotted name string from a CST expression.

        Recursively traverses ``Name`` and ``Attribute`` nodes to build
        a fully qualified dotted name (e.g., ``"foo.bar.baz"``).

        Args:
            expr: CST expression node.

        Returns:
            Dotted name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            root = FlextInfraUtilitiesRefactor.dotted_name(expr.value)
            if not root:
                return ""
            return f"{root}.{expr.attr.value}"
        return ""

    @staticmethod
    def root_name(expr: cst.BaseExpression) -> str:
        """Extract the root (leftmost) name from a CST expression.

        Handles ``Name``, ``Attribute``, and ``Call`` nodes to find the
        base identifier.

        Args:
            expr: CST expression node.

        Returns:
            Root name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return FlextInfraUtilitiesRefactor.root_name(expr.value)
        if isinstance(expr, cst.Call):
            return FlextInfraUtilitiesRefactor.root_name(expr.func)
        return ""

    @staticmethod
    def asname_to_local(asname: cst.AsName | None) -> str | None:
        """Extract the local alias name from a CST AsName node.

        Args:
            asname: CST AsName node, or None.

        Returns:
            Local alias string, or None if no alias or unsupported node.

        """
        if asname is None:
            return None
        if isinstance(asname.name, cst.Name):
            return asname.name.value
        return None

    @staticmethod
    def module_path(*, file_path: Path, project_root: Path) -> str:
        """Compute dotted module path relative to a project root.

        Strips the ``src/`` directory component and file extension.

        Args:
            file_path: Absolute path to a Python file.
            project_root: Root directory of the project.

        Returns:
            Dotted module path (e.g., ``"flext_infra.refactor.engine"``).

        """
        rel = file_path.relative_to(project_root)
        parts = [part for part in rel.with_suffix("").parts if part != "src"]
        return ".".join(parts)

    @staticmethod
    def module_family_from_path(path: str) -> str:
        """Resolve module family key from a source file path."""
        normalized = path.replace("\\", "/")
        if "_models" in normalized:
            return "_models"
        if "_utilities" in normalized:
            return "_utilities"
        if "_dispatcher" in normalized:
            return "_dispatcher"
        if "_decorators" in normalized:
            return "_decorators"
        if "_runtime" in normalized:
            return "_runtime"
        return "other_private"

    @staticmethod
    def entry_list(value: t.Infra.InfraValue | None) -> list[dict[str, str]]:
        """Normalize class-nesting config entries to a strict list."""
        if value is None:
            return []
        try:
            return TypeAdapter(list[dict[str, str]]).validate_python(value)
        except ValidationError:
            msg = "class nesting entries must be a list"
            raise ValueError(msg) from None

    @staticmethod
    def string_list(value: t.ContainerValue | None) -> list[str]:
        """Normalize policy fields that should contain string collections."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    msg = "expected list[str] value"
                    raise TypeError(msg)
                items.append(item)
            return items
        msg = "expected list[str] value"
        raise ValueError(msg)

    @staticmethod
    def mapping_list(
        value: t.ContainerValue | None,
    ) -> list[dict[str, t.ContainerValue]]:
        """Normalize policy fields that should contain mapping collections."""
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        msg = "expected list[dict[str, ContainerValue]] value"
        raise ValueError(msg)

    @staticmethod
    def has_required_fields(
        entry: t.ContainerValue,
        required_fields: list[str],
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(key in entry for key in required_fields)

    @staticmethod
    def normalize_module_path(path_value: Path) -> str:
        normalized = path_value.as_posix().replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    @staticmethod
    def project_scope_tokens(path_value: Path) -> set[str]:
        normalized = path_value.as_posix().replace("\\", "/")
        parts = Path(normalized).parts
        if not parts:
            return set()
        tokens: set[str] = set()
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            if src_index > 0:
                tokens.add(parts[src_index - 1])
            if src_index + 1 < len(parts):
                tokens.add(parts[src_index + 1])
        return tokens

    @staticmethod
    def rewrite_scope(entry: t.Infra.StrMap) -> str:
        raw_scope = entry.get(c.Infra.ReportKeys.REWRITE_SCOPE, c.Infra.ReportKeys.FILE)
        scope = raw_scope.strip().lower()
        if scope in {
            c.Infra.ReportKeys.FILE,
            c.Infra.Toml.PROJECT,
            c.Infra.ReportKeys.WORKSPACE,
        }:
            return scope
        msg = f"unsupported rewrite_scope: {raw_scope}"
        raise ValueError(msg)

    @staticmethod
    def scope_applies_to_file(
        entry: t.Infra.StrMap,
        current_file: Path,
        candidate_file: Path,
    ) -> bool:
        rewrite_scope = FlextInfraUtilitiesRefactor.rewrite_scope(entry)
        if rewrite_scope == c.Infra.ReportKeys.WORKSPACE:
            return True
        current_module = FlextInfraUtilitiesRefactor.normalize_module_path(current_file)
        candidate_module = FlextInfraUtilitiesRefactor.normalize_module_path(
            candidate_file
        )
        if rewrite_scope == c.Infra.ReportKeys.FILE:
            return current_module == candidate_module
        current_tokens = FlextInfraUtilitiesRefactor.project_scope_tokens(current_file)
        candidate_tokens = FlextInfraUtilitiesRefactor.project_scope_tokens(
            candidate_file
        )
        if current_tokens and candidate_tokens:
            return bool(current_tokens & candidate_tokens)
        return current_module == candidate_module

    @staticmethod
    def policy_document_schema_valid(
        loaded: dict[str, t.ContainerValue],
        schema_path: Path,
    ) -> bool:
        schema_result = FlextInfraUtilitiesIo.read_json(schema_path)
        if schema_result.is_failure:
            return False
        raw_schema: Mapping[str, t.ContainerValue] = schema_result.value
        schema: dict[str, t.ContainerValue] = dict(raw_schema.items())
        top_required = FlextInfraUtilitiesRefactor.string_list(
            schema.get("required", [])
        )
        if not FlextInfraUtilitiesRefactor.has_required_fields(loaded, top_required):
            return False
        definitions_raw = schema.get("definitions", {})
        if not isinstance(definitions_raw, dict):
            return False
        policy_entry_raw = definitions_raw.get("policyEntry", {})
        class_rule_raw = definitions_raw.get("classRule", {})
        if not isinstance(policy_entry_raw, dict):
            return False
        if not isinstance(class_rule_raw, dict):
            return False
        policy_entry_required = FlextInfraUtilitiesRefactor.string_list(
            policy_entry_raw.get("required", []),
        )
        class_rule_required = FlextInfraUtilitiesRefactor.string_list(
            class_rule_raw.get("required", []),
        )
        for entry in FlextInfraUtilitiesRefactor.mapping_list(
            loaded.get("policy_matrix")
        ):
            if not FlextInfraUtilitiesRefactor.has_required_fields(
                entry,
                policy_entry_required,
            ):
                return False
        for rule in FlextInfraUtilitiesRefactor.mapping_list(
            loaded.get(c.Infra.ReportKeys.RULES),
        ):
            if not FlextInfraUtilitiesRefactor.has_required_fields(
                rule, class_rule_required
            ):
                return False
        return True

    @staticmethod
    def load_validated_policy_document(policy_path: Path) -> t.Infra.ContainerDict:
        try:
            loaded = FlextInfraUtilitiesYaml.safe_load_yaml(policy_path)
        except (OSError, TypeError) as exc:
            msg = f"failed to read policy document: {policy_path}"
            raise ValueError(msg) from exc
        loaded_dict = dict(loaded.items())
        schema_path = policy_path.with_name("class-policy-v2.schema.json")
        if not FlextInfraUtilitiesRefactor.policy_document_schema_valid(
            loaded_dict,
            schema_path,
        ):
            msg = "policy document failed schema validation"
            raise ValueError(msg)
        return loaded_dict

    @staticmethod
    def _is_docstring_statement(stmt: cst.CSTNode) -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        if len(stmt.body) != 1:
            return False
        expr = stmt.body[0]
        return isinstance(expr, cst.Expr) and isinstance(
            expr.value,
            (cst.SimpleString, cst.ConcatenatedString),
        )

    @staticmethod
    def _is_import_statement(stmt: cst.CSTNode) -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        return any(isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body)

    @staticmethod
    def _is_future_import_statement(stmt: cst.CSTNode) -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        for small in stmt.body:
            if not isinstance(small, cst.ImportFrom):
                continue
            module = small.module
            if isinstance(module, cst.Name) and module.value == "__future__":
                return True
        return False

    @staticmethod
    def insert_import_statement(source: str, import_stmt: str) -> str:
        normalized_import = import_stmt.strip()
        if not normalized_import:
            return source
        try:
            module = cst.parse_module(source)
            parsed_stmt = cst.parse_statement(f"{normalized_import}\n")
        except cst.ParserSyntaxError:
            return source
        if not isinstance(parsed_stmt, cst.SimpleStatementLine):
            return source
        for stmt in module.body:
            if not isinstance(stmt, cst.SimpleStatementLine):
                continue
            if cst.Module(body=[stmt]).code.strip() == normalized_import:
                return source
        insert_idx = 0
        for idx, stmt in enumerate(module.body):
            if (
                FlextInfraUtilitiesRefactor._is_docstring_statement(stmt)
                or FlextInfraUtilitiesRefactor._is_future_import_statement(stmt)
                or FlextInfraUtilitiesRefactor._is_import_statement(stmt)
            ):
                insert_idx = idx + 1
                continue
            break
        new_body = [*module.body[:insert_idx], parsed_stmt, *module.body[insert_idx:]]
        return module.with_changes(body=new_body).code

    # ── Generic AST introspection ─────────────────────────────────────

    @staticmethod
    def extract_public_methods_from_dir(
        package_dir: Path,
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Extract public methods from all .py files in a package directory.

        Returns:
            ``{class_name: [(method_name, method_type, source_file), ...]}``.

        """
        result: dict[str, list[tuple[str, str, str]]] = {}
        for py_file in sorted(package_dir.glob(c.Infra.Extensions.PYTHON_GLOB)):
            if py_file.name == c.Infra.Files.INIT_PY:
                continue
            result.update(
                FlextInfraUtilitiesRefactor._extract_classes_ast(py_file),
            )
        return result

    @staticmethod
    def extract_public_methods_from_file(
        file_path: Path,
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Extract public methods from a single .py file.

        Returns:
            ``{class_name: [(method_name, method_type, source_file), ...]}``.

        """
        if not file_path.exists():
            return {}
        return FlextInfraUtilitiesRefactor._extract_classes_ast(file_path)

    @staticmethod
    def _extract_classes_ast(
        py_file: Path,
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Internal: extract all public methods from classes using stdlib ast."""
        try:
            source = py_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}
        result: dict[str, list[tuple[str, str, str]]] = {}
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods: list[tuple[str, str, str]] = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    decs = [
                        d.id
                        if isinstance(d, ast.Name)
                        else (d.attr if isinstance(d, ast.Attribute) else "")
                        for d in item.decorator_list
                    ]
                    mtype = (
                        "static"
                        if "staticmethod" in decs
                        else "class"
                        if "classmethod" in decs
                        else "instance"
                    )
                    entry = (item.name, mtype, py_file.name)
                    if not any(e[0] == item.name for e in methods):
                        methods.append(entry)
                elif isinstance(item, ast.ClassDef) and not item.name.startswith("_"):
                    for inner in ast.iter_child_nodes(item):
                        if isinstance(
                            inner, ast.FunctionDef
                        ) and not inner.name.startswith("_"):
                            entry = (
                                f"{item.name}.{inner.name}",
                                "static",
                                py_file.name,
                            )
                            if not any(e[0] == entry[0] for e in methods):
                                methods.append(entry)
            if methods:
                result[node.name] = methods
        return result

    @staticmethod
    def build_facade_alias_map(
        facade_path: Path,
        facade_class_name: str,
    ) -> dict[str, tuple[str, str]]:
        """Parse a facade class to build flat alias → (class, method) map.

        Inspects ``staticmethod(...)`` assignments in the facade class.
        """
        try:
            source = facade_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}

        alias_map: dict[str, tuple[str, str]] = {}
        for node in ast.iter_child_nodes(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == facade_class_name):
                continue
            for item in ast.iter_child_nodes(node):
                if not isinstance(item, ast.Assign):
                    continue
                for target in item.targets:
                    if not isinstance(target, ast.Name) or not isinstance(
                        item.value, ast.Call
                    ):
                        continue
                    call = item.value
                    if not (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "staticmethod"
                        and call.args
                    ):
                        continue
                    arg = call.args[0]
                    if isinstance(arg, ast.Attribute):
                        if isinstance(arg.value, ast.Name):
                            alias_map[target.id] = (arg.value.id, arg.attr)
                        elif isinstance(arg.value, ast.Attribute) and isinstance(
                            arg.value.value, ast.Name
                        ):
                            alias_map[target.id] = (
                                arg.value.value.id,
                                f"{arg.value.attr}.{arg.attr}",
                            )
        return alias_map

    @staticmethod
    def build_facade_inner_class_map(
        facade_path: Path,
        facade_class_name: str,
    ) -> dict[str, str]:
        """Map inner class names → base class names in a facade.

        E.g. ``{"Conversion": "FlextUtilitiesConversion", ...}``.
        """
        try:
            source = facade_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}

        name_map: dict[str, str] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == facade_class_name:
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, ast.ClassDef):
                        for base in item.bases:
                            if isinstance(base, ast.Name):
                                name_map[item.name] = base.id
        return name_map

    @staticmethod
    def identify_project_by_roots(
        file_path: Path,
        project_roots: list[Path],
    ) -> str:
        """Identify project name for a file path (most-specific root wins)."""
        best: Path | None = None
        for root in project_roots:
            try:
                file_path.relative_to(root)
            except ValueError:
                continue
            if best is None or len(root.parts) > len(best.parts):
                best = root
        return best.name if best else c.Infra.Defaults.UNKNOWN

    @staticmethod
    def build_mro_target(
        family: str,
        core_project: str = c.Infra.Refactor.Census.CORE_PROJECT,
    ) -> m.Infra.Refactor.MROFamilyTarget:
        """Create a generic target config object from a family code."""
        if family not in c.Infra.Refactor.MRO_FAMILIES:
            msg = f"Invalid MRO family {family}"
            raise ValueError(msg)
        sf = c.Infra.Refactor.FAMILY_SUFFIXES[family]
        return m.Infra.Refactor.MROFamilyTarget(
            family=family,
            class_suffix=sf,
            package_dir=c.Infra.Refactor.MRO_FAMILY_PACKAGE_DIRS[family],
            facade_module=c.Infra.Refactor.MRO_FAMILY_FACADE_MODULES[family],
            facade_class_prefix=f"Flext{sf}",
            core_project=core_project,
        )

    @staticmethod
    def aggregate_usage_metrics(
        methods: dict[str, list[m.Infra.Refactor.CensusMethodInfo]],
        records: list[m.Infra.Refactor.CensusUsageRecord],
        files_scanned: int,
        parse_errors: int,
    ) -> m.Infra.Refactor.CensusReport:
        """Pivot raw AST method visit occurrences into a structured usage report."""
        cnt: Counter[tuple[str, str, str]] = Counter()
        pcnt: Counter[tuple[str, str, str, str]] = Counter()

        for rec in records:
            cnt[rec.class_name, rec.method_name, rec.access_mode] += 1
            pcnt[rec.project, rec.class_name, rec.method_name, rec.access_mode] += 1

        cls_sums: list[m.Infra.Refactor.CensusClassSummary] = []
        unused = 0
        for cls, items in sorted(methods.items()):
            m_list: list[m.Infra.Refactor.CensusMethodSummary] = []
            for m_info in items:
                af = cnt.get(
                    (cls, m_info.name, c.Infra.Refactor.Census.MODE_ALIAS_FLAT), 0
                )
                an = cnt.get(
                    (cls, m_info.name, c.Infra.Refactor.Census.MODE_ALIAS_NS), 0
                )
                dr = cnt.get((cls, m_info.name, c.Infra.Refactor.Census.MODE_DIRECT), 0)
                tot = af + an + dr
                if tot == 0:
                    unused += 1
                m_list.append(
                    m.Infra.Refactor.CensusMethodSummary(
                        name=m_info.name,
                        method_type=m_info.method_type,
                        alias_flat=af,
                        alias_namespaced=an,
                        direct=dr,
                        total=tot,
                    )
                )
            cls_sums.append(
                m.Infra.Refactor.CensusClassSummary(
                    class_name=cls,
                    source_file=items[0].source_file if items else "",
                    methods=m_list,
                )
            )

        pj_sums: dict[str, list[m.Infra.Refactor.CensusProjectMethodUsage]] = (
            defaultdict(list)
        )
        for (pj, cls, mx, mo), co in sorted(pcnt.items()):
            pj_sums[pj].append(
                m.Infra.Refactor.CensusProjectMethodUsage(
                    class_name=cls,
                    method_name=mx,
                    access_mode=mo,
                    count=co,
                )
            )

        return m.Infra.Refactor.CensusReport(
            classes=cls_sums,
            projects=[
                m.Infra.Refactor.CensusProjectSummary(
                    project_name=p, usages=us, total=sum(u.count for u in us)
                )
                for p, us in sorted(pj_sums.items())
            ],
            total_classes=len(methods),
            total_methods=sum(len(v) for v in methods.values()),
            total_usages=len(records),
            total_unused=unused,
            files_scanned=files_scanned,
            parse_errors=parse_errors,
        )

    @staticmethod
    def export_pydantic_json(model_payload: BaseModel, export_path: Path) -> None:
        """Serialize any Pydantic model payload to a JSON file."""
        # Fallback to pure path string write since model_dump_json takes care of formatting
        export_path.write_text(
            model_payload.model_dump_json(indent=2),
            encoding=c.Infra.Encoding.DEFAULT,
        )

    @staticmethod
    def scan_cst_with_visitors(
        file_path: Path,
        *visitors: cst.CSTVisitor,
    ) -> cst.Module | None:
        """Parse CST and sequentially apply an arbitrary number of visitors."""
        tree = flext_infra.u.Infra.parse_module_cst(file_path)
        if not tree:
            return None
        for visitor in visitors:
            tree.visit(visitor)
        return tree


__all__ = ["FlextInfraUtilitiesRefactor"]
